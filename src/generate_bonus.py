"""Generate piano accompaniment from the supplied melody, never GT piano.

By default the quantized MELODY is forced as context before sampling PIANO.
The empty-melody ablation removes melody note conditioning for analysis only.
Both modes export the exact original melody note and control events, together
with piano sampled from our own randomly initialized/trained roles Transformer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pretty_midi
import torch

from infer import CachedDecoder, sample_token, portable_path
from model import load_checkpoint
from music_codec import (Vocabulary, tokenize_midi, parse_tokens, decode_tokens,
                         restore_midi_endpoint)


def melody_condition_bars(encoded, vocab, bars=16, conditioning="normal"):
    """Keep bar/track headers and optionally remove melody note conditioning."""
    if conditioning not in ("normal", "empty-melody"):
        raise ValueError("Conditioning must be normal or empty-melody")
    result = []
    piano_marker = vocab.id("Track_Piano")
    melody_marker = vocab.id("Track_Melody")
    for bar in range(bars):
        tokens = encoded.tokens[encoded.bar_offsets[bar]:encoded.bar_offsets[bar + 1]]
        end = tokens.index(piano_marker)
        if conditioning == "empty-melody":
            start = tokens.index(melody_marker)
            if start >= end:
                raise ValueError("Expected melody events before the piano track marker")
            result.append(tokens[:start + 1] + [piano_marker])
        else:
            result.append(tokens[:end + 1])
    return result


def generate_accompaniment(model, metadata, encoded, vocab, seed, temperature=1., top_k=0,
                          top_p=.95, bars=16, device="cpu", max_notes_per_bar=64, min_notes_per_bar=1,
                          conditioning="normal"):
    rng = np.random.default_rng(seed)
    decoder = CachedDecoder(model, device, metadata.get("inference_context"))
    tokens = [vocab.bos_id]
    decoder.feed(tokens)
    positions0, pitch0, dur0 = [vocab.ranges[key][0] for key in ("Pos", "Pitch", "Dur")]
    active_until = {}
    total_beats = encoded.beat_map.bar_indices[bars]
    counters = {"forced_bar_limit_count": 0, "infeasible_min_notes_count": 0}
    for bar, header in enumerate(melody_condition_bars(encoded, vocab, bars, conditioning)):
        tokens.extend(header)
        decoder.feed(header)
        meter = encoded.beat_map.bar_beats[bar]
        start = encoded.beat_map.bar_indices[bar]
        count, last_pos, last_pitch = 0, 0, -1
        while True:
            choices = {}
            for pos in range(last_pos, meter * 8):
                lower_pitch = last_pitch + 1 if pos == last_pos else 21
                pitches = [pitch for pitch in range(max(21, lower_pitch), 109)
                           if active_until.get(pitch, -1.) <= start + pos / 8 + 1e-9]
                if pitches:
                    choices[pos] = pitches
            allowed = [positions0 + pos for pos in choices]
            if count >= min_notes_per_bar:
                allowed.append(vocab.bar_id)
            if count >= max_notes_per_bar:
                allowed = [vocab.bar_id]
                counters["forced_bar_limit_count"] += 1
            if not allowed:
                allowed = [vocab.bar_id]
                counters["infeasible_min_notes_count"] += 1
            event = sample_token(decoder.logits, allowed, rng, temperature, top_k, top_p)
            if event == vocab.bar_id:
                break
            position = event - positions0
            tokens.append(event)
            decoder.feed([event])
            pitch = sample_token(decoder.logits, [pitch0 + p for p in choices[position]], rng, temperature, top_k, top_p)
            tokens.append(pitch)
            decoder.feed([pitch])
            velocity = sample_token(decoder.logits, vocab.group_ids("Vel"), rng, temperature, top_k, top_p)
            tokens.append(velocity)
            decoder.feed([velocity])
            remaining = int(round((total_beats - start) * 8 - position))
            duration = sample_token(decoder.logits, list(range(dur0, dur0 + min(256, remaining))),
                                    rng, temperature, top_k, top_p)
            tokens.append(duration)
            decoder.feed([duration])
            active_until[pitch - pitch0] = start + (position + duration - dur0 + 1) / 8
            last_pos, last_pitch, count = position, pitch - pitch0, count + 1
        print(json.dumps({"bar": bar + 1, "piano_notes": count, "tokens": len(tokens)}), flush=True)
    tokens.append(vocab.eos_id)
    parsed = parse_tokens(tokens, vocab)
    piano = [note for note in parsed["notes"] if note["track"] == "Piano"]
    if len(parsed["bars"]) != bars or parsed["invalid_events"] or not piano:
        raise ValueError("Invalid or empty accompaniment; no template fallback is permitted")
    counters.update(seed=seed, n_bars=bars, generated_piano_notes=len(piano), n_tokens=len(tokens),
                    context_reprefills=decoder.reprefills, invalid_events=parsed["invalid_events"],
                    conditioning=conditioning,
                    min_notes_per_bar=min_notes_per_bar, max_notes_per_bar=max_notes_per_bar,
                    pitch_range=[21, 108], same_pitch_overlap_policy="Forbid a new PIANO onset while its previous note is active")
    return tokens, counters


def _piano_export_tokens(tokens, vocab):
    """Discard quantized melody before decoding to keep one PIANO instrument."""
    parsed = parse_tokens(tokens, vocab)
    result = [vocab.bos_id]
    for b, bar in enumerate(parsed["bars"]):
        result.extend([vocab.bar_id, vocab.id(f"Meter_{bar['meter']}"), vocab.id(f"Tempo_{int(bar['tempo'])}")])
        for note in parsed["notes"]:
            if note["bar"] == b and note["track"] == "Piano":
                result.extend([vocab.id(f"Pos_{note['pos']}"), vocab.id(f"Pitch_{note['pitch']}"),
                               vocab.id(f"Vel_{note['velocity'] // 4}"), vocab.id(f"Dur_{note['duration']}")])
    result.append(vocab.eos_id)
    return result


def export_with_original_melody(tokens, melody_path, beat_map, output_path, vocab):
    decoded = decode_tokens(_piano_export_tokens(tokens, vocab), output_path, vocab,
                            beat_map=beat_map, clip_at_end=True)
    generated = pretty_midi.PrettyMIDI(str(output_path))
    original = pretty_midi.PrettyMIDI(str(melody_path))
    melody_tracks = [copy.deepcopy(inst) for inst in original.instruments if inst.name.strip().upper() == "MELODY"]
    if len(melody_tracks) != 1 or len(generated.instruments) != 1:
        raise ValueError("Official bonus export requires one original MELODY and one generated PIANO instrument")
    generated.instruments[0].name = "PIANO"
    generated.instruments = melody_tracks + generated.instruments
    generated.write(str(output_path))
    restore_midi_endpoint(output_path, decoded["duration_sec"])
    readback = pretty_midi.PrettyMIDI(str(output_path))
    source = sorted((n.pitch, n.velocity, n.start, n.end) for n in melody_tracks[0].notes)
    actual = sorted((n.pitch, n.velocity, n.start, n.end) for n in readback.instruments[0].notes)
    if len(source) != len(actual) or any(a[:2] != b[:2] for a, b in zip(source, actual)):
        raise ValueError("MIDI export changed the original melody")
    error = max((max(abs(a[2] - b[2]), abs(a[3] - b[3])) for a, b in zip(source, actual)), default=0.)
    if error > .002:
        raise ValueError(f"Original melody timing error exceeds 2 ms: {error}")
    return decoded, {"verified": True, "notes": len(source), "max_timing_error_seconds": error,
                     "source_instrument_name": melody_tracks[0].name,
                     "control_change_count": len(melody_tracks[0].control_changes),
                     "pitch_bend_count": len(melody_tracks[0].pitch_bends)}


def generate_bonus(args):
    torch.set_num_threads(1 if args.device == 'cpu' else 4)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable to this process")
    if not args.temperature > 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("Invalid sampling configuration")
    minimum, maximum = getattr(args, "min_notes_per_bar", 1), getattr(args, "max_notes_per_bar", 64)
    if not 0 <= minimum <= maximum or maximum < 1:
        raise ValueError("Require 0 <= minimum notes <= maximum notes, maximum >= 1")
    conditioning = getattr(args, "conditioning", "normal")
    if conditioning not in ("normal", "empty-melody"):
        raise ValueError("Conditioning must be normal or empty-melody")
    vocab = Vocabulary()
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    if metadata["vocabulary_sha256"] != vocab.sha256 or metadata.get("mode") != "roles":
        raise ValueError("Bonus needs a roles checkpoint with the current vocabulary")
    prompt_dir = getattr(args, "prompt_dir", None) or args.data_root / "prompts/bonus"
    jobs = sorted(folder for folder in prompt_dir.iterdir() if folder.is_dir())
    if len(jobs) != 10:
        raise ValueError("Expected all ten official bonus prompts")
    # infer.py defaults to 32 for basic tasks; accompaniment is always 16 bars.
    bars = 16
    if getattr(args, "smoke_bars", None) is not None:
        bars = args.smoke_bars
        jobs = jobs[:1]
    if not 1 <= bars <= 16:
        raise ValueError("Bonus length must be between 1 and 16 bars")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started, samples = time.monotonic(), []
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    config = {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
              "seed": args.seed, "device": args.device, "bars": bars, "conditioning": conditioning}
    for index, folder in enumerate(jobs):
        # Never open piano.mid or chord_midi.txt here: those are evaluation GT.
        melody_path, beats_path = folder / "melody.mid", folder / "beat_midi.txt"
        encoded = tokenize_midi(melody_path, beats_path, mode="roles", max_bars=16, vocab=vocab)
        if encoded.metadata["n_bars"] != 16:
            raise ValueError(f"Expected a 16-bar melody prompt: {folder}")
        tokens, diagnostics = generate_accompaniment(model, metadata, encoded, vocab, args.seed + index,
                                                    args.temperature, args.top_k, args.top_p, bars, args.device,
                                                    getattr(args, "max_notes_per_bar", 64),
                                                    getattr(args, "min_notes_per_bar", 1),
                                                    conditioning=conditioning)
        output_path, wav_path = args.output_dir / f"{folder.name}.mid", args.output_dir / f"{folder.name}.wav"
        if bars != 16:
            # Smoke exports are deliberately incomplete and cannot be submitted.
            decoded = decode_tokens(_piano_export_tokens(tokens, vocab), output_path, vocab,
                                    beat_map=encoded.beat_map, clip_at_end=True)
            melody_check = {"verified": False, "reason": "Explicit incomplete smoke test"}
        else:
            decoded, melody_check = export_with_original_melody(tokens, melody_path, encoded.beat_map, output_path, vocab)
        record = {"song_name": folder.name, "tokens": [vocab.tokens[token] for token in tokens],
                  "midi_path": portable_path(output_path), "wav_path": portable_path(wav_path) if args.render else None,
                  "bars": bars, "bar_times": decoded["bar_times"], "bar_beats": decoded["bar_beats"],
                  "diagnostics": diagnostics, "melody_verification": melody_check,
                  "input_melody_sha256": hashlib.sha256(melody_path.read_bytes()).hexdigest(),
                  "input_beats_sha256": hashlib.sha256(beats_path.read_bytes()).hexdigest(),
                  "midi_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()}
        if args.render:
            from render_audio import render_midi
            record["audio_audit"] = render_midi(output_path, wav_path)
        samples.append(record)
        manifest = {"task": "accompaniment", "checkpoint": portable_path(args.checkpoint),
                    "checkpoint_sha256": checkpoint_sha, "vocabulary_sha256": vocab.sha256,
                    "checkpoint_training_mode": metadata.get("mode"),
                    "all_official_training_songs_used": metadata.get("all_official_training_songs_used", False),
                    "checkpoint_seen_training_songs": len(metadata.get("seen_song_ids", [])),
                    "configuration": config, "samples": samples,
                    "conditioning": conditioning,
                    "conditioning_policy": (
                        "Original melody.mid and beat_midi.txt only; GT piano and chords never loaded by generator"
                        if conditioning == "normal" else
                        "Original bar/meter/tempo and track markers only; all melody note tokens removed from model context. "
                        "The full original melody is restored only in MIDI export; GT piano and chords never loaded."),
                    "interpretation_limitations": (
                        "Empty melody is an out-of-distribution condition. Differences can reflect distribution shift and "
                        "changed context length as well as melody dependence; improvement is not assumed."
                        if conditioning != "normal" else None),
                    "complete": len(samples) == len(jobs),
                    "submission_eligible": (bars == 16 and len(samples) == 10
                                            and conditioning == "normal"
                                            and metadata.get("all_official_training_songs_used", False)
                                            and len(metadata.get("seen_song_ids", [])) == 896),
                    "elapsed_seconds": time.monotonic() - started}
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"generated": folder.name, "piano_notes": diagnostics["generated_piano_notes"],
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, default=Path(__file__).resolve().parents[1] / "POP909/prompts/bonus")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--max-notes-per-bar", type=int, default=64)
    parser.add_argument("--min-notes-per-bar", type=int, default=1)
    parser.add_argument("--conditioning", choices=["normal", "empty-melody"], default="normal",
                        help="Empty-melody removes input melody notes for an analysis-only ablation")
    parser.add_argument("--smoke-bars", type=int, help="Explicit incomplete one-prompt test; never a submission")
    parser.add_argument("--render", action="store_true")
    generate_bonus(parser.parse_args())


if __name__ == "__main__":
    main()
