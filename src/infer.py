"""Generate genuine model samples and preserve supplied continuation prefixes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from model import load_checkpoint
from music_codec import (Vocabulary, BeatMap, tokenize_midi, parse_tokens,
                         decode_tokens, preserve_prompt_notes)


def portable_path(path):
    try:
        return str(Path(path).resolve().relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(path)


def sample_token(logits, allowed, rng, temperature=1., top_k=0, top_p=0.95):
    allowed = np.asarray(allowed, dtype=np.int64)
    if not len(allowed):
        raise ValueError('No valid next token')
    values = np.asarray(logits, dtype=np.float64)[allowed] / temperature
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite model scores')
    order = np.argsort(-values, kind='stable')
    if top_k:
        order = order[:min(top_k, len(order))]
    probabilities = np.exp(values[order] - values[order[0]])
    probabilities /= probabilities.sum()
    if top_p < 1:
        retain = max(1, int(np.searchsorted(np.cumsum(probabilities), top_p)) + 1)
        order, probabilities = order[:retain], probabilities[:retain]
        probabilities /= probabilities.sum()
    return int(allowed[order[int(rng.choice(len(order), p=probabilities))]])


class CachedDecoder:
    def __init__(self, model, device, context=None):
        self.model, self.device = model, device
        self.context = min(context or model.config.context, model.config.context)
        self.history, self.cache, self.logits = [], None, None
        self.reprefills = 0

    @torch.inference_mode()
    def feed(self, new_tokens):
        if not new_tokens:
            return self.logits
        self.history.extend(int(x) for x in new_tokens)
        current_length = 0 if self.cache is None else self.cache[0][0].shape[2]
        if current_length + len(new_tokens) > self.context:
            retained = self.history[-(self.context - 1):]
            try:
                start = retained.index(3)  # Bar ID is part of the fixed vocabulary.
            except ValueError:
                raise ValueError('No complete bar fits in the context window')
            new_tokens = [1] + retained[start:]
            self.cache = None
            self.reprefills += 1
        ids = torch.tensor([new_tokens], dtype=torch.long, device=self.device)
        scores, _, self.cache = self.model(ids, past=self.cache, use_cache=True)
        self.logits = scores[0, -1].float().cpu().numpy()
        return self.logits


def generate_sequence(model, metadata, vocab, seed, temperature=1., top_k=0, top_p=0.95,
                      bars=32, prefix=None, meter=4, tempo=120, device='cpu', max_notes_per_bar=64):
    rng = np.random.default_rng(seed)
    decoder = CachedDecoder(model, device, metadata.get('inference_context'))
    tokens = [vocab.bos_id] if prefix is None else [int(t) for t in prefix if t != vocab.eos_id]
    completed = tokens.count(vocab.bar_id)
    decoder.feed(tokens)
    group = {name: vocab.group_ids(name) for name in ('Pos', 'Pitch', 'Vel', 'Dur')}
    pitch0, pos0, dur0 = [vocab.ranges[x][0] for x in ('Pitch', 'Pos', 'Dur')]
    tempo = int(min(240, max(8, 2 * round(tempo / 2))))
    diagnostics = {'seed': seed, 'temperature': temperature, 'top_k': top_k, 'top_p': top_p,
                   'initial_bars': completed, 'forced_bar_limit_count': 0,
                   'fixed_generated_meter': meter, 'fixed_generated_tempo': tempo}
    for bar in range(completed, bars):
        header = [vocab.bar_id, vocab.id(f'Meter_{meter}'), vocab.id(f'Tempo_{tempo}')]
        tokens.extend(header); decoder.feed(header)
        last_position, last_pitch, count = 0, -1, 0
        while True:
            positions = list(range(last_position, meter * 8))
            if last_pitch >= 108 and positions and positions[0] == last_position:
                positions.pop(0)
            allowed = [vocab.bar_id] + [pos0 + p for p in positions]
            if count >= max_notes_per_bar:
                allowed = [vocab.bar_id]; diagnostics['forced_bar_limit_count'] += 1
            event = sample_token(decoder.logits, allowed, rng, temperature, top_k, top_p)
            if event == vocab.bar_id:
                # Bar denotes the end of this bar; next iteration feeds it once.
                break
            position = event - pos0
            tokens.append(event); decoder.feed([event])
            minimum_pitch = last_pitch + 1 if position == last_position else 21
            pitches = list(range(max(21, minimum_pitch), 109))
            pitch = sample_token(decoder.logits, [pitch0 + p for p in pitches], rng, temperature, top_k, top_p)
            tokens.append(pitch); decoder.feed([pitch])
            velocity = sample_token(decoder.logits, group['Vel'], rng, temperature, top_k, top_p)
            tokens.append(velocity); decoder.feed([velocity])
            remaining_steps = (bars - bar) * meter * 8 - position
            allowed_durations = list(range(dur0, dur0 + min(256, remaining_steps)))
            duration = sample_token(decoder.logits, allowed_durations, rng, temperature, top_k, top_p)
            tokens.append(duration); decoder.feed([duration])
            last_position, last_pitch, count = position, pitch - pitch0, count + 1
    tokens.append(vocab.eos_id)
    parsed = parse_tokens(tokens, vocab)
    if len(parsed['bars']) != bars or parsed['invalid_events']:
        raise ValueError('Invalid bar count or malformed generation')
    if not parsed['notes']:
        raise ValueError('The model generated an empty piece; no silent fallback is permitted')
    generated_notes = [note for note in parsed['notes'] if note['bar'] >= completed]
    if not generated_notes:
        raise ValueError('The continuation contains no new notes; prompt notes do not count as generation')
    diagnostics.update(n_bars=bars, n_notes=len(parsed['notes']), invalid_events=parsed['invalid_events'],
                       generated_note_count=len(generated_notes),
                       context_reprefills=decoder.reprefills, n_tokens=len(tokens))
    return tokens, diagnostics


def extended_clock(source, total_bars):
    times, boundaries = source.beat_times.copy(), source.bar_indices.copy()
    meter = source.bar_beats[-1]
    delta = float(np.median(np.diff(source.beat_times[-min(9, len(source.beat_times)):])) )
    while len(boundaries) - 1 < total_bars:
        boundary = boundaries[-1] + meter
        while len(times) <= boundary:
            times.append(times[-1] + delta)
        boundaries.append(boundary)
    return BeatMap(times, boundaries)


def verify_prefix(prompt_path, output_path, cutoff):
    import pretty_midi
    def notes(path):
        m = pretty_midi.PrettyMIDI(str(path))
        return sorted((n.pitch, n.velocity, n.start, n.end) for inst in m.instruments for n in inst.notes
                      if n.start < cutoff - 1e-6)
    wanted, actual = notes(prompt_path), notes(output_path)
    if len(wanted) != len(actual):
        raise ValueError('Continuation changed prompt note count')
    max_error = 0.
    for a, b in zip(wanted, actual):
        if a[:2] != b[:2]:
            raise ValueError('Continuation changed prompt pitch or velocity')
        max_error = max(max_error, abs(a[2] - b[2]), abs(a[3] - b[3]))
    if max_error > 0.002:
        raise ValueError(f'Continuation changed prefix timing: {max_error}')
    return {'verified': True, 'prefix_notes': len(wanted), 'max_timing_error_seconds': max_error}


def run(args):
    torch.set_num_threads(1 if args.device == 'cpu' else 4)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable to this process')
    if not 0 < args.temperature or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError('Invalid sampling configuration')
    if args.task == 'accompaniment':
        from generate_bonus import generate_bonus
        return generate_bonus(args)
    vocab = Vocabulary()
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    if metadata['vocabulary_sha256'] != vocab.sha256:
        raise ValueError('Wrong tokenizer for this checkpoint')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started, samples = time.monotonic(), []
    jobs = [(f'sample_{i + 1:03d}', None) for i in range(args.num_samples)]
    if args.task == 'continuation':
        prompt_dir = args.data_root / 'prompts/basic-b'
        jobs = [(p.name, p) for p in sorted(prompt_dir.iterdir()) if p.is_dir()]
        if len(jobs) != 3:
            raise ValueError('Expected three official continuation prompts')
    for index, (name, folder) in enumerate(jobs):
        prefix, beat_map, prefix_path = None, None, None
        meter, tempo = args.meter, args.tempo
        if folder is not None:
            prefix_path = next(folder.glob('*.mid'))
            encoded = tokenize_midi(prefix_path, folder / 'beat_midi.txt', max_bars=8, vocab=vocab)
            if encoded.metadata['n_bars'] != 8:
                raise ValueError('Expected an eight-bar prompt')
            prefix, meter = encoded.tokens, encoded.beat_map.bar_beats[7]
            tempo = 60 * meter / (encoded.beat_map.bar_times[8] - encoded.beat_map.bar_times[7])
            beat_map = extended_clock(encoded.beat_map, 32)
        seed = args.seed + index
        tokens, diagnostics = generate_sequence(model, metadata, vocab, seed, args.temperature,
                                                args.top_k, args.top_p, args.bars, prefix,
                                                meter, tempo, args.device)
        midi_path, wav_path = args.output_dir / f'{name}.mid', args.output_dir / f'{name}.wav'
        decoded = decode_tokens(tokens, midi_path, vocab, beat_map=beat_map, clip_at_end=True)
        prefix_check = None
        if prefix_path:
            preserve_prompt_notes(prefix_path, midi_path, midi_path, encoded.beat_map.bar_times[8])
            prefix_check = verify_prefix(prefix_path, midi_path, encoded.beat_map.bar_times[8])
        record = {'song_name': name, 'tokens': [vocab.tokens[t] for t in tokens],
                  'midi_path': portable_path(midi_path), 'wav_path': portable_path(wav_path) if args.render else None,
                  'bars': args.bars, 'bar_times': decoded['bar_times'], 'bar_beats': decoded['bar_beats'],
                  'diagnostics': diagnostics, 'prefix_verification': prefix_check,
                  'midi_sha256': hashlib.sha256(midi_path.read_bytes()).hexdigest()}
        if args.render:
            from render_audio import render_midi
            record['audio_audit'] = render_midi(midi_path, wav_path)
        samples.append(record)
        print(json.dumps({'generated': name, 'n_tokens': len(tokens), 'notes': diagnostics['n_notes'],
                          'elapsed_seconds': time.monotonic() - started}), flush=True)
        manifest = {'task': args.task, 'checkpoint': portable_path(args.checkpoint),
                    'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    'configuration': {'temperature': args.temperature, 'top_k': args.top_k, 'top_p': args.top_p,
                                      'seed': args.seed, 'device': args.device},
                    'samples': samples, 'elapsed_seconds': time.monotonic() - started,
                    'complete': len(samples) == len(jobs)}
        (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--task', choices=['unconditional', 'continuation', 'accompaniment'], default='unconditional')
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'POP909')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--bars', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--meter', type=int, default=4)
    parser.add_argument('--tempo', type=int, default=120)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
