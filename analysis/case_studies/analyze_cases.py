"""Reproduce bar-indexed symbolic case studies from the unchanged submission.

Run from any directory with the HW2 environment:
  .venv/bin/python -B analysis/case_studies/analyze_cases.py

This is numerical evidence, not a claim that anyone listened to the WAVs.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hw2-case-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "hw2-case-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import BONUS, parse_bars


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "hw2":
        path = Path(*path.parts[1:])
    return ROOT / path


def rel(path):
    return str(Path(path).relative_to(ROOT))


def numeric_summary(notes, n_bars):
    pitches = np.array([n["pitch"] for n in notes])
    velocity = np.array([n["velocity"] for n in notes])
    histogram = np.bincount(pitches % 12, minlength=12) if len(pitches) else np.zeros(12, dtype=int)
    return {
        "notes": len(notes),
        "notes_per_bar": len(notes) / n_bars,
        "mean_pitch": float(pitches.mean()) if len(pitches) else None,
        "p95_pitch": float(np.percentile(pitches, 95)) if len(pitches) else None,
        "pitch_min": int(pitches.min()) if len(pitches) else None,
        "pitch_max": int(pitches.max()) if len(pitches) else None,
        "mean_velocity": float(velocity.mean()) if len(velocity) else None,
        "fraction_pitch_ge_96": float(np.mean(pitches >= 96)) if len(pitches) else 0.0,
        "pc_histogram": histogram.tolist(),
        "pc_histogram_normalized": (histogram / max(1, len(notes))).tolist(),
    }


def segment(bars, times, start, stop):
    notes = [n for b in bars[start:stop] for n in b["notes"]]
    rhythms = [tuple(sorted({n["pos"] for n in b["notes"]})) for b in bars[start:stop]]
    run = longest = 1
    for before, after in zip(rhythms, rhythms[1:]):
        run = run + 1 if before == after else 1
        longest = max(longest, run)
    return {
        "bar_range": [start + 1, stop],
        "time_range": [float(times[start]), float(times[stop])],
        **numeric_summary(notes, stop - start),
        "unique_onset_position_sets": len(set(rhythms)),
        "longest_identical_onset_set_run": longest,
        "rhythm_adjacent_equal_fraction": sum(a == b for a, b in zip(rhythms, rhythms[1:])) / max(1, len(rhythms) - 1),
    }


def bar_rows(record):
    bars = parse_bars(record["tokens"])
    times = np.array(record["bar_times"])
    assert len(times) == len(bars) + 1
    midi_path, wav_path = resolve(record["midi_path"]), resolve(record["wav_path"])
    assert digest(midi_path) == record["midi_sha256"]
    if record.get("audio_audit", {}).get("wav_sha256"):
        assert digest(wav_path) == record["audio_audit"]["wav_sha256"]
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    midi_notes = [n for inst in midi.instruments for n in inst.notes]
    assert Counter(n.pitch for n in midi_notes) == Counter(n["pitch"] for b in bars for n in b["notes"])
    signal, rate = sf.read(wav_path, dtype="float32", always_2d=True)
    assert len(signal) / rate >= times[-1]
    rows = []
    for i, bar in enumerate(bars):
        t0, t1 = map(float, times[i:i+2])
        exact = [n for n in midi_notes if t0 - 0.001 <= n.start < t1 - 0.001]
        # The 1 ms boundary tolerance compensates only sub-millisecond MIDI export
        # rounding. Token statistics remain the primary published definitions.
        sample = signal[round(t0 * rate):round(t1 * rate)].astype(np.float64)
        rows.append({
            "bar": i + 1,
            "phase": "original_prompt" if i < 8 and record.get("prefix_verification") else "generated",
            "t0": t0, "t1": t1, "meter_annotated_beats": bar["meter"],
            **numeric_summary(bar["notes"], 1),
            "onset_positions_8_per_beat": sorted({n["pos"] for n in bar["notes"]}),
            "onset_pattern_with_multiplicity": sorted(n["pos"] for n in bar["notes"]),
            "exported_midi_onset_notes": len(exact),
            "exported_midi_mean_velocity": float(np.mean([n.velocity for n in exact])) if exact else None,
            "wav_rms": float(np.sqrt(np.mean(sample * sample))),
            "wav_peak": float(np.max(np.abs(sample))),
            "wav_clipped_sample_fraction": float(np.mean(np.abs(sample) >= 32767 / 32768)),
        })
    assert sum(r["exported_midi_onset_notes"] for r in rows) == len(midi_notes)
    return bars, rows, {
        "midi_path": rel(midi_path), "midi_sha256": digest(midi_path),
        "wav_path": rel(wav_path), "wav_sha256": digest(wav_path),
        "sample_rate": rate, "wav_duration_seconds": len(signal) / rate,
    }


def chord_label(index):
    if index < 0:
        return {-1: "N", -2: "unsupported", -3: "no_estimate"}[index]
    return ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"][index % 12] + (":min" if index >= 12 else ":maj")


def draw_roll(ax, record, bars, title):
    times = record["bar_times"]
    for i, bar in enumerate(bars):
        beat_seconds = (times[i+1] - times[i]) / bar["meter"]
        for n in bar["notes"]:
            t0 = times[i] + n["pos"] / 8 * beat_seconds
            t1 = min(times[-1], t0 + n["duration"] / 8 * beat_seconds)
            ax.plot([t0, t1], [n["pitch"], n["pitch"]], lw=1.3,
                    color="#c23232" if n["pitch"] >= 96 else "#465973", solid_capstyle="butt")
    for i in (0, 8, 16, 24, 32):
        ax.axvline(times[i], color="#9a9a9a", ls="--", lw=.6)
        if i < 32:
            ax.text(times[i] + .4, 109, f"bar {i+1}", fontsize=8)
    ax.axhline(96, color="#c23232", ls=":", lw=.9, label="MIDI 96 (C7)")
    ax.set(xlabel="WAV time (seconds; no lead-in)", ylabel="MIDI pitch", ylim=(20, 112), xlim=(0, times[-1]), title=title)
    ax.legend(loc="lower right", fontsize=8)


def piano_roll(record, bars, output, title):
    fig, ax = plt.subplots(figsize=(11.8, 3.2), constrained_layout=True)
    draw_roll(ax, record, bars, title)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    quality = load(ROOT / "analysis/grading_review/music_quality_review.json")
    comparison = {"scope": "Symbolic and signal measurements of unchanged submitted files; no human or model listening claim.",
                  "definitions": {
                      "bar_numbering": "1-based from the first submitted bar; B bars 1-8 are the original prompt, 9-32 are generated.",
                      "primary_stats": "Original manifest tokens, identical to music_quality_review: onsets count once each; MIDI pitch; velocity = 4*Vel_token+2; unweighted note-onset PC histogram; NumPy linear-interpolated 95th percentile.",
                      "rhythm": "SET of Pos8 onset positions per annotated-beat bar, ignoring pitch/duration/velocity/multiplicity; matching sets do not mean identical music.",
                      "wav_time": "Time since start of submitted WAV; waveform starts at MIDI time 0. MIDI musical endpoint excludes the approximately 2-second sound-release tail.",
                      "exact_midi": "All actual MIDI pitch multisets and note totals are asserted equal to tokens. Per-bar MIDI onsets use 1 ms boundary tolerance for export rounding; original prompt velocity may differ from the quantized token reconstruction.",
                      "register": "MIDI 96 = C7; a descriptive register marker, not a universal quality limit. Pitch comparisons do not establish perceived quality.",
                      "chord": "Original, unmodified HW2_scripts/eval_bonus.py: 24 major/minor triads, duration*velocity weighted chroma with BASS_WEIGHT, cosine similarity; reference is maximum-overlap-duration annotation label per bar. Chord annotations are read for post-generation evaluation only.",
                      "reference": "The 45 center-nearest 32-bar source crops were included in final all896-song fitting. Register comparisons are descriptive, not held-out generalization."},
                  "basic_a": {}, "basic_b": {"rows": []}, "bonus": [],
                  "evidence_files": ["analysis/case_studies/basic_b_segment_comparison.csv",
                                     "analysis/case_studies/bonus_chord_bar_comparison.csv",
                                     "analysis/grading_review/music_quality_review.json"]}

    # A: deterministic selection of an extreme and a typical register case.
    manifest = load(ROOT / "results/basic_a/t12/manifest.json")
    sample_stats = []
    for record in manifest["samples"]:
        bars = parse_bars(record["tokens"])
        row = numeric_summary([n for b in bars for n in b["notes"]], len(bars))
        sample_stats.append({"sample_id": record["song_name"], **row})
    extreme = sorted(sample_stats, key=lambda r: (-r["p95_pitch"], r["sample_id"]))[0]
    median = float(np.median([r["p95_pitch"] for r in sample_stats]))
    typical = sorted(sample_stats, key=lambda r: (abs(r["p95_pitch"] - median), r["sample_id"]))[0]
    source = quality["basic_a"]
    comparison["basic_a"]["register_summary"] = {
        "configuration_id": "t12", "n_generated": 20, "n_reference": 45,
        "generated_piece_mean_pitch": source["generated_summary"]["pitch_mean"]["piece_mean"],
        "reference_piece_mean_pitch": source["real45_summary"]["pitch_mean"]["piece_mean"],
        "generated_piece_mean_p95_pitch": source["generated_summary"]["pitch_p95"]["piece_mean"],
        "reference_piece_mean_p95_pitch": source["real45_summary"]["pitch_p95"]["piece_mean"],
        "generated_piece_mean_fraction_pitch_ge_96": source["generated_summary"]["fraction_pitch_ge_96"]["piece_mean"],
        "reference_piece_mean_fraction_pitch_ge_96": source["real45_summary"]["fraction_pitch_ge_96"]["piece_mean"],
        "sample_selection": "Extreme: greatest whole-piece p95 pitch, ties lexical sample ID. Typical: nearest whole-piece p95 pitch to the 20-piece median, ties lexical ID. Selection diagnoses register only, not overall quality.",
        "median_generated_piece_p95_pitch": median,
    }
    comparison["basic_a"]["all_piece_register"] = sample_stats
    comparison["basic_a"]["cases"] = []
    for selected, rule in ((extreme, "maximum piece p95"), (typical, "nearest median piece p95")):
        record = next(r for r in manifest["samples"] if r["song_name"] == selected["sample_id"])
        bars, perbar, provenance = bar_rows(record)
        figure = OUT / f'a_{selected["sample_id"]}_register.png'
        piano_roll(record, bars, figure, f'Basic A t12 {selected["sample_id"]}: {rule}')
        most_high = sorted(perbar, key=lambda r: (-r["fraction_pitch_ge_96"], -r["p95_pitch"], r["bar"]))[:2]
        item = {**selected, "selection_rule": rule, "bars": [1, 32], "time_range": [0, record["bar_times"][-1]],
                "figure_path": rel(figure), "segments": [segment(bars, record["bar_times"], i, i+8) for i in (0, 8, 16, 24)],
                "bar_examples": most_high, "all_bars": perbar, "provenance": provenance}
        comparison["basic_a"]["cases"].append(item)
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 5.6), constrained_layout=True)
    for ax, case in zip(axes, comparison["basic_a"]["cases"]):
        record = next(r for r in manifest["samples"] if r["song_name"] == case["sample_id"])
        draw_roll(ax, record, parse_bars(record["tokens"]),
                  f'{case["sample_id"]}: {case["selection_rule"]}; piece p95={case["p95_pitch"]:g}; '
                  f'{100*case["fraction_pitch_ge_96"]:.2f}% of onsets >= C7')
    combined = OUT / "basic_a_register_cases.png"
    fig.savefig(combined, dpi=170); plt.close(fig)
    comparison["basic_a"]["combined_figure"] = {"figure_path": rel(combined)}

    # B: all nine continuations, all 32 bars, and the same four 8-bar windows.
    for config in ("t08", "t10", "t12"):
        path = ROOT / "results/basic_b" / config / "manifest.json"
        data = load(path)
        for record in data["samples"]:
            bars, perbar, provenance = bar_rows(record)
            segments = [segment(bars, record["bar_times"], i, i+8) for i in (0, 8, 16, 24)]
            a, b, c = segments[1:]
            comment = (f'Bars 9-16 / 17-24 / 25-32 have {a["notes_per_bar"]:.3f} / {b["notes_per_bar"]:.3f} / {c["notes_per_bar"]:.3f} '
                       f'notes per bar and mean MIDI pitch {a["mean_pitch"]:.2f} / {b["mean_pitch"]:.2f} / {c["mean_pitch"]:.2f}. '
                       f'The last generated window changes density by {100*(c["notes_per_bar"]/a["notes_per_bar"]-1):+.1f}% and mean pitch by '
                       f'{c["mean_pitch"]-a["mean_pitch"]:+.2f} semitones relative to the first generated window. '
                       'These statistics describe long-range density/register drift; they are not listening judgments.')
            comparison["basic_b"]["rows"].append({"prompt_id": record["song_name"], "configuration_id": config,
                "selected_by_original_boundary_rule": quality["basic_b"]["selected_per_prompt"][record["song_name"]] == config,
                "prompt_end_seconds": record["bar_times"][8], "musical_endpoint_seconds": record["bar_times"][-1],
                "segments": segments, "bars": perbar, "comment": comment, "provenance": provenance})
    comparison["basic_b"]["rows"].sort(key=lambda r: (r["prompt_id"], r["configuration_id"]))

    # Bonus: two pre-specified cases; run the exact official estimator per bar.
    bonus_manifest = load(ROOT / "results/bonus/manifest.json")
    for song in ("004", "009"):
        record = next(r for r in bonus_manifest["samples"] if r["song_name"] == song)
        folder = ROOT / "POP909/prompts/bonus" / song
        times, down = BONUS.load_gt_beats(str(folder / "beat_midi.txt"))
        grid = BONUS.GTBars(times, down)
        chords = BONUS.load_gt_chords(str(folder / "chord_midi.txt"))
        gen = pretty_midi.PrettyMIDI(str(resolve(record["midi_path"])))
        gt = pretty_midi.PrettyMIDI(str(folder / "piano.mid"))
        gen_notes = BONUS.pick_track(gen, "PIANO", 1).notes
        gt_notes = BONUS.pick_track(gt, "PIANO", 0).notes
        gen_arr, gt_arr = BONUS.note_arrays(gen_notes), BONUS.note_arrays(gt_notes)
        rows = []
        for i in range(grid.n_bars):
            t0, t1 = float(grid.bar_start(i)), float(grid.bar_end(i))
            ref = int(BONUS.dominant_chord(chords, t0, t1))
            estimated = int(BONUS.estimate_chord(gen_arr, t0, t1))
            gt_estimated = int(BONUS.estimate_chord(gt_arr, t0, t1))
            active = [n for n in gen_notes if n.start < t1 and n.end > t0]
            def triad_pcs(index):
                if index < 0:
                    return []
                root = index % 12
                return sorted({root, (root + (3 if index >= 12 else 4)) % 12, (root + 7) % 12})
            ref_pcs, gen_pcs = triad_pcs(ref), triad_pcs(estimated)
            rows.append({"bar": i+1, "t0": t0, "t1": t1,
                "reference_chord": chord_label(ref), "generated_chord": chord_label(estimated),
                "gt_piano_estimated_chord": chord_label(gt_estimated), "match": ref == estimated,
                "gt_estimator_matches_reference": ref == gt_estimated,
                "reference_supported_major_minor": ref >= 0,
                "reference_triad_pitch_classes": ref_pcs,
                "generated_estimated_triad_pitch_classes": gen_pcs,
                "shared_estimated_and_reference_triad_tones": len(set(ref_pcs) & set(gen_pcs)),
                "n_generated_piano_overlapping_notes": len(active),
                "generated_piano_pitch_classes": sorted({n.pitch % 12 for n in active}),
                "generated_piano_pitches": sorted({n.pitch for n in active})})
        ca, gtca = BONUS.chord_accuracy(gen_arr, gt_arr, chords, grid)
        expected = next(r for r in quality["bonus"]["per_piece"] if r["song_name"] == song)
        assert ca == expected["CA"] and gtca == expected["GT_estimator_CA"]
        # First two mismatches where the GT piano's estimator agrees with label,
        # and first two successes: deterministic and includes estimator context.
        failures = [r for r in rows if not r["match"] and r["gt_estimator_matches_reference"]][:2]
        if not failures:
            failures = [r for r in rows if not r["match"]][:2]
        successes = [r for r in rows if r["match"]][:2]
        comparison["bonus"].append({"prompt_id": song, "CA": ca, "GT_estimator_CA": gtca,
            "generated_matches": sum(r["match"] for r in rows), "total_bars": len(rows),
            "supported_major_minor_bars": sum(r["reference_supported_major_minor"] for r in rows),
            "case_selection": "Pre-specified from the independent review: 004 has zero official CA; 009 has CA 0.25. Bar examples are first two failures for which GT-estimator agrees with the annotation and first two generated successes, in chronological order.",
            "bar_examples": sorted(failures + successes, key=lambda r:r["bar"]), "all_bars": rows,
            "discussion": "The estimator reports agreement with the supplied bar-dominant major/minor chord label. A mismatch does not prove every note is wrong or exclude a valid reharmonization; the comparison with the same estimator on GT piano helps expose estimator limitations. Repeated onset-position patterns are not repeated full music bars.",
            "provenance": {"midi_path": rel(resolve(record["midi_path"])), "midi_sha256": digest(resolve(record["midi_path"])),
                           "beat_sha256": digest(folder / "beat_midi.txt"), "chord_annotation_sha256": digest(folder / "chord_midi.txt"),
                           "official_estimator_sha256": digest(ROOT / "HW2_scripts/eval_bonus.py")}})

    comparison["report_discussion"] = {
        "register": "The t12 setting minimizes the preregistered combined mean H4/GS reference gap among the tested settings, yet its register remains broader and higher than the 45-song reference: mean piece-level pitch p95 is 92.39 versus 78.36, a 14.02-semitone gap. The mean per-piece share of onsets at MIDI 96 or above is 3.50% versus 0.033%. H4 collapses octave information into pitch classes and GS only uses onset positions, so neither proxy can expose this register mismatch. The extreme and median-p95 examples are selected by stated rules rather than listening preference; no pitch clipping or other postprocessing has been applied.",
        "harmony": "Bonus 004 matches none of the 16 annotated bar chords; 009 matches 4/16. The exact same 24-triad estimator on GT piano is shown alongside each example, so agreement failures are not automatically equated with perceptual dissonance. The relatively high pitch-class OA reflects distribution overlap, not correct note or chord alignment. Better conditioning should be tested on training-only validation tunes using chord-aware or phrase-aware objectives before sampling settings are selected.",
        "continuation": "All nine continuations are compared across the original 8-bar prompt and three generated 8-bar windows. This extends the local two-bar seam criterion to density, pitch, dynamics, and onset-set repetition across the full 24-bar continuation. For prompt 003 at t12, density falls from 33.875 to 9.250 notes/bar and mean pitch rises from 69.08 to 85.27 between the first and last generated windows. The original selected setting for this prompt remains t10. Numerical comparisons cannot establish musical phrasing or listener preference; any actual audition observations must be documented separately."}

    comparison["provenance"] = {"analysis_script": rel(Path(__file__)), "analysis_script_sha256": digest(Path(__file__)),
        "quality_review_sha256": digest(ROOT / "analysis/grading_review/music_quality_review.json"),
        "n_B_pieces": len(comparison["basic_b"]["rows"]), "n_B_bar_rows": sum(len(r["bars"]) for r in comparison["basic_b"]["rows"]),
        "listening_claim": False}
    checks = 0
    for row in comparison["basic_b"]["rows"]:
        old = next(r for r in quality["basic_b"]["per_piece"]
                   if r["config"] == row["configuration_id"] and r["song_name"] == row["prompt_id"])
        assert sum(b["notes"] for b in row["bars"][:8]) == old["prompt8"]["notes"]
        assert sum(b["notes"] for b in row["bars"][8:]) == old["generated24"]["notes"]
        for new_seg, old_seg in zip(row["segments"][1:], old["generated24"]["chunks_8bar"]):
            for new_name, old_name in (("notes_per_bar", "notes_per_bar"), ("mean_pitch", "pitch_mean"), ("mean_velocity", "velocity_mean")):
                assert abs(new_seg[new_name] - old_seg[old_name]) < 1e-10
                checks += 1
    for case in comparison["basic_a"]["cases"]:
        old = next(r for r in quality["basic_a"]["generated_per_piece"] if r["song_name"] == case["sample_id"])
        assert abs(case["p95_pitch"] - old["pitch_p95"]) < 1e-10
        assert abs(case["fraction_pitch_ge_96"] - old["fraction_pitch_ge_96"]) < 1e-10
        checks += 2
    comparison["validation"] = {"pass": True, "quality_review_numeric_crosschecks": checks,
        "actual_midi_pitch_multisets_verified": 11, "wav_manifest_hashes_verified": 11,
        "bonus_official_CA_recomputed": 2, "basic_b_bar_rows": 288,
        "no_source_midi_or_wav_modified": True}
    output = OUT / "bar_case_studies.json"
    output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
    with (OUT / "basic_b_segment_comparison.csv").open("w", newline="") as f:
        fields = ["prompt_id", "configuration_id", "selected_by_original_boundary_rule", "bar_start", "bar_end", "t0", "t1", "notes_per_bar", "mean_pitch", "p95_pitch", "mean_velocity", "unique_onset_position_sets", "longest_identical_onset_set_run"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in comparison["basic_b"]["rows"]:
            for seg in row["segments"]:
                writer.writerow({"prompt_id":row["prompt_id"], "configuration_id":row["configuration_id"],
                    "selected_by_original_boundary_rule":row["selected_by_original_boundary_rule"],
                    "bar_start":seg["bar_range"][0], "bar_end":seg["bar_range"][1],
                    "t0":seg["time_range"][0], "t1":seg["time_range"][1],
                    **{k:seg[k] for k in fields[7:]}})
    with (OUT / "bonus_chord_bar_comparison.csv").open("w", newline="") as f:
        fields = ["prompt_id", "bar", "t0", "t1", "reference_chord", "generated_chord", "gt_piano_estimated_chord", "match", "gt_estimator_matches_reference"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for item in comparison["bonus"]:
            for row in item["all_bars"]:
                writer.writerow({"prompt_id":item["prompt_id"], **{k:row[k] for k in fields[1:]}})
    print(json.dumps({"output": rel(output), "basic_a_cases": [r["sample_id"] for r in comparison["basic_a"]["cases"]], "basic_b_pieces": 9, "basic_b_bars": 288, "bonus": [{"id":r["prompt_id"], "CA":r["CA"], "GT_estimator_CA":r["GT_estimator_CA"], "match_bars":[b["bar"] for b in r["all_bars"] if b["match"]]} for r in comparison["bonus"]]}, indent=2))


if __name__ == "__main__":
    main()
