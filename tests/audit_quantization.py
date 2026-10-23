"""Representation-only resolution ablation; does not train or alter corpora."""
import json
from pathlib import Path
import sys

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_codec import BeatMap


def extrapolate(values, xp, fp):
    out = np.interp(values, xp, fp)
    before, after = values < xp[0], values > xp[-1]
    out[before] = fp[0] + (values[before] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    out[after] = fp[-1] + (values[after] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return out


def run():
    root = Path(__file__).resolve().parents[1]
    results = {grid: {"onset": [], "offset": [], "clipped": 0, "oov": 0,
                      "before_first_beat": 0, "exact_same_pitch_grid_collisions": 0} for grid in (4, 8, 16)}
    folders = sorted((root / "POP909/training").iterdir())
    count = 0
    for index, folder in enumerate(folders):
        if not folder.is_dir():
            continue
        midi = pretty_midi.PrettyMIDI(str(next(folder.glob("*.mid"))))
        notes = [n for instrument in midi.instruments for n in instrument.notes]
        starts = np.asarray([n.start for n in notes])
        ends = np.asarray([n.end for n in notes])
        pitch = np.asarray([n.pitch for n in notes])
        beat_map = BeatMap.from_file(folder / "beat_midi.txt", end_time=float(max(starts)))
        times = np.asarray(beat_map.beat_times)
        beat_ids = np.arange(len(times))
        sb, eb = extrapolate(starts, times, beat_ids), extrapolate(ends, times, beat_ids)
        bar = np.clip(np.searchsorted(beat_map.bar_times, starts + 1e-7, side="right") - 1, 0, beat_map.n_bars - 1)
        bar_start = np.asarray(beat_map.bar_indices)[bar]
        bar_beats = np.asarray(beat_map.bar_beats)[bar]
        for grid, metrics in results.items():
            pos = np.maximum(0, np.minimum(bar_beats * grid - 1, np.rint((sb - bar_start) * grid)))
            qb = bar_start + pos / grid
            raw_duration = np.maximum(1, np.rint((eb - sb) * grid))
            duration = np.minimum(32 * grid, raw_duration)
            qt = extrapolate(qb, beat_ids, times)
            qe = extrapolate(qb + duration / grid, beat_ids, times)
            metrics["onset"].append(abs(qt - starts))
            metrics["offset"].append(abs(qe - ends))
            metrics["clipped"] += int(sum(raw_duration > 32 * grid))
            metrics["before_first_beat"] += int(sum(sb < 0))
            metrics["oov"] += int(sum((pitch < 0) | (pitch > 127) | (pos >= 16 * grid)))
            unique = np.unique(np.column_stack([bar, pos, pitch]), axis=0)
            metrics["exact_same_pitch_grid_collisions"] += len(notes) - len(unique)
        count += len(notes)
        if (index + 1) % 100 == 0:
            print(f"Resolution audit: {index + 1}/896", flush=True)
    report = {"experiment": "Representation reconstruction at 4, 8 and 16 subdivisions per annotation beat",
              "method": "All 896 original training songs; identical annotation beat maps, rounding, velocity bins, merged tracks, 32-beat maximum duration. Position vocabulary supports 16 beats per bar at every resolution. No Transformer is retrained for this ablation.",
              "n_training_songs": 896, "n_notes": count, "results": {}}
    for grid, metrics in results.items():
        row = {key: value for key, value in metrics.items() if key not in ("onset", "offset")}
        row["vocabulary_size"] = 299 + 48 * grid
        for label in ("onset", "offset"):
            values = np.concatenate(metrics[label])
            row[label + "_error_seconds"] = {"mae": float(np.mean(values)), "p95": float(np.quantile(values, .95)),
                                              "max": float(np.max(values))}
        report["results"][str(grid)] = row
    target = root / "artifacts/quantization_resolution_audit.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run()
