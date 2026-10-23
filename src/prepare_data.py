"""Tokenize all supplied training songs with a fixed, song-disjoint split."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from music_codec import Vocabulary, tokenize_midi


def load_corpus(path):
    """Load packed sequences; offsets delimit songs without cross-song targets."""
    return np.load(path, allow_pickle=False)


def prepare(data_root, output_dir, seed=20261022, val_fraction=0.05):
    data_root, output_dir = Path(data_root), Path(output_dir)
    training = data_root / "training"
    folders = sorted(path for path in training.iterdir() if path.is_dir())
    if len(folders) != 896:
        raise ValueError(f"Expected all 896 official training songs, found {len(folders)}")
    ids = [path.name for path in folders]
    order = ids.copy()
    random.Random(seed).shuffle(order)
    validation_ids = set(order[:round(len(order) * val_fraction)])
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary()
    vocab.save(output_dir / "vocabulary.json")
    audit = {"seed": seed, "validation_fraction": val_fraction, "n_songs": len(folders),
             "training_ids": [song_id for song_id in ids if song_id not in validation_ids],
             "validation_ids": sorted(validation_ids), "vocabulary_sha256": vocab.sha256,
             "steps_per_beat": 8, "source": "Official training/ only; prompts never read.",
             "split_note": "Every supplied song is tokenized; fixed song-disjoint validation is excluded from optimization.",
             "modes": {}}
    for mode in ("basic", "roles"):
        started = time.time()
        all_tokens, offsets, all_bar_offsets, bar_song_offsets = [], [0], [], [0]
        metadata, hashes = [], []
        stats = Counter()
        for index, folder in enumerate(folders):
            midis = sorted(folder.glob("*.mid"))
            if len(midis) != 1:
                raise ValueError(f"Expected exactly one MIDI in {folder}")
            encoded = tokenize_midi(midis[0], folder / "beat_midi.txt", mode=mode, vocab=vocab)
            all_tokens.extend(encoded.tokens)
            offsets.append(len(all_tokens))
            all_bar_offsets.extend(encoded.bar_offsets)
            bar_song_offsets.append(len(all_bar_offsets))
            metadata.append(encoded.metadata)
            hashes.append({"song_id": folder.name,
                           "midi_sha256": hashlib.sha256(midis[0].read_bytes()).hexdigest(),
                           "beats_sha256": hashlib.sha256((folder / "beat_midi.txt").read_bytes()).hexdigest()})
            for key, value in encoded.metadata["stats"].items():
                if key.endswith("_max"):
                    stats[key] = max(stats[key], value)
                else:
                    stats[key] += value
            if (index + 1) % 100 == 0:
                print(f"{mode}: {index + 1}/{len(folders)} songs", flush=True)
        np.savez_compressed(output_dir / f"tokens_{mode}.npz",
                            tokens=np.asarray(all_tokens, dtype=np.uint16), offsets=np.asarray(offsets, dtype=np.int64),
                            bar_offsets=np.asarray(all_bar_offsets, dtype=np.int64),
                            bar_song_offsets=np.asarray(bar_song_offsets, dtype=np.int64),
                            song_ids=np.asarray(ids), splits=np.asarray(["val" if song in validation_ids else "train" for song in ids]),
                            vocabulary_sha256=np.asarray(vocab.sha256))
        (output_dir / f"metadata_{mode}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        audit["modes"][mode] = {"total_tokens": len(all_tokens), "stats": dict(stats),
                                  "elapsed_sec": time.time() - started,
                                  "bar_offset_convention": "Per song relative token indices, includes final EOS endpoint; bar_song_offsets delimits these lists."}
        if mode == "basic":
            audit["input_hashes"] = hashes
    (output_dir / "tokenization_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"n_songs": len(folders), "vocab_size": vocab.size,
                      "train": len(folders) - len(validation_ids), "val": len(validation_ids),
                      "modes": audit["modes"]}, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "POP909")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()
    prepare(args.data_root, args.output_dir, args.seed, args.val_fraction)
