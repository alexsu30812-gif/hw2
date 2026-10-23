"""Diagnose next-token prediction on reproducible official validation windows.

Main cross-entropy and top-1 accuracy use the whole vocabulary without sampling
masks. The type-conditioned diagnostics additionally assume the target token
family is known; they distinguish learning the event grammar from choosing the
pitch, position, velocity, or duration within that family.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from model import load_checkpoint
from music_codec import Vocabulary
from train import SongWindows


def _families(vocab):
    families = {"Bar": [vocab.bar_id]}
    families.update({name: vocab.group_ids(name)
                     for name in ("Meter", "Tempo", "Pos", "Pitch", "Vel", "Dur", "Track")})
    families.update({name: [vocab.id(name)] for name in ("<BOS>", "<EOS>")})
    return families


def _totals():
    return dict(count=0, cross_entropy_sum=0., correct=0, type_correct=0,
                conditional_cross_entropy_sum=0., conditional_correct=0)


def _summary(totals):
    count = totals["count"]
    if not count:
        return dict(targets=0, cross_entropy_nats=None, perplexity=None,
                    accuracy=None, predicted_type_accuracy=None,
                    type_conditioned_cross_entropy_nats=None, type_conditioned_accuracy=None)
    ce = totals["cross_entropy_sum"] / count
    return dict(targets=count, cross_entropy_nats=ce, perplexity=math.exp(ce),
                accuracy=totals["correct"] / count,
                predicted_type_accuracy=totals["type_correct"] / count,
                type_conditioned_cross_entropy_nats=totals["conditional_cross_entropy_sum"] / count,
                type_conditioned_accuracy=totals["conditional_correct"] / count)


@torch.inference_mode()
def evaluate(args):
    started = time.monotonic()
    if args.batches < 1 or args.batch_size < 1 or args.context < 1:
        raise ValueError("Batches, batch size, and context must be positive")
    torch.set_num_threads(args.threads)
    vocab = Vocabulary()
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    if metadata["vocabulary_sha256"] != vocab.sha256:
        raise ValueError("Checkpoint tokenizer does not match the current vocabulary")
    if args.context > model.config.context:
        raise ValueError("Evaluation context exceeds checkpoint position capacity")
    source = SongWindows(args.corpus, "val", args.context, args.seed, augment=False)
    validation_ids = sorted(str(source.corpus["song_ids"][i]) for i in source.songs)
    if len(validation_ids) != args.expected_validation_songs:
        raise ValueError(f"Expected {args.expected_validation_songs} validation songs, found {len(validation_ids)}")
    families = _families(vocab)
    type_map = torch.full((vocab.size,), -1, dtype=torch.long, device=args.device)
    for number, (_, ids) in enumerate(families.items()):
        type_map[torch.tensor(ids, device=args.device)] = number
    totals = {name: _totals() for name in families}
    digest = hashlib.sha256()
    batch_counts = []
    for batch in range(args.batches):
        x, y = source.batch(args.batch_size)
        # Hash inputs and targets before transfer so every checkpoint can prove
        # it was evaluated on identical windows, including the ignored padding.
        digest.update(x.numpy().astype("<i8", copy=False).tobytes())
        digest.update(y.numpy().astype("<i8", copy=False).tobytes())
        x, y = x.to(args.device), y.to(args.device)
        logits, _, _ = model(x)
        valid = (y != -100) & (y != vocab.pad_id)
        targets = y[valid]
        scores = logits[valid].float()
        if not len(targets):
            raise ValueError("Validation batch has no non-padding targets")
        if not bool(torch.isfinite(scores).all().cpu()):
            raise ValueError("Nonfinite model scores")
        if bool((type_map[targets] < 0).any().cpu()):
            raise ValueError("An evaluation target has no token family")
        losses = F.cross_entropy(scores, targets, reduction="none")
        predictions = scores.argmax(dim=-1)
        batch_counts.append(int(targets.numel()))
        for number, (name, ids) in enumerate(families.items()):
            mask = type_map[targets] == number
            count = int(mask.sum().cpu())
            if not count:
                continue
            family_ids = torch.tensor(ids, device=args.device)
            family_scores = scores[mask][:, family_ids]
            family_targets = targets[mask] - ids[0]
            conditional_losses = F.cross_entropy(family_scores, family_targets, reduction="sum")
            entry = totals[name]
            entry["count"] += count
            entry["cross_entropy_sum"] += float(losses[mask].double().sum().cpu())
            entry["correct"] += int((predictions[mask] == targets[mask]).sum().cpu())
            entry["type_correct"] += int((type_map[predictions[mask]] == number).sum().cpu())
            entry["conditional_cross_entropy_sum"] += float(conditional_losses.double().cpu())
            entry["conditional_correct"] += int((family_scores.argmax(dim=-1) == family_targets).sum().cpu())
        print(json.dumps({"batch": batch + 1, "batches": args.batches,
                          "targets": batch_counts[-1], "elapsed_seconds": time.monotonic() - started}), flush=True)
    combined = _totals()
    for entry in totals.values():
        for name, value in entry.items():
            combined[name] += value
    training_ids = set(metadata.get("training_song_ids", []))
    for stage in metadata.get("training_stages", []):
        training_ids.update(stage.get("training_song_ids", []))
    overlap = sorted(training_ids.intersection(validation_ids))
    result = dict(
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint_step=metadata.get("step"), checkpoint_mode=metadata.get("mode"),
        checkpoint_model_config=asdict(model.config),
        corpus=str(args.corpus), corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        vocabulary_sha256=vocab.sha256,
        evaluation=dict(split="val", context=args.context, seed=args.seed,
                        batches=args.batches, batch_size=args.batch_size,
                        windows=args.batches * args.batch_size, augmentation=False,
                        padding_ignored=True, device=args.device, torch_cpu_threads=torch.get_num_threads(),
                        validation_song_ids=validation_ids,
                        observed_song_ids=sorted(source.seen),
                        validation_songs=len(validation_ids), observed_songs=len(source.seen),
                        windows_sha256=digest.hexdigest(), batch_target_counts=batch_counts,
                        training_overlap_song_ids=overlap,
                        held_out_from_checkpoint_training=(not overlap),
                        note="Deterministic bar-aligned windows, cycling through a shuffled song order before revisiting songs."),
        definitions=dict(
            cross_entropy_nats="Mean negative log probability of the true next token, over the full vocabulary; no grammar/sampling mask.",
            accuracy="Fraction whose full-vocabulary top-1 token equals the true target.",
            predicted_type_accuracy="Fraction whose full-vocabulary top-1 token belongs to the true target family.",
            type_conditioned_cross_entropy_nats="Cross-entropy after renormalizing scores within the known target family; diagnostic, not generation.",
            type_conditioned_accuracy="Top-1 accuracy restricted to the known target family; the singleton Bar family is trivially correct.",
            target_weighting="Each non-padding target has equal weight; aggregate values are not unweighted averages of token families."),
        overall=_summary(combined), by_target_type={name: _summary(value) for name, value in totals.items()},
        elapsed_seconds=time.monotonic() - started,
        versions=dict(torch=str(torch.__version__), numpy=str(np.__version__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "overall": result["overall"],
                      "windows_sha256": digest.hexdigest(), "held_out": not overlap}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/tokens_basic.npz")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261039)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--expected-validation-songs", type=int, default=45)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
