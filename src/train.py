"""Train a causal music Transformer using official training songs only."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from model import ModelConfig, MusicTransformer, load_checkpoint, save_checkpoint
from music_codec import Vocabulary


def recorded_path(path):
    try:
        return str(Path(path).resolve().relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(path)


class SongWindows:
    def __init__(self, path, split, length, seed, augment=False):
        self.corpus = dict(np.load(path, allow_pickle=False))
        self.length, self.augment = length, augment
        self.rng = np.random.default_rng(seed)
        self.vocab = Vocabulary()
        if str(self.corpus['vocabulary_sha256']) != self.vocab.sha256:
            raise ValueError('Tokenizer hash mismatch')
        self.songs = np.flatnonzero(np.ones(len(self.corpus['song_ids']), dtype=bool)
                                   if split == 'all' else self.corpus['splits'] == split)
        if not len(self.songs):
            raise ValueError('No songs in selected split')
        self.order = self.rng.permutation(self.songs).tolist()
        self.cursor = 0
        self.seen = set()

    def batch(self, count):
        X = np.full((count, self.length), self.vocab.pad_id, dtype=np.int64)
        Y = np.full((count, self.length), -100, dtype=np.int64)
        for row in range(count):
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(self.songs).tolist()
                self.cursor = 0
            song = int(self.order[self.cursor]); self.cursor += 1
            self.seen.add(str(self.corpus['song_ids'][song]))
            start, end = self.corpus['offsets'][song:song + 2]
            a, b = self.corpus['bar_song_offsets'][song:song + 2]
            boundaries = self.corpus['bar_offsets'][a:b - 1]
            if not len(boundaries):
                raise ValueError('Song without bars')
            # Each window begins at a complete bar; no target crosses songs.
            window_start = int(boundaries[self.rng.integers(len(boundaries))])
            remaining = self.corpus['tokens'][start + window_start:min(end, start + window_start + self.length)]
            tokens = np.concatenate(([self.vocab.bos_id], remaining)).astype(np.int64)
            if self.augment:
                low, high = self.vocab.ranges['Pitch']
                mask = (tokens >= low) & (tokens < high)
                if mask.any():
                    pitches = tokens[mask] - low
                    shifts = [s for s in range(-5, 6) if pitches.min() + s >= 21 and pitches.max() + s <= 108]
                    if shifts:
                        tokens[mask] += self.rng.choice(shifts)
            usable = min(self.length, len(tokens) - 1)
            X[row, :usable], Y[row, :usable] = tokens[:usable], tokens[1:usable + 1]
        return torch.from_numpy(X), torch.from_numpy(Y)


@torch.inference_mode()
def validate(model, source, device, batches=24, batch_size=2):
    model.eval()
    loss_sum, total = 0., 0
    for _ in range(batches):
        x, y = source.batch(batch_size)
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, y)
        count = int((y != -100).sum().cpu())
        loss_sum += float(loss.cpu()) * count
        total += count
    model.train()
    return loss_sum / total


def train(args):
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(4)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable: this process needs access to the GPU')
    device = args.device
    vocab = Vocabulary()
    if args.initialize_from:
        model, parent = load_checkpoint(args.initialize_from, device)
        if parent['vocabulary_sha256'] != vocab.sha256:
            raise ValueError('Initial checkpoint vocabulary mismatch')
        origin = {'own_checkpoint': recorded_path(args.initialize_from),
                  'sha256': hashlib.sha256(args.initialize_from.read_bytes()).hexdigest(),
                  'original_random_initialization': True}
        ancestors = list(parent.get('training_stages', [])) + [{
            'mode': parent['mode'], 'step': parent['step'],
            'processed_tokens': parent['processed_tokens'],
            'all_official_training_songs_used': parent['all_official_training_songs_used'],
            'training_song_ids': parent['training_song_ids']}]
        if not args.all_songs and any(stage['all_official_training_songs_used'] for stage in ancestors):
            raise ValueError('A diagnostic validation split was already seen by an ancestor checkpoint')
    else:
        model = MusicTransformer(ModelConfig(vocab.size, args.width, args.layers, args.heads,
                                             args.model_context, args.dropout)).to(device)
        origin = {'random_initialization': True, 'pretrained_weights': False}
        ancestors = []
    if args.context > model.config.context:
        raise ValueError('Training length exceeds position capacity')
    source = SongWindows(args.corpus, 'all' if args.all_songs else 'train', args.context, args.seed, args.augment)
    data_hash = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, best, started, processed = [], math.inf, time.monotonic(), 0
    parameters = {k: recorded_path(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata = {'vocabulary_sha256': vocab.sha256, 'origin': origin, 'arguments': parameters,
                'corpus_sha256': data_hash, 'training_song_ids': [str(source.corpus['song_ids'][i]) for i in source.songs],
                'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
                'inference_context': args.context, 'all_official_training_songs_used': args.all_songs,
                'mode': args.mode, 'parameter_count': sum(p.numel() for p in model.parameters()),
                'training_stages': ancestors}
    for step in range(1, args.steps + 1):
        fraction = min(1., processed / args.token_budget) if args.token_budget else (step - 1) / max(1, args.steps - 1)
        warmup = min(1., step / args.warmup)
        lr = args.lr * warmup * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * fraction)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        running = 0.
        for _ in range(args.accumulate):
            x, y = source.batch(args.batch_size)
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, y)
            if not bool(torch.isfinite(loss).cpu()):
                raise RuntimeError(f'Nonfinite loss at {step}')
            (loss / args.accumulate).backward()
            processed += int((y != -100).sum().cpu())
            running += float(loss.detach().cpu()) / args.accumulate
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % args.log_every == 0 or step == 1:
            record = {'step': step, 'train_loss': running, 'lr': lr, 'processed_tokens': processed,
                      'elapsed_sec': time.monotonic() - started,
                      'gradient_norm': float(norm.detach().cpu()), 'seen_songs': len(source.seen)}
            history.append(record)
            print(json.dumps(record), flush=True)
        complete = step == args.steps or bool(args.token_budget and processed >= args.token_budget)
        if step % args.eval_every == 0 or complete:
            metadata.update(step=step, processed_tokens=processed, elapsed_sec=time.monotonic() - started,
                            seen_song_ids=sorted(source.seen),
                            total_processed_tokens_including_ancestors=processed + sum(s['processed_tokens'] for s in ancestors))
            if not args.all_songs:
                validation = SongWindows(args.corpus, 'val', min(args.context, args.eval_context), args.seed + 17)
                value = validate(model, validation, device, args.eval_batches, min(2, args.batch_size))
                metadata['validation_loss'] = value
                record = {'step': step, 'validation_loss': value, 'validation_context': min(args.context, args.eval_context),
                          'validation_song_ids': [str(validation.corpus['song_ids'][i]) for i in validation.songs],
                          'validation_seed': args.seed + 17}
                history.append(record); print(json.dumps(record), flush=True)
                if value < best:
                    best = value
                    save_checkpoint(args.output_dir / 'best.pt', model, metadata)
            else:
                metadata['validation_loss'] = None  # No held-out claim after full-data fitting.
            save_checkpoint(args.output_dir / 'last.pt', model, metadata)
            if step in args.keep_steps or complete:
                save_checkpoint(args.output_dir / f'step_{step:06d}.pt', model, metadata)
            (args.output_dir / 'history.json').write_text(json.dumps({'metadata': metadata, 'history': history}, indent=2) + '\n')
        if complete:
            break
    if args.token_budget and processed < args.token_budget:
        raise RuntimeError('Maximum steps reached before the specified token budget')
    if set(source.seen) != set(metadata['training_song_ids']):
        raise RuntimeError('Run ended without exposing every selected song')
    print(json.dumps({'completed': True, 'steps': step, 'output_dir': str(args.output_dir),
                      'best_validation_loss': None if args.all_songs else best}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--mode', choices=['basic', 'roles'], default='basic')
    parser.add_argument('--initialize-from', type=Path)
    parser.add_argument('--all-songs', action='store_true')
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--token-budget', type=int, default=0,
                        help='Stop after this many non-padding targets; steps becomes a safety limit')
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--model-context', type=int, default=4096)
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--eval-batches', type=int, default=24)
    parser.add_argument('--eval-context', type=int, default=2048)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--keep-steps', nargs='*', type=int, default=[500, 2000])
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
