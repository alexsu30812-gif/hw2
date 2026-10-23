"""Make figures and a local listening index from actual generated samples."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from music_codec import Vocabulary, parse_tokens

ROOT = Path(__file__).resolve().parents[1]


def plot_selected(manifest_path, output_dir):
    manifest = json.loads(Path(manifest_path).read_text())
    if not manifest.get('complete'):
        raise ValueError('Cannot present an incomplete run as selected music')
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    sample = manifest['samples'][0]  # Fixed first seed; not cherry-picked by appearance.
    parsed = parse_tokens(sample['tokens'])
    bars = parsed['bars']; n = len(bars)
    chroma = np.zeros((n, 12))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), width_ratios=(1.8, 1), constrained_layout=True)
    for note in parsed['notes']:
        bar = note['bar']; width = bars[bar]['meter'] * 8
        start = bar + note['pos'] / width
        end = min(n, start + note['duration'] / width)
        axes[0].plot([start, end], [note['pitch']] * 2, color='black', lw=1.2, alpha=.65)
        chroma[bar, note['pitch'] % 12] += 1
    axes[0].set(xlim=(0, n), xlabel='Bar (zero-based)', ylabel='MIDI pitch',
                title=f"Fixed first-seed sample: {sample['song_name']}")
    axes[0].set_xticks(range(0, n + 1, 4)); axes[0].grid(axis='x', alpha=.2)
    norm = np.linalg.norm(chroma, axis=1)
    normalized = np.divide(chroma, norm[:, None], out=np.zeros_like(chroma), where=norm[:, None] > 0)
    similarity = normalized @ normalized.T
    im = axes[1].imshow(similarity, vmin=0, vmax=1, origin='lower', cmap='Greys', aspect='equal')
    axes[1].set(xlabel='Bar', ylabel='Bar', title='Bar pitch-class similarity\n(cosine; separate from SI)')
    fig.colorbar(im, ax=axes[1], label='Cosine similarity')
    fig.savefig(output_dir / 'selected_structure.png', dpi=170); plt.close(fig)
    (output_dir / 'selected_structure_notes.json').write_text(json.dumps({
        'sample': sample['song_name'], 'selection': 'first saved seed, not chosen by its structure plot',
        'similarity_definition': 'cosine of per-bar12-pitch-class note-count vectors; emptybar vectors remain0',
        'limitation': 'This is a diagnostic plot, not MusDr SI, and similarity alone does not establish musical form.',
        'n_bars': n, 'notes': len(parsed['notes'])}, indent=2) + '\n')


def comparisons(output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    series = {}
    reference = ROOT / 'results/evaluation/real_reference/real_reference_per_piece.csv'
    if reference.exists():
        series['Real reference'] = pd.read_csv(reference)
    for name in ('t08', 't10', 't12', 'k16', 'k32'):
        path = ROOT / f'results/evaluation/basic_a/{name}/{name}_per_piece.csv'
        if path.exists(): series[name] = pd.read_csv(path)
    if len(series) < 2: return
    cols = [('H4', 'H4: pitch-class entropy'), ('GS', 'GS: official normalized onset similarity')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, (short, title) in zip(axes, cols):
        values = []
        for frame in series.values():
            matching = [c for c in frame if c == short or c.startswith(short + ' ') or c.startswith(short + '(')]
            if not matching: raise ValueError(f'Metric column {short} not found: {list(frame)}')
            values.append(frame[matching[0]].to_numpy())
        axis.boxplot(values, tick_labels=list(series), showmeans=True)
        axis.set_title(title); axis.grid(axis='y', alpha=.2)
    fig.savefig(output_dir / 'sampling_distributions.png', dpi=170); plt.close(fig)


def listening_index(destination):
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    groups = []
    for folder in ('basic_a', 'basic_b', 'bonus'):
        for manifest_path in sorted((ROOT / 'results' / folder).rglob('manifest.json')):
            manifest = json.loads(manifest_path.read_text())
            items = []
            for sample in manifest.get('samples', []):
                midi = Path(sample.get('midi_path', ''))
                if not midi.is_absolute(): midi = ROOT / midi
                wav = midi.with_suffix('.wav')
                if not wav.is_file(): continue
                import os
                url = html.escape(os.path.relpath(wav, destination.parent), quote=True)
                label = html.escape(f"{folder}/{manifest_path.parent.name}/{sample['song_name']}")
                items.append(f'<p>{label}<br><audio controls preload="none" src="{url}"></audio></p>')
            if items: groups.append(''.join(items))
    destination.write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>HW2 試聽</title>'
        '<style>body{font-family:system-ui;max-width:960px;margin:40px auto;line-height:1.8}audio{width:100%}section{border-top:1px solid #bbb;padding:20px}</style>'
        '<h1>HW2 生成成果試聽</h1><p>所有音訊由自己的模型生成 MIDI 後渲染。請比較節奏、和聲、樂句發展與續寫邊界；此頁不代表已進行人工評分。</p>'
        + ''.join('<section>' + group + '</section>' for group in groups) + '</html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures')
    parser.add_argument('--listening-index', type=Path, default=ROOT / 'listening.html')
    args = parser.parse_args()
    if args.selected_manifest: plot_selected(args.selected_manifest, args.output_dir)
    comparisons(args.output_dir)
    listening_index(args.listening_index)


if __name__ == '__main__':
    main()
