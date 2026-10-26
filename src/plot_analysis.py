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
    case_path = ROOT / 'analysis/case_studies/bar_case_studies.json'
    cases = json.loads(case_path.read_text()) if case_path.is_file() else {}
    continuation_segments = {(row['prompt_id'], row['configuration_id']): row['segments']
                             for row in cases.get('basic_b', {}).get('rows', [])}
    groups = []
    audio_count = 0
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
                audio_id = f'audio-{audio_count}'
                audio_count += 1
                segment_table = ''
                if folder == 'basic_b':
                    segments = continuation_segments.get((sample['song_name'], manifest_path.parent.name), [])
                    if segments:
                        if [segment['bar_range'] for segment in segments] != [[1, 8], [9, 16], [17, 24], [25, 32]]:
                            raise ValueError('Continuation comparison must retain all four original 8-bar windows')
                        rows = []
                        for segment in segments:
                            start_bar, end_bar = segment['bar_range']
                            t0, t1 = segment['time_range']
                            phase = '原提示' if start_bar == 1 else '生成'
                            rows.append(f'<tr><td><button type="button" class="bar-seek" '
                                f'data-audio-id="{audio_id}" data-time="{float(t0)!r}" '
                                f'aria-label="{label} 從第 {start_bar} 小節播放">▶ 第 {start_bar} 小節</button></td>'
                                f'<td>{start_bar}–{end_bar}（{phase}）</td><td>{t0:.3f}–{t1:.3f} 秒</td>'
                                f'<td>{segment["notes_per_bar"]:.3f}</td><td>{segment["mean_pitch"]:.2f}</td></tr>')
                        segment_table = ('<div class="segment-comparison"><table><caption>每 8 小節比較：'
                            '第 1–8 小節是原提示，第 9 小節起續寫。時間從此 WAV 開頭計算。</caption>'
                            '<thead><tr><th>跳播</th><th>小節</th><th>WAV 時間</th><th>音符數／小節</th>'
                            '<th>平均 MIDI 音高</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
                            '<p class="note">數值來自儲存的 token 音符起音；平均音高不是聽感評分。'
                            '按鈕會暫停其他曲目，再從所選小節播放；可在同一提示的三種設定間逐段比較。</p></div>')
                items.append(f'<article><p>{label}<br><audio id="{audio_id}" controls preload="none" src="{url}"></audio></p>'
                             + segment_table + '</article>')
            if items: groups.append(''.join(items))
    destination.write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>HW2 試聽</title>'
        '<style>body{font-family:system-ui;max-width:960px;margin:40px auto;line-height:1.8}audio{width:100%}section{border-top:1px solid #bbb;padding:20px}'
        'article{margin-bottom:28px}.segment-comparison{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem;text-align:left}'
        'caption{text-align:left;margin-bottom:8px}th,td{border-bottom:1px solid #ddd;padding:6px 8px;white-space:nowrap}'
        'button{font:inherit;cursor:pointer;padding:5px 9px}.note{font-size:.88rem;color:#444}#playback-status{min-height:1.8em}</style>'
        '<h1>HW2 生成成果試聽</h1><p>所有音訊由自己的模型生成 MIDI 後渲染。請比較節奏、和聲、樂句發展與續寫邊界；此頁不代表已進行人工評分。</p>'
        '<p><strong>沒有真人試聽評分。</strong>先前 AI audio critic 未通過 silence control（靜音對照檢查），'
        '其聽感文字不採信，也不作為報告結論。此頁供下載／解壓後在本機逐段聽比，不代表已有人完整聽完。</p>'
        '<p id="playback-status" role="status" aria-live="polite"></p>'
        + ''.join('<section>' + group + '</section>' for group in groups)
        + '<script>\n'
        'function pauseOtherAudio(target) {\n'
        '  document.querySelectorAll("audio").forEach(audio => { if (audio !== target) audio.pause(); });\n'
        '}\n'
        'document.addEventListener("play", event => {\n'
        '  if (event.target.tagName === "AUDIO") pauseOtherAudio(event.target);\n'
        '}, true);\n'
        'document.addEventListener("click", async event => {\n'
        '  const button = event.target.closest("button.bar-seek");\n'
        '  if (!button) return;\n'
        '  const audio = document.getElementById(button.dataset.audioId);\n'
        '  const time = Number(button.dataset.time);\n'
        '  const status = document.getElementById("playback-status");\n'
        '  if (!audio || !Number.isFinite(time) || time < 0) return;\n'
        '  pauseOtherAudio(audio);\n'
        '  try {\n'
        '    if (audio.readyState === 0) {\n'
        '      await new Promise((resolve, reject) => {\n'
        '        const cleanup = () => { audio.removeEventListener("loadedmetadata", loaded); audio.removeEventListener("error", failed); };\n'
        '        const loaded = () => { cleanup(); resolve(); };\n'
        '        const failed = () => { cleanup(); reject(new Error("無法載入音訊")); };\n'
        '        audio.addEventListener("loadedmetadata", loaded); audio.addEventListener("error", failed);\n'
        '        audio.preload = "metadata"; audio.load();\n'
        '      });\n'
        '    }\n'
        '    pauseOtherAudio(audio);\n'
        '    audio.currentTime = Number.isFinite(audio.duration) ? Math.min(time, audio.duration) : time;\n'
        '    await audio.play();\n'
        '    status.textContent = button.getAttribute("aria-label") + "（" + time.toFixed(3) + " 秒）";\n'
        '  } catch (error) {\n'
        '    status.textContent = "播放未開始，請確認在本機開啟並保留 WAV 的相對位置，或按播放器播放鍵。";\n'
        '  }\n'
        '});\n'
        '</script></html>')


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
