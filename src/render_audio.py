"""Render real piano WAVs: macOS native sampler or FluidSynth with a user soundfont.

macOS uses Apple's installed General MIDI bank (not distributed here). The portable
FluidSynth backend requires --soundfont or HW2_SOUNDFONT; no model weights are used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import wave
import numpy as np
import pretty_midi
import mido

ROOT = Path(__file__).resolve().parents[1]


def inspect_wav(path: str | Path) -> dict:
    with wave.open(str(path), 'rb') as f:
        if f.getsampwidth() != 2 or f.getcomptype() != 'NONE':
            raise ValueError(f'{path}: expected uncompressed PCM16 WAV')
        rate, channels, n_frames = f.getframerate(), f.getnchannels(), f.getnframes()
        audio = np.frombuffer(f.readframes(n_frames), dtype='<i2').astype(np.float64) / 32768
    if not n_frames or not np.isfinite(audio).all():
        raise ValueError(f'{path}: empty or nonfinite audio')
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms <= 1e-6:
        raise ValueError(f'{path}: silent rendering')
    return dict(sample_rate=rate, channels=channels, frames=n_frames,
                duration_seconds=n_frames/rate, peak=peak, rms=rms,
                clipped_fraction=float(np.mean(np.abs(audio) >= 32766/32768)))


def native_binary() -> Path:
    source = ROOT / 'tools/render_native.swift'
    binary = ROOT / 'tools/render_native'
    if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
        cache = ROOT / 'cache/swift-module-cache'
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(['swiftc', '-module-cache-path', str(cache), str(source), '-o', str(binary)], check=True)
    return binary


def render_midi(midi_path: str | Path, output_path: str | Path,
                sample_rate: int = 22050, tail_seconds: float = 2.0,
                backend: str = 'auto', soundfont: str | Path | None = None) -> dict:
    midi_path, output_path = Path(midi_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    if not any(i.notes for i in midi.instruments):
        raise ValueError(f'{midi_path}: no notes')
    if backend == 'auto':
        backend = 'native' if platform.system() == 'Darwin' else 'fluidsynth'
    if backend == 'native':
        if platform.system() != 'Darwin':
            raise ValueError('Native rendering requires macOS; use --backend fluidsynth --soundfont bank.sf2')
        events = []
        channels = [c for c in range(16) if c != 9]
        if len(midi.instruments) > len(channels):
            raise ValueError('More than 15 melodic instruments are unsupported by the native renderer')
        for inst, channel in zip(midi.instruments, channels):
            for n in inst.notes:
                for seconds, on in ((n.start, True), (n.end, False)):
                    events.append(dict(frame=round(seconds*sample_rate), pitch=int(n.pitch),
                                       velocity=int(n.velocity), channel=channel, on=on))
        events.sort(key=lambda e: (e['frame'], e['on'], e['channel'], e['pitch']))
        request = dict(output=str(output_path.resolve()), sample_rate=sample_rate,
                       duration_frames=round((max(midi.get_end_time(), mido.MidiFile(str(midi_path)).length)+tail_seconds)*sample_rate), events=events)
        with tempfile.TemporaryDirectory(prefix='hw2-render-') as td:
            request_path = Path(td)/'request.json'
            request_path.write_text(json.dumps(request))
            subprocess.run([str(native_binary()), str(request_path)], check=True, capture_output=True, text=True)
        sound_source = 'macOS CoreAudio gs_instruments.dls, General MIDI acoustic grand piano'
    elif backend == 'fluidsynth':
        sf = soundfont or os.environ.get('HW2_SOUNDFONT')
        if not sf or not Path(sf).is_file():
            raise ValueError('FluidSynth requires an existing --soundfont or HW2_SOUNDFONT .sf2/.sf3 file')
        exe = shutil.which('fluidsynth')
        if not exe:
            raise ValueError('fluidsynth executable not found; see README audio-rendering instructions')
        subprocess.run([exe, '-ni', '-g', '0.4', '-r', str(sample_rate), '-F', str(output_path), str(sf), str(midi_path)],
                       check=True, capture_output=True, text=True)
        sound_source = f'FluidSynth; soundfont SHA256 {hashlib.sha256(Path(sf).read_bytes()).hexdigest()}'
    else:
        raise ValueError(f'Unknown backend: {backend}')
    audit = inspect_wav(output_path)
    if audit['clipped_fraction'] > 1e-4:
        raise ValueError(f'{output_path}: excessive clipping ({audit["clipped_fraction"]:.4%})')
    audit.update(midi=str(midi_path), wav=str(output_path), backend=backend, sound_source=sound_source,
                 midi_sha256=hashlib.sha256(midi_path.read_bytes()).hexdigest(),
                 wav_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest())
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--midi', type=Path)
    inputs.add_argument('--midi-dir', type=Path)
    p.add_argument('--output', type=Path, required=True, help='WAV path for --midi; output root for --midi-dir')
    p.add_argument('--backend', choices=['auto','native','fluidsynth'], default='auto')
    p.add_argument('--soundfont', type=Path)
    p.add_argument('--sample-rate', type=int, default=22050)
    p.add_argument('--tail-seconds', type=float, default=2.0)
    args = p.parse_args()
    if args.midi:
        pairs = [(args.midi, args.output)]
        manifest = args.output.with_suffix('.render.json')
    else:
        paths = sorted(args.midi_dir.rglob('*.mid'))
        if not paths:
            p.error('No MIDI files found')
        pairs = [(f, args.output/f.relative_to(args.midi_dir).with_suffix('.wav')) for f in paths]
        manifest = args.output/'render_manifest.json'
    rows = []
    for src, dst in pairs:
        rows.append(render_midi(src, dst, args.sample_rate, args.tail_seconds, args.backend, args.soundfont))
        print(f'Rendered {len(rows)}/{len(pairs)}: {dst}', flush=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(dict(count=len(rows), failures=0, files=rows), indent=2)+'\n')

if __name__ == '__main__':
    main()
