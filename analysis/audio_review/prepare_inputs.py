"""Prepare complete, blind-labelled audio inputs without omitting any WAV frame."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/audio_review'
CACHE = ROOT / 'cache/audio_review_inputs'
RATE = 16000
LIMIT = 30 * RATE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    pieces = []
    for config in ('t08', 't10', 't12'):
        manifest = json.loads((ROOT / f'results/basic_b/{config}/manifest.json').read_text())
        assert manifest['complete'] and len(manifest['samples']) == 3
        for sample in manifest['samples']:
            wav = ROOT / f"results/basic_b/{config}/{sample['song_name']}.wav"
            x, source_rate = sf.read(wav, dtype='float32', always_2d=True)
            assert np.isfinite(x).all() and x.shape[1] == 2
            divisor = int(np.gcd(source_rate, RATE))
            audio = resample_poly(x.mean(axis=1), RATE // divisor,
                                  source_rate // divisor).astype(np.float32)
            boundaries = [int(round(float(t) * RATE)) for t in sample['bar_times']]
            assert len(boundaries) == 33 and boundaries[0] == 0
            assert boundaries[-1] <= len(audio)
            start, first_bar, segments, rebuilt = 0, 0, [], []
            while start < len(audio):
                maximum = min(len(audio), start + LIMIT)
                if len(audio) <= maximum:
                    end, final_bar = len(audio), 32
                else:
                    target = min(first_bar + 8, 32)
                    candidates = [b for b in range(first_bar + 1, target + 1)
                                  if boundaries[b] <= maximum]
                    if not candidates:
                        raise ValueError('A bar exceeds the audio-model clip limit')
                    final_bar = max(candidates)
                    # Bundle the last bar with its release, instead of sending
                    # a two-second silent tail as an isolated semantic clip.
                    if final_bar == 32 and len(audio) > maximum:
                        final_bar -= 1
                    end = boundaries[final_bar]
                assert start < end <= maximum and final_bar > first_bar
                segment = audio[start:end]
                path = CACHE / f"{config}_{sample['song_name']}_{len(segments)+1:02d}.wav"
                sf.write(path, segment, RATE, subtype='FLOAT')
                readback, sr = sf.read(path, dtype='float32')
                assert sr == RATE and np.array_equal(readback, segment)
                rebuilt.append(readback)
                segments.append({'clip': len(segments) + 1,
                    'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                    'sample_rate': RATE, 'frames': len(segment),
                    'start_frame': start, 'end_frame': end,
                    'start_seconds': start / RATE, 'end_seconds': end / RATE,
                    'first_bar_1based': first_bar + 1, 'last_bar_1based': final_bar,
                    'includes_release_tail': end == len(audio)})
                start, first_bar = end, final_bar
            combined = np.concatenate(rebuilt)
            assert np.array_equal(combined, audio)
            assert sum(s['frames'] for s in segments) == len(audio)
            assert all(a['end_frame'] == b['start_frame'] for a, b in zip(segments, segments[1:]))
            pieces.append({'prompt_id': sample['song_name'], 'configuration_id': config,
                'source_wav': str(wav.relative_to(ROOT)), 'source_wav_sha256': sha(wav),
                'original_frames': len(x), 'original_sample_rate': source_rate,
                'original_seconds': len(x) / source_rate,
                'resampled_frames': len(audio), 'resampled_rate': RATE,
                'resampled_seconds': len(audio) / RATE,
                'resampled_sha256_float32': hashlib.sha256(audio.tobytes()).hexdigest(),
                'full_frame_coverage_verified': True, 'max_chunk_frames': max(s['frames'] for s in segments),
                'bars': 32, 'segments': segments})
    random.Random(20261025).shuffle(pieces)
    for index, piece in enumerate(pieces, 1):
        piece['blind_id'] = f'R{index:02d}'
    payload = {'status': 'inputs_complete', 'pieces': pieces,
        'piece_count': len(pieces), 'segment_count': sum(len(p['segments']) for p in pieces),
        'source_seconds_total': sum(p['original_seconds'] for p in pieces),
        'complete_wav_frame_coverage': all(p['full_frame_coverage_verified'] for p in pieces),
        'method': 'Original stereo PCM WAV -> channel mean -> polyphase resampling to mono16kHz float32. Bar-aligned clips at most30s include the entire release tail. Concatenated FLOAT WAV clips must exactly equal every resampled frame.',
        'blinding': 'Fixed shuffled order seed20261025. Only clip waveforms, chronological time ranges and bar labels are provided to the audio model; no configuration, best label, MIDI statistics, chord labels or expected failure observations.',
        'chunk_cache_excluded_from_submission': True}
    assert len(pieces) == 9 and payload['complete_wav_frame_coverage']
    (OUTPUT / 'input_coverage.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({key: value for key, value in payload.items() if key != 'pieces'}, indent=2))


if __name__ == '__main__':
    main()
