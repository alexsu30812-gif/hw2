"""Review every complete continuation WAV using a local audio-language model.

This is machine auditory analysis, not a human listening study. Audio-only
responses are retained verbatim; measurable bar claims are verified separately.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/audio_review'
MODEL = ROOT / 'cache/audio_review_models/Qwen2-Audio-7B-Instruct-4bit'
os.environ['HF_HOME'] = str(ROOT / 'cache/audio_review_models/hf')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.stt.utils import load_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prompt_for(piece):
    parts = [f"Audio {s['clip']} covers {s['start_seconds']:.3f}–{s['end_seconds']:.3f} seconds, "
             f"bars {s['first_bar_1based']}–{s['last_bar_1based']}"
             + (' and the final release tail.' if s['includes_release_tail'] else '.')
             for s in piece['segments']]
    return (
        'The audio inputs are consecutive, non-overlapping parts of ONE instrumental performance, '
        'in chronological order; concatenate them mentally to assess the whole recording. '
        + ' '.join(parts) + '\n'
        'Review the actual audio, including every supplied part. For each Audio N give one concise '
        'observation about audible texture, register, rhythmic regularity, repetition or change, '
        'and say if the evidence is uncertain. Then describe: (a) the transition from the first '
        'eight bars into the following material; (b) whether later parts develop or remain similar; '
        '(c) the ending: a resolved close, a fade, an abrupt stop, or unclear. '
        'Do not invent note names, exact chords, exact tempo, composer, title, genre, or quality scores. '
        'Do not assume an abrupt change is an error. If an aspect is not reliably audible, say it is unclear. '
        'Return JSON with keys clips (list of {audio,observation,confidence}), opening_transition, '
        'later_development, ending, uncertainties. Keep the answer under 450 words. '
        'You have no MIDI, parameter settings, reference performance, or preferred-version label.'
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coverage = json.loads((OUT / 'input_coverage.json').read_text())
    if not coverage['complete_wav_frame_coverage'] or coverage['piece_count'] != 9:
        raise ValueError('All nine complete inputs are required')
    setup_path = ROOT / 'analysis/audio_review_model_setup.json'
    setup = json.loads(setup_path.read_text())
    source = Path(__import__('mlx_audio.stt.models.qwen2_audio.qwen2_audio',
                             fromlist=['Model']).__file__)
    version_names = ('mlx-audio', 'mlx', 'mlx-metal', 'transformers', 'huggingface-hub', 'numpy', 'scipy', 'soundfile')
    provenance = {'model_repository': 'mlx-community/Qwen2-Audio-7B-Instruct-4bit',
        'download_setup': setup, 'runtime_versions': {n: importlib.metadata.version(n) for n in version_names},
        'backend_source_sha256': digest(source), 'input_coverage_sha256': digest(OUT / 'input_coverage.json'),
        'driver_sha256': digest(__file__), 'decoding': {'temperature': 0., 'max_tokens': 900, 'prefill_step_size': 512}}
    results_path = OUT / 'raw_audio_observations.json'
    previous = json.loads(results_path.read_text()) if results_path.exists() else {}
    if previous and previous.get('provenance') != provenance:
        raise ValueError('Existing observations use different inputs/model/code; preserve them before a new run')
    completed = {p['blind_id']: p for p in previous.get('pieces', [])}
    print('Loading local evaluation-only audio model', flush=True)
    model = load_model(str(MODEL))
    mx.random.seed(20261025)
    extractions = []

    def official_features(self, audio):
        wave = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not 0 < len(wave) <= 480000:
            raise ValueError('No audio clip may be silently truncated')
        features = self._processor.feature_extractor(
            wave, sampling_rate=16000, return_tensors='np', padding='max_length',
            max_length=480000, truncation=True)['input_features']
        if features.shape != (1, 128, 3000):
            raise ValueError(f'Unexpected official Whisper features: {features.shape}')
        extractions.append({'input_frames': len(wave),
            'input_sha256_float32': hashlib.sha256(wave.tobytes()).hexdigest(),
            'feature_shape': list(features.shape),
            'official_feature_sha256_float32': hashlib.sha256(features.astype(np.float32).tobytes()).hexdigest(),
            'audio_tokens': 750})
        return mx.array(features, dtype=self.audio_tower.conv1.weight.dtype), 750

    model._extract_features = MethodType(official_features, model)
    for piece in coverage['pieces']:
        if piece['blind_id'] in completed:
            continue
        prompt = prompt_for(piece)
        waves = []
        for segment in piece['segments']:
            path = ROOT / segment['path']
            if digest(path) != segment['sha256']:
                raise ValueError('Input clip changed after coverage verification')
            wave, rate = sf.read(path, dtype='float32')
            if rate != 16000 or wave.ndim != 1 or len(wave) != segment['frames']:
                raise ValueError('Input clip format changed')
            waves.append(wave)
        extractions.clear()
        began = time.monotonic()
        print(json.dumps({'reviewing': piece['blind_id'], 'clips': len(waves),
                          'source_seconds': piece['original_seconds']}), flush=True)
        response = model.generate(waves, prompt=prompt, temperature=0., max_tokens=900,
                                  prefill_step_size=512)
        if len(extractions) != len(waves):
            raise ValueError('The model did not encode every provided clip')
        if [x['input_frames'] for x in extractions] != [s['frames'] for s in piece['segments']]:
            raise ValueError('The model saw different clip lengths')
        for extraction, wave in zip(extractions, waves):
            if extraction['input_sha256_float32'] != hashlib.sha256(wave.tobytes()).hexdigest():
                raise ValueError('The model frontend received different waveform samples')
        parsed = None
        try:
            text = response.text.strip()
            if text.startswith('```'):
                text = text.split('\n', 1)[1].rsplit('```', 1)[0]
            parsed = json.loads(text)
        except (ValueError, IndexError):
            pass  # The complete raw response is the source of truth.
        record = {'blind_id': piece['blind_id'], 'prompt_id': piece['prompt_id'],
            'configuration_id': piece['configuration_id'], 'source_wav': piece['source_wav'],
            'source_wav_sha256': piece['source_wav_sha256'], 'full_frame_coverage_verified': True,
            'all_frontend_waveform_hashes_match': True,
            'original_seconds': piece['original_seconds'], 'segments': piece['segments'],
            'prompt': prompt, 'frontend_extractions': list(extractions),
            'raw_response': response.text, 'parsed_response': parsed,
            'prompt_tokens': response.prompt_tokens, 'generation_tokens': response.generation_tokens,
            'possibly_token_limited': response.generation_tokens >= 900,
            'review_seconds': time.monotonic() - began, 'peak_memory_bytes': mx.get_peak_memory(),
            'review_type': 'machine_audio_language_analysis', 'human_listening': False}
        completed[piece['blind_id']] = record
        payload = {'completed': len(completed) == 9, 'provenance': provenance,
            'method': 'Every complete WAV is encoded as ordered, bar-aligned <=30s clips in one multimodal request per piece. Official HF Whisper frontend replaces the numerically different MLX native frontend. Labels, MIDI metrics, chords and selection results are withheld from the model.',
            'pieces': list(completed.values()), 'pieces_completed': len(completed),
            'human_listening_performed': False,
            'limitations': ['Audio-language descriptions may be generic or hallucinated and are not listener ratings.',
                'Each audio encoder clip has at most30s context, although all clip embeddings enter one language-model request.',
                'Time and bar coordinates come from verified waveform segmentation, not the backend STTOutput.segments latency field.',
                'Model text is cross-checked with MIDI evidence; model words alone do not verify a chord, a cadence or a preferred configuration.']}
        write(results_path, payload)
        print(json.dumps({'reviewed': piece['blind_id'], 'completed': len(completed),
                          'seconds': record['review_seconds'], 'generation_tokens': response.generation_tokens}), flush=True)
        gc.collect()
        mx.clear_cache()
    print('ALL_NINE_COMPLETE_AUDIO_INPUTS_REVIEWED', flush=True)


if __name__ == '__main__':
    main()
