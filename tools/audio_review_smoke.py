"""A partial-clip audio input smoke test, explicitly not a listening study."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "cache" / "audio_review_models" / "Qwen2-Audio-7B-Instruct-4bit"
os.environ["HF_HOME"] = str(ROOT / "cache" / "audio_review_models" / "hf")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import mlx.core as mx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from mlx_audio.stt.utils import load_model


def official_extract_features(self, audio):
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(waveform) > 480000:
        raise ValueError("Audio review clips must not exceed 30 seconds at 16 kHz")
    inputs = self._processor.feature_extractor(
        waveform, sampling_rate=16000, return_tensors="np",
        padding="max_length", max_length=480000, truncation=True,
    )["input_features"]
    if inputs.shape != (1, 128, 3000):
        raise ValueError(f"Unexpected official frontend shape: {inputs.shape}")
    return mx.array(inputs, dtype=self.audio_tower.conv1.weight.dtype), 750


def main():
    source = ROOT / "results" / "basic_b" / "t08" / "001.wav"
    waveform, rate = sf.read(source, dtype="float32", always_2d=True)
    clip = waveform[:20 * rate].mean(axis=1)
    divisor = int(np.gcd(rate, 16000))
    audio = resample_poly(clip, 16000 // divisor, rate // divisor).astype(np.float32)
    prompt = (
        "The two audio clips are consecutive parts of one recording. "
        "Describe the audible sounds in this recording. Identify the main instrument "
        "or instruments and whether there is speech or singing. Then describe the "
        "rhythmic texture and any audible changes. Base the answer only on the audio. "
        "Do not infer a composer, title, exact key, exact chords, or exact tempo. "
        "If something is uncertain, say so."
    )
    began = time.monotonic()
    print("Loading local evaluation model", flush=True)
    model = load_model(str(MODEL))
    model._extract_features = MethodType(official_extract_features, model)
    loaded = time.monotonic()
    print("Reviewing one 20-second waveform smoke clip", flush=True)
    clips = [audio[:10 * 16000], audio[10 * 16000:]]
    result = model.generate(clips, prompt=prompt, max_tokens=220, temperature=0.0)
    reviewed = time.monotonic()
    print("Reviewing matched 20-second all-zero silence control", flush=True)
    silent_clips = [np.zeros_like(clip) for clip in clips]
    control = model.generate(silent_clips, prompt=prompt, max_tokens=220, temperature=0.0)
    record = {
        "test_is_full_piece": False,
        "purpose": "Verify non-ASR audio understanding and local runtime only",
        "source_wav": str(source.relative_to(ROOT)),
        "source_wav_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_full_seconds": len(waveform) / rate,
        "input_interval_seconds": [0, 20],
        "input_sample_rate": 16000,
        "input_samples": len(audio),
        "input_waveform_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "input_clip_intervals_seconds": [[0, 10], [10, 20]],
        "multiple_clips_one_context": True,
        "input_features": "Official local Hugging Face WhisperFeatureExtractor",
        "prompt": prompt,
        "raw_response": result.text,
        "generation_tokens": result.generation_tokens,
        "output_reached_token_limit": result.generation_tokens >= 220,
        "prompt_tokens": result.prompt_tokens,
        "load_seconds": loaded - began,
        "review_seconds": reviewed - loaded,
        "silence_control": {
            "input_samples": len(audio),
            "input_waveform_sha256": hashlib.sha256(np.zeros_like(audio).tobytes()).hexdigest(),
            "zero_waveform": True,
            "same_prompt_and_decode": True,
            "prompt_tokens": control.prompt_tokens,
            "raw_response": control.text,
            "generation_tokens": control.generation_tokens,
            "output_reached_token_limit": control.generation_tokens >= 220,
            "review_seconds": time.monotonic() - reviewed,
            "interpretation": "Audible music described for silence is hallucination evidence",
        },
        "peak_memory_bytes": mx.get_peak_memory(),
        "limitation": "Machine audio description is not a human listening judgment",
    }
    output = ROOT / "analysis" / "audio_review_smoke.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
