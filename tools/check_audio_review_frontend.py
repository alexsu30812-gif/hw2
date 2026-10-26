"""Numerically compare the local MLX audio frontend with the official HF one."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "cache" / "audio_review_models" / "hf")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import mlx.core as mx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor
from mlx_audio.stt.models.qwen2_audio.config import ModelConfig
from mlx_audio.stt.models.qwen2_audio.qwen2_audio import Model


def main() -> None:
    mx.set_default_device(mx.cpu)
    source = ROOT / "results" / "basic_b" / "t08" / "001.wav"
    waveform, source_rate = sf.read(source, dtype="float32", always_2d=True)
    source_seconds = len(waveform) / source_rate
    mono = waveform[:20 * source_rate].mean(axis=1)
    divisor = int(np.gcd(source_rate, 16000))
    audio = resample_poly(mono, 16000 // divisor, source_rate // divisor).astype(np.float32)
    assert len(audio) <= 480000
    model_dir = ROOT / "cache" / "audio_review_models" / "Qwen2-Audio-7B-Instruct-4bit"
    frontend = WhisperFeatureExtractor.from_pretrained(str(model_dir), local_files_only=True)
    official = frontend(audio, sampling_rate=16000, return_tensors="np",
                        padding="max_length", max_length=480000, truncation=True)["input_features"]
    stub = SimpleNamespace(config=ModelConfig(), audio_tower=SimpleNamespace(
        conv1=SimpleNamespace(weight=mx.array([0.0], dtype=mx.float32))))
    Model._init_mel_constants(stub)
    native, native_tokens = Model._extract_features(stub, audio)
    native_np = np.asarray(native)
    common_frames = min(native_np.shape[-1], official.shape[-1])
    delta = native_np[..., :common_frames] - official[..., :common_frames]
    record = {
        "source_wav": str(source.relative_to(ROOT)),
        "source_wav_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_full_seconds": source_seconds,
        "test_interval_seconds": [0.0, len(mono) / source_rate],
        "test_is_full_piece": False,
        "sample_rate": 16000,
        "input_samples": len(audio),
        "native_shape": list(native_np.shape),
        "official_shape": list(official.shape),
        "native_audio_tokens": native_tokens,
        "compared_frames": common_frames,
        "feature_rmse": float(np.sqrt(np.mean(delta ** 2))),
        "feature_max_abs_error": float(np.max(np.abs(delta))),
        "official_feature_extractor": frontend.to_dict(),
        "decision": "Use official Hugging Face WhisperFeatureExtractor for all audio review",
        "scope": "Evaluation frontend only; no MIDI, composition weights, or generation changes",
    }
    output = ROOT / "analysis" / "audio_review_frontend_comparison.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: val for key, val in record.items()
                      if key != "official_feature_extractor"}, indent=2))


if __name__ == "__main__":
    main()
