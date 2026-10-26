"""Download a public, evaluation-only audio model into the isolated local cache."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "audio_review_models"
os.environ["HF_HOME"] = str(CACHE / "hf")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    repo_id = "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
    info = HfApi(token=False).model_info(repo_id, files_metadata=True)
    local = CACHE / "Qwen2-Audio-7B-Instruct-4bit"
    print(json.dumps({"repo_id": repo_id, "revision": info.sha,
                      "download_dir": str(local)}, indent=2), flush=True)
    snapshot_download(
        repo_id=repo_id,
        revision=info.sha,
        token=False,
        local_dir=local,
        allow_patterns=["*.json", "*.txt", "*.md", "*.safetensors"],
    )
    files = []
    source_files = {item.rfilename: item for item in info.siblings}
    for path in sorted(local.glob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            item = {"name": path.name, "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest()}
            lfs = getattr(source_files.get(path.name), "lfs", None)
            expected = getattr(lfs, "sha256", None)
            if expected:
                item["expected_source_sha256"] = expected
                item["matches_source_sha256"] = item["sha256"] == expected
                if not item["matches_source_sha256"]:
                    raise ValueError(f"Downloaded weight digest mismatch: {path.name}")
            files.append(item)
    record = {
        "repo_id": repo_id,
        "revision": info.sha,
        "source_url": "https://huggingface.co/" + repo_id,
        "base_model": "Qwen/Qwen2-Audio-7B-Instruct",
        "model_license": "Apache-2.0",
        "runtime_license": "MIT",
        "purpose": "Evaluation-only WAV input; no training or generation changes",
        "local_directory": str(local.relative_to(ROOT)),
        "download_bytes": sum(f["bytes"] for f in files),
        "files": files,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx-audio", "mlx", "mlx-metal", "transformers",
                         "huggingface-hub", "numpy", "scipy", "soundfile",
                         "jinja2", "markupsafe")
        },
        "audio_input": {
            "sample_rate": 16000,
            "channels": 1,
            "per_clip_max_samples": 480000,
            "per_clip_max_seconds": 30,
            "multiple_clips": "List of waveforms merged into one multi-audio context",
            "timestamp_warning": "STTOutput.segments records inference time, not audio time",
        },
        "evaluation_frontend_adjustment": {
            "frontend": "Official local Hugging Face WhisperFeatureExtractor",
            "replaces": "MLX runtime _extract_features only",
            "reason": "Numerical frontend differences measured before review",
            "comparison_evidence": "analysis/audio_review_frontend_comparison.json",
            "composition_or_training_changes": False,
        },
    }
    output = ROOT / "analysis" / "audio_review_model_setup.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ready": True, "record": str(output),
                      "download_bytes": record["download_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
