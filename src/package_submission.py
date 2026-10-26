"""Validate measured HW2 outputs and assemble an allowlisted inference package.

No training MIDI, tokenized training arrays, downloaded models or caches are
copied. A draft package is explicitly named DRAFT and is never called final.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import zipfile

from build_report import build_report, hosting_warnings, read_json, sample_path, validate_payload

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def owned_file(root, value):
    """Resolve only files inside this project; reject accidental dataset/cache paths."""
    path = Path(value)
    path = root / path if not path.is_absolute() else path
    if path.is_symlink():
        raise ValueError(f"Symlinked package input: {value}")
    path = path.resolve()
    try:
        relative = path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Package input is outside the project: {value}") from exc
    forbidden = {"POP909", "training", "cache", "caches", "__pycache__", ".venv", ".git", ".github_auth", "preprocessed"}
    if any(part in forbidden for part in relative.parts) or path.suffix.lower() in {".npz", ".npy", ".pkl"}:
        raise ValueError(f"Excluded training/cache input: {relative}")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or symlinked package input: {relative}")
    return path, relative


def wav_info(path):
    """Read basic RIFF WAV integrity without assuming only integer PCM."""
    with path.open("rb") as handle:
        head = handle.read(12)
        if len(head) != 12 or head[:4] != b"RIFF" or head[8:] != b"WAVE":
            raise ValueError(f"Not a RIFF WAV: {path}")
        fmt = None
        data_bytes = 0
        while True:
            raw = handle.read(8)
            if not raw:
                break
            if len(raw) != 8:
                raise ValueError(f"Truncated WAV chunk: {path}")
            kind, size = struct.unpack("<4sI", raw)
            if kind == b"fmt ":
                content = handle.read(size)
                if len(content) != size or size < 16:
                    raise ValueError(f"Invalid WAV fmt chunk: {path}")
                fmt = struct.unpack("<HHIIHH", content[:16])
            elif kind == b"data":
                data_bytes += size
                handle.seek(size, 1)
                if handle.tell() > path.stat().st_size:
                    raise ValueError(f"Truncated WAV data: {path}")
            else:
                handle.seek(size, 1)
            if size % 2:
                handle.seek(1, 1)
        if not fmt or not data_bytes:
            raise ValueError(f"WAV has no format/audio data: {path}")
        encoding, channels, rate, byte_rate, align, bits = fmt
        # The handout does not prescribe a sample rate. 8 kHz is only a basic
        # technical sanity check, not an assignment-specific requirement.
        if encoding not in (1, 3, 65534) or channels not in (1, 2) or rate < 8000 or align <= 0 or data_bytes % align:
            raise ValueError(f"Unsupported or inconsistent WAV format: {path}")
        seconds = data_bytes / align / rate
        if seconds <= 0:
            raise ValueError(f"Empty WAV: {path}")
        import numpy as np
        import soundfile as sf
        energy, count, peak = 0.0, 0, 0.0
        with sf.SoundFile(path) as audio:
            for block in audio.blocks(blocksize=65536, dtype="float64", always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite WAV samples: {path}")
                energy += float(np.sum(block * block))
                count += block.size
                peak = max(peak, float(np.max(np.abs(block))))
        rms = (energy / count) ** 0.5 if count else 0.0
        if rms <= 1e-6:
            raise ValueError(f"Silent WAV rendering: {path}")
        return {"sample_rate": rate, "channels": channels, "bits": bits, "seconds": seconds,
                "peak": peak, "rms": rms}


def midi_info(path):
    import mido
    midi = mido.MidiFile(path)
    if midi.type not in (0, 1) or midi.ticks_per_beat <= 0:
        raise ValueError(f"Unsupported MIDI type/timebase: {path}")
    notes = sum(msg.type == "note_on" and msg.velocity > 0 for track in midi.tracks for msg in track)
    names = sorted({str(msg.name).upper() for track in midi.tracks for msg in track if msg.type == "track_name"})
    if not notes or midi.length <= 0:
        raise ValueError(f"Empty MIDI: {path}")
    return {"notes": notes, "seconds": midi.length, "track_names": names}


def submission_samples(payload):
    a = payload.get("basic_a", {})
    selected = next((c for c in a.get("configurations", []) if c.get("id") == a.get("selected_configuration")), None)
    if selected is None:
        raise ValueError("No selected basic-a configuration")
    samples = [("basic_a", s) for s in selected.get("submission_samples", selected.get("samples", []))]
    samples += [("basic_b", s) for s in payload.get("basic_b", {}).get("results", [])]
    if payload.get("bonus", {}).get("attempted"):
        samples += [("bonus", s) for s in payload.get("bonus", {}).get("results", [])]
    counts = {task: sum(t == task for t, _ in samples) for task in ("basic_a", "basic_b", "bonus")}
    expected = {"basic_a": 20, "basic_b": 9, "bonus": 10 if payload.get("bonus", {}).get("attempted") else 0}
    if counts != expected:
        raise ValueError(f"Submission sample counts do not match {expected}: {counts}")
    return samples, counts


def validate_sample_files(root, payload):
    samples, counts = submission_samples(payload)
    sources, checks, seen = [], [], set()
    for task, item in samples:
        mid, midrel = owned_file(root, sample_path(item, "midi") or "MISSING_MIDI")
        wav, wavrel = owned_file(root, sample_path(item, "wav") or "MISSING_WAV")
        if mid.suffix.lower() not in (".mid", ".midi") or wav.suffix.lower() != ".wav":
            raise ValueError("Listening samples must have MIDI and WAV extensions")
        if str(midrel) in seen or str(wavrel) in seen:
            raise ValueError(f"A sample path was reused: {midrel}")
        seen.update((str(midrel), str(wavrel)))
        if item.get("midi_sha256") and sha256(mid) != item["midi_sha256"]:
            raise ValueError(f"MIDI changed after generation/evaluation: {midrel}")
        mi, wi = midi_info(mid), wav_info(wav)
        # Synthesizers may add a release tail, but audio must not truncate music.
        if wi["seconds"] + 0.15 < mi["seconds"] or wi["seconds"] > mi["seconds"] + 12:
            raise ValueError(f"WAV and MIDI durations disagree: {midrel}")
        if task == "bonus" and not {"MELODY", "PIANO"}.issubset(mi["track_names"]):
            raise ValueError(f"Bonus output must contain MELODY and PIANO tracks: {midrel}")
        checks.append({"task": task, "midi": str(midrel), "wav": str(wavrel), "midi_info": mi, "wav_info": wi})
        sources.extend((mid, wav))
    return sources, {"sample_counts": counts, "midi_wav_pairs": checks}


def collect_sources(root, payload, report_path, results_path):
    samples, checks = validate_sample_files(root, payload)
    files = samples + [report_path, results_path]
    for name in ("README.md", "requirements.txt", "requirements-experiments.txt", "requirements-audio-review.txt"):
        path = root / name
        if path.is_file():
            files.append(path)
        elif name in ("README.md", "requirements.txt"):
            raise ValueError(f"Missing required file: {name}")
    for name in ("infer.py", "model.py"):
        if not (root / "src" / name).is_file():
            raise ValueError(f"Missing required inference source: src/{name}")
    files += sorted((root / "src").glob("*.py"))
    files += sorted((root / "tools").glob("*.swift"))
    files += sorted((root / "tools").glob("*.sh"))
    # Optional audio-only criticism has separate dependencies and never ships
    # downloaded external weights, environments, caches, or clipped WAV copies.
    for name in ("prepare_audio_review_model.py", "check_audio_review_frontend.py", "audio_review_smoke.py"):
        path = root / "tools" / name
        if path.is_file():
            files.append(path)
    for name in ("eval_basic.py", "eval_bonus.py", "tokenization.py", "README.md"):
        path = root / "HW2_scripts" / name
        if path.is_file():
            files.append(path)
    # Public source code is allowed with attribution; pretrained weights are not.
    vendor = root / "vendor"
    if vendor.is_dir():
        files += [p for p in vendor.rglob("*") if p.is_file() and
                  (p.suffix in (".py", ".md", ".txt") or p.name in ("UPSTREAM.json",) or p.name.upper().startswith("LICENSE")) and
                  not any(x in p.parts for x in (".git", "__pycache__"))]
    reproduce = payload.get("reproducibility", {})
    checkpoints = reproduce.get("checkpoints", [])
    tokenizer_files = reproduce.get("tokenizer_files", [])
    if not checkpoints or not tokenizer_files:
        raise ValueError("Explicit checkpoint and tokenizer-file lists are required")
    for entry in checkpoints:
        if not isinstance(entry, dict) or entry.get("trained_from_scratch") is not True or not entry.get("sha256"):
            raise ValueError("Each own checkpoint needs trained_from_scratch=true and a recorded SHA256")
        path, _ = owned_file(root, entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {entry['path']}")
        import torch
        content = torch.load(path, map_location="cpu", weights_only=True)
        metadata = content["metadata"]
        origin = metadata.get("origin", {})
        if not (origin.get("random_initialization") is True or origin.get("original_random_initialization") is True):
            raise ValueError(f"Checkpoint random-initialization provenance is missing: {entry['path']}")
        training_ids, seen_ids = set(metadata.get("training_song_ids", [])), set(metadata.get("seen_song_ids", []))
        if metadata.get("all_official_training_songs_used") is not True or len(training_ids) != 896 or seen_ids != training_ids:
            raise ValueError(f"Submitted checkpoint has not observed all896 official songs: {entry['path']}")
        if metadata.get("validation_loss") is not None:
            raise ValueError(f"Final all-song checkpoint incorrectly claims held-out validation: {entry['path']}")
        del content
        files.append(path)
    for name in tokenizer_files:
        path, _ = owned_file(root, name)
        if path.suffix.lower() not in (".json", ".txt", ".yaml", ".yml"):
            raise ValueError("Tokenizer files must be configuration/vocabulary text, not training arrays")
        files.append(path)
    for name in reproduce.get("source_artifacts", []):
        path, relative = owned_file(root, name)
        if path.suffix.lower() not in (".json", ".csv") or "sequences" in path.stem or "tokens" in path.stem:
            raise ValueError(f"Evidence must be result summaries/manifests, not preprocessed sequences: {relative}")
        recorded = reproduce.get("source_sha256", {}).get(str(relative))
        if not recorded or sha256(path) != recorded:
            raise ValueError(f"Measured evidence hash missing or changed: {relative}")
        files.append(path)
    for name in reproduce.get("figure_artifacts", []):
        path, relative = owned_file(root, name)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".svg"):
            raise ValueError(f"Unsupported case-study figure: {relative}")
        recorded = reproduce.get("figure_sha256", {}).get(str(relative))
        if not recorded or sha256(path) != recorded:
            raise ValueError(f"Case-study figure hash missing or changed: {relative}")
        files.append(path)
    files += [path for path in sorted((root / "tests").glob("*.py"))
              if path.name != "test_update_published_report.py"]
    # Include the code behind reported audits, not their raw/preprocessed data.
    files += sorted((root / "analysis").glob("*.py"))
    files += sorted((root / "analysis/case_studies").glob("*.py"))
    files += sorted((root / "analysis/audio_review").glob("*.py"))
    # Figures and the listening index are produced from the submitted outputs.
    for name in ("figures/selected_structure.png", "figures/sampling_distributions.png",
                 "figures/selected_structure_notes.json", "listening.html", "HW2_說明.html", "analysis/report_data_template.json",
                 "results/report_inputs.json", "results/evaluation/watch_evaluation.py"):
        path = root / name
        if path.is_file():
            files.append(path)
    return sorted(set(files)), checks


def verify_archive(path):
    """Read back every ZIP member and validate the independent file inventory."""
    with zipfile.ZipFile(path) as handle:
        if handle.testzip():
            raise ValueError("Archive CRC verification failed")
        names = handle.namelist()
        roots = {Path(name).parts[0] for name in names}
        if len(roots) != 1 or len(names) != len(set(names)):
            raise ValueError("Archive root or duplicate-member inventory is invalid")
        prefix = next(iter(roots)) + "/"
        hashes = json.loads(handle.read(prefix + "FILE_SHA256.json"))
        expected = {prefix + name for name in hashes} | {prefix + "FILE_SHA256.json", prefix + "SUBMISSION_CHECKS.json"}
        if set(names) != expected:
            raise ValueError("Archive has unlisted or missing files")
        for name, wanted in hashes.items():
            if hashlib.sha256(handle.read(prefix + name)).hexdigest() != wanted:
                raise ValueError(f"Archived file hash mismatch: {name}")
        checks = json.loads(handle.read(prefix + "SUBMISSION_CHECKS.json"))
        expected_midis = {row["midi"] for row in checks["midi_wav_pairs"]}
        expected_wavs = {row["wav"] for row in checks["midi_wav_pairs"]}
        midi_paths = {name for name in hashes if Path(name).suffix.lower() in (".mid", ".midi")}
        wav_paths = {name for name in hashes if Path(name).suffix.lower() == ".wav"}
        if midi_paths != expected_midis or wav_paths != expected_wavs:
            raise ValueError("Archive listening files differ from the checked sample inventory")
        forbidden = {"POP909", "training", "__pycache__", ".venv", ".git", "cache", "caches", "preprocessed"}
        if any(any(part in forbidden for part in Path(name).parts) or Path(name).suffix.lower() in (".npz", ".npy", ".pkl") for name in hashes):
            raise ValueError("Archive contains excluded training/preprocessed/cache files")
        return {"verified_files": len(names), "all_file_hashes_match": True,
                "midi_files": len(midi_paths), "wav_files": len(wav_paths),
                "excluded_paths_absent": True}


def package(root, payload, results_path, allow_draft=False):
    reasons = validate_payload(payload)
    if reasons and not allow_draft:
        raise ValueError("Not ready for final submission: " + "; ".join(reasons))
    student_id = payload["student"]["student_id"]
    if not student_id.isalnum():
        raise ValueError("Invalid student ID")
    report_path = root / f"{student_id}_report.pdf"
    report_info = build_report(payload, report_path)
    files, checks = collect_sources(root, payload, report_path, results_path)
    label = f"{student_id}_HW2" + ("_DRAFT" if reasons else "")
    destination = root / "submission"
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hw2-package-", dir=destination) as temporary:
        stage = Path(temporary) / label
        stage.mkdir()
        hashes = {}
        for source in files:
            source, relative = owned_file(root, source)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = sha256(target)
        (stage / "FILE_SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        checks.update({"report": report_info, "public_cloud_drive_url": payload.get("cloud_drive_url", ""),
                       "user_provided_artifact_url": payload.get("artifact_url", ""),
                       "cloud_drive_requirement_unverified": payload.get("cloud_drive_requirement_unverified", True),
                       "user_selected_repository_hosting": payload.get("user_selected_repository_hosting", False),
                       "hosting_warnings": hosting_warnings(payload),
                       "ta_acceptance_of_github_substitution_verified": False,
                       "public_access_independently_verified": False,
                       "excluded": ["training MIDI", "preprocessed sequences", "third-party model weights", "caches"],
                       "package_is_draft": bool(reasons)})
        (stage / "SUBMISSION_CHECKS.json").write_text(json.dumps(checks, indent=2) + "\n")
        archive = root / f"{label}.zip"
        temporary_zip = archive.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(stage.parent))
        integrity = verify_archive(temporary_zip)
        temporary_zip.replace(archive)
        final = destination / label
        if final.exists():
            shutil.rmtree(final)
        shutil.move(str(stage), final)
    return {"archive": str(archive), "bytes": archive.stat().st_size,
            "sha256": sha256(archive), "files": len(hashes)+2, "sample_counts": checks["sample_counts"],
            "draft": bool(reasons), "draft_reasons": reasons, "report": report_info,
            "archive_integrity": integrity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/report_data.json")
    parser.add_argument("--cloud-url", help="Required public cloud-drive folder for the report first page")
    parser.add_argument("--artifact-url", help="User-provided artifact location; recorded separately from the cloud-drive folder")
    parser.add_argument("--allow-draft", action="store_true", help="Clearly label an otherwise complete archive DRAFT")
    args = parser.parse_args()
    payload = read_json(args.results)
    if args.cloud_url:
        payload["cloud_drive_url"] = args.cloud_url
        payload["cloud_drive_requirement_unverified"] = False
    if args.artifact_url:
        payload["artifact_url"] = args.artifact_url
    if args.cloud_url or args.artifact_url:
        args.results.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        print(json.dumps(package(ROOT, payload, args.results, args.allow_draft), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Packaging failed: {exc}\n")


if __name__ == "__main__":
    main()
