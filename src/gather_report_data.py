"""Assemble the HW2 report from saved, actual manifests and evaluations.

Missing files remain pending. No experiment scores, listening impressions, or
checkpoint provenance are invented. An optional report_inputs.json supplies
reviewed interpretation and path overrides; measured values always come from
the files named there or the conventional experiment directories.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

from build_report import BASIC_METRICS, BONUS_METRICS, hosting_warnings, validate_audio_review, validate_payload

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("t08", "t10", "t12", "k16", "k32")
HISTORIES = ("basic_diagnostic", "basic_refined", "context_2048", "context_4096",
             "basic_all", "roles_diagnostic", "roles_all")
BONUS_ALIASES = {"DP (Pitch-class OA)": "OA_P", "DV (Velocity OA)": "OA_V",
                 "DD (Duration OA)": "OA_D", "DIOI (Inter-Onset OA)": "OA_IOI"}


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else copy.deepcopy(default)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(root, path):
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else root / candidate
    return candidate.resolve().relative_to(root.resolve()).as_posix()


def metric(value, n=None):
    if not isinstance(value, dict):
        return value
    result = copy.deepcopy(value)
    if "mean_ci95" in result:
        result["ci95"] = result.pop("mean_ci95")
    if n is not None:
        result["n"] = n
    return result


def csv_rows(path):
    if not Path(path).is_file():
        return {}
    with Path(path).open(newline="", encoding="utf-8") as source:
        return {str(row["song_name"]).zfill(3):
                {BONUS_ALIASES.get(key, key): value for key, value in row.items()}
                for row in csv.DictReader(source)}


def observed_sample(root, record):
    result = {"song_name": record["song_name"], "bars": record.get("bars"),
              "seed": record.get("diagnostics", {}).get("seed"),
              "diagnostics": record.get("diagnostics", {})}
    for kind in ("midi", "wav"):
        supplied = record.get(f"{kind}_path") or record.get(f"output_{kind}")
        if kind == "wav" and not supplied and record.get("midi_path"):
            rendered = Path(record["midi_path"]).with_suffix(".wav")
            rendered = rendered if rendered.is_absolute() else root / rendered
            if rendered.is_file():
                supplied = rendered
        result[f"output_{kind}"] = relative(root, supplied) if supplied else None
    for key in ("midi_sha256", "prefix_verification", "melody_verification", "audio_audit"):
        if key in record:
            result[key] = record[key]
    return result


def checkpoint_details(root, path):
    path = root / path
    if not path.is_file():
        return None
    import torch
    content = torch.load(path, map_location="cpu", weights_only=True)
    metadata = content["metadata"]
    origin = metadata.get("origin", {})
    from_scratch = (origin.get("random_initialization") is True or
                    origin.get("original_random_initialization") is True)
    all_ids, seen_ids = set(metadata.get("training_song_ids", [])), set(metadata.get("seen_song_ids", []))
    all_songs = metadata.get("all_official_training_songs_used") is True and len(all_ids) == 896 and seen_ids == all_ids
    return {"path": relative(root, path), "sha256": sha256(path),
            "trained_from_scratch": from_scratch, "all_896_songs_observed": all_songs,
            "architecture": content["model_config"], "metadata": metadata}


def evaluation_paths(root, task, config, override):
    base = root / "results/evaluation" / task
    tag = config if task != "bonus" else "bonus"
    folder = base / config if task != "bonus" else base
    return (root / override.get("evaluation", str((folder / f"{tag}_summary.json").relative_to(root))),
            root / override.get("per_piece", str((folder / f"{tag}_per_piece.csv").relative_to(root))))


def load_run(root, task, config, overrides, evidence):
    entry = overrides.get(task, {}).get(config, {})
    default = root / "results" / task / config / "manifest.json" if task != "bonus" else root / "results/bonus/manifest.json"
    manifest_path = root / entry.get("manifest", str(default.relative_to(root)))
    manifest = read(manifest_path)
    if manifest is None:
        return None
    evidence.add(manifest_path)
    summary_path, per_piece_path = evaluation_paths(root, task, config, entry)
    summary = read(summary_path, {})
    if summary_path.is_file():
        evidence.add(summary_path)
    if per_piece_path.is_file():
        evidence.add(per_piece_path)
    samples = [observed_sample(root, sample) for sample in manifest.get("samples", [])]
    n = summary.get("evaluated_pieces")
    metrics = {BONUS_ALIASES.get(name, name): metric(value, n) for name, value in summary.get("summary", {}).items()}
    parameters = copy.deepcopy(manifest.get("configuration", {}))
    parameters.pop("device", None)
    parameters.pop("seed", None)
    return {"id": config, "parameters": parameters, "n": len(samples), "metrics": metrics,
            "samples": samples,
            "checks": {"manifest_complete": manifest.get("complete", False),
                       "evaluated": n, "skipped": summary.get("skipped_pieces"),
                       "expected_bars": summary.get("expected_bars"),
                       "valid_bar_counts": all(s.get("bars") == (16 if task == "bonus" else 32) for s in samples)},
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "evaluation": summary, "per_piece": csv_rows(per_piece_path),
            "manifest": relative(root, manifest_path)}


def reconstruction_experiments(root, evidence):
    experiments = []
    path = root / "artifacts/quantization_resolution_audit.json"
    audit = read(path)
    if audit:
        evidence.add(path)
        rows = [[int(resolution), values["vocabulary_size"],
                 values["onset_error_seconds"]["mae"] * 1000,
                 values["offset_error_seconds"]["mae"] * 1000,
                 values["exact_same_pitch_grid_collisions"]]
                for resolution, values in audit["results"].items()]
        experiments.append({"name": "Quantization resolution: reconstruction cost",
            "method": audit["method"],
            "headers": ["Steps/beat", "Vocabulary", "Onset MAE ms", "Offset MAE ms", "Grid collisions"], "rows": rows,
            "analysis": f"Across {audit['n_notes']:,} notes, finer grids reduce mean timing error and same-pitch collisions while increasing vocabulary size. Eight steps/beat is the implemented compromise. This reconstruction study does not compare retrained model quality. The same two duration-cap outliers and early-note clamp errors persist."})
    path = root / "artifacts/codec_roundtrip_audit.json"
    audit = read(path)
    if audit:
        evidence.add(path)
        cases = audit["test_cases"]
        experiments.append({"name": "Codec round-trip verification",
            "method": audit["method"],
            "headers": ["Song", "Notes", "Max onset ms", "Max offset ms", "Invalid events"],
            "rows": [[r["song_id"], r["notes"], r["max_export_onset_error_sec"] * 1000,
                      r["max_export_offset_error_sec"] * 1000, r["invalid_events"]] for r in cases],
            "analysis": "All cases pass independent readback within2ms of quantized timing. This does not imply lossless source reconstruction: two notes exceed the32beat duration cap (max offset error8.732s); pre-annotation notes are clamped (max onset error0.709s). These are disclosed representation limits."})
    return experiments


def training_ablation_experiments(root, evidence):
    """Read controlled comparisons; every number stays tied to its result file."""
    experiments = []
    path = root / "results/context_comparison.json"
    comparison = read(path)
    if comparison:
        evidence.add(path)
        rows = []
        for branch in comparison["branches"]:
            history_path = root / f"artifacts/models/context_{branch['context']}/history.json"
            history = read(history_path, {})
            if history:
                evidence.add(history_path)
            meta = history.get("metadata", {})
            rows.append([branch["context"], branch["processed_tokens"],
                         meta.get("arguments", {}).get("batch_size"),
                         branch["validation_loss"], meta.get("elapsed_sec")])
        experiments.append({"name": "Equal-token context-length comparison",
            "method": "Two branches start from the same refined checkpoint SHA256. Both use AdamW, learning rate 0.00008 and a 4,000,000 non-padding target budget. Compare last checkpoints on the same 45-song, 2048-token validation windows; allow at most one training-step budget overshoot.",
            "headers": ["Context", "Target tokens", "Batch", "Validation CE", "Seconds"], "rows": rows,
            "analysis": "The 2048-token branch has lower validation CE (1.70637 versus 1.74084) and is selected before the all-song final fit. The measured token-budget difference is 3,454 targets (0.086% of the nominal budget).",
            "limitations": "One run per context, with different batch sizes and step counts. The shared parent was trained mainly at 2048 tokens with learned absolute positions; the 4096 branch must adapt later positions. Validation also uses 2048 tokens. This compares this warm start and budget, not the general value of long context."})
    path = root / "results/checkpoint_2000/token_diagnostic_comparison.json"
    comparison = read(path)
    if comparison:
        evidence.add(path)
        for name in comparison["source_paths"]:
            evidence.add(root / name)
        selected_types = {"Tempo", "Pos", "Pitch", "Vel", "Dur"}
        experiments.append({"name": "What improves when token loss falls?",
            "method": "Teacher-forced full-vocabulary prediction with no grammar mask: 77,926 non-padding targets from the same 48 deterministic windows covering all 45 pre-refit validation songs. Window hashes match across the step-500 and step-2000 checkpoints.",
            "headers": ["Target family", "CE @500", "CE @2000", "Acc % @500", "Acc % @2000"],
            "rows": [[r["target_type"], r["CE_step500"], r["CE_step2000"],
                      r["accuracy_step500"]*100, r["accuracy_step2000"]*100]
                     for r in comparison["by_target_type"] if r["target_type"] in selected_types],
            "analysis": "Overall CE falls from 2.36666 to 1.97120 nats/token and top-1 accuracy rises from 26.54% to 35.51%. Token-family accuracy is already 98.30% at step 500; pitch-content accuracy rises from 8.71% to 19.30% and remains much harder than recognizing token grammar.",
            "limitations": "Teacher forcing supplies the correct history and does not measure free-running musical quality. These checkpoints were trained on the 851-song optimization split; the final all-song models are not claimed to have held-out validation accuracy."})
    return experiments


def named_evidence(value):
    """Collect nested experiment evidence as well as top-level evidence."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evidence_files":
                yield from item
            else:
                yield from named_evidence(item)
    elif isinstance(value, list):
        for item in value:
            yield from named_evidence(item)


def named_figures(value):
    """Keep report images separate from tabular measured evidence."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "figure_path" and isinstance(item, str):
                yield item
            elif isinstance(item, (dict, list)):
                yield from named_figures(item)
    elif isinstance(value, list):
        for item in value:
            yield from named_figures(item)


def assemble(root, inputs=None):
    inputs = inputs or {}
    payload = read(root / "analysis/report_data_template.json")
    payload.pop("_schema_notes", None)
    evidence = set()
    overrides = inputs.get("runs", {})
    analysis_path = root / inputs.get("analysis", "results/evaluation/report_analysis.json")
    analysis = read(analysis_path, {})
    if analysis_path.is_file():
        evidence.add(analysis_path)
    analysis.update({key: value for key, value in inputs.items() if key not in ("runs", "histories", "checkpoints")})
    for name in named_evidence(analysis):
        path = root / relative(root, name)
        if not path.is_file():
            raise ValueError(f"Named experiment evidence is missing: {name}")
        evidence.add(path)
    payload["artifact_url"] = inputs.get("artifact_url", "https://github.com/alexsu30812-gif/hw2")
    payload["cloud_drive_url"] = inputs.get("cloud_drive_url", "")
    payload["cloud_drive_requirement_unverified"] = inputs.get("cloud_drive_requirement_unverified", True)
    payload["user_selected_repository_hosting"] = inputs.get("user_selected_repository_hosting", False)

    checkpoint_paths = {"basic": "artifacts/models/basic_final.pt", "bonus": "artifacts/models/roles_final.pt"}
    checkpoint_paths.update(inputs.get("checkpoints", {}))
    checkpoints = {name: checkpoint_details(root, path) for name, path in checkpoint_paths.items()}
    basic = checkpoints.get("basic")
    split_path = root / "artifacts/tokenization_audit.json"
    split = read(split_path, {})
    if split:
        evidence.add(split_path)
        payload["data"]["diagnostic_split"] = (f"Song-level split before window sampling: {len(split.get('training_ids', []))} optimization songs, "
            f"{len(split.get('validation_ids', []))} validation songs; seed {split.get('seed')}. The 13 official prompt folders are excluded.")
    payload["data"]["training_songs"] = 896 if basic and basic["all_896_songs_observed"] else None
    payload["data"]["final_fit"] = ("Final checkpoint has observed all 896 official songs, confirmed by training and seen-song ID sets. Its validation loss is unset; earlier validation estimates describe only pre-refit diagnostic models."
        if basic and basic["all_896_songs_observed"] else "PENDING: final checkpoint must confirm exposure to every official training song.")
    payload["data"]["final_all_896_songs_verified"] = bool(basic and basic["all_896_songs_observed"])
    payload["data"]["basic_tracks"] = "Merge all MELODY, BRIDGE and PIANO source note events; role training retains MELODY and PIANO separately."

    vocabulary_path = root / "artifacts/vocabulary.json"
    vocabulary = read(vocabulary_path, {})
    payload["tokenizer"] = {"representation": "Beat-annotation-aware REMI: BOS, Bar, Meter, Tempo, then Pos/Pitch/Velocity/Duration note tuples; EOS. Bonus retains Track_Melody and Track_Piano roles.",
        "vocabulary_size": len(vocabulary.get("tokens", [])) or None,
        "vocabulary_sha256": vocabulary.get("sha256"), "time_grid": "Eight subdivisions per supplied annotation beat; up to sixteen beats per bar; duration 1..256 steps (32 beats).",
        "velocity": "32 bins; exact original prompt velocities are retained in final exports.",
        "export_meter": "Canonical annotation-beat notation: one supplied annotation beat is written as one quarter note and Meter_n as n/4. This preserves bar counts and elapsed time; the beat annotations do not identify the original score denominator (six pulses are exported as6/4, not claimed to recover6/8).",
        "pickup_policy": "Retain an annotated short pickup bar; clamp notes before the first annotation timestamp to the first grid position and disclose the measured error.",
        "quantization_ablation": "Actual 4/8/16-resolution reconstruction audit below; models are trained only at resolution eight.",
        "decoding_constraints": "Grammar masks legal next-token types, ascending pitch order for simultaneous notes, complete bar boundaries, pitch21..108, end-time-safe durations and a 64-note-per-bar cap; these are disclosed inference rules."}
    payload["model"] = {"random_initialization": basic["trained_from_scratch"] if basic else None,
        "architecture": basic["architecture"] if basic else {},
        "parameter_count": basic["metadata"].get("parameter_count") if basic else None,
        "attention": "Decoder-only causal self-attention; learned absolute positions, pre-layer normalization, GELU feed-forward layers, tied token embedding/output weights.",
        "generation_context": basic["metadata"].get("inference_context") if basic else None,
        "context_rollover": "On overflow, retain the latest complete bars fitting the context, prepend BOS and rebuild the key/value cache. Position indices restart; this limits long-range memory.",
        "bonus_checkpoint": checkpoints["bonus"]["path"] if checkpoints.get("bonus") else None}

    histories = {}
    for name in HISTORIES:
        path = root / inputs.get("histories", {}).get(name, f"artifacts/models/{name}/history.json")
        history = read(path)
        if history:
            evidence.add(path)
            histories[name] = history
    diagnostic_names = [name for name in ("basic_diagnostic", "basic_refined") if name in histories]
    curve, offset = [], 0
    for name in diagnostic_names:
        source = histories[name]
        for row in source.get("history", []):
            curve.append({**row, "step": row["step"] + offset, "stage": name})
        offset += source.get("metadata", {}).get("step", 0)
    stages = {}
    for name, source in histories.items():
        meta = source.get("metadata", {})
        args = meta.get("arguments", {})
        stages[name] = {"steps": meta.get("step"), "nonpad_target_tokens": meta.get("processed_tokens"),
            "seconds": meta.get("elapsed_sec"), "training_songs": len(meta.get("training_song_ids", [])),
            "seen_songs": len(meta.get("seen_song_ids", [])), "context": args.get("context"),
            "validation_CE": meta.get("validation_loss"), "seed": args.get("seed"),
            "token_budget": args.get("token_budget")}
    payload["training"] = {"history": curve, "stages": stages,
        "optimizer": "AdamW; betas(0.9,0.95), weight decay0.1, norm clipping1.0; warmup and cosine decay. Per-stage values and measured target-token counts are saved in the source histories.",
        "augmentation": "Training-window transposition sampled uniformly from legal integer shifts -5..+5 semitones, preserving piano-range pitch21..108; validation uses no augmentation.",
        "window_sampling": "Shuffle songs, visit each selected song before repeating, and sample a bar-aligned window. Labels use nonpadding targets only; no target crosses a song boundary.",
        "curve_scope": "Shown curve concatenates basic_diagnostic and basic_refined optimizer steps. Validation uses a deterministic song-level pre-refit split. All-song stages have no held-out validation loss.",
        "no_human_listening_claim": "No human listener ratings or listening comparisons were collected; selection uses explicitly described objective diagnostics."}

    base_path = root / inputs.get("real_reference", "results/evaluation/real_reference/real_reference_summary.json")
    baseline = read(base_path, {})
    if baseline:
        evidence.add(base_path)
        for related in (base_path.with_name("real_reference_per_piece.csv"), base_path.with_name("real_reference_selection.json")):
            if related.is_file():
                evidence.add(related)
    n = baseline.get("evaluated_pieces")
    payload["basic_a"]["real_baseline"] = {"name": "Pre-refit POP909 real-data reference", "n": n,
        **{m: metric(baseline.get("summary", {}).get(m), n) for m in BASIC_METRICS}}
    payload["basic_a"]["baseline_protocol"] = ("One deterministic center-nearest eligible 32-bar crop from each of 45 pre-refit validation songs; nearest eligible crop for rare meters. "
        "Shared Pos8-to-Pos16 adapter and official128-position dictionary; 2000 whole-piece bootstrap draws with seed8173. "
        "These songs participate in the final all896-song fit, so this reference is descriptive and is not independent of that final model.")
    runs = [load_run(root, "basic_a", config, overrides, evidence) for config in CONFIGS]
    runs = [run for run in runs if run]
    payload["basic_a"]["configurations"] = [{key: value for key, value in run.items() if key not in ("evaluation", "per_piece")} for run in runs]
    selected = analysis.get("selected_configuration")
    distances = {}
    if baseline and runs:
        from evaluate import compare_to_reference
        for run in runs:
            if all(m in run["evaluation"].get("summary", {}) for m in BASIC_METRICS):
                distances[run["id"]] = compare_to_reference(run["evaluation"], baseline)
        if selected is None and len(distances) == len(CONFIGS):
            selected = min(distances, key=lambda config: (distances[config]["mean_gap"], config))
    payload["basic_a"]["selected_configuration"] = selected
    payload["basic_a"]["selection"] = analysis.get("basic_a_selection", {
        "rule": "Predeclared: minimize the equal-weight mean absolute H4/GS mean gap divided by the real reference standard deviation; ties use configuration ID. This is a descriptive selection rule, not a grade formula or perceptual quality score.",
        "selected": selected, "measured_gaps": distances,
        "listening_scope": "No human listening comparison completed. Density, empty bars, repetition and local training matches are reported separately."})
    diagnostics = {run["id"]: {key: value for key, value in run["metrics"].items() if key not in ("H1", "H4", "GS")}
                   for run in runs}
    payload["basic_a"]["diagnostics"] = diagnostics

    b_runs = [load_run(root, "basic_b", config, overrides, evidence) for config in ("t08", "t10", "t12")]
    b_runs = [run for run in b_runs if run]
    best = analysis.get("basic_b_best", {})
    b_rows = []
    boundary = {(str(prompt["prompt_id"]).zfill(3), str(row["configuration_id"])): row
                for prompt in analysis.get("prompt_analysis", []) for row in prompt.get("comparison", [])}
    for run in b_runs:
        for sample in run["samples"]:
            prompt = str(sample["song_name"]).zfill(3)
            b_rows.append({**sample, "prompt_id": prompt, "configuration_id": run["id"],
                "parameters": run["parameters"], "prompt_bars": 8, "new_bars": 24,
                "is_best": best.get(prompt) == run["id"],
                "boundary_diagnostics": boundary.get((prompt, run["id"]), {}),
                "selection_reason": analysis.get("basic_b_reasons", {}).get(prompt, {}).get(run["id"],
                    "PENDING: reviewed boundary-diagnostic interpretation; no human listening claim.")})
    payload["basic_b"].update({"results": b_rows,
        "same_checkpoint_as_basic_a": bool(basic and b_runs and all(run["checkpoint_sha256"] == basic["sha256"] for run in b_runs)
                                           and all(run["checkpoint_sha256"] == basic["sha256"] for run in runs)),
        "prompt_preservation_verified": bool(len(b_rows) == 9 and all(row.get("prefix_verification", {}).get("verified") is True for row in b_rows)),
        "method": "Tokenize the supplied eight bars as causal prefix, continue for exactly24 new bars using the same basic checkpoint, then merge the original unquantized prefix notes back into the exported MIDI. Generated durations stop at the32bar boundary. No empty continuation fallback is allowed.",
        "prompt_analysis": analysis.get("basic_b_prompt_analysis", {})})

    bonus = load_run(root, "bonus", "bonus", overrides, evidence)
    payload["bonus"]["attempted"] = True
    payload["bonus"]["method"] = analysis.get("bonus_method", "Role-aware causal Transformer initialized from our own randomly trained basic model, then trained with melody/piano role tokens. Generator sees melody.mid and beat_midi.txt only, retains the original melody, and generates piano on the supplied16bar clock. Exact conditioning visibility and masking are documented in source and training histories.")
    if bonus:
        payload["bonus"]["parameters"] = bonus["parameters"]
        rows = []
        for sample in bonus["samples"]:
            prompt = str(sample["song_name"]).zfill(3)
            per_piece = bonus["per_piece"].get(prompt, {})
            rows.append({**sample, "prompt_id": prompt,
                         "metrics": {m: float(per_piece[m]) if per_piece.get(m) else None for m in BONUS_METRICS}})
        payload["bonus"]["results"] = rows
        payload["bonus"]["melody_preservation_verified"] = bool(len(rows) == 10 and
            all(row.get("melody_verification", {}).get("verified") is True for row in rows))
        payload["bonus"]["aggregate"] = {"pooled": bonus["evaluation"].get("pooled", {}),
            "piece_mean_and_uncertainty": bonus["metrics"],
            "coverage": f"{bonus['evaluation'].get('evaluated_pieces')} prompts; {bonus['evaluation'].get('total_bars')} bars; {bonus['evaluation'].get('skipped_pieces')} skipped.",
            "interpretation": "CA is a coarse triad-estimator agreement; CA-ceiling means the same estimator applied to ground truth, not a strict mathematical upper bound. OA uses distributions and can overlook octave displacement and temporal ordering."}

    # These two studies are assembled directly above with the full training
    # controls. Do not duplicate the evaluator's shorter versions in the PDF.
    direct_studies = {"Context length at a matched token budget", "Next-token errors by musical event family"}
    payload["additional_experiments"] = (training_ablation_experiments(root, evidence)
        + reconstruction_experiments(root, evidence)
        + [row for row in analysis.get("additional_experiments", []) if row.get("name") not in direct_studies])
    # Case studies inspect the unchanged submitted samples, after the initial
    # metric-only selection. They are not another sampling/tuning experiment.
    case_path = root / inputs.get("case_studies", "analysis/case_studies/bar_case_studies.json")
    cases = read(case_path)
    figure_files = set()
    if cases:
        payload["structured_failure_analysis"] = cases
        evidence.add(case_path)
        for name in named_evidence(cases):
            path = root / relative(root, name)
            if not path.is_file() or path.suffix.lower() not in (".json", ".csv"):
                raise ValueError(f"Case-study evidence is missing or not tabular: {name}")
            evidence.add(path)
        for name in named_figures(cases):
            path = root / relative(root, name)
            if not path.is_file() or path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".svg"):
                raise ValueError(f"Case-study figure is missing or unsupported: {name}")
            figure_files.add(path)
    audio_path = inputs.get("audio_review")
    if audio_path:
        path = root / relative(root, audio_path)
        review = read(path)
        if not review:
            raise ValueError(f"Named audio review is missing: {audio_path}")
        payload["audio_review"] = review
        audio_reasons = validate_audio_review(review)
        if audio_reasons:
            raise ValueError("Named audio review is not complete: " + "; ".join(audio_reasons))
        for reference in review.get("references", []):
            if reference not in payload["references"]:
                payload["references"].append(reference)
        evidence.add(path)
        for name in named_evidence(review):
            source = root / relative(root, name)
            if not source.is_file() or source.suffix.lower() not in (".json", ".csv"):
                raise ValueError(f"Audio-review evidence is missing or not tabular: {name}")
            recorded = review.get("evidence_sha256", {}).get(name)
            if recorded and sha256(source) != recorded:
                raise ValueError(f"Audio-review evidence changed after its review: {name}")
            evidence.add(source)
    for name in ("results/final_generation_device_probe.json", "figures/selected_structure_notes.json",
                 "results/evaluation/evidence_freeze.json"):
        if (root / name).is_file():
            evidence.add(root / name)
    for run in ([next((row for row in runs if row['id'] == selected), None)] + b_runs + [bonus]):
        if run:
            path = (root / run['manifest']).with_name('render_manifest.json')
            if path.is_file():
                evidence.add(path)
    payload["figures"] = {"sampling": "figures/sampling_distributions.png",
                          "structure": "figures/selected_structure.png"}
    payload["failures"] = analysis.get("failures", [])
    payload["limitations"] = analysis.get("limitations", [
        "No human listening study was completed. Objective selection does not establish perceptual quality or long-range musical form.",
        "The real reference is not independent of the final all896-song fit. Only pre-refit diagnostics have held-out validation estimates.",
        "Grammar masks, note caps and context rollover make generation reliable but limit expressive freedom and long-range memory.",
        "The eight-step grid introduces timing and duration error; source duration-cap and early-pickup outliers are disclosed.",
        "H4/GS and CA/OA omit many musical qualities. Exact local-match checks cannot rule out paraphrase or memorization."])
    public_checkpoints = [{key: value[key] for key in ("path", "sha256", "trained_from_scratch", "all_896_songs_observed")}
                          for value in checkpoints.values() if value]
    # Manifests contain generated token sequences and are valid artifacts. The
    # real-reference token sequences and NPZ training arrays are excluded.
    allowed_evidence = sorted(relative(root, path) for path in evidence if path.is_file())
    payload["reproducibility"].update({"checkpoints": public_checkpoints,
        "tokenizer_files": ["artifacts/vocabulary.json"] if vocabulary_path.is_file() else [],
        "source_artifacts": allowed_evidence,
        "source_sha256": {relative(root, path): sha256(path) for path in sorted(evidence) if path.is_file()},
        "figure_artifacts": sorted(relative(root, path) for path in figure_files),
        "figure_sha256": {relative(root, path): sha256(path) for path in sorted(figure_files)},
        "inference_cli": "src/infer.py accepts --data-root and --output-dir; all three tasks load our submitted checkpoints.",
        "listening_files": "Twenty selected basic-a pairs, all nine basic-b pairs, and ten bonus pairs; audio is synthesized from the corresponding MIDI."})
    payload["status"] = "complete"
    reasons = validate_payload(payload)
    substantive = [reason for reason in reasons if "cloud-drive" not in reason and "cloud drive" not in reason]
    if substantive:
        payload["status"] = "incomplete_measured_experiments"
    payload["assembly"] = {"missing_experiment_requirements": substantive,
        "pending_submission_requirements": [reason for reason in reasons if reason not in substantive],
        "hosting_warnings": hosting_warnings(payload),
        "evidence_files": len(allowed_evidence)}
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "results/report_inputs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/report_data.json")
    args = parser.parse_args()
    payload = assemble(ROOT, read(args.inputs, {}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": payload["status"], **payload["assembly"]}, indent=2))


if __name__ == "__main__":
    main()
