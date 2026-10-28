"""Build an English, 16:9 HW2 report using measured JSON results only.

Missing required experiments or submission checks remain DRAFT. The user's
explicit repository-hosting choice is recorded separately from unverified
acceptance of that host by the TA. No missing score is imputed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
BASIC_METRICS = ("H4", "GS")
BONUS_METRICS = ("CA", "OA_P", "OA_V", "OA_D", "OA_IOI")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metric_mean(value):
    return value.get("mean") if isinstance(value, dict) else value


def valid_metric(name, value):
    value = metric_mean(value)
    maximum = math.log2(12) if name == "H4" else 1.0
    return finite_number(value) and 0 <= value <= maximum + 1e-9


def format_value(value, precision=4):
    if value is None:
        return "PENDING"
    if isinstance(value, dict) and "mean" in value:
        mean = metric_mean(value)
        if not finite_number(mean):
            return "PENDING"
        suffix = f" +/- {value['std']:.{precision}f}" if finite_number(value.get("std")) else ""
        if isinstance(value.get("ci95"), list) and len(value["ci95"]) == 2:
            suffix += f"; CI [{format_value(value['ci95'][0])}, {format_value(value['ci95'][1])}]"
        return f"{mean:.{precision}f}{suffix}"
    if isinstance(value, float):
        return f"{value:.{precision}f}" if math.isfinite(value) else "INVALID"
    if isinstance(value, (str, int)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(format_value(v, precision) for v in value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def sample_path(sample, kind):
    return sample.get(f"output_{kind}") or sample.get(kind)


def configurations(payload):
    return payload.get("basic_a", {}).get("configurations", [])


def repository_hosting_selected(payload):
    """An explicit user choice is separate from TA approval of the host."""
    parsed = urlparse(payload.get("artifact_url", ""))
    return (payload.get("user_selected_repository_hosting") is True
            and parsed.scheme == "https" and parsed.hostname in ("github.com", "www.github.com")
            and len([part for part in parsed.path.split("/") if part]) == 2)


def hosting_warnings(payload):
    if repository_hosting_selected(payload):
        return ["Public artifacts use the user-selected GitHub repository. GitHub substitution for the slides' cloud-drive wording is unverified; no TA acceptance is claimed."]
    return []


def validate_audio_review(review):
    """A model review is complete only when every full WAV was consumed."""
    if not review:
        return []
    checks = (
        (review.get("completed") is True, "audio review is not complete"),
        (review.get("coverage", {}).get("n_pieces") == 9, "audio review does not cover all nine pieces"),
        (review.get("coverage", {}).get("full_frame_coverage_verified") is True,
         "full audio-frame coverage is not verified"),
        (review.get("source_model_input_frames_verified") is True,
         "all audio frames reaching the model are not verified"),
        (review.get("human_listening_performed") is False,
         "audio-model review must remain distinct from human listening"),
    )
    return [reason for passed, reason in checks if not passed]


def validate_payload(payload):
    """Check report completeness; filesystem/sample integrity is checked at packaging."""
    reasons = []
    if payload.get("status") != "complete":
        reasons.append("experiment status is not complete")
    student = payload.get("student", {})
    if student.get("student_id") != "r14942154":
        reasons.append("student ID is missing or incorrect")
    url = payload.get("cloud_drive_url", student.get("cloud_drive_url", ""))
    parsed = urlparse(url)
    if repository_hosting_selected(payload):
        if payload.get("cloud_drive_requirement_unverified") is not True:
            reasons.append("GitHub hosting choice must retain cloud-drive acceptance as unverified")
    else:
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            reasons.append("public cloud-drive folder URL is pending")
        elif parsed.hostname in ("github.com", "www.github.com", "raw.githubusercontent.com"):
            reasons.append("GitHub is not the required cloud-drive folder")
        if payload.get("cloud_drive_requirement_unverified") is True:
            reasons.append("acceptance of the provided artifact location as the required cloud drive is unverified")
    if payload.get("data", {}).get("training_songs") != 896:
        reasons.append("all 896 official training songs are not confirmed")
    if payload.get("data", {}).get("final_all_896_songs_verified") is not True:
        reasons.append("final checkpoint exposure to all 896 songs is not verified")
    if payload.get("model", {}).get("random_initialization") is not True:
        reasons.append("training from random initialization is not confirmed")
    baseline = payload.get("basic_a", {}).get("real_baseline", {})
    if not all(valid_metric(m, baseline.get(m)) for m in BASIC_METRICS):
        reasons.append("real-data H4/GS baseline is missing")
    configs = configurations(payload)
    eligible = []
    seen_ids = []
    for item in configs:
        cid = item.get("id")
        seen_ids.append(cid)
        samples = item.get("samples", [])
        good = (len(samples) >= 20 and item.get("n", len(samples)) == len(samples)
                and all(s.get("bars") == 32 and sample_path(s, "midi") for s in samples)
                and all(valid_metric(m, item.get("metrics", {}).get(m)) for m in BASIC_METRICS))
        checks = item.get("checks", {})
        if checks and (checks.get("manifest_complete") is not True or checks.get("skipped") != 0
                       or checks.get("evaluated") != len(samples) or checks.get("valid_bar_counts") is not True):
            good = False
        if good:
            eligible.append(cid)
    if len(set(seen_ids)) != len(seen_ids):
        reasons.append("basic-a configuration IDs are duplicated")
    parameter_sets = {json.dumps(c.get("parameters", {}), sort_keys=True) for c in configs if c.get("id") in eligible}
    if len(parameter_sets) < 3:
        reasons.append("three distinct basic-a inference-parameter combinations are not recorded")
    if len(eligible) < 3:
        reasons.append("fewer than three measured basic-a configurations with 20 x 32-bar samples")
    selected = payload.get("basic_a", {}).get("selected_configuration")
    selected_row = next((c for c in configs if c.get("id") == selected), None)
    if selected not in eligible:
        reasons.append("selected basic-a configuration is missing or incomplete")
    elif len(selected_row.get("submission_samples", selected_row.get("samples", []))) != 20:
        reasons.append("selected basic-a submission must contain exactly 20 samples")
    elif not all(sample_path(s, "wav") for s in selected_row.get("submission_samples", selected_row["samples"])):
        reasons.append("selected basic-a WAV samples are missing")
    b = payload.get("basic_b", {})
    rows = b.get("results", [])
    expected_prompts = {"001", "002", "003"}
    if len(rows) != 9 or {str(r.get("prompt_id", "")).zfill(3) for r in rows} != expected_prompts:
        reasons.append("all nine basic-b outputs are not present")
    for prompt in sorted(expected_prompts):
        group = [r for r in rows if str(r.get("prompt_id", "")).zfill(3) == prompt]
        if len({r.get("configuration_id") for r in group}) != 3:
            reasons.append(f"basic-b prompt {prompt}: three distinct configurations are required")
        if sum(r.get("is_best") is True for r in group) != 1:
            reasons.append(f"basic-b prompt {prompt}: exactly one best version must be marked")
    if b.get("same_checkpoint_as_basic_a") is not True:
        reasons.append("basic-b use of the same basic-a checkpoint is not confirmed")
    if any(r.get("bars") != 32 or r.get("prompt_bars") != 8 or r.get("new_bars") != 24
           or not sample_path(r, "midi") or not sample_path(r, "wav") for r in rows):
        reasons.append("basic-b output lengths or audio paths are incomplete")
    if b.get("prompt_preservation_verified") is not True:
        reasons.append("basic-b original prompt preservation is not verified")
    bonus = payload.get("bonus", {})
    if bonus.get("attempted"):
        rows = bonus.get("results", [])
        if len(rows) != 10 or {str(r.get("prompt_id", "")).zfill(3) for r in rows} != {f"{i:03d}" for i in range(1, 11)}:
            reasons.append("bonus attempted but ten prompt outputs are not complete")
        if any(r.get("bars") != 16 or not sample_path(r, "midi") or not sample_path(r, "wav")
               or not all(valid_metric(m, r.get("metrics", {}).get(m)) for m in BONUS_METRICS)
               for r in rows):
            reasons.append("bonus attempted but 16-bar outputs or CA/OA metrics are incomplete")
        if bonus.get("melody_preservation_verified") is not True:
            reasons.append("bonus original melody preservation is not verified")
    if not payload.get("references"):
        reasons.append("references are missing")
    reasons.extend(validate_audio_review(payload.get("audio_review")))
    return list(dict.fromkeys(reasons))


class Slides:
    W, H = 960, 540

    def __init__(self, path, draft_reasons, student_id):
        self.c = Canvas(str(path), pagesize=(self.W, self.H), pageCompression=1)
        self.c.setTitle(f"HW2 - Symbolic Music Generation - {student_id}")
        self.c.setAuthor(student_id)
        self.c.setCreator("HW2 measured-results report builder; AI assistance disclosed")
        self.page = 0
        self.draft_reasons = draft_reasons
        self.student_id = student_id

    def text(self, value, x, top, width, height, size=15, bold=False, minimum=10, raw=False):
        content = str(value) if raw else escape(str(value)).replace("\n", "<br/>")
        while size >= minimum:
            style = ParagraphStyle("body", fontName="Helvetica-Bold" if bold else "Helvetica",
                                   fontSize=size, leading=size * 1.28, textColor=colors.black)
            p = Paragraph(content, style)
            _, h = p.wrap(width, height)
            if h <= height:
                p.drawOn(self.c, x, self.H - top - h)
                return h
            size -= 0.5
        raise ValueError(f"Text does not fit on page {self.page}: {str(value)[:100]!r}")

    def start(self, title, section="HW2 | Symbolic Music Generation", subtitle=""):
        if self.page:
            self.c.showPage()
        self.page += 1
        self.c.setFillColor(colors.white)
        self.c.rect(0, 0, self.W, self.H, stroke=0, fill=1)
        self.c.setFillColor(colors.black)
        self.text(section, 40, 21, 880, 18, 10, minimum=10)
        self.text(title, 40, 52, 880, 48, 27, bold=True, minimum=21)
        if subtitle:
            self.text(subtitle, 40, 104, 880, 40, 12, minimum=10)
        self.c.setStrokeGray(0.65)
        self.c.setLineWidth(0.5)
        self.c.line(40, 36, 920, 36)
        label = self.student_id + (" | DRAFT - incomplete experiments or public folder link" if self.draft_reasons else " | Measured experiments")
        self.text(label, 40, 512, 820, 15, 9, minimum=9)
        self.text(str(self.page), 885, 512, 35, 15, 9, minimum=9)

    def bullets(self, items, top=158, size=16, gap=19):
        y = top
        for item in items:
            h = self.text("- " + str(item), 48, y, 864, 490-y, size, minimum=11)
            y += h + gap
        return y

    def table(self, headers, rows, widths, top=158, height=38, size=12):
        x0 = 40
        if top + (len(rows) + 1) * height > 495:
            raise ValueError("Too many table rows: paginate before rendering")
        self.c.setStrokeGray(0.65)
        self.c.line(x0, self.H - top, x0 + sum(widths), self.H - top)
        for i, row in enumerate([headers] + list(rows)):
            x = x0
            for value, width in zip(row, widths):
                self.text(value, x + 7, top + i*height + 7, width - 14, height - 11,
                          size, bold=(i == 0), minimum=8.5)
                x += width
            self.c.line(x0, self.H - top - (i+1)*height, x0 + sum(widths), self.H - top - (i+1)*height)

    def finish(self):
        self.c.save()


def chunks(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]


def description_pages(slides, title, data, section="METHODS"):
    """Render every supplied field, rather than silently dropping experiment details."""
    if not data:
        slides.start(title, section)
        slides.bullets(["PENDING: measured implementation details have not been supplied."])
        return
    items = [f"{key.replace('_', ' ')}: {format_value(value)}" for key, value in data.items()]
    slides.start(title, section)
    top = 158
    continuation = 1
    gap = 12
    for body_size in (14, 13, 12):
        style = ParagraphStyle("measure", fontName="Helvetica", fontSize=body_size, leading=body_size * 1.28)
        total_height = sum(Paragraph(escape("- " + item), style).wrap(864, 10000)[1] + gap for item in items)
        if total_height <= 332:
            break
    for value in items:
        p = Paragraph(escape("- " + value), style)
        _, needed = p.wrap(864, 500)
        if needed > 320:
            # Field values are split across slides rather than truncated.
            words = value.split()
            parts, current = [], []
            for word in words:
                candidate = " ".join(current + [word])
                if len(candidate) > 1100 and current:
                    parts.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
            if current:
                parts.append(" ".join(current))
        else:
            parts = [value]
        for part in parts:
            p = Paragraph(escape("- " + part), style)
            _, needed = p.wrap(864, 500)
            if top + needed > 490:
                continuation += 1
                slides.start(title + f" ({continuation})", section)
                top = 158
            used = slides.text("- " + part, 48, top, 864, 490-top, size=body_size, minimum=11)
            top += used + gap


def plot_training(slides, history):
    usable = [r for r in history if finite_number(r.get("step")) and
              (finite_number(r.get("train_loss", r.get("loss"))) or finite_number(r.get("validation_loss")))]
    if not usable:
        slides.bullets(["PENDING: no recorded training curve.", "Synthetic throughput probes are not music-model training results."])
        return
    x0, y0, w, h = 95, 140, 785, 230
    xs = [r["step"] for r in usable]
    all_losses = [r.get("train_loss", r.get("loss")) for r in usable if finite_number(r.get("train_loss", r.get("loss")))]
    all_losses += [r["validation_loss"] for r in usable if finite_number(r.get("validation_loss"))]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(all_losses), max(all_losses)
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1
    def xy(x, y):
        return x0 + (x-xmin)/(xmax-xmin)*w, y0 + (y-ymin)/(ymax-ymin)*h
    c = slides.c
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.7)
    c.line(x0, y0, x0+w, y0)
    c.line(x0, y0, x0, y0+h)
    for label, key, dashed in [("Train", "train_loss", False), ("Validation", "validation_loss", True)]:
        points = []
        for r in usable:
            loss = r.get(key, r.get("loss") if key == "train_loss" else None)
            if finite_number(loss):
                points.append(xy(r["step"], loss))
        if not points:
            continue
        c.setDash(4, 3) if dashed else c.setDash()
        c.setLineWidth(1.2)
        p = c.beginPath()
        p.moveTo(*points[0])
        for point in points[1:]:
            p.lineTo(*point)
        c.drawPath(p)
        c.setDash()
        slides.text(label + (" (dashed)" if dashed else " (solid)"), 690, 150 if not dashed else 172, 200, 18, 10)
    for f in (0, 0.5, 1):
        slides.text(f"{ymin+(ymax-ymin)*f:.3f}", 40, slides.H-(y0+h*f)-6, 50, 18, 10)
        slides.text(f"{xmin+(xmax-xmin)*f:.0f}", x0+w*f-25, slides.H-y0+10, 60, 18, 10)
    slides.text("Optimizer step", 420, 438, 250, 18, 11)
    slides.text("Cross-entropy (nats/token)", 95, 132, 400, 18, 11)


def training_pages(slides, training):
    settings = {key: value for key, value in training.items() if key not in ("history", "stages")}
    description_pages(slides, "Optimization, checkpoints and validation", settings)
    stages = list(training.get("stages", {}).items())
    for group in chunks(stages, 7):
        slides.start("Measured training stages", "TRAINING | COST AND EXPOSURE",
                     "Target tokens exclude padding. Validation CE belongs only to pre-refit stages.")
        rows = [[name, values.get("steps"), values.get("nonpad_target_tokens"),
                 values.get("context"), values.get("seen_songs"), values.get("validation_CE")]
                for name, values in group]
        slides.table(["Stage", "Steps", "Target tokens", "Context", "Songs seen", "Val CE"],
                     [[format_value(value) for value in row] for row in rows],
                     [240, 80, 160, 105, 125, 170], height=37, size=11)


def diagnostic_pages(slides, diagnostics):
    if not diagnostics:
        description_pages(slides, "Density, repetition and non-empty generation", {}, "BASIC A | FAILURE CHECKS")
        return
    keys = ("notes_per_bar", "empty_bar_fraction", "unique_bar_fraction", "longest_identical_bar_run", "GS_active")
    rows = [[name] + [format_value(metric_mean(metrics.get(key)), 3) for key in keys]
            for name, metrics in diagnostics.items()]
    for group in chunks(rows, 6):
        slides.start("Density, repetition and non-empty generation", "BASIC A | FAILURE CHECKS",
                     "These are whole-piece diagnostic means. Detailed distributions and confidence intervals are saved in report_data.json.")
        slides.table(["Config", "Notes/bar", "Empty bars", "Unique bars", "Max same run", "GS active"],
                     group, [130, 140, 150, 150, 160, 150], height=39, size=11)
        slides.text("Official GS includes a fixed128-position vector. GS active uses the native grid and the active bar length. High GS can coexist with empty or repetitive bars; neither measure establishes musical quality.",
                    40, 450, 880, 42, size=12, minimum=10)


def reproduction_pages(slides, reproduction):
    entries = reproduction.get("checkpoints", [])
    slides.start("Reproduction and submission inventory", "REPRODUCIBILITY")
    slides.bullets([reproduction.get("submission_inventory", "PENDING: listening-file inventory"),
                   reproduction.get("inference_cli", "PENDING: inference commands"),
                   f"Tokenizer files: {format_value(reproduction.get('tokenizer_files', []))}",
                   f"Measured source artifacts: {len(reproduction.get('source_artifacts', []))}; full paths and SHA256 hashes in results/report_data.json.",
                   "FILE_SHA256.json and SUBMISSION_CHECKS.json independently record packaged file integrity and all MIDI/WAV pair checks.",
                   "The archive includes our checkpoints, inference/evaluation source and selected MIDI/WAV pairs. Training MIDI, token arrays and caches are excluded."],
                  size=14, gap=16)
    if entries:
        slides.start("Submitted checkpoints and provenance", "REPRODUCIBILITY | OWN WEIGHTS")
        rows = [[entry.get("path"), str(entry.get("trained_from_scratch")),
                 str(entry.get("all_896_songs_observed", "PENDING")), entry.get("sha256", "PENDING")]
                for entry in entries]
        slides.table(["Checkpoint", "From scratch", "All896 seen", "SHA256"], rows,
                     [285, 110, 105, 380], height=80, size=10.5)
        slides.text("Model weights originate from random initialization. Own-checkpoint continuation stages preserve ancestry in checkpoint metadata. Final all-song fitting has no independent held-out validation estimate.",
                    40, 420, 880, 65, size=13)


def render_detailed(slides, payload, reasons):
    student = payload.get("student", {})
    slides.start("Symbolic Music Generation", "HW2 | Report", "POP909 | Transformer models trained from random initialization")
    url = payload.get("cloud_drive_url", student.get("cloud_drive_url", ""))
    artifact_url = payload.get("artifact_url", student.get("artifact_url", ""))
    slides.bullets([f"Student ID: {student.get('student_id', 'r14942154')}",
                   f"User-provided code/artifact location: {artifact_url or 'PENDING'}",
                   f"Public cloud-drive folder: {url or 'PENDING - required before submission'}",
                   "Basic a: unconditional 32-bar generation; Basic b: 8-bar prompt + 24-bar continuation.",
                   "Bonus: 16-bar melody-conditioned piano accompaniment.",
                   "DRAFT: " + "; ".join(reasons[:3]) if reasons else "All required result and submission checks are complete."], size=15, gap=16)
    if reasons:
        for i, part in enumerate(chunks(reasons, 7)):
            slides.start("Remaining requirements" + (f" ({i+1})" if len(reasons)>7 else ""), "STATUS | DRAFT")
            slides.bullets(part, size=13, gap=12)
    slides.start("Research questions and controlled comparisons", "EXPERIMENT DESIGN")
    questions = payload.get("research_questions") or [
        "How do sampling temperature and truncation affect H4, GS, note density and repetition?",
        "Does a later checkpoint improve musical structure as well as token prediction?",
        "How much does available context affect continuation coherence and generation cost?",
        "Can a model produce accompaniment while preserving the complete supplied melody?",
        "High metric values are not a musical-quality guarantee; compare real data and inspect failures."]
    slides.bullets(questions, size=15, gap=17)
    description_pages(slides, "Data, splits and leakage controls", payload.get("data", {}))
    description_pages(slides, "Token representation and round-trip checks", payload.get("tokenizer", {}))
    description_pages(slides, "Architecture and training setup", payload.get("model", {}))
    training_pages(slides, payload.get("training", {}))
    slides.start("Training and validation loss", "TRAINING | MEASURED CURVE")
    plot_training(slides, payload.get("training", {}).get("history", []))
    a = payload.get("basic_a", {})
    base = a.get("real_baseline", {})
    slides.start("Evaluation protocol and real-data reference", "BASIC A | METRICS")
    slides.table(["Reference", "N", "H4", "GS"], [[base.get("name", "Real POP909"), str(base.get("n", "PENDING")),
                  format_value(base.get("H4")), format_value(base.get("GS"))]], [330, 70, 240, 240], height=44)
    slides.bullets([a.get("baseline_protocol", "PENDING: identify real-song/window selection, bar grid, common tokenizer and aggregation."),
                   "H4: pitch-class entropy over four-bar windows; GS: similarity of binary onset patterns across bars.",
                   "Compare to real-data distributions. Random pitches can increase H4; repeated or empty bars can inflate GS.",
                   "Overlapping H4 windows are not independent songs. Report uncertainty at the piece level."], top=273, size=13, gap=13)
    configs = configurations(payload)
    if not configs:
        slides.start("Unconditional generation comparison", "BASIC A | PENDING")
        slides.bullets(["PENDING: at least three configurations x twenty 32-bar pieces."])
    for i, group in enumerate(chunks(configs, 6)):
        slides.start("Unconditional generation comparison" + (f" ({i+1})" if len(configs)>6 else ""), "BASIC A | ALL REQUIRED CONFIGURATIONS", "Metrics are mean +/- standard deviation when available; no missing score is filled in.")
        rows = [[str(c.get("id", "?")), str(c.get("n", len(c.get("samples", [])))),
                 format_value(c.get("metrics", {}).get("H4")), format_value(c.get("metrics", {}).get("GS")),
                 "YES" if c.get("id") == a.get("selected_configuration") else ""] for c in group]
        slides.table(["Configuration", "N", "H4", "GS", "Selected"], rows, [260, 55, 235, 235, 95], height=42, size=11.5)
    for group in chunks(configs, 4):
        slides.start("Decoding settings and output checks", "BASIC A | REPRODUCIBILITY")
        rows = [[str(c.get("id")), format_value(c.get("parameters", {})), format_value(c.get("checks", {}))] for c in group]
        slides.table(["Configuration", "Sampling / checkpoint", "Measured integrity checks"], rows, [180, 350, 350], height=70, size=10.5)
    description_pages(slides, "Why the submitted configuration was selected", a.get("selection", {}), "BASIC A | ANALYSIS")
    diagnostics = a.get("diagnostics", {})
    diagnostic_pages(slides, diagnostics)
    b = payload.get("basic_b", {})
    slides.start("Prompt continuation protocol", "BASIC B | SAME CHECKPOINT")
    slides.bullets([f"Same checkpoint as basic a: {b.get('same_checkpoint_as_basic_a', 'PENDING')}",
                   f"Original eight-bar prefix preserved: {b.get('prompt_preservation_verified', 'PENDING')}",
                   b.get("method", "PENDING: explain beat-aligned conditioning, raw prefix retention and 24-bar stopping."),
                   "All three configurations for each of the three prompts are submitted; one best version is marked per prompt.",
                   "The assignment requires no objective metrics for this task. Best labels here use boundary diagnostics; no human listening comparison was completed."], size=15, gap=17)
    rows = b.get("results", [])
    for prompt in ("001", "002", "003"):
        group = [r for r in rows if str(r.get("prompt_id", "")).zfill(3) == prompt]
        slides.start(f"Prompt {prompt}: three continuation settings", "BASIC B | RESULTS AND SELECTION")
        table = [[str(r.get("configuration_id")), "BEST" if r.get("is_best") else "", str(r.get("output_midi", "PENDING")),
                  str(r.get("selection_reason", r.get("analysis", "PENDING")))] for r in group]
        slides.table(["Configuration", "Choice", "MIDI / matching WAV", "Selection evidence"], table or [["PENDING"]*4],
                     [150, 60, 310, 360], height=80, size=10.5)
        slides.text(b.get("prompt_analysis", {}).get(prompt, "PENDING: discuss continuity across bar 8 to 9, rhythmic stability, register and repetition."),
                    40, 440, 880, 52, size=12, minimum=10)
    for exp in payload.get("additional_experiments", []):
        slides.start(str(exp.get("name", "Additional experiment")), "ADDITIONAL EXPERIMENT | MEASURED RESULTS")
        slides.text(exp.get("method", "PENDING: controlled comparison method"), 40, 147, 880, 58, size=13)
        headers, table = exp.get("headers"), exp.get("rows")
        if headers and table:
            if len(table) > 6:
                raise ValueError("Additional-experiment tables must be paginated into <=6 rows")
            row_height = min(34, 190 / (len(table)+1))
            slides.table(headers, [[format_value(v) for v in row] for row in table], [880/len(headers)]*len(headers), top=220, height=row_height, size=11)
        else:
            slides.text(format_value(exp.get("results", "PENDING")), 40, 220, 880, 170, size=13)
        slides.text(exp.get("analysis", "PENDING: explain the observed effect and uncertainty."), 40, 425, 880, 65, size=12, minimum=10)
        if exp.get("discussion") or exp.get("limitations"):
            slides.start(str(exp.get("name", "Additional experiment")) + ": interpretation", "ADDITIONAL EXPERIMENT | DISCUSSION")
            slides.bullets([exp.get("discussion", "No additional discussion supplied."), exp.get("limitations", "No limitation supplied.")], size=15)
    bonus = payload.get("bonus", {})
    slides.start("Melody-conditioned accompaniment", "BONUS | METHOD")
    slides.bullets([f"Attempted: {bonus.get('attempted', False)}",
                   bonus.get("method", "PENDING: model, representation, conditioning visibility and training recipe."),
                   f"Original melody preserved: {bonus.get('melody_preservation_verified', 'PENDING')}",
                   "PIANO ground truth is evaluation-only for the ten prompts; it is not a generation input.",
                   "CA uses the supplied chord annotations. OA compares pitch class, velocity, duration and inter-onset-interval distributions."], size=15)
    for i, group in enumerate(chunks(bonus.get("results", []), 5)):
        slides.start("Accompaniment results" + (f" ({i+1})" if len(bonus.get("results", []))>5 else ""), "BONUS | ALL TEN PROMPTS")
        table = [[str(r.get("prompt_id"))] + [format_value(r.get("metrics", {}).get(m)) for m in BONUS_METRICS] for r in group]
        slides.table(["Prompt"]+list(BONUS_METRICS), table, [100, 156, 156, 156, 156, 156], height=43, size=11)
        slides.text("CA/OA are incomplete quality measures: OA discards octave and much timing order; empty/empty bars can score 1. All 16 bars and all ten IDs must be evaluated without skipped files.",
                    40, 443, 880, 46, size=12)
    if bonus.get("aggregate"):
        description_pages(slides, "Accompaniment aggregate and interpretation", bonus["aggregate"], "BONUS | ANALYSIS")
    failures = payload.get("failures", [])
    if failures:
        for i, part in enumerate(chunks(failures, 3)):
            slides.start("Failure cases and listening evidence" + (f" ({i+1})" if len(failures)>3 else ""), "ANALYSIS | LIMITATIONS")
            slides.bullets([format_value(v) for v in part], size=14, gap=22)
    slides.start("Limitations and what the results establish", "DISCUSSION")
    slides.bullets(payload.get("limitations") or ["PENDING: measured limitations, including listening and training-budget limits.",
                    "Metric proximity is not proof of perceptual quality or long-range form.",
                    "No claim is made about unmeasured experiments or subjective ratings."], size=15, gap=19)
    reproduction_pages(slides, payload.get("reproducibility", {}))
    slides.start("AI assistance and attribution", "DISCLOSURE")
    slides.bullets([payload.get("ai_disclosure", "Codex assisted with code, execution, result aggregation, analysis and English report drafting. All numerical results come from saved experiment outputs; missing experiments are not invented."),
                   "The course HW2 slides explicitly permit using the internet and AI (p.34). Public code is cited; all learned music-model weights are trained from random initialization on the official training set.",
                   "The student should understand the method and verify the final submission. External training data and third-party model weights are excluded."], size=15, gap=22)
    references = payload.get("references", [])
    for i, part in enumerate(chunks(references, 5)):
        slides.start("References" + (f" ({i+1})" if len(references)>5 else ""), "REFERENCES")
        slides.bullets([format_value(v) for v in part], size=12.5, gap=17)



def experiment(payload, name):
    return next((row for row in payload.get("additional_experiments", []) if name.lower() in row.get("name", "").lower()), {})


def mean_text(value, precision=4):
    mean = metric_mean(value)
    return f"{mean:.{precision}f}" if finite_number(mean) else "PENDING"


def mean_sd(value, precision=4):
    if not isinstance(value, dict):
        return mean_text(value, precision)
    return mean_text(value, precision) + (f" +/- {value['std']:.{precision}f}" if finite_number(value.get("std")) else "")


def figure_page(slides, title, path, caption, section):
    from reportlab.lib.utils import ImageReader
    slides.start(title, section)
    path = ROOT / path
    if not path.is_file():
        raise ValueError(f"Missing report figure: {path}")
    image = ImageReader(str(path))
    width, height = image.getSize()
    scale = min(880/width, 325/height)
    shown_w, shown_h = width*scale, height*scale
    slides.c.drawImage(image, (960-shown_w)/2, 540-139-shown_h,
                       width=shown_w, height=shown_h, preserveAspectRatio=True)
    slides.text(caption, 40, 474, 880, 30, size=11, minimum=10)


def register_case_pages(slides, payload):
    """Show register failures independently of pitch-class/onset proxies."""
    study = payload.get("structured_failure_analysis", {}).get("basic_a", {})
    if not study:
        return
    summary = study["register_summary"]
    slides.start("A failure hidden by H4 / GS: register", "BASIC A | SPECIFIC FAILURE CASES",
                 "Same selected t12 outputs and descriptive 45-song reference. Every song has equal weight; pitches use MIDI numbers.")
    rows = []
    for label, field, percentage in (
        ("Mean note pitch, averaged by piece", "piece_mean_pitch", False),
        ("95th-percentile pitch, averaged by piece", "piece_mean_p95_pitch", False),
        ("Onset share at pitch >=96 (C7)", "piece_mean_fraction_pitch_ge_96", True),
    ):
        generated, real = summary.get("generated_" + field), summary.get("reference_" + field)
        if generated is None or real is None:
            raise ValueError(f"Register case summary is incomplete: {field}")
        rows.append([label, f"{generated*100:.3f}%" if percentage else f"{generated:.3f}",
                     f"{real*100:.3f}%" if percentage else f"{real:.3f}",
                     f"{(generated-real)*100:+.3f} percentage points" if percentage else f"{generated-real:+.3f} semitones"])
    slides.table(["Descriptor", "Generated (20)", "Real reference (45)", "Gap"], rows,
                 [345, 165, 175, 195], top=146, height=31, size=11.5)
    bar_rows = []
    for case in study["cases"]:
        for bar in case["bar_examples"]:
            bar_rows.append([case["sample_id"], f"{bar['bar']} ({bar['t0']:.1f}-{bar['t1']:.1f}s)",
                             str(bar["notes"]), f"{bar['pitch_min']}-{bar['pitch_max']}",
                             f"{bar['mean_pitch']:.2f}", f"{bar['fraction_pitch_ge_96']*100:.1f}%"])
    slides.table(["Sample", "Bar (WAV time)", "Onsets", "Pitch range", "Mean pitch", "Pitch >=96"],
                 bar_rows, [130, 235, 95, 140, 140, 140], top=284, height=29, size=11.5)
    slides.text("Cases: maximum piece-p95 (004), and nearest median-p95 (001); sample-ID ties. Each shows its two bars with the greatest onset share at >=96, then p95/bar-number ties. H4 discards octave and GS uses onset positions. This measured mismatch does not prove poor sound; no pitch clipping was applied.",
                40, 441, 880, 56, size=11.5, minimum=10.5)
    combined = study.get("combined_figure", {})
    if combined.get("figure_path"):
        figure_page(slides, "Where the high register appears", combined["figure_path"],
                    "Unchanged t12 samples 004 and 001. Red dotted line: MIDI 96 (C7), a descriptive marker. Bar numbers and WAV times support checking the actual audio; this is not a listening preference.",
                    "BASIC A | REGISTER CASE INSPECTION")


def continuation_case_pages(slides, payload):
    """Compare every continuation over the complete 24 generated bars."""
    study = payload.get("structured_failure_analysis", {}).get("basic_b", {})
    if not study:
        return
    comments = {
        "001": "t12 remains the local-boundary choice, yet density grows from 24.250 to 30.125 notes/bar (prompt: 15.500). t08 velocity rises from 101.73 to 120.96. A small seam score does not establish stable density or dynamics later.",
        "002": "t12 density stays closer to the prompt's 22.125 notes/bar than the other settings across all three windows. Its mean pitch moves from 68.14 to 65.73. By contrast, t08 reaches 42.875 notes/bar in bars 25-32. This is a descriptor comparison, not a preference rating.",
        "003": "t12 drops from 33.875 to 9.250 notes/bar while mean pitch rises 69.08 to 85.27 (+16.19 semitones). The original t10 choice changes much less between the first/last generated windows: 30.625 to 27.250 notes/bar and 63.58 to 64.33.",
    }
    for prompt in ("001", "002", "003"):
        group = sorted([r for r in study["rows"] if r["prompt_id"] == prompt],
                       key=lambda r: r["configuration_id"])
        if len(group) != 3:
            raise ValueError(f"Full-continuation case table needs all three settings: {prompt}")
        slides.start(f"Continuation {prompt}: compare the full 32 bars", "BASIC B | COMPLETE SEGMENT COMPARISON",
                     "Saved-token onset statistics; bars are 1-based. Prompt velocity is quantized in this table; the submitted MIDI retains its original velocities. * = original boundary-rule choice.")
        segments = [("Prompt", group[0]["segments"][0])]
        for row in group:
            label = row["configuration_id"] + (" *" if row["selected_by_original_boundary_rule"] else "")
            segments.extend((label, segment) for segment in row["segments"][1:])
        rows = []
        for label, s in segments:
            lo, hi = s["bar_range"]
            t0, t1 = s["time_range"]
            rows.append([label, f"{lo}-{hi} ({t0:.1f}-{t1:.1f}s)", f"{s['notes_per_bar']:.3f}",
                         f"{s['mean_pitch']:.2f}", f"{s['p95_pitch']:.2f}", f"{s['mean_velocity']:.2f}"])
        slides.table(["Set", "Bars (WAV time)", "Notes/bar", "Mean pitch", "p95 pitch", "Mean velocity"], rows,
                     [85, 265, 130, 130, 130, 140], top=146, height=27, size=11)
        slides.text(comments[prompt] + " Full per-bar values are retained in the CSV/JSON evidence.",
                    40, 452, 880, 48, size=11.3, minimum=10.5)


def harmony_case_page(slides, payload):
    cases = payload.get("structured_failure_analysis", {}).get("bonus", [])
    if not cases:
        return
    slides.start("Harmony failures: which bars disagree?", "BONUS | EXACT CHORD CASES",
                 "Official 24-triad estimator; reference = maximum-duration annotation per bar. Ground-truth piano is evaluation-only, never input to the generator.")
    rows = []
    for case in cases:
        for bar in case["bar_examples"]:
            rows.append([case["prompt_id"], f"{bar['bar']} ({bar['t0']:.2f}-{bar['t1']:.2f}s)",
                         bar["reference_chord"], bar["generated_chord"], bar["gt_piano_estimated_chord"],
                         "match" if bar["match"] else "mismatch"])
    slides.table(["Prompt", "Bar (WAV time)", "Reference", "Generated", "GT estimator", "Result"], rows,
                 [75, 250, 145, 145, 145, 120], top=146, height=33, size=11.5)
    slides.bullets([
        "Overall CA: 18/160 bars (11.25%). Prompt 004: 0/16; 009: 4/16 (bars 3, 4, 5, 14). The same GT-piano estimator matches 13/16 in both prompts: this proxy is imperfect.",
        "004 bars 1-2 receive relative-minor estimates, sharing two of three triad tones with the reference. A CA mismatch need not be audible dissonance; high pitch-class OA does not establish correct chord order.",
        "Examples: first two mismatches with a correct GT estimate, then first two successes (none for 004). Future models should test train-only chord/phrase objectives. The submitted generation and original selection remain unchanged."
    ], top=389, size=11.5, gap=10)


def audio_review_page(slides, payload):
    """Render actual audio-model observations only after complete coverage."""
    review = payload.get("audio_review", {})
    if review.get("completed") is not True:
        return
    reasons = validate_audio_review(review)
    if reasons:
        raise ValueError("Incomplete audio-model review: " + "; ".join(reasons))
    rows = review.get("report_rows", [])
    if len(rows) != 3 or {str(r.get("prompt_id", "")).zfill(3) for r in rows} != {"001", "002", "003"}:
        raise ValueError("Completed audio review requires one reviewed summary for each of the three prompts")
    if not review.get("report_summary") or not review.get("report_limitations"):
        raise ValueError("Completed audio review requires methods, coverage and limitations text")
    slides.start("Complete audio input, failed critic calibration", "BASIC B | AI-ASSISTED AUDIO REVIEW",
                 "The critic consumes the WAV signal. This is an AI-model review, not human listening or a listener-preference study.")
    slides.text(review["report_summary"], 40, 145, 880, 80, size=13, minimum=11.5)
    slides.table(["Prompt", "Exploratory model text and verification limits"],
                 [[str(r["prompt_id"]).zfill(3), r["observation"]] for r in sorted(rows, key=lambda r: str(r["prompt_id"]))],
                 [95, 785], top=235, height=51, size=12)
    slides.text(review["report_limitations"], 40, 449, 880, 48, size=11.5, minimum=10.5)


def render_complete(slides, p, reasons):
    """Twelve self-contained slides; full tables and audit studies stay in the package."""
    from reportlab.lib.utils import ImageReader

    a, b, bonus = p["basic_a"], p["basic_b"], p["bonus"]
    configs, base = configurations(p), a["real_baseline"]
    selected = a["selected_configuration"]
    chosen = next(row for row in configs if row["id"] == selected)
    failures = p["structured_failure_analysis"]
    register = failures["basic_a"]["register_summary"]
    pooled = bonus["aggregate"]["pooled"]

    # 1. One overview replaces the former cover and separate results overview.
    slides.start("Symbolic Music Generation", "HW2 | r14942154",
                 "POP909 | Causal Transformer models trained from random initialization")
    slides.text("Student ID: r14942154", 40, 147, 880, 30, size=18, bold=True)
    slides.text("Code, own weights and all listening files: " + p["artifact_url"],
                40, 190, 880, 30, size=15)
    slides.table(["Task", "Required result", "Submitted files"], [
        ["Basic a", "5 settings x 20 complete 32-bar samples; select t12", "20 MIDI + 20 WAV"],
        ["Basic b", "3 prompts x 3 settings; original 8 + 24 new bars", "9 MIDI + 9 WAV"],
        ["Bonus", "10 melody-conditioned 16-bar piano accompaniments", "10 MIDI + 10 WAV"],
    ], [110, 530, 240], top=238, height=44, size=13)
    slides.text("Beyond the minimum: an equal-token context experiment and a paired melody-conditioning ablation, with uncertainty and concrete register/harmony failures.",
                40, 431, 880, 44, size=15, bold=True)
    if hosting_warnings(p):
        slides.text("Hosting note: GitHub replaces the slides' cloud-drive folder at the user's request; TA acceptance of that substitution is unverified.",
                    40, 483, 880, 22, size=10.5)
    if reasons:
        slides.text("DRAFT: " + "; ".join(reasons), 40, 450, 880, 50, size=11)

    # 2. Keep the full model/method recipe, with short labeled blocks.
    slides.start("Data, representation and model", "METHOD | REPRODUCIBLE DESIGN")
    slides.text("Data and timing", 40, 145, 420, 27, size=18, bold=True)
    slides.text("Only the 896 supplied training songs; exclude all 13 prompt songs. Diagnostic split: 851/45 songs (seed 20261022); final own checkpoints observe all 896. The final fit has no held-out validation score.\n\nUse beat_midi.txt for beat/bar boundaries: source MIDI meters can be wrong. Export n annotation beats as n/4; retain pickup bars.",
                40, 181, 420, 148, size=13.5)
    slides.text("REMI-style event tokens", 40, 344, 420, 27, size=18, bold=True)
    slides.text("BOS, [Bar, Meter, Tempo, (Pos, Pitch, Vel, Dur)*], EOS. Vocabulary: 683; 8 subdivisions/beat, 32 velocity bins, durations 1..256 grid steps. Basic merges MELODY/BRIDGE/PIANO; bonus retains MELODY/PIANO roles.",
                40, 381, 420, 104, size=13.5)
    slides.text("Causal Transformer", 500, 145, 420, 27, size=18, bold=True)
    arch = p["model"]["architecture"]
    slides.text(f"{arch['layers']} layers, width {arch['width']}, {arch['heads']} heads; {p['model']['parameter_count']:,} parameters. Pre-LN, GELU, dropout 0.1, tied embeddings, learned absolute positions (capacity 4096; effective context 2048).\n\nRandom initialization; predict the next token with cross-entropy, ignoring padding. Bonus continues our own refined basic weights.",
                500, 181, 420, 148, size=13.5)
    slides.text("Optimization and decoding", 500, 344, 420, 27, size=18, bold=True)
    slides.text("AdamW (betas .9/.95, weight decay .1), norm clip 1; warmup + cosine LR; legal transposition -5..+5. Decode with grammar masks, pitch 21..108, at most 64 notes/bar. Overflow retains recent complete bars and resets positions/KV cache.",
                500, 381, 420, 104, size=13.5)

    # 3. Training progress and the strongest controlled training experiment share a page.
    slides.start("Training lineage and the context experiment", "TRAINING | MEASURED TOKEN BUDGETS",
                 "New non-padding targets at each stage; optimizer resets between stages. M4 Pro / PyTorch MPS.")
    stage_labels = {
        "basic_diagnostic": "Basic diagnostic", "basic_refined": "Basic refined",
        "context_2048": "Context 2048", "context_4096": "Context 4096",
        "basic_all": "Basic final", "roles_diagnostic": "Roles diagnostic", "roles_all": "Roles final",
    }
    lrs = ["3e-4", "1.8e-4", "8e-5", "8e-5", "5e-5", "1.5e-4", "5e-5"]
    rows = []
    for i, (name, d) in enumerate(p["training"]["stages"].items()):
        rows.append([stage_labels[name], str(d["steps"]), f"{d['nonpad_target_tokens']/1e6:.3f}M",
                     str(d["context"]), str(d["seen_songs"]), "2" if name == "context_4096" else "4",
                     lrs[i], mean_text(d["validation_CE"], 5) if d["validation_CE"] is not None else "not held out"])
    slides.table(["Stage", "Steps", "New targets", "Context", "Songs", "Batch", "LR", "Val CE"], rows,
                 [185, 70, 110, 85, 65, 65, 95, 205], top=148, height=26, size=11)
    slides.text("Basic: diagnostic -> refined -> context 2048 -> final. Roles: refined -> roles diagnostic -> roles final. Validation CE falls from 2.36666 (step 500) to 1.71925 (refined).", 
                40, 365, 880, 34, size=11.5)
    context = experiment(p, "Equal-token context")
    c2048, c4096 = context["rows"]
    slides.text("Extra experiment: same refined parent, LR 8e-5, ~4M new targets per branch.",
                40, 413, 880, 24, size=13.5, bold=True)
    slides.text(f"2048: CE {c2048[3]:.5f} in {c2048[4]:.1f}s; 4096: CE {c4096[3]:.5f} in {c4096[4]:.1f}s. Both use the same 45-song, 2048-token validation; token budgets differ by 0.086%. Select 2048 before final refitting. One run each; batches/steps differ, the parent favors 2048, and validation uses 2048. This does not establish that longer context is generally worse.",
                40, 446, 880, 59, size=12.5)

    # 4. All required unconditional configurations and the reference remain visible.
    slides.start("Basic a: all sampling settings and H4 / GS", "RESULTS | 100 COMPLETE 32-BAR PIECES",
                 "Same basic_final checkpoint, CPU float32; seed 20261022+i (i=0..19). Unconditional 4/4, 120 BPM; means +/- piece SD.")
    rows = [["Real reference (45)", "same grid adapter", mean_sd(base["H4"]), mean_sd(base["GS"], 6)]]
    for c in configs:
        q = c["parameters"]
        setting = f"T={q['temperature']:g}; p={q['top_p']:g}; k={q['top_k']}"
        rows.append([c["id"] + (" [selected]" if c["id"] == selected else ""), setting,
                     mean_sd(c["metrics"]["H4"]), mean_sd(c["metrics"]["GS"], 6)])
    slides.table(["Set", "T / top-p / top-k", "H4", "GS (official)"], rows,
                 [205, 225, 220, 230], top=153, height=37, size=12)
    slides.text("H4: pitch-class entropy in 4-bar windows. GS: bar-onset pattern similarity. Reference: one deterministic eligible 32-bar crop per diagnostic song, included in final fitting (descriptive reference); all use the same Pos8-to-Pos16 adapter. All 100 generated pieces pass length/note checks; zero skips.",
                40, 422, 880, 45, size=12)
    slides.text("Selection rule: minimize the mean absolute H4/GS gap in real-reference SD units. t12: 0.6155; k32: 0.6882; k16: 0.7593. This is selection on the reported samples, not an independent test or a musical-quality score.",
                40, 473, 880, 27, size=11.5)

    # 5. One clear failure figure replaces distributions and two register slides.
    slides.start("Basic a: similar metrics can hide register drift", "ANALYSIS | DENSITY, RHYTHM AND REGISTER")
    diag_rows = []
    for label, metrics in [("t08", next(c for c in configs if c["id"] == "t08")["metrics"]), ("t12", chosen["metrics"])]:
        diag_rows.append([label, mean_text(metrics["notes_per_bar"], 3), mean_text(metrics["GS_active"], 5)])
    slides.table(["Set", "Notes/bar", "GS active"], diag_rows,
                 [125, 140, 145], top=150, height=28, size=12)
    slides.text(f"GS active is native-grid onset similarity, not official GS. t08 is dense with nearly fixed onsets; t12 reduces both (real density: 22.953 notes/bar). Mean piece-p95 pitch is {register['generated_piece_mean_p95_pitch']:.3f} vs real {register['reference_piece_mean_p95_pitch']:.3f} (+14.022 semitones). H4 discards octave; GS ignores pitch.\n\n004 has the largest piece-p95 (100). Bar 26, 50-52 s: 7 of 9 onsets reach >=96 (C7). This is a descriptive warning, not proof of poor sound.",
                490, 148, 430, 130, size=12.5)
    slides.text("Selected sample 004 | whole-piece piano roll; dotted C7 line is a register marker, not a correctness cutoff.",
                40, 284, 880, 22, size=11.5)
    image = ImageReader(str(ROOT / failures["basic_a"]["cases"][0]["figure_path"]))
    iw, ih = image.getSize()
    scale = min(880 / iw, 194 / ih)
    w, h = iw * scale, ih * scale
    slides.c.drawImage(image, (960-w)/2, 540-303-h, width=w, height=h, preserveAspectRatio=True)

    # 6. The chosen files remain explicit and all nine seam scores are reported.
    slides.start("Basic b: all nine continuations and best labels", "RESULTS | SAME BASIC CHECKPOINT",
                 "8 supplied bars + 24 generated bars. Restore every original prefix note; all nine preserve pitch/velocity/timing identity within export rounding.")
    rows = []
    for r in sorted(b["results"], key=lambda row: (row["prompt_id"], row["configuration_id"])):
        d = r["boundary_diagnostics"]
        rows.append([r["prompt_id"], r["configuration_id"], f"{r['parameters']['temperature']:g}",
                     mean_text(d["boundary_selection_score"], 4), mean_text(d["note_density_ratio"], 3),
                     mean_text(d["pitch_mean_jump"], 3), "BEST" if r["is_best"] else ""])
    slides.table(["Prompt", "Set", "T", "Boundary score", "Density ratio", "Pitch jump", "Choice"], rows,
                 [85, 85, 65, 185, 160, 160, 140], top=153, height=28, size=12)
    slides.text("All use top-p=.95, top-k=0 and CPU float32; fixed seed per prompt (20261022 + prompt index 0..2). Compare 2 bars before/after the seam: |log density ratio| + |mean pitch jump|/12 + (1 - pitch-class Jaccard); lowest wins within each prompt.",
                40, 444, 880, 43, size=12)
    slides.text("Best labels measure local symbolic continuity. The assignment requires no objective metrics here; no human listening preference is claimed.",
                40, 487, 880, 20, size=10.5)

    # 7. All three generated windows retain full-sequence comparison in one table.
    slides.start("Basic b: what changes after the boundary?", "ANALYSIS | ALL NINE COMPLETE CONTINUATIONS",
                 "Each cell lists bars 9-16 / 17-24 / 25-32, in that order. Saved-token statistics; * = original boundary-rule choice.")
    group = sorted(failures["basic_b"]["rows"], key=lambda r: (r["prompt_id"], r["configuration_id"]))
    rows = []
    for r in group:
        segments = r["segments"][1:]
        rows.append([r["prompt_id"], r["configuration_id"] + (" *" if r["selected_by_original_boundary_rule"] else ""),
                     " / ".join(f"{s['notes_per_bar']:.3f}" for s in segments),
                     " / ".join(f"{s['mean_pitch']:.2f}" for s in segments),
                     " / ".join(f"{s['mean_velocity']:.2f}" for s in segments)])
    slides.table(["Prompt", "Set", "Notes/bar: early / mid / late", "Mean pitch: early / mid / late", "Velocity: early / mid / late"], rows,
                 [70, 70, 245, 245, 250], top=148, height=27, size=11.5)
    clocks, baselines = [], []
    for prompt in ("001", "002", "003"):
        r = next(r for r in group if r["prompt_id"] == prompt)
        seg = r["segments"]
        times = [seg[1]["time_range"][0]] + [s["time_range"][1] for s in seg[1:]]
        clocks.append(prompt + ": " + "/".join(f"{t:.1f}" for t in times) + "s")
        baselines.append(f"{prompt}: {seg[0]['notes_per_bar']:.2f} notes/bar, pitch {seg[0]['mean_pitch']:.2f}")
    slides.text("Window time boundaries: " + "; ".join(clocks) + ". Prompt bars 1-8: " + "; ".join(baselines) + ".",
                40, 428, 880, 33, size=10.5)
    slides.text("003/t12: late density falls 72.7% and pitch rises 16.19 semitones relative to bars 9-16; selected t10 stays much closer (30.625 -> 27.250 notes/bar, pitch 63.58 -> 64.33). A smooth seam does not establish later stability. Full per-bar CSV/JSON is included.",
                40, 465, 880, 38, size=11.5)

    # 8. Melody conditioning and all ten official result rows fit together.
    slides.start("Bonus: melody-conditioned accompaniment", "METHOD AND RESULTS | ALL TEN 16-BAR PROMPTS",
                 "Own roles_final: insert each bar's MELODY, then sample PIANO (T1, p=.95, k=0), seeing past bars/current melody. Input: melody.mid + beat_midi.txt; no reference piano/chords. Restore original melody; export MELODY + PIANO.")
    rows = [[r["prompt_id"]] + [mean_text(r["metrics"][m], 4) for m in BONUS_METRICS]
            for r in sorted(bonus["results"], key=lambda row: row["prompt_id"])]
    aggregate = [pooled["CA"], pooled["DP (Pitch-class OA)"], pooled["DV (Velocity OA)"],
                 pooled["DD (Duration OA)"], pooled["DIOI (Inter-Onset OA)"]]
    rows.append(["Mean"] + [mean_text(v, 4) for v in aggregate])
    slides.table(["Prompt"] + list(BONUS_METRICS), rows, [100, 156, 156, 156, 156, 156],
                 top=153, height=26, size=11.5)
    slides.text("CA: 24-triad agreement with bar annotations. OA_P/V/D/IOI: distribution overlap in pitch class / velocity / duration / inter-onset interval. All 16 bars evaluated per prompt; zero skips.",
                40, 477, 880, 29, size=11)

    # 9. Preserve the informative paired intervention and all five intervals.
    slides.start("Extra experiment: remove melody conditioning", "BONUS | PAIRED ABLATION")
    ablation = experiment(p, "Bonus melody-conditioning")
    slides.text("Same roles checkpoint, 10 prompts, seed schedule and T=1 / p=.95. Remove melody-note tokens, retain bar/meter/tempo/track headers; restore the complete original melody in both exports. Evaluate official CA/OA; bootstrap 5,000 paired whole-prompt resamples.",
                40, 145, 880, 63, size=13)
    slides.table(["Metric", "Normal", "Empty melody", "Difference", "95% low", "95% high"],
                 [[format_value(v, 4) for v in row] for row in ablation["rows"]],
                 [120, 150, 155, 155, 150, 150], top=215, height=31, size=12.5)
    slides.text("CA rises by 0.0750, OA_V by 0.2352 and OA_D by 0.1356; their intervals stay above zero. OA_P and OA_IOI intervals include zero. Melody access affects measured descriptors, with weaker evidence for pitch-class and onset-distribution overlap.",
                40, 414, 880, 43, size=13)
    slides.text("Limits: only 10 prompts; five unadjusted intervals. Empty melody is out of distribution and changes context length/sample history. This does not isolate perceived quality. The empty condition is evaluation-only and is not submitted.",
                40, 467, 880, 35, size=11)

    # 10. Keep exact harmony examples rather than a second generic discussion.
    slides.start("Harmony failures: which bars disagree?", "BONUS | SPECIFIC BAR COMPARISON",
                 "Official 24-triad estimator vs maximum-duration annotation per bar. Ground-truth piano is evaluated separately, never supplied to generation.")
    rows = []
    for case in failures["bonus"]:
        for bar in case["bar_examples"]:
            rows.append([case["prompt_id"], f"{bar['bar']} ({bar['t0']:.2f}-{bar['t1']:.2f}s)",
                         bar["reference_chord"], bar["generated_chord"], bar["gt_piano_estimated_chord"],
                         "match" if bar["match"] else "mismatch"])
    slides.table(["Prompt", "Bar (WAV time)", "Reference", "Generated", "GT estimator", "Result"], rows,
                 [75, 250, 145, 145, 145, 120], top=151, height=31, size=12)
    slides.bullets([
        "Overall CA: 18/160 bars. 004: 0/16; 009: 4/16 (bars 3, 4, 5, 14). The GT estimator matches 13/16 for both and 0.7625 overall, so it is an imperfect comparison, not a strict upper bound.",
        "004 bars 1-2 are estimated as relative minors (Ab:maj -> F:min; Eb:maj -> C:min): each pair shares 2 of 3 triad tones. A CA mismatch need not mean audible dissonance.",
        "High OA_P can coexist with wrong chord order. Test train-only chord/phrase supervision next; the current outputs and original selections remain unchanged."
    ], top=382, size=12.5, gap=11)

    # 11. State findings, honest limits, secondary tuning and exact artifact coverage.
    slides.start("Discussion and reproduction", "CONCLUSION | WHAT THE RESULTS SUPPORT")
    slides.bullets([
        "Training: at this warm start/budget, 2048 context gives lower validation CE than 4096. It costs less here, but batch size and the 2048-trained parent confound a general context-length claim.",
        "Generation: warmer sampling reduces dense, nearly fixed onsets; H4/GS still hide register drift. Melody conditioning improves CA/velocity/duration descriptors, yet only 18/160 bonus chord labels match. A chord/phrase objective and better long-range memory are useful next experiments.",
        "Secondary temperature check: the custom equal-weight mean of 5 bonus metrics selects T1 (.5771) over T1.2 (.5726). Paired 95% CI [-.0216, .0295] includes zero; this is in-sample tuning, not proven better music or a course-grade formula.",
        "Audio limit: all nine WAVs (15 min 25 s) reached an evaluation-only Qwen critic, but it invented music on silence. Its prose is not used as listening evidence. No human listening ratings or official SI experiment are claimed.",
        "Reproduce: pip install -r requirements.txt, then src/infer.py; exact seeds/settings, output paths and WAV commands are in README. Package: 39 MIDI/WAV pairs + 2 own checkpoints, vocabulary, code, figures and full experiment JSON/CSV. Training data, external critic weights and caches are excluded."
    ], top=145, size=13.5, gap=12)

    # 12. References and assistance remain visible without an audit appendix.
    slides.start("Sources and AI assistance", "ATTRIBUTION | FULL HASHES AND TABLES IN THE PACKAGE")
    refs = [
        "Course Homework 2 (Bo-Rui Chen, 2026): task pp.11-18,22-28; submission pp.29-32; grading p.33; AI permission p.34; rules p.35.",
        "Wang et al. POP909: A Pop-Song Dataset for Music Arrangement Generation. ISMIR 2020. https://arxiv.org/abs/2008.07142",
        "Huang & Yang. Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions. ACM MM 2020. https://arxiv.org/abs/2002.00212",
        "Ren et al. PopMAG: Pop Music Accompaniment Generation. ACM MM 2020. Inspiration for role-aware accompaniment; our weights are trained from scratch.",
        "Course HW2_scripts/tokenization.py, eval_basic.py, eval_bonus.py; MusDr side_utils.py: https://github.com/slSeanWU/MusDr/blob/master/musdr/side_utils.py",
        "Chu et al. Qwen2-Audio Technical Report (2024). https://arxiv.org/abs/2407.10759 ; evaluation-only model: https://huggingface.co/mlx-community/Qwen2-Audio-7B-Instruct-4bit (Apache-2.0).",
        "Blaizzy/mlx-audio v0.6.1 (MIT): https://github.com/Blaizzy/mlx-audio . Official HF audio frontend substituted after numerical comparison; full model revision/provenance is saved."
    ]
    slides.bullets(refs, top=145, size=12, gap=10)
    slides.text("AI assistance: Codex assisted implementation, execution, result aggregation, analysis and English drafting. All numerical results come from saved runs. The external Qwen critic is evaluation-only, failed calibration, and is not either submitted composition model. Full source hashes, checkpoint ancestry, per-piece results and package checks remain in results/report_data.json, FILE_SHA256.json and SUBMISSION_CHECKS.json.",
                40, 455, 880, 50, size=11.5)


def render(slides, payload, reasons):
    if payload.get("status") == "complete" and not payload.get("assembly",{}).get("missing_experiment_requirements"):
        render_complete(slides, payload, reasons)
    else:
        render_detailed(slides, payload, reasons)


def build_report(payload, output):
    reasons = validate_payload(payload)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".hw2-report-", suffix=".pdf", dir=output.parent)
    os.close(handle)
    try:
        slides = Slides(temporary, reasons, payload.get("student", {}).get("student_id", "r14942154"))
        render(slides, payload, reasons)
        slides.finish()
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"path": str(output), "pages": slides.page, "aspect_ratio": "16:9", "draft": bool(reasons), "draft_reasons": reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/report_data.json")
    parser.add_argument("--output", type=Path, default=ROOT / "r14942154_report.pdf")
    parser.add_argument("--cloud-url", help="Public cloud-drive folder; do not substitute a GitHub URL")
    parser.add_argument("--artifact-url", help="Separate user-provided artifact URL; does not satisfy the cloud-drive rule")
    args = parser.parse_args()
    payload = read_json(args.input)
    if args.cloud_url:
        payload["cloud_drive_url"] = args.cloud_url
        payload["cloud_drive_requirement_unverified"] = False
    if args.artifact_url:
        payload["artifact_url"] = args.artifact_url
    print(json.dumps(build_report(payload, args.output), indent=2))


if __name__ == "__main__":
    main()
