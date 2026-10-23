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


def render_complete(slides, p, reasons):
    """Compact main report; full per-piece evidence remains in the package."""
    a, b, bonus = p["basic_a"], p["basic_b"], p["bonus"]
    configs, base = configurations(p), a["real_baseline"]
    selected = a["selected_configuration"]
    chosen = next(row for row in configs if row["id"] == selected)
    extras = p.get("additional_experiments", [])
    cloud_note = hosting_warnings(p)
    slides.start("Symbolic Music Generation", "HW2 | r14942154", "A causal Transformer trained from scratch on POP909")
    slides.bullets([
        "Student ID: r14942154",
        "Code, own model weights and MIDI/WAV results:",
        p.get("artifact_url", "PENDING"),
        "Basic a: 5 settings x 20 complete 32-bar pieces; submit the selected 20 MIDI/WAV pairs.",
        "Basic b: all 9 prompt continuations. Bonus: all 10 melody-conditioned accompaniments.",
        "Extra experiments: context length, token prediction, quantization, sampling, checkpoint learning and melody conditioning."
    ], size=16, gap=14)
    if cloud_note:
        slides.text("Hosting note: GitHub substitution for the slides' cloud-drive wording is unverified; no TA acceptance is claimed.", 40, 468, 880, 32, size=10.5)
    if reasons:
        slides.text("DRAFT: " + "; ".join(reasons), 40, 450, 880, 50, size=11)

    slides.start("Main results and how to read them", "OVERVIEW")
    rows = [
        ["Basic a", f"{selected}: H4 {mean_text(chosen['metrics']['H4'])}; GS {mean_text(chosen['metrics']['GS'],6)}", "20 selected MIDI + 20 WAV"],
        ["Basic b", "Best by local boundary rule: 001:t12, 002:t12, 003:t10", "9 MIDI + 9 WAV; original 8 bars retained"],
        ["Bonus", "10 prompts; required CA and four OA measures reported", "10 MIDI + 10 WAV; original melody retained"],
    ]
    slides.table(["Task", "Measured result", "Submitted listening files"], rows, [100, 450, 330], height=54, size=12)
    slides.bullets([
        "The report compares actual saved outputs. All final checkpoints have observed the complete 896-song training set; no pretrained weights or extra training data are used.",
        "A 45-song split supports pre-refit diagnostics. Those songs enter the final all-song fit, so the real-data reference is descriptive, not an independent test of the final model.",
        "H4/GS and CA/OA are proxies. No human preference ratings were collected, and none of these values is a course-grade formula."
    ], top=391, size=12, gap=10)

    slides.start("Data preparation: trust the beat annotations", "METHOD | DATA AND CLOCKS")
    slides.bullets([
        "Use only the supplied 896 training songs. Merge MELODY, BRIDGE and PIANO for basic tasks; retain separate MELODY/PIANO roles for accompaniment. All 13 official prompt songs are excluded from fitting.",
        p["data"]["diagnostic_split"],
        "The TA announcement warns that many source MIDI time signatures are wrong. Bar boundaries and beat times come from beat_midi.txt; source MIDI time-signature metadata is not authoritative.",
        "Canonical export: one annotation beat becomes one quarter note; a bar with n supplied beats is written as n/4. This preserves the supplied clock and bar count, including mixed meters. It does not recover an original denominator such as 6/8.",
        "Short pickup bars are retained. Notes before the first annotated beat are clamped; representation error is measured rather than called lossless. Original prompt notes are restored after generation."
    ], size=14.5, gap=16)

    slides.start("Token representation and causal model", "METHOD | IMPLEMENTATION")
    slides.text("REMI-style event sequence", 40, 148, 410, 30, size=18, bold=True)
    slides.text("BOS, [Bar, Meter, Tempo, (Pos, Pitch, Vel, Dur)*], ..., EOS\n\n683 tokens; 8 subdivisions per supplied beat; 32 velocity bins; note duration 1..256 grid steps (up to 32 beats).\n\nFor bonus, each bar contains Track_Melody + its notes, then Track_Piano + its notes. Future-bar melody is not visible while that bar's piano is sampled.",40,190,410,245,size=15)
    arch = p["model"]["architecture"]
    slides.text("Decoder-only Transformer", 500, 148, 420, 30, size=18, bold=True)
    slides.text(f"{arch.get('layers')} layers; width {arch.get('width')}; {arch.get('heads')} attention heads; {p['model'].get('parameter_count',0):,} parameters.\n\nLearned absolute positions (4096 capacity), pre-layer normalization, GELU, dropout 0.1 and tied embedding/output weights. Selected effective training/inference context: 2048 tokens.\n\nTrain by next-token cross-entropy, ignoring padding. Causal attention sees only preceding tokens. All music-model weights originate from random initialization.",500,190,420,245,size=15)
    slides.text("Reliable decoding uses legal event-type masks, ascending pitches at one onset, piano-range pitches 21..108, a 64-note/bar cap and exact end boundaries. Overflow retains recent complete bars and rebuilds the KV cache; this limits long-range memory.",40,451,880,43,size=12)

    slides.start("Optimization and checkpoint lineage", "TRAINING | ACTUAL EXPOSURE", "Apple M4 Pro / PyTorch MPS training. Own checkpoint continuation resets the optimizer at each stage.")
    stages = p["training"]["stages"]
    rows = []
    for name, d in stages.items():
        rows.append([name, str(d.get("steps")), f"{d['nonpad_target_tokens']/1e6:.3f}M", str(d.get("context")), str(d.get("seen_songs")), mean_text(d.get("validation_CE")) if d.get("validation_CE") is not None else "not held out"])
    slides.table(["Stage", "Steps", "New targets", "Context", "Songs seen", "Val CE"], rows, [240,80,145,105,125,185], top=149,height=33,size=11)
    slides.text("AdamW: betas (0.9, 0.95), weight decay 0.1, gradient norm clip 1.0; warmup + cosine decay. Learning rates by row: 3e-4, 1.8e-4, 8e-5, 8e-5, 5e-5, 1.5e-4, 5e-5. Batch 4, except context_4096 batch 2. Legal pitch transposition -5..+5 semitones; no validation augmentation.",40,423,880,62,size=12)

    slides.start("Training and validation loss", "TRAINING | MEASURED CURVE")
    plot_training(slides, p["training"]["history"])
    slides.text("The fixed 851/45 song split uses deterministic 2048-token validation windows. The optimizer restarts after step 2000. CE falls from 2.36666 at step 500 to 1.71925 after refinement. Final all-song fitting has no held-out validation score.",40,466,880,36,size=11.5)

    context = experiment(p, "Equal-token context")
    slides.start("Extra experiment: does more context help?", "EXPERIMENT | CONTROLLED TOKEN BUDGET")
    slides.text(context["method"],40,145,880,76,size=14)
    slides.table(context["headers"], [[format_value(v,5) for v in row] for row in context["rows"]], [125,190,100,230,235],top=231,height=36,size=13)
    slides.bullets([context["analysis"], context["limitations"]],top=365,size=13,gap=13)

    token = experiment(p, "What improves")
    slides.start("Extra experiment: grammar versus note content", "EXPERIMENT | TEACHER-FORCED PREDICTION")
    slides.text(token["method"],40,143,880,65,size=13)
    slides.table(token["headers"], [[format_value(v,3) for v in row] for row in token["rows"]], [180,175,175,175,175],top=221,height=32,size=12)
    slides.text(token["analysis"] + " " + token["limitations"],40,432,880,65,size=12,minimum=10.5)

    slides.start("Basic a: sampling settings and required metrics", "RESULTS | 100 COMPLETE 32-BAR PIECES", "Each setting uses the same final checkpoint, CPU float32 and piece seed 20261022+i (i=0..19). Means +/- piece standard deviations.")
    rows = [["Real reference (45)", "same adapter", mean_sd(base["H4"]), mean_sd(base["GS"],6)]]
    for c in configs:
        q=c["parameters"]
        setting = f"T={q['temperature']:g}, " + (f"k={q['top_k']}" if q.get('top_k') else f"p={q['top_p']:g}")
        rows.append([c['id']+(' [selected]' if c['id']==selected else ''),setting,mean_sd(c['metrics']['H4']),mean_sd(c['metrics']['GS'],6)])
    slides.table(["Set", "Sampling", "H4", "GS (official)"], rows, [230,170,230,250],top=160,height=40,size=12)
    slides.text("H4 is pitch-class entropy over 4-bar windows; GS compares bar onset patterns. Unconditional generation fixes 4/4 and 120 BPM. All 100 pieces pass 32-bar/ID/hash/note checks with zero skips. Real/generated music uses the same Pos8-to-Pos16 mapping; 2000 whole-piece bootstrap draws provide saved confidence intervals.",40,457,880,43,size=12)

    slides.start("Why select t12? Inspect density and rhythm too", "BASIC A | ANALYSIS AND FAILURE MODES")
    rows=[]
    for c in configs:
        m=c['metrics']; rows.append([c['id'], mean_text(m['notes_per_bar'],3),mean_text(m['GS_active'],4),mean_text(m['velocity_std'],3),mean_text(m['mean_duration_beats'],3)])
    slides.table(["Set", "Notes/bar", "GS active", "Velocity SD", "Mean duration/beat"], rows, [100,170,170,190,250],top=148,height=34,size=12)
    slides.bullets([
        f"Predeclared choice: minimize the mean of absolute H4 and GS mean gaps divided by real-reference SDs. t12 scores 0.6155, versus k32 0.6882 and k16 0.7593. This rule selects on the same reported samples; it is not an independent test.",
        f"t12 has {mean_text(chosen['metrics']['notes_per_bar'],3)} notes/bar versus real 22.953. T=0.8 is much denser (38.952), with nearly identical onset patterns (GS active 0.98995) and little velocity variation (SD 3.833). Exact note-level bars are still unique; the collapse is rhythmic/expressive, not literal repeated bars.",
        "t12 minus t10: GS -0.02244, paired 95% bootstrap CI [-0.03428, -0.01233]. H4 +0.05914, CI [-0.01600, 0.13447], so the H4 difference is less certain. No human listening preference is inferred."
    ],top=371,size=12.5,gap=12)

    figure_page(slides,"Sampling changes distributions, not just one mean",p['figures']['sampling'],
                "All 20 pieces per setting are shown. The real reference uses the same event grid. These are metric distributions, not listener scores.","BASIC A | DISTRIBUTIONS")
    figure_page(slides,"One generated piece: notes and bar similarity",p['figures']['structure'],
                "t12 / sample_001: the first saved seed, not a cherry-picked example. Cosine similarity of bar pitch-class counts is a diagnostic; this is not the optional MusDr SI score.","BASIC A | STRUCTURE INSPECTION")

    slides.start("Basic b: all 9 continuations and chosen versions", "RESULTS | SAME BASIC CHECKPOINT", "Supply 8 original bars, sample 24 new bars, then restore every original prefix note. All nine retain the 8-bar note identity and contain new notes.")
    boundary = experiment(p,"Continuation boundary")
    entries=p.get('boundary_details',[])
    # Evidence is loaded by the gatherer; no boundary values are inferred from prose.
    rows=[]
    for r in sorted(b['results'],key=lambda row:(row['prompt_id'],row['configuration_id'])):
        d=r.get('boundary_diagnostics',{})
        rows.append([r['prompt_id'],r['configuration_id'],f"T={r['parameters']['temperature']:g}; p=.95",mean_text(d.get('boundary_selection_score'),4),mean_text(d.get('note_density_ratio'),3),mean_text(d.get('pitch_mean_jump'),3),'BEST' if r['is_best'] else ''])
    slides.table(["Prompt","Set","Sampling","Boundary score","Density ratio","Pitch jump","Choice"],rows,[85,80,155,150,145,150,115],top=155,height=29,size=11)
    slides.text("Selection compares the two bars before/after the boundary: |log density ratio| + |mean pitch jump|/12 + (1 - pitch-class Jaccard). Lower wins within each prompt. All use the same basic checkpoint and CPU; only T=0.8/1.0/1.2 changes. Best labels are local continuity preferences, not listener judgments.",40,459,880,42,size=11.5)

    slides.start("Continuation: what the chosen labels establish", "BASIC B | SPECIFIC ANALYSIS")
    slides.bullets([
        "001 / t12: boundary density ratio 1.333, mean pitch shift -0.296 semitones, pitch-class Jaccard 0.857. Its score 0.4552 is lower than t08 1.0342 and t10 1.0499; the other settings nearly double the local density.",
        "002 / t12: density ratio 0.913, pitch shift +2.588, Jaccard 0.750; score 0.5566 versus 0.8008 and 0.7137. It preserves local density better, although it does not minimize every descriptor separately.",
        "003 / t10: density ratio 1.113, pitch shift +0.941, Jaccard 0.857; score 0.3285. t12 instead jumps +7.108 semitones at the seam, illustrating that the best unconditional temperature need not be best for every prompt.",
        "Boundary continuity does not evaluate the remaining 24-bar form. The model sees only its recent complete bars after cache rollover. Full-32-bar and continuation-only-24-bar metrics are saved separately; the supplied prefix is not counted as newly generated music."
    ],size=14,gap=19)

    slides.start("Bonus: melody-conditioned piano generation", "METHOD | BAR-INTERLEAVED ROLES")
    bonus_q=bonus.get('parameters',{})
    slides.bullets([
        "Start from our own refined basic checkpoint, then learn MELODY/PIANO bar interleaving on 851 songs; refit on all 896. This is an own-weight transfer stage, not use of a pretrained external model.",
        "For each of 16 bars, insert that bar's melody tokens and sample its piano events. The model sees past bars and the current melody bar, not the full future melody. BRIDGE is excluded from role training.",
        "Generation reads only melody.mid and beat_midi.txt. Ground-truth piano and chord annotations are evaluation-only, so generated piano is not given the reference accompaniment or the target chords.",
        f"Formal bonus configuration: {format_value(bonus_q)}. The original melody pitch/velocity is retained, with sub-millisecond export rounding checked independently. Exactly MELODY + PIANO tracks and all 16 bars are validated.",
        "The source melody is restored after token generation, including mixed-meter prompts and the long ending of prompt 007. Generated piano uses bounded durations and no active same-pitch overlap."
    ],size=14,gap=17)

    slides.start("Bonus: all 10 prompt results", "RESULTS | OFFICIAL CA AND OA", "Each row evaluates all 16 bars. OA_P is the supplied pitch-class overlap measure; higher overlap alone does not establish harmonic correctness.")
    rows=[[r['prompt_id']]+[mean_text(r['metrics'][m],4) for m in BONUS_METRICS] for r in sorted(bonus['results'],key=lambda r:r['prompt_id'])]
    slides.table(["Prompt"]+list(BONUS_METRICS),rows,[100,156,156,156,156,156],top=152,height=27,size=12)
    pooled=bonus.get('aggregate',{}).get('pooled',{})
    slides.text("Means over all ten prompts: CA "+mean_text(pooled.get('CA'))+"; OA_P "+mean_text(pooled.get('DP (Pitch-class OA)'))+"; OA_V "+mean_text(pooled.get('DV (Velocity OA)'))+"; OA_D "+mean_text(pooled.get('DD (Duration OA)'))+"; OA_IOI "+mean_text(pooled.get('DIOI (Inter-Onset OA)'))+". Zero skipped files. CA-ceiling "+mean_text(pooled.get('CA-ceiling'))+" is the same chord estimator on ground truth, not a strict upper bound.",40,467,880,36,size=11.5)

    tuning=experiment(p,'Bonus sampling temperature')
    if tuning:
        slides.start("Bonus: select the inference setting", "ADDITIONAL EXPERIMENT | SAMPLING")
        slides.text(tuning['method'],40,145,880,75,size=13)
        slides.table(tuning['headers'],[[format_value(v,4) for v in row] for row in tuning['rows']],[880/len(tuning['headers'])]*len(tuning['headers']),top=230,height=34,size=12)
        slides.text("The predeclared equal-weight composite selects T=1.0 (0.57708), narrowly above T=1.2 (0.57258). T=1.2 has higher CA and IOI overlap, but lower velocity/duration overlap. We retain T=1.0 for the formal files and conditioning ablation; no setting dominates every descriptor.",40,387,880,62,size=13)
        slides.text("All three settings use the same ten prompts and evaluation ground truth: this is in-sample tuning, with no independent test. The custom composite is not the course grading formula and does not establish listener preference. All per-prompt scores and paired bootstrap differences are retained.",40,458,880,41,size=11.5)

    ablation=experiment(p,'Bonus melody-conditioning')
    slides.start("Extra experiment: remove melody conditioning", "BONUS | PAIRED ABLATION AT T=1.0")
    slides.text("Same roles checkpoint, ten prompts, seed schedule and T=1.0 / top-p=0.95. Remove melody-note tokens while keeping bar, meter, tempo and track headers; restore the original melody in both exported versions. Evaluate official CA/OA and bootstrap 5000 paired whole-prompt resamples.",40,143,880,72,size=13)
    slides.table(ablation['headers'],[[format_value(v,4) for v in row] for row in ablation['rows']],[120,150,155,155,150,150],top=230,height=32,size=12)
    slides.text("Normal melody improves CA by 0.0750 (95% CI [0.0250, 0.1313]), velocity overlap by 0.2352 and duration overlap by 0.1356. Pitch-class and IOI intervals include zero. This provides evidence of a conditioning effect on measured descriptors, with weak evidence for some metrics.",40,435,880,42,size=11.5)
    slides.text("Caution: deleting all melody is out of distribution and changes context length/sampled histories. Both exports restore the full melody; the empty condition is not submitted. This does not isolate an effect on perceived musical quality.",40,479,880,26,size=10)

    slides.start("Discussion, limits and reproduction", "CONCLUSION")
    slides.bullets([
        "Useful changes: more training improves note-content prediction; the controlled 2048-token branch predicts better at this budget; warmer sampling reduces the dense, nearly fixed onset behavior of low temperature.",
        "What remains weak: accompaniment CA is low even when pitch-class OA is high. The model has no explicit chord input or harmonic objective. Distribution overlap can hide wrong chord order, register or voice leading; a high OA_P should not be read as correct harmony.",
        "Long-range form is unverified. Absolute-position resets, finite recent-bar context and grammar constraints can limit phrasing. No human listening study, architecture sweep or official SI experiment is claimed.",
        "Reproduce inference with pip install -r requirements.txt, then src/infer.py; README gives exact checkpoint, seed, settings, output-path and WAV-rendering commands. All 39 submitted MIDI/WAV pairs are synthesized from matching MIDI and checked for duration, finite audio and non-silence.",
        "The public package includes two own checkpoints, vocabulary, code, licenses, measured tables/manifests, figures and hashes. It excludes training MIDI, preprocessed training arrays, external weights and caches. See the appendix for audit experiments and references."
    ],size=14,gap=15)

    slides.start("Appendix A: quantization and export error", "EXTRA EXPERIMENT | REPRESENTATION")
    quant=experiment(p,'Quantization resolution')
    slides.text(quant['method'],40,143,880,63,size=13)
    slides.table(quant['headers'],[[format_value(v,3) for v in row] for row in quant['rows']],[140,160,190,190,200],top=209,height=32,size=12)
    slides.bullets([
        "Across 1,508,536 notes, 16 subdivisions lowers onset/offset MAE but increases vocabulary from 683 to 1067 tokens. We implement 8; this is a reconstruction audit, not a comparison of models retrained at each grid.",
        "Six codec round-trip cases preserve all represented notes and have maximum MIDI export timing error below 0.317 ms. This small export error is separate from source-to-grid quantization error.",
        "The duration cap clips two long source notes (worst offset error 8.732 s); 196 early notes are clamped before the first annotation (worst onset error 0.709 s). The representation is therefore not lossless. Full per-case audits and reproduction code are provided."
    ],top=354,size=12,gap=12)

    slides.start("Appendix B: checkpoint progress and GS sensitivity", "EXTRA EXPERIMENT | METRIC INTERPRETATION")
    learned=experiment(p,'Checkpoint learning')
    rows=[[row[0],row[2],row[3],row[4],row[5],row[6]] for row in learned['rows']]
    slides.table(['Checkpoint','H4','GS','GS active','Notes/bar','Empty fraction'],[[format_value(v,4) for v in row] for row in rows],[200,135,135,135,140,135],top=147,height=40,size=12)
    slides.bullets([
        "At step 500, GS 0.97074 already resembles the real reference 0.97083, but density is only 11.559 notes/bar versus 22.953 and 3.28% of bars are empty. At step 2000, density rises to 28.164 while H4/GS move away from the real means. Lower prediction loss alone does not optimize these descriptors.",
        "The required official GS uses 128 binary positions. Mapping our 8 subdivisions to the official 16-subdivision IDs leaves odd slots zero. GS active separately normalizes onset XOR by max(bar beats) x 8; it does not replace official GS.",
        "For the real reference GS/GS active are 0.97083/0.88679; selected t12 gives 0.97017/0.88066. Shared grids make the comparison fair, but onset-only similarity ignores pitches, durations, velocities and the number of simultaneous notes.",
        "Checkpoint generation uses completed 20-piece sets. These pre-refit checkpoints are diagnostics; context/learning stage effects are not an isolated controlled quality comparison."
    ],top=295,size=13,gap=16)

    slides.start("Appendix C: reference sensitivity and local matches", "EXTRA EXPERIMENT | ROBUSTNESS")
    ref=experiment(p,'Real-data reference sensitivity')
    slides.text("Take one deterministic center-nearest eligible 32-bar crop per song using the same official metric adapter. Consider all 896 training songs; list individually the five with no eligible crop. Keep the original 45-song reference and configuration-selection rule fixed.",40,145,880,65,size=13)
    slides.table(ref['headers'],[[format_value(v,4) for v in row] for row in ref['rows']],[195,105,105,145,145,185],top=226,height=32,size=12)
    slides.bullets([
        "The all-eligible reference includes 891/896 songs; five songs have no eligible 32-bar crop. Its H4/GS/density means fall inside the 45-song primary-reference bootstrap intervals. The original reference and t12 choice are retained; the two reference sets overlap and are not independent tests.",
        "No duplicate full pieces and no exact non-empty 1/2/4/8-bar training-window matches are found in any final basic-a setting. Windows match position, pitch and duration while ignoring tempo/velocity/track names. Shared fully silent windows are reported separately.",
        "This narrow exact-match check cannot rule out transposition, approximation, partial copying or memorization. Zero exact matches is not proof of originality. Selection and uncertainty estimates use reported output sets, not a new hidden test."
    ],top=340,size=12.5,gap=13)

    slides.start("References, source evidence and AI disclosure", "APPENDIX D | ATTRIBUTION")
    refs=p.get('references',[])
    slides.bullets(refs,size=11.5,gap=13,top=143)
    slides.text("AI assistance: Codex assisted implementation, execution, result aggregation, analysis and English drafting. The HW2 slides permit internet and AI use (p.34). All reported numbers come from saved actual outputs. Full SHA256 inventories, checkpoint ancestry and per-piece tables are in results/report_data.json, FILE_SHA256.json and SUBMISSION_CHECKS.json.",40,443,880,58,size=11)


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
