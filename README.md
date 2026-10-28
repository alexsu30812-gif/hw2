# HW2: Symbolic Music Generation

Student ID: **r14942154**

User-provided repository: https://github.com/alexsu30812-gif/hw2

Public files use the student's explicitly selected GitHub repository. The
slides say “cloud drive”; acceptance of GitHub as that substitute has not been
confirmed with the TA. This hosting uncertainty is recorded separately from
completion of the experiments: `user_selected_repository_hosting=true` and
`cloud_drive_requirement_unverified=true`. No TA acceptance is claimed.

This project trains Transformer music-generation models from random initialization
on the complete official POP909 training set (896 songs). It implements
unconditional generation, prompt continuation, and melody-conditioned piano
accompaniment. The final results and model details are recorded in
`results/report_data.json`. All required experiments, the bonus and the 39
submitted MIDI/WAV pairs are complete. Substantive missing work would still
prevent a complete package.

## Environment

Use Python 3.10. Create a fresh environment and install the inference packages:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch runs on CPU; `--device mps` can use Apple Silicon when supported by the
inference CLI. Composition inference loads our own trained checkpoints; it
requires no pretrained model download. macOS audio synthesis uses the included Swift tool and
system General MIDI piano sounds. Rendering on other platforms requires a
compatible synthesizer such as FluidSynth and a user-provided soundfont; the
MIDI generation itself does not depend on a soundfont.

To render existing MIDI files separately:

```sh
# macOS: requires Apple's Command Line Tools (swiftc) and system MIDI sounds.
python src/render_audio.py --midi-dir results/basic_a/t12 --output results/basic_a/t12

# Linux or another FluidSynth installation: provide a legally obtained bank.
python src/render_audio.py --midi-dir results/basic_a/t12 --output results/basic_a/t12 \
  --backend fluidsynth --soundfont /path/to/bank.sf2
```

Install [Apple's Command Line Tools](https://developer.apple.com/xcode/resources/)
if `swiftc` is unavailable:

```sh
xcode-select --install
```

On Debian/Ubuntu, install [FluidSynth](https://www.fluidsynth.org/) and a separate
General MIDI soundfont:

```sh
sudo apt-get update
sudo apt-get install fluidsynth fluid-soundfont-gm
```

The soundfont package has its own license. Set `HW2_SOUNDFONT` to use
an existing bank without repeating `--soundfont`. The native renderer generates
22,050 Hz PCM16 audio and a two-second release tail. This sample rate is an
implementation choice; the handout does not require a particular WAV rate.
The archive contains our renderer source, not Apple's sound bank or another
party's soundfont. All tracks use acoustic grand piano with the generated note
velocities. `render_manifest.json` records actual audio checks and file hashes.

Install `requirements-experiments.txt` only to retrain, calculate metrics, or
regenerate the PDF. The provided POP909 dataset is supplied by the course and is
not included in the submission archive.

## Inference

Run commands from this directory. `--output-dir` is always configurable. The
same basic checkpoint is used for unconditional generation and continuation.
The final submitted sampling settings are saved in `results/report_data.json`.

```sh
# Reproduce the submitted basic-a configuration (t12): twenty 32-bar pieces.
python src/infer.py --checkpoint artifacts/models/basic_final.pt \
  --task unconditional --output-dir /tmp/hw2-unconditional \
  --num-samples 20 --temperature 1.2 --top-p 0.95 --seed 20261022 \
  --device cpu --render

# Eight-bar prompts followed by twenty-four generated bars.
# data-root points to the course POP909 directory containing prompts/.
python src/infer.py --checkpoint artifacts/models/basic_final.pt \
  --task continuation --data-root /path/to/POP909 \
  --output-dir /tmp/hw2-continuation --temperature 1.0 --top-p 0.95 \
  --seed 20261022 --device cpu --render

# Ten sixteen-bar melody prompts with generated piano accompaniment.
python src/infer.py --checkpoint artifacts/models/roles_final.pt \
  --task accompaniment --data-root /path/to/POP909 \
  --output-dir /tmp/hw2-accompaniment --temperature 1.0 --top-p 0.95 \
  --seed 20261022 --device cpu --render
```

The continuation example uses T=1.0. To reproduce all nine submitted variants,
run it at T=0.8, 1.0 and 1.2 in three different output directories. All use
`--top-p 0.95 --seed 20261022 --device cpu` and the **same basic checkpoint**.
The marked choices are prompt **001:t12**, **002:t12**, **003:t10**. “Best” is the
fixed two-bar boundary-continuity rule in the report, without human ratings.

The 20 unconditional pieces use seeds 20261022 through 20261041. Prompt tasks
use the same base seed and stable prompt order; each manifest records the exact
per-piece seed. All formal generation used CPU float32, one Torch thread and
the pinned package versions. A different device/version can change sampled
music despite numerically close logits. The commands use the checkpoint names
shipped with the submission. Exact settings and hashes are in
`results/report_data.json` and the generation manifests.
The decoder preserves supplied prompt notes instead of replacing them with
quantized reconstructions. Bonus prompt `PIANO` ground truth is evaluation-only
and must never be passed to generation.

## Results and evaluation

The report includes five basic-a settings (100 complete 32-bar pieces), all
nine continuation variants and ten bonus pieces. The submitted basic-a set is
**t12**: H4=2.784589, GS=0.970166, 21.045 notes/bar, compared with the primary
real reference H4=2.639774, GS=0.970831, 22.953 notes/bar. This is metric
proximity, not a listening score.

The concise report prioritizes two additional studies: equal-token context
length and paired melody-conditioning removal. Each includes the method,
measured results, interpretation and limitations. Register drift, late
continuation changes and bar-level chord mismatches show what the required
metrics miss. Full token-family, quantization, early-checkpoint, temperature,
reference-sensitivity and local-match results remain in `results/report_data.json`
and its named evidence files.

Bonus T=1.0 is retained by the predeclared average of CA and the four OA metrics,
narrowly above T=1.2. The paired composite-difference interval includes zero;
this is in-sample tuning, not evidence of a perceptual advantage. Its CA is only
0.1125 despite OA_P=0.8552; the report explains this harmonic weakness.

Open **`listening.html`** locally for all 39 formal listening samples and the
marked continuation choices. All nine continuations have jump buttons at bars
1, 9, 17 and 25, with segment timing, density and mean-pitch tables for direct comparison.
**`HW2_說明.html`** explains the implementation,
principles and results in Chinese; the formal submission report is the English
16:9 **`r14942154_report.pdf`**.

H4 is pitch-class diversity, and GS is onset-pattern consistency. Neither is
a listening-quality score. OA histograms ignore octave and much temporal order;
empty bars can produce deceptively high overlap. The report cross-checks metrics
against density, repetition and completeness; perceived musical quality remains
unverified by a human listening study.

The model uses eight positions per annotated beat. The metric adapter maps
`Pos_q` to `Pos_(2*q)` on the official 16-subdivision grid, with 128 consecutive
position IDs, identically for real and generated pieces. `GS_active` is an
additional diagnostic on the native eight-subdivision grid: pairwise onset XOR
is divided by `max(beats_in_bar_a, beats_in_bar_b) * 8`. It removes both empty
odd-position slots introduced by the metric adapter and the unused tail of the
128-position vector; it does not replace the required official GS. Real-reference
crops must be compatible with the official 128-position grid (at most eight
annotated beats per bar); when necessary, a nearby eligible 32-bar crop is used
and recorded. This metric-specific crop selection does not exclude any song
from model training. Uncertainty is bootstrapped over whole pieces, not
over overlapping H4 windows. The provided official evaluation scripts remain
unchanged. The pinned MusDr source and MIT license are included under `vendor/`.

The MIDI exporter uses canonical annotation-beat notation: one supplied beat is
one quarter note, and a bar containing `n` annotated beats is written as `n/4`.
This preserves bar counts and elapsed time, including variable meters. The beat
annotations do not provide the score's original denominator; six pulses are
exported as 6/4, without claiming to recover an original 6/8 spelling. Original
prompt note timing is checked independently of this notation choice.

Automated note, rhythm and waveform inspection is not a human listening test.
Any report statement about a human listening comparison must identify an actual
completed listening procedure; no listener ratings are inferred from metrics.

## Optional audio-only evaluation

This additional review uses the pretrained
[Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
audio-language model through its
[MLX 4-bit conversion](https://huggingface.co/mlx-community/Qwen2-Audio-7B-Instruct-4bit)
and the [MLX Audio runtime](https://github.com/Blaizzy/mlx-audio).
It is an **evaluation-only tool**: it does not train or change the composition
models, generated MIDI/WAV, or previously selected versions. All submitted
composition weights still originate from random initialization on the official
training set. The pretrained audio-review weights are not submitted.

On Apple Silicon/macOS, create a separate Python 3.10 environment. These optional
dependencies are not part of `requirements.txt` and are unnecessary for music
generation or the required official metrics:

```sh
python3.10 -m venv .venv_audio_review
.venv_audio_review/bin/python -m pip install -r requirements-audio-review.txt

# Download approximately 6.56 GB into the excluded local cache.
.venv_audio_review/bin/python tools/prepare_audio_review_model.py

# Check the input frontend, then a short real clip and an all-zero control.
.venv_audio_review/bin/python tools/check_audio_review_frontend.py
.venv_audio_review/bin/python tools/audio_review_smoke.py

# Prepare every WAV frame, then submit all clips of each piece together.
.venv_audio_review/bin/python analysis/audio_review/prepare_inputs.py
.venv_audio_review/bin/python analysis/audio_review/review_all.py
```

`prepare_inputs.py` covers all nine full continuation WAVs, including release
tails, using bar-aligned clips of at most 30 seconds. Each piece's chronological
clips enter one request. Blind prompts include the waveform and segment
coordinates; they omit sampling settings, chosen-version labels, MIDI statistics,
chord labels, ground-truth piano and expected failures. The frontend uses the
official Hugging Face Whisper feature extractor after a numerical comparison
with the native MLX frontend. Model revision, file hashes, dependencies, input
frame coverage and frontend waveform hashes are recorded in the result evidence.

**The silence control failed:** the model invented musical sounds on all-zero
audio. Its textual descriptions are therefore exploratory and unreliable, not
proof of sound quality, correct harmony or a preferred version. Full input
coverage establishes which audio reached the model; it does not establish that
the model understood it correctly. This is not human listening or a listening
study. Specific bar-level conclusions in the report use independently checkable
MIDI and waveform measurements.

Raw responses are saved in `results/audio_review/raw_audio_observations.json`;
`completed=true` is set only after all nine pieces have been processed. The
report separately records the completed review's coverage and limitations.
The package contains the analysis code, optional dependency list and measured
JSON evidence. Downloaded weights, the isolated environment, caches and clipped
WAV copies are excluded. Exact model/runtime revisions matter when rerunning
this optional analysis.

## Regenerate the report and assemble the submission

```sh
python src/gather_report_data.py

python src/build_report.py --input results/report_data.json \
  --output r14942154_report.pdf

python src/build_explainer.py

# Uses the explicit GitHub choice saved in report_inputs/report_data.
python src/package_submission.py
```

`gather_report_data.py` reads actual generation manifests, complete official
metric summaries, per-piece CSV files, training histories and the two submitted
checkpoints. Missing work stays pending. Optional `results/report_inputs.json`
can name reviewed interpretations or override paths; it cannot supply scores.
All 896 training-song IDs must appear in each submitted checkpoint's observed
song set. The all-song checkpoint has no held-out validation score. Earlier
diagnostic validation and the descriptive real-data reference are labeled
separately.

The package tool checks all selected MIDI/WAV pairs, verifies checkpoint hashes,
regenerates the English 16:9 PDF, and creates `r14942154_HW2.zip`. It requires an
explicit list of our checkpoint and tokenizer files and includes only selected
listening samples: 20 unconditional pairs, all 9 continuation pairs, and 10 bonus
pairs if the bonus is included. It excludes training data, preprocessed training
sequences, downloaded model weights and caches. `FILE_SHA256.json` records the
contents for integrity checking.

The completed package is `r14942154_HW2.zip`; the upload tree is
`submission/r14942154_HW2/`. The archive is for local download or a separate
file host; individual source/model/audio files are what GitHub stores.
`SUBMISSION_CHECKS.json` keeps the explicit repository choice and the hosting
warning. Choosing GitHub does not mark TA acceptance as verified and does not
waive any experiment or file-integrity check.

A package with substantive missing work is refused by default. `--allow-draft`
creates a clearly named review archive for incomplete work; it is not a way to
claim missing results are complete. The package reads back every ZIP member,
checks all SHA256 hashes and verifies exactly 39 matching MIDI/WAV pairs. Real
reference token sequences, training arrays and caches are excluded.

Upload `r14942154_report.pdf` to NTU COOL and place the public artifact URL in
the assignment comments as requested by the slides. The report cover includes
the URL and the hosting note. Confirm public files open while signed out. If
the TA requires a conventional cloud-drive folder, mirror this same complete
package there and update the cover/comment link; no retraining is needed.

## Attribution and AI assistance

The course's HW2 slides permit using the internet and AI (p.34). Codex assisted
with implementation, execution, result aggregation, analysis, and English
drafting. The optional Qwen audio critic is a separate pretrained evaluation
aid with a failed silence control; its descriptions do not constitute human
listening evidence. All reported scores must come from actual saved outputs.
All submitted composition weights are learned from scratch on the official training set; the source code
and report cite the public methods and evaluation code used.

Primary task references: POP909 (Wang et al., ISMIR 2020), Pop Music Transformer
(Huang and Yang, ACM MM 2020), the course HW2 slides and starter scripts, and
MusDr's metric implementations. Full references and implementation details are
included in the report and saved experiment results.
