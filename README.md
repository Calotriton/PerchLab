# PerchLab

**PerchLab** identifies species from audio recordings, generates embeddings, and
benchmarks Google's **Perch V2** model — built on top of
[Perch Hoplite](https://github.com/google-research/perch-hoplite). It is a clean,
modular, extensible command-line tool designed to be shared with other
researchers and to slot into existing bioacoustics workflows (its detection
tables are drop-in compatible with Raven/BirdNET selection tables).

- **Species Identification** — Perch V2's built-in classifier, top-k per window, confidence threshold, optional clip extraction.
- **Embedding Generation** — store embeddings in a Perch Hoplite SQLite database (optionally exported to Parquet/NPZ).
- **Benchmark** — evaluate Perch on a labelled dataset with accuracy, precision/recall/F1, confusion matrices, ROC/PR curves and AUCs, and a full scikit-learn classification report.
- **Optimal Confidence Threshold Detection** — from human-validated detections, estimate the species-specific confidence threshold giving a target (default 95%) probability of correct identification, with a precision-by-confidence table and a probability-of-correct plot per species.

---

## Table of contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
4. [Concepts](#4-concepts)
5. [Benchmark](#5-benchmark)
6. [Optimal confidence threshold](#6-optimal-confidence-threshold)
7. [Project structure](#7-project-structure)
8. [Scientific background](#8-scientific-background)
9. [References](#9-references)

---

## 1. Introduction

### Perch V2

**Perch** is a bioacoustics model from Google trained to classify ~15,000
species (of which ~10,000 are birds) and to produce general-purpose audio
**embeddings**. Perch 2.0 improves embedding and prediction quality over the
original and adds support for many non-avian taxa. It achieves state-of-the-art
results on bioacoustics benchmarks such as BirdSet and BEANS.

**Input contract.** Perch V2 consumes **5-second windows** of **mono, 32 kHz,
float32** waveform samples. It outputs, per window:

- a **1536-dimensional embedding**, and
- **logits** over its ~15k classes (iNaturalist taxonomy).

### Perch Hoplite

[Perch Hoplite](https://github.com/google-research/perch-hoplite) is Google's
system for storing large volumes of machine-perception embeddings and combining
vector search with active learning. PerchLab reuses Hoplite for model loading
(`zoo`), audio IO, the embedding database (`db`), and evaluation metrics rather
than reimplementing them.

### Model architecture, embeddings, and transfer learning

Perch is a deep audio classifier trained on a very large, taxonomically diverse
corpus. Its penultimate layer produces **embeddings** — dense vectors that
summarise the acoustic content of a window. Because these embeddings were trained
to be **linearly separable**, a simple linear classifier on top of them transfers
remarkably well to *new* tasks with little data. This is **transfer learning**:
reuse a model trained on a broad task (global birdsong) as a feature extractor for
a narrow one (your local species, individual ID, call types).

### Softmax vs. sigmoid, and the inference workflow

Perch V2's species-classifier head is trained with a **softmax** activation and
cross-entropy loss (per the Perch V2 paper, which found this trains faster than
the sigmoid binary cross-entropy usually used for multi-label problems). So
PerchLab converts logits to confidence with a **softmax over classes** by default
— this matches the model's training objective and recovers the probabilities it
was optimised to produce. Classes compete for a shared unit of probability mass,
so when several species co-occur in a window the mass splits between them (e.g.
three co-singing *Acrocephalus* each land near 0.3 rather than all reading ~1.0).

A per-class **sigmoid** (`1 / (1 + e^-logit)`, independent probability per class)
is still selectable via `model.activation: sigmoid`, but it is **not recommended
for Perch V2**: the model's raw logits are large (winning logits ≈ +10), so
sigmoid saturates almost every top prediction to ~1.0 and makes the confidence
threshold ineffective. PerchLab's inference workflow is:

```
audio file → preprocess (mono/32 kHz/float32) → frame into windows
           → Perch V2 forward pass (embeddings + logits)
           → softmax → top-k per window → confidence threshold → outputs
```

Inference is run **once** per file; the confidence threshold is a cheap
post-processing filter (so a whole multi-threshold sweep needs only one pass).
Because softmax scores are lower than saturated sigmoids, the default threshold
is **0.1** — high enough to drop noise, low enough to keep co-occurring species.

### One inference path for every workflow

All four workflows share the same core: **discover** audio files → load Perch V2
**once** → **preprocess** each file to the input contract → **frame** it into
windows → run the **forward pass** in batches → hand the per-window
embeddings/logits to the workflow. Only the final step differs — top-k detections
(Identification), an embedding database (Embedding), evaluation metrics
(Benchmark), or a per-species logistic fit (Optimal Threshold). This shared
engine is why interactive and command-line modes give identical results, and why
a whole threshold sweep costs a single inference pass.

---

## 2. Installation

PerchLab runs on **Linux** and on **Windows via WSL**. Python **3.11** is
required (TensorFlow/Perch Hoplite do not yet support 3.14+).

### 2.1 Install WSL (Windows users)

In an elevated PowerShell:

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, then launch **Ubuntu** from the Start menu and create your
Linux user. All remaining steps run *inside* the WSL Ubuntu shell.

### 2.2 System dependencies

Perch Hoplite needs `libsndfile` and `ffmpeg` for audio decoding:

```bash
sudo apt-get update
sudo apt-get install -y libsndfile1 ffmpeg
```

### 2.3 Configure VS Code (optional but recommended)

Install [VS Code](https://code.visualstudio.com/) and the **WSL** and **Python**
extensions. Open the project in WSL with **"WSL: Connect to WSL"**
(`Ctrl+Shift+P`), then **File → Open Folder → PerchLab**. A ready-made task
(**Terminal → Run Task → "PerchLab: Run"**) launches `uv run perchlab`.

### 2.4 Install uv

[`uv`](https://docs.astral.sh/uv/) manages the environment and dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.5 Clone, create the environment, install

```bash
git clone https://github.com/Calotriton/PerchLab.git
cd PerchLab
uv venv --python 3.11
source .venv/bin/activate
# Choose an inference backend extra:
uv pip install -e ".[onnx]"     # ONNX runtime — light, good default (CPU or modest GPU)
# uv pip install -e ".[tf]"     # TensorFlow (CPU)
# uv pip install -e ".[gpu]"    # TensorFlow with CUDA
# add ,dev for the test/lint toolchain, e.g. ".[onnx,dev]"
```

> **Running the guided notebooks?** Pick `tf` or `gpu`, not `onnx`. The notebooks in
> `notebooks/` call the Perch SavedModel through TensorFlow directly, and the `onnx`
> extra does not install TensorFlow — the notebook's import cell fails with
> `ModuleNotFoundError: No module named 'tensorflow'`. The CLI is unaffected and works
> on any backend.

> **If you use `uv sync` instead, pass your extras.** The commands above use
> `uv pip install`, which *adds* to the environment. `uv sync` makes the
> environment **exactly** match the lockfile, so a bare `uv sync` **uninstalls
> TensorFlow and the dev tools** — they live in optional groups (`gpu`/`tf`/
> `onnx`, `dev`), not the core runtime. Always name the extras you need:
>
> ```bash
> uv sync --extra gpu --extra dev    # GPU backend + test/lint tools
> ```

### 2.6 GPU TensorFlow (optional)

For CUDA acceleration install the `gpu` extra (`uv pip install -e ".[gpu]"`),
which pulls `tensorflow[and-cuda] ~=2.20` and the bundled CUDA/cuDNN wheels. Then
**select the GPU backend explicitly** — the default `auto` backend resolves to
the CPU build:

```bash
export PERCHLAB_MODEL__BACKEND=gpu   # or set model.backend: gpu in a config file
```

PerchLab automatically puts the bundled CUDA libraries on `LD_LIBRARY_PATH`
(re-execing once at startup), so GPU works out of the box under WSL2 — no manual
`LD_LIBRARY_PATH` export is needed. On small GPUs (e.g. a 4 GB laptop card) keep
`model.batch_size` modest; if you hit out-of-memory errors, fall back to
`PERCHLAB_MODEL__BACKEND=cpu` (still TensorFlow) or the ONNX backend. See the
official [TensorFlow GPU guide](https://www.tensorflow.org/install/pip) for
driver/CUDA background.

**Log noise on GPU runs.** TensorFlow/XLA/CUDA print a fixed set of harmless
lines every run (oneDNN and CPU-instruction notices, GPU/cuDNN/XLA init, ptxas
register-spill notes, repeated `Delay kernel timed out` timing warnings), and
perch-hoplite adds a numpy `np.divide` warning plus a duplicate-eBird-class-list
warning. **None affect results.** PerchLab filters exactly these known-benign
lines out of `stderr` by content, so any unrecognised line — including a genuine
error — still shows. To see the raw, unfiltered output (e.g. when debugging),
set `PERCHLAB_LOG_FILTER=0`.

### 2.7 Kaggle credentials (first model download)

Perch V2 is downloaded from Kaggle on first use via `kagglehub`. Kaggle offers
two credential styles:

- **New access token (recommended)** — Kaggle → *Settings* → *API* → *Create New
  Token* gives a `KGAT_…` token. Export it:
  ```bash
  export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```
  These tokens require `kagglehub >= 1.0`; PerchLab enforces this with a uv
  dependency override (perch-hoplite's own pin is older), so a normal install
  already has a compatible version.
- **Classic token** — a downloaded `kaggle.json` placed at `~/.kaggle/kaggle.json`
  (or `KAGGLE_USERNAME` + `KAGGLE_KEY`) also works.

Accept the model's terms once on the
[model page](https://www.kaggle.com/models/google/bird-vocalization-classifier).
The model is cached under `~/.cache/kagglehub` afterwards.

> **Tip.** To persist settings across shells, add the exports to `~/.bashrc`:
> ```bash
> export KAGGLE_API_TOKEN=KGAT_...        # your token
> export PERCHLAB_MODEL__BACKEND=gpu      # if using the GPU build
> ```
> Treat the token as a secret; rotate it in Kaggle settings if it is ever exposed.

---

## 3. Usage

### 3.1 Quick start

After installing (§2) and setting a Kaggle token (§2.7), the simplest run is the
interactive menu. The **first run downloads Perch V2** (cached under
`~/.cache/kagglehub` afterwards, so later runs start immediately):

```bash
uv run perchlab
```

Or identify species in a folder of recordings in a single line:

```bash
uv run perchlab id --input recordings --output results
```

That writes, under `results/threshold_0.10/`, one CSV + Raven table per
recording, an aggregated `all_detections.csv`, a per-species `summary.txt`, and a
`manifest.json` capturing the exact run.

### 3.2 Interactive mode

```bash
uv run perchlab
```

You are shown a menu:

```
Select a workflow:
  1) Species Identification
  2) Embedding Generation
  3) Benchmark
  4) Optimal Confidence Threshold Detection
```

Pick one and PerchLab prompts for each parameter, showing the configured default
in brackets — press Enter to accept it. Folder prompts offer **path
autocompletion**, so it works over SSH/WSL. For example, Species Identification
asks:

```
? Input folder (recordings): ./recordings
? Output folder: data/outputs/perchlab_ID_20260714_101500
? Window size (s): 5.0
? Hop size (s): 5.0
? Top-k (predictions per window): 3
? Run multiple confidence thresholds? No
? Confidence threshold: 0.1
? Extract audio segments? Yes
?   Confidence-bin width: 0.1
?   Max samples per bin: 20
?   Clip duration (s): 5.0
?   Add context seconds around each clip? Yes
?     Context seconds on each side: 1.0
?   Random seed (blank = none):
```

Interactive and command-line modes build the same configuration and run the same
code, so results never diverge between them.

### 3.3 Command-line mode (scripting)

The same workflows run non-interactively:

```bash
# Species identification
uv run perchlab id \
    --input recordings \
    --output perchID \
    --window 5 --hop 5 \
    --top-k 3 \
    --threshold 0.1 \
    --format csv,raven

# Multi-threshold sweep + clip extraction
uv run perchlab id --input recordings --output perchID \
    --threshold-start 0.1 --threshold-end 1.0 --threshold-step 0.1 \
    --extract --max-per-bin 20 --clip-duration 5 --context 1 --extract-seed 0
# --context 1 pads each 5 s clip with 1 s on both sides -> 7 s clips for context

# Embedding generation (labelled: one folder per species)
uv run perchlab embed --input clips --output embeddings --labeled --export parquet

# Benchmark on a labelled dataset
uv run perchlab benchmark --input labelled_dataset --output report \
    --threshold-start 0.0 --threshold-end 1.0 --threshold-step 0.1

# Optimal confidence threshold from human-validated detections
uv run perchlab threshold --input validated --output thresholds \
    --species "Periparus ater" --target-probability 0.95
# (omit --species to estimate one threshold per species subfolder)
```

Run `uv run perchlab <command> --help` for the complete, up-to-date flag list of
any subcommand. Every command also accepts `--config run.yaml`; see
[Configuration](#34-configuration) for how flags, environment variables, and YAML
combine.

### 3.4 Configuration

Every parameter has a default in `perchlab.config`, so most runs need no
configuration at all. When you do override something, values resolve with the
precedence **CLI flags > environment variables > `--config` YAML > built-in
defaults**.

**Config file.** Copy `configs/default.yaml` (a fully documented mirror of the
defaults), edit what you need, and pass it — you only have to list the keys you
change:

```bash
uv run perchlab id --input recordings --config run.yaml
```

**Environment variables.** Prefix with `PERCHLAB_` and join nested keys with a
double underscore — handy for CI or one-off tweaks without editing a file:

```bash
export PERCHLAB_MODEL__BACKEND=onnx     # -> model.backend = onnx
export PERCHLAB_SEED=0                    # -> global RNG seed
export PERCHLAB_IDENTIFY__TOP_K=5         # -> identify.top_k = 5
```

**Audio preprocessing (normalization + resampler).** Two knobs let you match a
different pipeline or study the effect of preprocessing on scores — available on
every workflow subcommand, interactively (answer *yes* to "Customize audio
preprocessing?"), or in a config file:

```bash
uv run perchlab id --input recordings --normalize none --resampler soxr_hq
```

| Option | Values | Meaning |
| --- | --- | --- |
| `preprocess.normalize` | `hoplite` (default) / `none` | `hoplite` peak-normalizes each 5 s window to `target_peak = 0.25` (Perch's canonical preprocessing); `none` feeds **raw** audio to the model. |
| `preprocess.resampler` | `polyphase` (default) / `soxr_hq` | Resampling filter used to reach 32 kHz. `polyphase` is Hoplite's native choice; `soxr_hq` is librosa's default. |

The defaults (`hoplite` + `polyphase`) are the recommended, canonical Perch path
and should be left alone for normal use. Change them only to **reproduce another
tool's numbers** (e.g. a hand-rolled `serving_default` script that skips
normalization) — see [Reproducibility across tools](#reproducibility-across-tools).
`normalize` is the dominant factor; `resampler` is a minor secondary effect.

**Recorder filename parsing.** PerchLab reads each recording's start date/time and
device ID from its filename using a configurable regex with named groups
`device`/`date`/`time`. The default matches PIC recorders
(`PIC02_20250530_040000.wav`); override it for AudioMoth, Song Meter, and others:

```yaml
filename:
  pattern: '(?P<device>[A-Za-z0-9]+)_(?P<date>\d{8})_(?P<time>\d{6})'
  date_format: '%Y%m%d'
  time_format: '%H%M%S'
```

Unrecognised filenames still process — the `date`, `hour`, and `time` columns are
just left blank.

### 3.5 Interpreting outputs

- **Identification** writes, under `threshold_<value>/`: one CSV per recording, a
  matching `.selection.table.txt` (Raven), an `all_detections.csv` aggregate, a
  `summary.txt` of per-species counts, and (if enabled) `segments/` with clips in
  per-species subfolders. A `manifest.json` records the exact run.
- **Embedding** writes a Hoplite database under `hoplite_db/` (plus an optional
  `embeddings.parquet`/`.npz`).
- **Benchmark** writes `metrics.json`, `metrics.csv`, `classification_report.txt`,
  `confusion_matrix.csv`, `sweep.csv`, four PNG figures, and a human-readable
  `report.md`.
- **Optimal Confidence Threshold** writes `thresholds.csv`/`thresholds.json` (the
  estimated threshold + regression coefficients per species), `precision_table.csv`
  (detections/verified/precision by confidence category), a
  `probability_curves.png` plot, a human-readable `report.md`, and a `manifest.json`.

---

## 4. Concepts

| Concept | Meaning |
| --- | --- |
| **Window size** | Length of audio analysed at once. Perch V2 is trained on **5 s**; other values are padded/truncated to the model window and deviate from training. |
| **Hop size** | Stride between consecutive windows. `hop < window` produces overlapping windows; `hop = window` (default 5 s) is non-overlapping. |
| **Top-k** | How many highest-ranked species are kept per window (default 3). Selection happens **before** thresholding. |
| **Confidence threshold** | Minimum confidence (`[0,1]`) to keep a prediction. A pure **post-processing** filter — it never changes inference or predictions, only which are reported. |
| **Embedding** | The 1536-d vector Perch produces per window; the reusable, task-agnostic representation. |

### CSV output fields (identification)

Each row is one predicted species in one window (empty windows produce no rows;
multiple species in a window produce multiple rows).

| Column | Description |
| --- | --- |
| `date` | Recording start date parsed from the filename (`YYYY-MM-DD`). |
| `hour` | Recording start clock time parsed from the filename (`HH:MM:SS`). |
| `timestamp` | Offset of the window **within** the recording (`H:MM:SS`). |
| `window_size` | Analysis window length used, in seconds. |
| `hop_size` | Hop length used, in seconds. |
| `start` | Window start time within the recording, in seconds. |
| `end` | Window end time within the recording, in seconds. |
| `time` | Wall-clock time of the event = recording start + `timestamp`. |
| `top_k` | Rank of this prediction within its window (1 = most confident). |
| `expected_label` | Parent-folder label, when recordings are organised by species. |
| `label` | Predicted Perch label (scientific name). |
| `threshold` | Confidence threshold this file was produced under. |
| `confidence` | Prediction confidence in `[0, 1]` (softmax over class logits by default; see §1). |
| `file` | Path to the source audio file. |

### Benchmark metrics

Accuracy, precision, recall, F1 (per class + macro/micro), confusion matrix,
ROC curves + ROC-AUC, precision–recall curves + PR-AUC, and scikit-learn's
`classification_report`. See below.

---

## 5. Benchmark

**Methodology.** The benchmark evaluates Perch's **top-1** prediction against a
ground-truth label. It expects a **labelled dataset**: a folder named after a
species, or a folder containing one subfolder per species.

**Assumptions.** Every audio file inherits its **parent folder name** as its
ground-truth label. Evaluation is per **window** by default (each window is a
sample); use `--aggregate file` to mean-pool a file's windows into one sample —
appropriate for long recordings, whereas per-window suits short one-event clips
(such as those extracted by the identification workflow).

**Evaluation metrics.**
- **Top-1 accuracy** — fraction of windows whose top prediction (above threshold) matches the folder label.
- **Precision / recall / F1** — per class and macro/micro averaged.
- **Confusion matrix** — over the dataset's species.
- **ROC + ROC-AUC** and **PR + PR-AUC** — per class, one-vs-rest, using the continuous confidence of each class (threshold-independent).
- **`classification_report`** — produced on every run.
- A **threshold sweep** yields the accuracy/precision/recall/F1-vs-threshold curves.

**Interpretation.** High macro F1 with high per-class ROC-AUC indicates Perch
separates your species well. The metric-vs-threshold plot helps choose an
operating threshold. PR curves are more informative than ROC when classes are
imbalanced.

**Limitations.** Perch confidences are **uncalibrated** softmax scores and are
*not* directly comparable across species, so a single global threshold means
different things per class — treat thresholds as relative. Folder labels must match Perch's
class names (scientific names) to score as positives; non-matching labels still
appear in the confusion matrix but yield degenerate ROC/PR. Windows of silence in
a species-labelled file are still counted as that species, which can depress
apparent accuracy for long recordings (prefer `--aggregate file` or short clips).

---

## 6. Optimal confidence threshold

The confidence score is Perch's estimated probability that a detection is correct
(1 = perfect match). Used as a **threshold**, it filters the output: a higher
threshold raises the proportion of correctly classified detections but keeps
fewer detections overall. This workflow answers *"what threshold should I use for
this species?"* — and, crucially, it estimates a **separate threshold per
species**, because the relationship between confidence and correctness differs
between species (a single universal threshold is not appropriate).

### What you provide

A set of **human-validated detections** for the species. In practice you run
Species Identification (ideally with `--extract` to save the detection clips),
then a person listens to each clip and inspects its spectrogram to decide whether
the species is truly **present** (correct) or **absent** (a false positive). Sort
the clips into two folders:

```
validated/
  Periparus ater/
    correct/     ← species present (accepted synonyms: present, true, tp, 1, yes)
    incorrect/   ← species absent  (accepted synonyms: absent, false, fp, 0, no)
  Certhia brachydactyla/
    correct/ ...
    incorrect/ ...
```

For a single species you can point `--input` straight at a folder that contains
`correct/` and `incorrect/` and pass `--species "Periparus ater"`. PerchLab
re-runs Perch over each clip to recover the confidence it assigns that species
(the strongest window), pairing each confidence with its correct/incorrect verdict.

### Precision and the 95%-correct threshold

**Precision** here is the proportion of a species' detections that are correct:
`verified correct / total verified`. The workflow tabulates precision by
confidence category (mirroring the reference table below), so you can see
precision climb as confidence rises. This example is PerchLab's output on the
Coal Tit validation set of Bota et al. (2023), and reproduces their Table 1:

| Confidence category | Detections | Verified | Precision |
| --- | --- | --- | --- |
| [0.10, 0.30) | 157 | 134 | 85.4% |
| [0.30, 0.50) | 77 | 77 | 100.0% |
| [0.50, 1.00] | 75 | 75 | 100.0% |
| **TOTAL** | 309 | 286 | 92.6% |

The **threshold** is the confidence that corresponds to a chosen probability of
correct identification (default **95%**). Applied as a post-processing filter, it
keeps only detections with confidence ≥ the threshold — these are the detections
*predicted to be correct with ≥95% probability*. Note this does **not** retain
95% of detections; it retains those estimated to be 95% reliable. Raising the
threshold raises precision but lowers the number of detections (and thus the
proportion of calls/presences recovered).

### How it is computed

1. **Back-transform** each confidence score to the logit scale (the standard
   logit, i.e. the inverse sigmoid, following Wood et al. 2023):

   ```
   logit_score = ln( confidence / (1 - confidence) )
   ```

2. **Fit a logistic regression** per species, where the response is whether the
   detection was correct (1) or incorrect (0) and the predictor is the
   logit-transformed confidence. This calibrates raw confidence into an actual
   probability of correct identification:

   ```
   P(correct) = sigmoid( b0 + b1 · logit_score )
   ```

   (fitted by maximum likelihood, i.e. unregularised logistic regression).

3. **Solve for the target probability.** Setting `P(correct) = 0.95` and
   inverting gives the logit score, then the confidence, at which detections are
   95% likely to be correct:

   ```
   logit_score* = ( ln(0.95 / 0.05) − b0 ) / b1
   threshold    = sigmoid( logit_score* ) = 1 / (1 + exp( −logit_score* ))
   ```

   The resulting confidence is adopted as the species-specific threshold and can
   then be applied to the model's full set of predictions, yielding a filtered,
   high-confidence dataset — a statistically justified balance between reliability
   and the number of detections retained.

> **On the transform and inversion.** The method comes from Wood et al. (2023);
> Bota et al. (2023) applied it to two bird species. PerchLab uses the **standard
> logit** `ln(c/(1−c))` (Wood's transform and the inverse of the sigmoid that
> produces the score) and inverts the fitted model for the confidence at the
> target probability — the textbook way to invert a logistic calibration.
> Validated against Bota's raw dataset, PerchLab reproduces their Table 1 exactly
> and returns a threshold whose fitted P(correct) is precisely 0.95 (Coal Tit
> ≈0.25). Note Bota's *published* threshold formula, taken literally, is a garbled
> transcription that under-shoots the 95% target for one of their species; the
> standard inversion used here is the correct one. A separate threshold is
> estimated per species, since the confidence→accuracy relationship varies.

### Outputs

- **`probability_curves.png`** — one panel per species: the validated detections
  as a 0/1 scatter over confidence, the fitted logistic probability-of-correct
  curve with its 95% confidence band, the target-probability line, and the
  estimated threshold marked (like the figure in the reference study).
- **`precision_table.csv`** — the detections/verified/precision table above, per
  species, with a TOTAL row.
- **`thresholds.csv`** / **`thresholds.json`** — the estimated threshold, sample
  counts, overall precision, and the fitted `intercept`/`slope` per species.
- **`report.md`** — a human-readable summary embedding the table and plot.
- **`manifest.json`** — the exact run configuration.

When a species has too few validated clips, only correct **or** only incorrect
examples, or a confidence that is not positively associated with correctness
(slope ≤ 0), no threshold can be estimated: that species is reported with an
empty threshold and an explanatory `note`, and the rest still proceed.

---

## 7. Project structure

```
src/perchlab/
  config.py         Central, typed configuration (pydantic) + YAML/env loading.
  logging.py        Structured logging (rich); no bare print().
  errors.py         Exception hierarchy (recoverable AudioError, etc.).
  models.py         PerchModel: thin wrapper over the Hoplite zoo.
  preprocess.py     Perch input contract: mono/32 kHz/float32/validate + windowing.
  audio.py          File discovery + configurable filename->datetime parsing.
  inference.py      InferenceEngine: batch windows -> per-window embeddings/logits.
  classify.py       Top-k over logits (cached) + threshold filtering -> detections.
  detections.py     The Detection record and its CSV columns.
  taxonomy.py       Class-name <-> display/code mapping (extension seam).
  segments.py       Confidence-binned clip extraction into per-species folders.
  embedding.py      EmbeddingRunner: write embeddings/labels to a Hoplite DB.
  prompts.py        Interactive prompt helpers (path autocompletion, etc.).
  util.py           Determinism (seeding) and run manifests.
  cli.py            Typer CLI: interactive menu + id/embed/benchmark/threshold subcommands.
  workflows/        Workflow registry + the four workflows.
  io/               Raven selection tables + CSV/Parquet writers.
  benchmark/        Labelled-dataset eval: dataset/evaluate/metrics/sweep/plots/report.
  threshold/        Optimal-threshold estimation: dataset/collect/stats/plots/report.
configs/default.yaml  Documented default configuration.
notebooks/            Example notebooks mirroring the workflows (need the tf/gpu extra).
tests/                pytest suite (offline PlaceholderModel + real-model marker).
```

### Why Perch requires mono / 32 kHz / float32 / normalization

Perch V2 was trained on a fixed input representation, and inference must match it:

- **Mono** — the model has a single input channel; stereo is reduced to one channel.
- **32 kHz** — the sample rate the spectrogram front-end expects; resampling keeps the frequency content aligned with training.
- **float32** — the numeric type the network operates on; integer PCM is converted to floating point.
- **Normalization** — each 5-second window is peak-normalized (its mean removed, then scaled so the loudest sample sits at `target_peak = 0.25`, the value chosen by Perch's authors) so loud and quiet recordings are treated consistently. PerchLab **reuses Perch Hoplite's own per-window normalization** (`EmbeddingModel.normalize_audio`) rather than reimplementing it; PerchLab's preprocessing focuses on mono/32 kHz/float32 conversion and validation. See [`perch_hoplite/zoo/taxonomy_model_tf.py`](https://github.com/google-research/perch-hoplite/blob/main/perch_hoplite/zoo/taxonomy_model_tf.py).

### Reproducibility across tools

Perch V2 is **not amplitude-invariant**, so the confidence a window receives
depends on this preprocessing, not only on the audio content. PerchLab always
follows Hoplite's canonical path — resample to 32 kHz, then peak-normalize each
window to `0.25` — before every forward pass. A separate script that feeds
**raw, un-normalized** audio to the model, or that **resamples with a different
library**, will produce slightly different confidence scores (typically within
~0.05). The predicted species and window timings still agree; but because a fixed
global threshold turns a continuous score into a keep/drop decision, those small
differences can retain or discard a few **borderline** detections (those sitting
right around the threshold) and reshuffle the low-confidence 2nd/3rd-ranked
predictions. To reproduce another pipeline exactly, match **both** its resampler
and its peak-normalization; the normalization is the larger factor and is the one
that follows Perch's official inference path. Both are exposed as the
`preprocess.normalize` and `preprocess.resampler` options (see
[Configuration](#34-configuration)) — for example, `--normalize none --resampler
soxr_hq` reproduces a raw `librosa.load` + `serving_default` script that skips
Hoplite's normalization.

---

## 8. Scientific background

Perch matters for bioacoustics because a single, broadly-trained model provides a
strong foundation for many downstream tasks:

- **Transfer learning** — its linearly-separable embeddings let researchers build accurate classifiers for new signals from very little labelled data.
- **Species recognition** — state-of-the-art avian soundscape classification across thousands of species.
- **Call-type recognition** — embeddings capture within-species vocalisation structure, enabling call-type classifiers.
- **Dialect detection** — regional vocal variation is separable in embedding space.
- **General bioacoustic representation learning** — the same embeddings support individual identification (e.g. dogs, bats), event detection in coral reefs, and marine mammal tasks, making Perch a general feature extractor for ecology and conservation monitoring.

---

## 9. References

- **Perch Hoplite** — https://github.com/google-research/perch-hoplite
- **Perch V2 (Kaggle)** — https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2
- **Perch 2.0 paper** — *Perch 2.0: The Bittern Lesson for Bioacoustics* — https://research.google/pubs/perch-20-the-bittern-lesson-for-bioacoustics/
- **TensorFlow GPU install** — https://www.tensorflow.org/install/pip

---

## Development

```bash
uv pip install -e ".[onnx,dev]"
uv run pytest          # fast suite uses an offline placeholder model
uv run pytest -m slow  # runs the real Perch V2 model (needs Kaggle credentials)
ruff check src tests
mypy src/perchlab
```

## License

MIT — see [`LICENSE`](LICENSE).
