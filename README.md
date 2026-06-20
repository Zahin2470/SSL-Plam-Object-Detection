<div align="center">

# 🌴 PalmBYOL-YOLO
### Self-Supervised Object Detection for Palm Health Monitoring

*Bootstrapping a YOLOv12s detector with BYOL self-supervised pretraining — squeezing more signal out of labeled data by learning from the unlabeled training images first.*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.72-00FFFF?logo=yolo&logoColor=black)
![SSL](https://img.shields.io/badge/Self--Supervised-BYOL-8A2BE2)
![Domain](https://img.shields.io/badge/Domain-Precision%20Agriculture-2E8B57)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-76B900?logo=nvidia&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20%2F%20In%20Progress-yellow)

</div>

---

## 📚 Table of Contents

1. [✨ At a Glance](#-at-a-glance)
2. [💡 The Idea — Why BYOL?](#-the-idea--why-byol)
3. [🔁 Pipeline Overview](#-pipeline-overview)
4. [🗂️ Dataset Deep-Dive](#%EF%B8%8F-dataset-deep-dive)
5. [📑 Stage-by-Stage Notebook Map](#-stage-by-stage-notebook-map)
6. [🧬 BYOL Pretraining — Under the Hood](#-byol-pretraining--under-the-hood)
7. [🎯 Detection Fine-Tuning — Under the Hood](#-detection-fine-tuning--under-the-hood)
8. [🔬 Evaluation Suite](#-evaluation-suite)
9. [🛡️ Engineering Safeguards & Design Decisions](#%EF%B8%8F-engineering-safeguards--design-decisions)
10. [📦 Output Artifacts](#-output-artifacts)
11. [🧰 Tech Stack](#-tech-stack)
13. [▶️ Reproducing This Notebook](#%EF%B8%8F-reproducing-this-notebook)
14. [🙏 Acknowledgments](#-acknowledgments)

---

## ✨ At a Glance

| | |
|---|---|
| 🎯 **Task** | 3-class object detection — `abnormal_palm`, `dead_palm`, `healthy_palm` |
| 🧠 **Backbone** | YOLOv12s, pretrained via **BYOL** (Bootstrap Your Own Latent) |
| 🗂️ **Dataset** | Roboflow `palm-oil-vj5ii / dat-palm-fx` — 3,855 train · 200 val · 747 test images |
| 📦 **Annotations** | 9,819 train boxes · 608 val boxes · 2,366 test boxes |
| 🖼️ **SSL Image Size** | 160 × 160 (BYOL views) |
| 🖼️ **Detection Image Size** | 640 × 640 (fine-tuning & eval) |
| ⚙️ **Hardware** | Single Tesla T4 (14.9 GB), Kaggle environment |
| 🧪 **Notebook Structure** | 19 logical stages (39 raw cells incl. markdown headers) |
| 🔁 **Reproducibility** | Global seed 42, deterministic cuDNN, leakage-audited splits |

---

## 💡 The Idea — Why BYOL?

Labeled palm-health imagery is expensive to collect at scale — every box requires an expert to look at a tree and decide *healthy*, *abnormal*, or *dead*. The raw pixels, on the other hand, are cheap.

> **BYOL (Bootstrap Your Own Latent)** learns useful visual representations *without labels* by training a network to predict the representation of one augmented view of an image from a different augmented view of the *same* image — using two networks (an "online" network being trained, and a slowly-moving "target" network) instead of negative pairs or large batches like contrastive methods (SimCLR, MoCo) require.

The hypothesis this notebook tests: if the YOLOv12s backbone/neck is **pretrained with BYOL on the training images alone**, then **fine-tuned normally with labels**, the detector should learn more robust, less overfit features than training from a generic pretrained checkpoint or from scratch — without ever letting validation/test data leak into the self-supervised stage.

---

## 🔁 Pipeline Overview

```
 📥 DATA              🔍 SANITY            🧬 SELF-SUPERVISED        🎯 SUPERVISED          📊 EVALUATION
 ───────────          ───────────          ───────────────────       ──────────────         ─────────────
 Roboflow download      Visual checks        BYOL pretraining          Fine-tune detector      Official mAP
 COCO → YOLO        →   on raw + aug    →    160px · 70 epochs    →   640px · 70 epochs   →   Matched-event stats
 + offline augment      Class distros        EMA target, m=0.996      AdamW + cosine LR        Calibration & cost
                                              backbone/neck only       shape-safe weight        Threshold robustness
                                                                       transfer                 Paper-ready tables
```

Each phase is leakage-isolated: augmentation, BYOL pretraining, and fine-tuning all read exclusively from the **training** split; validation and test images are touched only at evaluation time, and only in their **original, unaugmented** form.

---

## 🗂️ Dataset Deep-Dive

**Source:** Roboflow project `palm-oil-vj5ii / dat-palm-fx-vfhm6`, version 1, COCO format.

### Conversion & Integrity Audit (Cell 3)

| Split | Images (JSON) | Annotations (JSON) | Copied | Valid Boxes | Invalid/Skipped Boxes | Images w/o Valid Labels |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3,855 | 9,819 | 3,855 | 9,819 | 0 | 0 |
| Valid | 200 | 608 | 200 | 608 | 0 | 0 |
| Test | 747 | 2,366 | 747 | 2,366 | 0 | 0 |

> **🛡️ Leakage audit:** MD5 image-hash comparison across all three splits found **zero duplicate images** between train/valid/test.

### Class Mapping

Category IDs are derived **from the training annotations only**. A 4th metadata category (`palms-obj`) appears in the raw COCO file but has **zero training instances**, so it is explicitly excluded from the class list rather than silently included as an empty class.

| Final Class ID | Class Name |
|---:|---|
| 0 | `abnormal_palm` |
| 1 | `dead_palm` |
| 2 | `healthy_palm` |

### Per-Class Instance Distribution

| Class | Train | Valid | Test |
|---|---:|---:|---:|
| `abnormal_palm` | 2,679 | 128 | 602 |
| `dead_palm` | 2,152 | 64 | 245 |
| `healthy_palm` | 4,988 | 416 | 1,519 |
| **Total** | **9,819** | **608** | **2,366** |

### Offline Training Augmentation (Cell 5)

Generates up to **1,000** extra training images (3 attempts per source image), using Albumentations with `BboxParams(format="yolo", min_visibility=0.25, min_area=4)`:

<details>
<summary><b>📋 Full augmentation policy (click to expand)</b></summary>

**Geometric**
- `HorizontalFlip(p=0.5)`
- `ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15, rotate_limit=12°, border_mode=REFLECT_101, p=0.60)`
- *(optional, disabled by default)* `VerticalFlip` / `RandomRotate90` — gated behind `USE_STRONG_ORIENTATION_AUG=False` since the imagery isn't confirmed nadir/top-down

**Photometric**
- `OneOf[RandomBrightnessContrast(±0.18), CLAHE(clip=2.0, tile=8×8)], p=0.55`
- `HueSaturationValue(hue=8, sat=25, val=18, p=0.35)`
- `OneOf[GaussianBlur(3–5), GaussNoise(var=5–20)], p=0.18`

Validation and test splits are copied through **unaugmented**.

</details>

---

## 📑 Stage-by-Stage Notebook Map

| # | Stage | What it does |
|---|---|---|
| 1 | 🛠️ Environment Setup | Installs deps, seeds Python/NumPy/Torch, deterministic cuDNN, verifies CUDA/GPU |
| 2 | 📥 Roboflow Download | Pulls the COCO-format dataset (skips re-download if already cached) |
| 3 | 🔄 COCO → YOLO Conversion | Train-derived category mapping, bbox clipping/validation, MD5 leakage audit |
| 4 | 👁️ Annotation Sanity Check | Draws boxes on sampled training images, flags malformed label lines |
| 5 | 🎨 Training-Only Augmentation | Geometric + photometric Albumentations pipeline, train split only |
| 6 | 📈 Distribution Audit | Before/after image & per-class instance counts, bar chart |
| 7 | 👁️ Augmented Sample Check | Visual QA pass on the augmented training set |
| 8 | ⚙️ BYOL + YOLOv12s Config | Canonical paths, reproducibility, required-file validation |
| 9 | 🧩 BYOL Dataset | `TwoView` two-augmentation dataset, train-only leakage guard, image-integrity check |
| 10 | 🧠 BYOL Loss & Hooks | Negative-cosine loss, Detect-head feature hooks, EMA + momentum schedule, MLP builders |
| 11 | 🚀 BYOL Pretraining | 70-epoch SSL run on backbone/neck only, early stopping, best-checkpoint saving |
| 12 | 🎯 Detection Fine-Tuning | Shape-safe BYOL weight transfer, full-detector training |
| 13 | ✅ Official Val/Test Eval | Native Ultralytics `.val()` on the *original* (non-augmented) split |
| 14 | 📉 Training Curves | Loss/metric/LR plots parsed from `results.csv` |
| 15 | 🔬 Matched-Event Evaluation | IoU-matched classification-style diagnostics: κ, ECE, Brier, ROC/PR |
| 16 | ⏱️ Cost Benchmarking | Params, GFLOPs, latency distribution, throughput, peak GPU memory |
| 17 | 🖼️ Qualitative Results | Sample detections rendered on test images |
| 18 | 🎚️ Threshold Robustness | F1 sweep across 9 confidence × 5 IoU thresholds (45 combinations) |
| 19 | 📄 Paper-Ready Tables | Aggregates everything into 6 publication tables (CSV + Markdown + LaTeX) |

---

## 🧬 BYOL Pretraining — Under the Hood

<details>
<summary><b>⚙️ Hyperparameters (click to expand)</b></summary>

| Parameter | Value |
|---|---|
| Epochs | 70 |
| Batch size | 8 |
| Image size | 160 × 160 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Base EMA momentum | 0.996 → 1.0 (cosine schedule) |
| Gradient clipping | max-norm 5.0 |
| Mixed precision | AMP (`GradScaler`) |
| Early stopping | patience 10, min Δ 1e-4 |
| Checkpoint scope | backbone + neck only (Detect head excluded) |

</details>

**View augmentation** — deliberately gentler than typical natural-image SSL:

```
RandomResizedCrop(160, scale=(0.45, 1.0))   # vs. the SSL-standard (0.20, 1.0)
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.08), p=0.75
RandomGrayscale(p=0.15)
GaussianBlur(kernel=3, sigma=0.1–1.5), p=0.20
Normalize(ImageNet mean/std)
```

> **💡 Design rationale:** the standard SSL crop range `(0.2, 1.0)` is tuned for natural-image classification, where the subject usually fills the frame. For small-object detection, an aggressive crop can remove the object entirely from both views — so the lower bound is raised to `0.45`.

**Architecture**

- **Feature extraction:** a forward pre-hook on the YOLO `Detect` module captures the P3/P4/P5 multi-scale feature maps *before* they reach the detect head, on both the online and target networks. The target's Detect head is forced into train-mode (while the rest of the target stays in eval-mode) specifically to bypass YOLO's inference-time box-decoding path and expose raw feature maps instead.
- **Projector / Predictor:** `Linear → BatchNorm → ReLU → Linear`, mapping pooled features → 512 hidden → 256 output. The target projector is a frozen copy of the online projector (no predictor on the target side — standard BYOL asymmetry).
- **Loss:**
  ```
  L = 2 − 2 · cos_sim(p, stopgrad(z))
  Total = L(p1, z2_target) + L(p2, z1_target)     # symmetrized over both views
  ```
- **EMA update order:** the target network is updated with EMA *after* the optimizer step on the online network — explicitly called out in-notebook since getting this ordering backwards is a common BYOL implementation bug.

---

## 🎯 Detection Fine-Tuning — Under the Hood

<details>
<summary><b>⚙️ Hyperparameters (click to expand)</b></summary>

| Parameter | Value |
|---|---|
| Epochs | 70 |
| Image size | 640 × 640 |
| Batch size | 8 |
| Optimizer | AdamW |
| `lr0` / `lrf` | 5e-4 / 0.01 (cosine schedule) |
| Weight decay | 5e-4 |
| Warmup epochs | 3 |
| Early stopping patience | 10 |
| Mosaic | 0.40 (closes at epoch 10) |
| Mixup / Copy-paste / Erasing | 0 (disabled) |
| HSV jitter (h/s/v) | 0.010 / 0.400 / 0.250 |
| Geometric jitter | degrees 5°, translate 0.05, scale 0.25, fliplr 0.50 |

</details>

> **💡 Design rationale — no double augmentation:** because offline augmentation already ran in Cell 5, YOLO's *online* augmentation is deliberately turned down here (mixup/copy-paste/erasing off, modest mosaic/HSV/geometric jitter) to avoid compounding two augmentation pipelines on top of each other.

**Weight transfer.** `load_byol_weights_safely` filters the BYOL checkpoint by **both** key existence *and* tensor-shape match before calling `load_state_dict` — stricter than plain `strict=False`, which silently skips key mismatches but does *not* catch shape mismatches on matching keys. Compatible / missing / unexpected / shape-mismatched keys are all reported, and the cell raises if zero tensors transferred successfully.

**Model resolution.** Both the BYOL cell and the fine-tuning cell try `yolo12s.pt → yolov12s.pt → yolo12s.yaml → yolov12s.yaml` in order, defensively, since Ultralytics' naming for YOLOv12 has shifted across package versions.

---

## 🔬 Evaluation Suite

Three complementary layers of evaluation, each answering a different question:

### 1️⃣ Official Detection Metrics (Cell 13)
Native Ultralytics `.val()` on the **original, non-augmented** val and test splits — mP, mR, mAP@0.50, mAP@0.50:0.95, F1, plus per-class breakdowns and Ultralytics' native PR/F1/P/R curves and confusion matrices.

### 2️⃣ Matched-Event Diagnostics (Cell 15)
Predictions are greedily IoU-matched (≥ 0.50) to ground truth, with an explicit **background** pseudo-class for unmatched boxes — turning detection into a classification-style problem on top of mAP:

| Metric | Detail |
|---|---|
| Accuracy / Macro & Weighted P-R-F1 | over object classes |
| Cohen's Kappa | agreement beyond chance |
| Bootstrap 95% CIs | 1,000 resamples, accuracy & macro-F1 |
| PPV / NPV | per class, from the full confusion matrix |
| Expected Calibration Error | 15-bin reliability diagram |
| Brier Score | confidence calibration |
| ROC-AUC / PR-AUC / EER | **binary** — "is this detection correct?", not multiclass OvR (explicitly noted: YOLO doesn't expose a full calibrated class-probability vector per event) |

### 3️⃣ Threshold Robustness Sweep (Cell 18)
Predicts **once** at `conf=0.001`, then post-hoc filters across **9 confidence thresholds × 5 IoU thresholds = 45 combinations** — efficient design that avoids 45 separate inference passes. Reports class-sensitive TP/FP/FN/precision/recall/F1 per combination (a correctly-located but wrong-class box counts as both an FP and an FN), an F1-vs-confidence curve per IoU level, a precision-recall tradeoff plot, and a 2D F1 heatmap across the full grid.

### 4️⃣ Computational Cost (Cell 16)
Model size (MB), parameter counts, GFLOPs/GMACs, CPU RAM delta, GPU warmup + 100-image timed inference loop (`torch.cuda.synchronize()`-bracketed), throughput (img/s), and peak GPU memory.

---

## 🛡️ Engineering Safeguards & Design Decisions

- [x] **Train-only SSL** — BYOL pretraining reads exclusively from `AUG_SPLIT/train/images`
- [x] **Train-only augmentation** — val/test are copied through untouched, never augmented
- [x] **Train-derived class mapping** — categories with zero training instances are excluded, not silently zero-filled
- [x] **MD5 cross-split duplicate audit** — confirmed zero leaked images
- [x] **Original split for final metrics** — Cell 13's official `.val()` explicitly uses the non-augmented data config
- [x] **Shape-safe weight transfer** — catches mismatches `strict=False` alone would miss
- [x] **Correct BYOL EMA ordering** — target updates *after* the optimizer step
- [x] **No double augmentation** — online YOLO augmentation reduced since offline augmentation already ran

<details>
<summary><b>🐛 Known issue & fix — CUDA OOM in Cell 15 (click to expand)</b></summary>

The matched-event evaluation cell originally crashed with `CUDA OutOfMemoryError` (*"Tried to allocate 5.13 GiB... 13.49 GiB already in use"*) while predicting on all 747 test images. Root causes:

1. `best_model.predict(..., batch=8)` without `stream=True` materializes **every** `Results` object into one list before returning, and each keeps its boxes tensor on the GPU for the whole call.
2. GPU memory left over from earlier cells in the same kernel session was never explicitly released.

**Fix:** switched to `stream=True` so results are yielded and discarded one at a time, added `gc.collect()` / `torch.cuda.empty_cache()` before the call and every 100 images during it. Cell 15 is fully self-contained (re-derives all paths and reloads `best.pt` from disk), so restarting the kernel and jumping straight to it — skipping the training cells — is also a valid quick fix.

</details>

---

## 📦 Output Artifacts

The notebook writes everything to `FIG_DIR` (figures) and `RESULTS_DIR` (data) under `palm_byol_ssl/`.

| Category | Approx. Count | Examples |
|---|---:|---|
| 🖼️ Figures (PNG) | ~20 | confusion matrices, calibration/reliability diagrams, ROC/PR curves, training curves, threshold heatmaps, qualitative detections |
| 📄 Metrics & Logs (CSV / JSON) | ~20 | per-class metrics, SSL training logs, matched-event tables, calibration bins, computational cost summary |
| 📑 Paper-Ready Tables (CSV + MD + LaTeX) | 6 tables + 1 executive summary | main detection performance, per-class performance, matched-event stats, calibration, computational cost, best threshold |
| 📝 Reporting Note | 1 | methodology note clarifying matched-event metrics are diagnostic, not a mAP replacement |

---

## 🧰 Tech Stack

`Ultralytics YOLOv12` · `PyTorch` · `Albumentations` · `Roboflow` · `scikit-learn` · `OpenCV` · `Matplotlib` · `Pandas` · `NumPy`

---

## 🙏 Acknowledgments

Dataset hosted via **Roboflow** (`palm-oil-vj5ii / dat-palm-fx-vfhm6`). Detector built on **Ultralytics YOLOv12**. Self-supervised pretraining follows the **BYOL** method (Grill et al., 2020), adapted here for detection backbones rather than image classifiers.

<div align="center">

*Made with 🧠 self-supervised learning and a lot of leakage audits.*

</div>
