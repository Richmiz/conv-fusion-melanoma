# Beyond AUROC in Dermoscopic Melanoma Detection

This release contains the original 60/40 benchmark workflow. The revised manuscript uses an equal-weight Conv-Fusion and a revised PROVe-AI evaluation cohort. This release does not reproduce those revised results.

Code for evaluating **Conv-Fusion** (`0.60 * Retained + 0.40 * Source-Balanced` ConvNeXt-Small logit blend) and comparators (**ADAE**, **Committee-29**, **SIIM-Retrained**, **Contrast+Gamma**, **Benign-Mimics**) across **ISIC 2020**, **DERM12345**, **PROVe-AI**, and **PH²**, with a common calibration allowance of up to seven alerts among 71 PROVe-AI nevi and patient-clustered paired bootstrap confidence intervals.

## Repository Structure

```text
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── final_external_benchmark.py      # Unified 4-cohort evaluation & shared calibration
│   ├── final_benchmark_intervals.py     # Patient/image-clustered paired bootstrap CIs
│   ├── benign_mimic_training_pilot.py   # Retained & Benign-Mimics ConvNeXt-Small training
│   ├── balanced_mimic_aux_pilot.py      # Source-Balanced dual-head ConvNeXt-Small training
│   ├── prove_ai_adae.py                 # ADAE comparator inference
│   ├── prove_ai_retrained_siim.py       # SIIM-Retrained comparator inference
│   ├── committee_ph2.py                 # Committee-29 comparator inference
│   └── ...                              # Supporting data, TTA, and store utilities
├── scripts/
│   └── run_paired_analysis.py           # CLI for calibration, cohort summaries, & paired CIs
├── notebooks/
│   ├── 01_external_benchmark.ipynb      # Full 4-cohort benchmark workflow
│   └── 02_paired_analysis.ipynb         # Paired bootstrap & threshold-transfer analysis
└── tests/
    └── test_benchmark.py                # Self-contained offline unit tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datasets

Public dermoscopic datasets are not redistributed in this repository. Download them from their official sources:
- **ISIC 2020** (`SIIM-ISIC Melanoma Classification`)
- **DERM12345** (`https://isic-archive.s3.amazonaws.com/dois/10.34970-705541/derm12345.zip`)
- **PROVe-AI** (ISIC Archive / PROVe-AI release)
- **PH²** (University of Porto Dermoscopy Database)

## Usage

### 1. Run Unit Tests

```bash
python -m unittest discover -s tests -v
```

### 2. Run Paired Bootstrap & Threshold-Transfer Analysis (from Saved Prediction CSVs)

```bash
python scripts/run_paired_analysis.py \
  --isic path/to/isic_predictions.csv \
  --derm path/to/derm_predictions.csv \
  --prove path/to/prove_predictions.csv \
  --ph2 path/to/ph2_predictions.csv \
  --out-dir outputs
```

### 3. Run Notebooks
- `notebooks/01_external_benchmark.ipynb`: Full GPU inference & 4-cohort benchmark (`FinalExternalBenchmark`).
- `notebooks/02_paired_analysis.ipynb`: Interactive threshold calibration and paired bootstrap CIs.

### Storage configuration for the historical GPU workflow

The full inference workflow requires the relevant model weights, score exports, and run receipts in folders you are authorized to access. Supply your own Google Drive folder IDs with the configuration variables below. The offline unit tests and analysis from saved scores do not require this setup. Model weights and prediction files are not included in this release.

- `CONV_FUSION_TRAIN_FOLDER`, `CONV_FUSION_NB08_FOLDER`, `CONV_FUSION_NB09_FOLDER`, and `CONV_FUSION_PARENT_FOLDER` for development inputs.
- `CONV_FUSION_BASELINE_FOLDER`, `CONV_FUSION_CONTRAST_GAMMA_FOLDER`, and `CONV_FUSION_PREFLIGHT_FOLDER` for earlier runs.
- `CONV_FUSION_NB15_FOLDER` and `CONV_FUSION_NB21_FOLDER` for the original benchmark inputs.
- `CONV_FUSION_RESEARCH_FOLDER` and `CONV_FUSION_PROJECT_FOLDER` for the Drive output store.
