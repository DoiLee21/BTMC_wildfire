# Wildfire Detection and Spread Monitoring from 2-Minute GK2A Observations Using Backward Time-Window Maximum Compositing and Spatiotemporal Features

> **Note:** This repository is currently associated with a paper submission. 
> It provides the core implementation code for the spatiotemporal wildfire detection framework using Geo-Kompsat-2A (GK2A) data.

---

## Abstract

Geostationary (GEO) satellites provide high-temporal-resolution observations that enable continuous monitoring of wildfire occurrence and spread. However, their relatively coarse spatial resolution limits early wildfire detection and accurate monitoring of fire spread boundaries. Herein, a spatiotemporal wildfire detection framework is developed using high-frequency data acquired by the Geo-Kompsat-2A Advanced Meteorological Imager. The framework integrates spatial contextual contrast features, which capture relative differences between the center pixel and its background, with a backward time-window maximum compositing strategy that reduces transient attenuation caused by atmospheric conditions and variability observation. To evaluate regional generalization under diverse topographic and environmental conditions, leave-one-region-out cross-validation was applied. Threshold-based cloud masking combined with guard-zone down-weighting provided the most stable preprocessing configuration. Among all evaluated configurations, the 10-min Scheme 5 setting produced the best results, yielding a mean F1 score of 0.800. According to the SHAP analysis, BT38-11, solar zenith angle, and the temporal standard deviation of BT38 were the dominant contributors, emphasizing the value of thermal infrared information and temporal dynamics in wildfire detection. Cumulative detection analysis demonstrated that the model reproduced the major spatial patterns of wildfire spread and detected most wildfire-affected areas larger than 30 ha, including those occurring outside the predefined study regions. By integrating high-temporal-resolution GEO observations with spatiotemporal feature-based detection, the model helps overcome the GEO spatial resolution limitations and shows strong potential for near-real-time wildfire spread monitoring and large-scale disaster response.

---

## Overview

This repository contains the core implementation of the XGBoost-based wildfire detection framework described in the paper. The code demonstrates:

- Backward Time-Window Maximum Compositing (BTMC): Reduces transient noise from atmospheric interference across multi-temporal GK2A observations
- Spatiotemporal feature extraction: Spatial contextual contrast (background difference/z-score) and temporal dynamics (time std, range, frame-to-frame difference)
- Guard-zone down-weighting: Reduces label ambiguity at fire perimeter boundaries during training
- Leave-One-Region-Out Cross-Validation (LORO-CV): Evaluates generalization across 14 wildfire events in South Korea (2022–2025)
- Internal validation-based threshold (τ) selection: Selects decision threshold on a held-out validation split without test-set peeking

---

## Repository Structure

```
BTMC_wildfire
├── README.md
├── requirements.txt
└── XGBoost_wildfire.py       # Main training and evaluation script
```

---

## Requirements

- Python >= 3.8
- See `requirements.txt` for package dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: The code includes fallback handling for XGBoost early stopping API differences across versions. Tested on XGBoost 3.2.0

---

## Data

The training data used in this study is based on GK2A AMI Level-1B observations processed into pixel-level tabular features. Due to data size and licensing, the raw dataset is not included in this repository.

The input CSV is expected to contain the following key columns:

| Column | Description |
|---|---|
| `label` | Binary fire label (1 = fire, 0 = non-fire) |
| `label_src` | Label source (filtered to `viirs`) |
| `region` | Region identifier (e.g., `2022_uljin`) |
| `ts_label` | Scene/timestamp identifier |
| `row`, `col` | Pixel grid coordinates |
| `lon`, `lat` | Geographic coordinates |
| `SZA` | Solar zenith angle |
| `BT38` | 3.8 µm brightness temperature (center pixel) |
| `BTD38_11` | BTD between 3.9 µm and 11 µm |
| `BTD38_12` | BTD between 3.9 µm and 12 µm |
| `*_mean3`, `*_std3` | 3 X 3 window background mean and standard deviation |
| `*_lci`, `*_z3` | Local Contrast Index (Center - BG) and Z-score anomaly |
| `*_Δmean`, `*_tstd`, `*_trange` | Temporal dynamics (Frame diff, Std, and Range) |

---

## Usage

### 1. Configure paths and settings

Edit the `USER SETTINGS` section at the top of `XGBoost_wildfire.py`:

```python
BASE_DIR = "path/to/your/dataset"   # Root directory containing input CSV files
OUT_DIR  = "path/to/your/results"   # Output directory
CSV_NAME = "your_dataset.csv"       # Input CSV filename
```

### 2. Run

```bash
python XGBoost_wildfire.py
```

### 3. Outputs

After execution, the following outputs are saved under `OUT_DIR`:

| File | Description |
|---|---|
| `summary_all.csv` | Macro-averaged metrics across all folds and feature schemes |
| `summary_by_region.csv` | Per-region metrics aggregated across seeds |
| `<tag>/feat_<scheme>/seed_<N>/metrics_lopo.csv` | Per-fold detailed metrics |
| `<tag>/feat_<scheme>/seed_<N>/model_holdout_<region>.pkl` | Trained model (joblib) |
| `<tag>/feat_<scheme>/seed_<N>/model_holdout_<region>.json` | Trained model (XGBoost native) |

---

## Feature Schemes

Five feature schemes are evaluated, progressively adding feature groups:

| Scheme | Features |
|---|---|
| Scheme 1 | Geometry + Thermal (BT38, BTD) |
| Scheme 2 | Scheme 1 + SPECTRAL (VIS/NIR) |
| Scheme 3 | Scheme 1 + SPATIAL_STAT (window mean/std) |
| Scheme 4 | Scheme 1 + SPATIAL_CONTRAST (local anomaly) |
| Scheme 5 | Scheme 1 + TEMPORAL_DYN (time std, range, frame diff) |

---

## Study Regions

14 wildfire events across South Korea (2022–2025) are used for LORO-CV evaluation:

`2022_uljin`, `2022_gangneung`, `2022_hapcheon`, `2022_yanggu`, `2022_gunwi`, `2022_uljin_2`, `2022_milyang`, `2023_geumsan`, `2023_hongseong`, `2023_hapcheon`, `2025_gyeongbuk`, `2025_sancheong`, `2025_ulsan_yangsan`, `2025_daegu`

---
