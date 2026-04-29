# -*- coding: utf-8 -*-
"""
XGBoost LORO-CV Wildfire Detection
- Backward time-window maximum compositing (BTMC)
- Guard-zone boundary downweighting
- Internal validation-based threshold (tau) selection

Outputs (per fold):
- metrics_lopo.csv
- model_holdout_<region>.pkl / .json
"""

import os
import glob
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import xgboost as xgb
from joblib import dump as joblib_dump

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    brier_score_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

# =========================
# USER SETTINGS
# =========================
BASE_DIR = "path/to/your/dataset"
OUT_DIR  = os.path.join(BASE_DIR, "results")

CSV_NAME     = "your_dataset.csv"
MODE         = "threshold"
WINDOW_MIN   = 10
WINDOW_TYPE  = "backward"
SEED         = 42

ALLOW_REGIONS = [
    "2022_uljin", "2022_gangneung", "2022_hapcheon", "2022_yanggu",
    "2022_gunwi", "2022_uljin_2", "2022_milyang",
    "2023_geumsan", "2023_hongseong", "2023_hapcheon",
    "2025_gyeongbuk", "2025_sancheong", "2025_ulsan_yangsan", "2025_deagu"
]
TEST_REGIONS = ALLOW_REGIONS[:]

TARGET_COL = "label"
GROUP_COL  = "region"

SAVE_MODELS_PKL  = True
SAVE_MODELS_JSON = True

META_COLS = {
    "label", "label_src", "region", "ts_label", "cand_time",
    "row", "col", "x_m", "y_m", "lon", "lat",
    "window_type", "tol_min",
}
STRIP_APOS_COLS = ["ts_label", "cand_time"]

# =========================
# Threshold selection (no test-peeking)
# =========================
USE_INTERNAL_VAL_FOR_TAU = True
VAL_FRAC            = 0.2
VAL_SCENE_COL       = "ts_label"
GLOBAL_TAU_FALLBACK = 0.5

# =========================
# Sample weight settings
# =========================
W_MIN        = 0.2
W_IDW_ALPHA  = 0.7
W_CLEAR_BETA = 0.3
USE_CLOUD_CLEAR_FRAC_IN_WEIGHT = True
USE_CLEAR_CNT_IN_WEIGHT        = True
CLEAR_CNT_CAP = 5

# =========================
# Guard-zone settings
# =========================
USE_GUARD_ZONE   = True
GUARD_PIX        = 1
GUARD_NEG_WEIGHT = 0.5
SCENE_COLS       = ["region", "ts_label"]

# =========================
# XGBoost parameters
# =========================
XGB_PARAMS = dict(
    n_estimators=700,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1.0,
    gamma=0.5,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=8,
    random_state=0,
)

SPW_MAX = 30.0

# =========================
# Feature definitions
# =========================
GROUPS = {
    "GEOMETRY": ["SZA"],
    "SPECTRAL": ["VI004", "VI005", "VI006", "VI008", "NR016"],
    "THERMAL": ["BT38", "BTD38_11", "BTD38_12"],
    
    # Spatial Context (Neighborhood Statistics)
    "SPATIAL_STAT": [
        "BT38_mean3", "BT38_std3",
        "BTD38_11_mean3", "BTD38_11_std3",
        "BTD38_12_mean3", "BTD38_12_std3",
    ],
    
    # Center-Background Contrast (LCI & Z-score)
    "SPATIAL_CONTRAST": [
        "BT38_lci", "BT38_z3",
        "BTD38_11_lci", "BTD38_11_z3",
        "BTD38_12_lci", "BTD38_12_z3",
    ],
    
    # Temporal Dynamics
    "TEMPORAL_DYN": [
        "BT38_Δmean", "BT38_tstd", "BT38_trange",
        "BTD38_11_Δmean", "BTD38_11_tstd", "BTD38_11_trange",
        "BTD38_12_Δmean", "BTD38_12_tstd", "BTD38_12_trange",
    ],
}

SCHEMES = {
    "Scheme_1": (GROUPS["GEOMETRY"] + GROUPS["THERMAL"]),
    "Scheme_2": (GROUPS["GEOMETRY"] + GROUPS["THERMAL"] + GROUPS["SPECTRAL"]),
    "Scheme_3": (GROUPS["GEOMETRY"] + GROUPS["THERMAL"] + GROUPS["SPATIAL_STAT"]),
    "Scheme_4": (GROUPS["GEOMETRY"] + GROUPS["THERMAL"] + GROUPS["SPATIAL_CONTRAST"]),
    "Scheme_5": (GROUPS["GEOMETRY"] + GROUPS["THERMAL"] + GROUPS["TEMPORAL_DYN"]) # Best
}

GROUPS.update(SCHEMES)
FEATURE_CASES = {k: [k] for k in SCHEMES.keys()}


def normalize_region(s: str) -> str:
    return str(s).strip().lower()


ALLOW_REGIONS_N = {normalize_region(r) for r in ALLOW_REGIONS}
TEST_REGIONS_N  = [normalize_region(r) for r in TEST_REGIONS]


def find_csvs(base_dir: str, csv_name: str) -> List[str]:
    return sorted(glob.glob(os.path.join(base_dir, "**", csv_name), recursive=True))


def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for c in STRIP_APOS_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lstrip("'")
    return df


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if "label_src" in df.columns:
        df["label_src"] = df["label_src"].astype(str).str.strip().str.lower()
        df = df[df["label_src"] == "viirs"].copy()
    if GROUP_COL in df.columns:
        df[GROUP_COL] = df[GROUP_COL].astype(str).map(normalize_region)
        df = df[df[GROUP_COL].isin(ALLOW_REGIONS_N)].copy()
    df = df.dropna(subset=[TARGET_COL, GROUP_COL]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def parse_dataset_info(csv_path: str) -> Dict:
    parts = os.path.normpath(csv_path).split(os.sep)[:-1]
    win_idx = None
    for i, p in enumerate(parts):
        p_low = p.lower()
        if p_low.endswith("min") and p_low[:-3].isdigit():
            win_idx = i
    if win_idx is None or win_idx < 1:
        return dict(mode="unk", win_min=None, fill="unk", wtype="unk", tag="__".join(parts[-4:]))
    mode    = parts[win_idx - 1].lower()
    win_str = parts[win_idx].lower()
    win_min = int(win_str[:-3]) if win_str[:-3].isdigit() else None
    fill    = parts[win_idx + 1].lower() if win_idx + 1 < len(parts) else "unk"
    wtype   = parts[win_idx + 2].lower() if win_idx + 2 < len(parts) else "unk"
    tag     = f"{mode}__{win_str}__{fill}__{wtype}"
    return dict(mode=mode, win_min=win_min, fill=fill, wtype=wtype, tag=tag)


def expand_case_to_columns(case_spec: List[str], df_columns: List[str]) -> List[str]:
    cols: List[str] = []
    for token in case_spec:
        if token in GROUPS:
            for c in GROUPS[token]:
                if c in df_columns and c not in cols:
                    cols.append(c)
        elif token in df_columns and token not in cols:
            cols.append(token)
    return cols


def compute_sample_weight(df: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(df), dtype=np.float32)
    if "idw_win3_frac" in df.columns:
        idw = pd.to_numeric(df["idw_win3_frac"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        w *= (1.0 - W_IDW_ALPHA * np.clip(idw, 0.0, 1.0))
    if USE_CLOUD_CLEAR_FRAC_IN_WEIGHT and "cloud_clear_frac" in df.columns:
        ccf = pd.to_numeric(df["cloud_clear_frac"], errors="coerce").fillna(1.0).to_numpy(np.float32)
        w *= (1.0 - W_CLEAR_BETA * (1.0 - np.clip(ccf, 0.0, 1.0)))
    if USE_CLEAR_CNT_IN_WEIGHT:
        cand_cols = [c for c in ["SW038_cen_clear_cnt", "IR123_cen_clear_cnt", "IR112_cen_clear_cnt"] if c in df.columns]
        if cand_cols:
            cc = pd.to_numeric(df[cand_cols[0]], errors="coerce").fillna(0.0).to_numpy(np.float32)
            cc_norm = np.clip(cc / float(max(CLEAR_CNT_CAP, 1)), 0.0, 1.0)
            w *= (0.7 + 0.3 * cc_norm)
    if USE_GUARD_ZONE and ("guard_neg" in df.columns):
        y      = df[TARGET_COL].astype(int).to_numpy(np.int32)
        is_neg = (y == 0)
        gn     = df["guard_neg"].astype(bool).to_numpy()
        w     *= np.where(is_neg & gn, float(GUARD_NEG_WEIGHT), 1.0)
    return np.clip(w, W_MIN, 1.0)


def split_train_val_by_scene(tr: pd.DataFrame,
                             scene_col: str,
                             val_frac: float,
                             seed: int,
                             min_pos_scenes: int = 3,
                             max_tries: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if (scene_col not in tr.columns) or (len(tr) < 10):
        rng = np.random.default_rng(int(seed))
        idx = np.arange(len(tr))
        rng.shuffle(idx)
        cut = max(1, int(len(tr) * (1.0 - float(val_frac))))
        return tr.iloc[idx[:cut]].copy(), tr.iloc[idx[cut:]].copy()
    g = tr.groupby(scene_col)[TARGET_COL].max()
    pos_scenes = g[g == 1].index.to_numpy()
    neg_scenes = g[g == 0].index.to_numpy()
    rng = np.random.default_rng(int(seed))
    all_scenes   = g.index.to_numpy()
    n_val_scenes = max(1, int(len(all_scenes) * float(val_frac)))
    if len(pos_scenes) == 0:
        y      = tr[TARGET_COL].astype(int).to_numpy()
        groups = tr[scene_col].astype(str).to_numpy()
        gss    = GroupShuffleSplit(n_splits=1, test_size=float(val_frac), random_state=int(seed))
        fit_idx, val_idx = next(gss.split(tr, y, groups))
        return tr.iloc[fit_idx].copy(), tr.iloc[val_idx].copy()
    for _ in range(int(max_tries)):
        n_pos_val = min(len(pos_scenes), max(int(round(len(pos_scenes) * float(val_frac))), int(min_pos_scenes)))
        n_pos_val = max(1, n_pos_val)
        n_neg_val = max(0, n_val_scenes - n_pos_val)
        n_neg_val = min(n_neg_val, len(neg_scenes))
        val_pos = rng.choice(pos_scenes, size=n_pos_val, replace=False)
        val_neg = rng.choice(neg_scenes, size=n_neg_val, replace=False) if n_neg_val > 0 else np.array([], dtype=object)
        val_set = set(val_pos.tolist() + val_neg.tolist())
        tr_val = tr[tr[scene_col].isin(val_set)].copy()
        tr_fit = tr[~tr[scene_col].isin(val_set)].copy()
        if len(tr_fit) == 0 or len(tr_val) == 0:
            continue
        if int(tr_val[TARGET_COL].sum()) == 0:
            continue
        return tr_fit, tr_val
    return tr_fit, tr_val


def metrics_at_tau(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> Tuple[float, float, float]:
    y_hat = (y_prob >= float(tau)).astype(np.int32)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true.astype(int), y_hat.astype(int),
        average="binary", zero_division=0
    )
    return float(f1), float(p), float(r)


def best_tau_on_val(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    if len(np.unique(y_true)) < 2:
        return float(GLOBAL_TAU_FALLBACK)
    try:
        precision, recall, thresh = precision_recall_curve(y_true, y_prob)
        if len(thresh) > 0:
            p  = precision[:-1]
            r  = recall[:-1]
            ok = np.isfinite(p) & np.isfinite(r)
            f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
            f1[~ok] = -np.inf
            i = int(np.nanargmax(f1))
            if np.isfinite(f1[i]) and (f1[i] > -np.inf):
                return float(thresh[i])
    except Exception:
        pass
    return float(GLOBAL_TAU_FALLBACK)

def _resolve_spw(neg: int, pos: int) -> float:
    spw = float(neg / max(pos, 1))
    return float(min(max(spw, 1.0), float(SPW_MAX)))


def mark_guard_zone_negatives(df: pd.DataFrame,
                              scene_cols: List[str],
                              row_col=("row", "col"),
                              label_col="label",
                              guard_pix: int = 1) -> pd.Series:
    rcol, ccol = row_col
    need = list(scene_cols) + [rcol, ccol, label_col]
    if any(c not in df.columns for c in need):
        return pd.Series(False, index=df.index)
    rr = pd.to_numeric(df[rcol], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
    cc = pd.to_numeric(df[ccol], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
    y  = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    keys    = (rr.astype(np.int64) << 20) + cc.astype(np.int64)
    scene   = df[scene_cols].astype(str).agg("||".join, axis=1).to_numpy()
    order   = np.argsort(scene)
    scene_s = scene[order]
    offsets = [(dr, dc) for dr in range(-guard_pix, guard_pix+1) for dc in range(-guard_pix, guard_pix+1)]
    out = np.zeros(len(df), dtype=bool)
    _, starts = np.unique(scene_s, return_index=True)
    starts = np.r_[starts, len(scene_s)]
    for i in range(len(starts)-1):
        a, b = starts[i], starts[i+1]
        idx  = order[a:b]
        y_g  = y[idx]
        if y_g.sum() == 0:
            continue
        pos_idx = idx[y_g == 1]
        neg_idx = idx[y_g == 0]
        if len(neg_idx) == 0:
            continue
        r_pos = rr[pos_idx]
        c_pos = cc[pos_idx]
        neigh = set()
        for dr, dc in offsets:
            kk = ((r_pos + dr).astype(np.int64) << 20) + (c_pos + dc).astype(np.int64)
            for k in kk.tolist():
                neigh.add(k)
        for j in neg_idx:
            if keys[j] in neigh:
                out[j] = True
    return pd.Series(out, index=df.index)


def train_lopo_one_seed(df: pd.DataFrame,
                        feature_cols: List[str],
                        out_dir_seed: str,
                        seed: int) -> pd.DataFrame:
    present = set(df[GROUP_COL].astype(str).unique().tolist())
    rows    = []
    for heldout in TEST_REGIONS_N:
        if heldout not in present:
            continue
        tr = df[df[GROUP_COL] != heldout].copy()
        te = df[df[GROUP_COL] == heldout].copy()
        if USE_INTERNAL_VAL_FOR_TAU:
            tr_fit, tr_val = split_train_val_by_scene(tr, scene_col=VAL_SCENE_COL, val_frac=VAL_FRAC, seed=seed)
        else:
            tr_fit, tr_val = tr, None
        y_fit = tr_fit[TARGET_COL].astype(int).to_numpy(np.int32)
        y_te  = te[TARGET_COL].astype(int).to_numpy(np.int32)
        X_fit = tr_fit[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        X_te  = te[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        col_mean = np.nanmean(X_fit, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
        bad = np.where(~np.isfinite(X_fit))
        X_fit[bad] = np.take(col_mean, bad[1])
        bad = np.where(~np.isfinite(X_te))
        X_te[bad] = np.take(col_mean, bad[1])
        w_fit   = compute_sample_weight(tr_fit)
        pos     = int((y_fit == 1).sum())
        neg     = int((y_fit == 0).sum())
        spw_use = _resolve_spw(neg, pos)
        params = dict(XGB_PARAMS)
        params["scale_pos_weight"] = float(spw_use)
        params["random_state"]     = int(seed)
        model = xgb.XGBClassifier(**params)
        if USE_INTERNAL_VAL_FOR_TAU and (tr_val is not None) and (len(tr_val) > 0):
            y_val_es = tr_val[TARGET_COL].astype(int).to_numpy(np.int32)
            X_val_es = tr_val[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            bad = np.where(~np.isfinite(X_val_es))
            X_val_es[bad] = np.take(col_mean, bad[1])
            w_val_es = compute_sample_weight(tr_val)
            fit_kwargs = dict(
                X=X_fit, y=y_fit,
                sample_weight=w_fit,
                eval_set=[(X_val_es, y_val_es)],
                verbose=False,
            )
            fit_kwargs["sample_weight_eval_set"] = [w_val_es]
            try:
                model.fit(**fit_kwargs, early_stopping_rounds=50)
            except TypeError:
                try:
                    cb = [xgb.callback.EarlyStopping(rounds=50, save_best=True)]
                    model.fit(**fit_kwargs, callbacks=cb)
                except TypeError:
                    print("[WARN] early stopping not supported. Training without it.")
                    model.fit(X_fit, y_fit, sample_weight=w_fit)
        else:
            model.fit(X_fit, y_fit, sample_weight=w_fit)
        if USE_INTERNAL_VAL_FOR_TAU and (tr_val is not None) and (len(tr_val) > 0):
            y_val = tr_val[TARGET_COL].astype(int).to_numpy(np.int32)
            X_val = tr_val[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            bad   = np.where(~np.isfinite(X_val))
            X_val[bad] = np.take(col_mean, bad[1])
            p_val = model.predict_proba(X_val)[:, 1].astype(np.float32)
            tau   = best_tau_on_val(y_val, p_val) if len(np.unique(y_val)) > 1 else float(GLOBAL_TAU_FALLBACK)
        else:
            tau = float(GLOBAL_TAU_FALLBACK)
        y_prob  = model.predict_proba(X_te)[:, 1].astype(np.float32)
        pr_auc  = float(average_precision_score(y_te, y_prob)) if len(np.unique(y_te)) > 1 else np.nan
        roc_auc = float(roc_auc_score(y_te, y_prob))          if len(np.unique(y_te)) > 1 else np.nan
        brier   = float(brier_score_loss(y_te, y_prob))
        f1, p, r = metrics_at_tau(y_te, y_prob, tau)
        rows.append(dict(
            seed=seed, heldout_region=heldout,
            n_train=len(tr), n_test=len(te),
            pos_train=int(pos), pos_test=int((y_te == 1).sum()),
            pr_auc=pr_auc, roc_auc=roc_auc, brier=brier,
            best_tau=float(tau), f1=float(f1), precision=float(p), recall=float(r),
            scale_pos_weight=float(spw_use),
        ))
        os.makedirs(out_dir_seed, exist_ok=True)
        if SAVE_MODELS_PKL:
            joblib_dump(model, os.path.join(out_dir_seed, f"model_holdout_{heldout}.pkl"))
        if SAVE_MODELS_JSON:
            model.save_model(os.path.join(out_dir_seed, f"model_holdout_{heldout}.json"))
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(os.path.join(out_dir_seed, "metrics_lopo.csv"), index=False)
    return fold_df


def macro_from_fold_df(fold_df: pd.DataFrame) -> Dict:
    return dict(
        pr_auc_macro    =float(np.nanmean(fold_df["pr_auc"]))    if "pr_auc"    in fold_df else np.nan,
        roc_auc_macro   =float(np.nanmean(fold_df["roc_auc"]))   if "roc_auc"   in fold_df else np.nan,
        f1_macro        =float(np.nanmean(fold_df["f1"]))        if "f1"        in fold_df else np.nan,
        precision_macro =float(np.nanmean(fold_df["precision"])) if "precision" in fold_df else np.nan,
        recall_macro    =float(np.nanmean(fold_df["recall"]))    if "recall"    in fold_df else np.nan,
        brier_macro     =float(np.nanmean(fold_df["brier"]))     if "brier"     in fold_df else np.nan,
    )


def write_region_summary(out_dir: str) -> str:
    paths = glob.glob(os.path.join(out_dir, "**", "seed_*", "metrics_lopo.csv"), recursive=True)
    if not paths:
        return ""
    rows = []
    for p in paths:
        parts       = os.path.normpath(p).split(os.sep)
        seed_dir    = parts[-2]
        feat_dir    = parts[-3]
        dataset_tag = parts[-4]
        try:
            seed = int(seed_dir.split("_")[1])
        except Exception:
            seed = -1
        featcase = feat_dir.replace("feat_", "")
        df = pd.read_csv(p)
        if df.empty or "heldout_region" not in df.columns:
            continue
        df["dataset_tag"] = dataset_tag
        df["featcase"]    = featcase
        df["seed"]        = seed
        rows.append(df)
    if not rows:
        return ""
    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df
           .groupby(["dataset_tag", "featcase", "heldout_region"], as_index=False)
           .agg(
               pr_auc_mean   =("pr_auc",    "mean"),
               pr_auc_std    =("pr_auc",    "std"),
               roc_auc_mean  =("roc_auc",   "mean"),
               roc_auc_std   =("roc_auc",   "std"),
               f1_mean       =("f1",        "mean"),
               f1_std        =("f1",        "std"),
               precision_mean=("precision", "mean"),
               precision_std =("precision", "std"),
               recall_mean   =("recall",    "mean"),
               recall_std    =("recall",    "std"),
               brier_mean    =("brier",     "mean"),
               brier_std     =("brier",     "std"),
               n_rows        =("f1",        "count"),
               n_seeds       =("seed",      "nunique"),
           ))
    out_csv = os.path.join(out_dir, "summary_by_region.csv")
    agg.to_csv(out_csv, index=False)
    return out_csv


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_paths = find_csvs(BASE_DIR, CSV_NAME)
    if not csv_paths:
        raise FileNotFoundError(f"No '{CSV_NAME}' found under: {BASE_DIR}")
    summary_rows = []
    for csv_path in csv_paths:
        info = parse_dataset_info(csv_path)
        if info["mode"] != MODE.lower():
            continue
        if info["win_min"] is None or info["win_min"] != WINDOW_MIN:
            continue
        if info["wtype"] != WINDOW_TYPE.lower():
            continue
        df = read_dataset(csv_path)
        df = filter_df(df)
        if df.empty:
            continue
        if USE_GUARD_ZONE:
            df["guard_neg"] = mark_guard_zone_negatives(
                df,
                scene_cols=SCENE_COLS,
                row_col=("row", "col"),
                label_col=TARGET_COL,
                guard_pix=int(GUARD_PIX),
            ).astype(bool)
            print(f"[GUARD] guard_neg: {int(df['guard_neg'].sum())}/{len(df)} (policy=downweight)")
        dataset_tag = info["tag"]
        all_cols    = df.columns.tolist()
        for featcase_name, spec in FEATURE_CASES.items():
            feat_cols = expand_case_to_columns(spec, all_cols)
            feat_cols = [c for c in feat_cols if c in df.columns and c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
            if not feat_cols:
                continue
            out_dir_feat = os.path.join(OUT_DIR, dataset_tag, f"feat_{featcase_name}")
            os.makedirs(out_dir_feat, exist_ok=True)
            feat_list_path = os.path.join(out_dir_feat, "feature_list.txt")
            if not os.path.exists(feat_list_path):
                with open(feat_list_path, "w", encoding="utf-8") as f:
                    for c in feat_cols:
                        f.write(c + "\n")
            print(f"\n=== {dataset_tag} | featcase={featcase_name} | rows={len(df)} | n_feat={len(feat_cols)} ===")
            print(f"    GUARD={USE_GUARD_ZONE}/downweight pix={GUARD_PIX}")
            out_dir_seed = os.path.join(out_dir_feat, f"seed_{SEED:03d}")
            fold_df = train_lopo_one_seed(df, feat_cols, out_dir_seed, seed=SEED)
            if fold_df.empty:
                continue
            mac = macro_from_fold_df(fold_df)
            mac.update(seed=SEED)
            print(f"  seed={SEED} | PR-AUC={mac['pr_auc_macro']:.4f} | F1={mac['f1_macro']:.4f}")
            summary_rows.append({
                "dataset_tag":    dataset_tag,
                "mode":           info["mode"],
                "win_min":        info["win_min"],
                "fill":           info["fill"],
                "wtype":          info["wtype"],
                "featcase":       featcase_name,
                "n_rows":         int(len(df)),
                "n_regions":      int(df[GROUP_COL].nunique()),
                "pr_auc_mean":    float(mac.get("pr_auc_macro",    np.nan)),
                "f1_mean":        float(mac.get("f1_macro",        np.nan)),
                "precision_mean": float(mac.get("precision_macro", np.nan)),
                "recall_mean":    float(mac.get("recall_macro",    np.nan)),
                "brier_mean":     float(mac.get("brier_macro",     np.nan)),
            })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(by=["pr_auc_mean", "f1_mean"], ascending=False)
        out_all    = os.path.join(OUT_DIR, "summary_all.csv")
        summary_df.to_csv(out_all, index=False)
        print("\nSaved:", out_all)
        out_reg = write_region_summary(OUT_DIR)
        if out_reg:
            print("Saved:", out_reg)


if __name__ == "__main__":
    main()
