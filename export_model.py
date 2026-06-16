# -*- coding: utf-8 -*-
"""
export_model.py  ——  Run once to build model_bundle.joblib for the web app.

What it does:
  Refits TabICLv2 on all 105 development-cohort patients (identical config to the
  paper / V14.py) and packages everything the Streamlit app needs:
    - the 105 training rows + TabICLv2 config (the app refits at start-up;
      identical seed -> identical predictions, avoids fragile pickling of torch)
    - Platt calibration parameters (paper's probability scale)
    - risk-tertile cut-points and the observed PMD rate within each tertile
    - per-feature summary statistics (min/p25/median/p75/max) for the input form
      and the patient-vs-cohort plot
    - TabICLv2 cross-validated performance (AUC + 95% CI, sensitivity, specificity,
      PPV, NPV, Brier, ECE) and the operating threshold, read from the pipeline cache

Run it in the SAME environment you used for V14.py (torch / tabicl installed,
internet available to download the TabICL checkpoint).

    python export_model.py
"""

import os
import json
import random
import pickle
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ════════════════════════════════════════════════════════════════════════════
# 0. Configuration — must match V14.py
# ════════════════════════════════════════════════════════════════════════════
DATA_PATH  = "data.5.30.txt"                                  # tab-separated raw data
LABEL_COL  = "quality"
CACHE_FILE = Path("ML_Results_V14") / "cache" / "pipeline_cache_v14.pkl"  # optional but recommended
OUT_BUNDLE = "model_bundle.joblib"
SEED       = 42

# Identical to TabICLv2 in V14.py build_models()
TABICL_KWARGS = dict(
    checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
    n_estimators=12,
    norm_methods=["none", "power", "quantile", "robust"],
    feat_shuffle_method="random",
    outlier_threshold=6.0,
    softmax_temperature=0.8,
    average_logits=True,
    kv_cache=False,
    n_jobs=-1,
    random_state=SEED,
)


# ════════════════════════════════════════════════════════════════════════════
# 1. Data loading — identical to V14.py load_data()
# ════════════════════════════════════════════════════════════════════════════
def load_data(path):
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={
        "SEP change time": "SEP_change_time",
        "MEP change time": "MEP_change_time",
        "SEP recovery time": "SEP_recovery_time",
        "MEP recovery time": "MEP_recovery_time",
        "temporary clipping duration": "temporary_clipping_duration",
    })
    df["Gender"]           = (df["Gender"] == "Male").astype(int)            # Male=1 Female=0
    df["Aneurysm_rupture"] = (df["Aneurysm_rupture"] == "Yes").astype(int)   # Yes=1 No=0
    df["NLA_on_CT"]        = (df["NLA_on_CT"] == "Yes").astype(int)          # Yes=1 No=0
    df["Hunt_Hess_grade"]  = df["Hunt_Hess_grade"].astype(int)
    raw_q = df[LABEL_COL].astype(int)
    df[LABEL_COL] = (raw_q > 0).astype(int)                                   # binarise: >0 -> 1
    return df


# ════════════════════════════════════════════════════════════════════════════
# Helpers: seeds / calibration
# ════════════════════════════════════════════════════════════════════════════
def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def apply_calib(p, calib):
    p = np.asarray(p, dtype=float)
    t = calib.get("type", "none")
    if t == "platt":
        return _sigmoid(calib["a"] * _logit(p) + calib["b"])
    if t == "temperature":
        return _sigmoid(_logit(p) / calib["T"])
    if t == "beta":
        pp = np.clip(p, 1e-7, 1 - 1e-7)
        return _sigmoid(calib["a"] * np.log(pp) - calib["b"] * np.log(1 - pp) + calib["c"])
    return p


def collect_versions():
    v = {"python": platform.python_version()}
    for mod in ["numpy", "pandas", "sklearn", "scipy", "torch", "tabicl", "joblib"]:
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            v[mod] = "NOT INSTALLED"
    # tabicl often lacks __version__; read it from package metadata instead
    if v.get("tabicl") in (None, "unknown"):
        try:
            import importlib.metadata as _im
            v["tabicl"] = _im.version("tabicl")
        except Exception:
            pass
    return v


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    set_all_seeds(SEED)

    print("[1/6] Loading data ...")
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"Data file {DATA_PATH} not found. Put export_model.py next to {DATA_PATH}.")
    df = load_data(DATA_PATH)
    feat_names = [c for c in df.columns if c != LABEL_COL]      # same order as training
    X = df[feat_names].values.astype(float)
    y = df[LABEL_COL].values.astype(int)
    print(f"      {len(df)} patients | {len(feat_names)} features:")
    print(f"      {feat_names}")
    print(f"      outcome prevalence = {y.mean():.1%}")

    print("[2/6] Fitting TabICLv2 on the full development cohort ...")
    from tabicl import TabICLClassifier
    model = TabICLClassifier(**TABICL_KWARGS)
    model.fit(X, y)
    p_insample = model.predict_proba(X)[:, 1]

    print("[3/6] Per-feature summary statistics ...")
    feat_stats = {}
    for j, name in enumerate(feat_names):
        col = X[:, j]
        feat_stats[name] = {
            "min":    float(np.nanmin(col)),
            "p25":    float(np.nanpercentile(col, 25)),
            "median": float(np.nanmedian(col)),
            "p75":    float(np.nanpercentile(col, 75)),
            "max":    float(np.nanmax(col)),
            "mean":   float(np.nanmean(col)),
        }

    print("[4/6] Reading calibration, threshold, performance from pipeline cache ...")
    calib       = {"type": "none"}
    tertiles    = None
    performance = {}
    threshold   = 0.5
    try:
        with open(CACHE_FILE, "rb") as f:
            C = pickle.load(f)

        # —— Platt / Beta / Temperature calibration (as used in the paper) ——
        ts = C.get("ts_models", {}).get("TabICLv2")
        if isinstance(ts, dict):
            if ts.get("type") == "platt":
                calib = {"type": "platt", "a": float(ts["a"]), "b": float(ts["b"])}
            elif ts.get("type") == "beta":
                calib = {"type": "beta", "a": float(ts["a"]),
                         "b": float(ts["b"]), "c": float(ts["c"])}
            elif ts.get("type") == "temperature":
                calib = {"type": "temperature", "T": float(ts.get("T", 1.0))}

        # —— operating threshold (Gmean-optimised, nested) ——
        threshold = float(C.get("opt_thrs", {}).get("TabICLv2", 0.5))

        # —— cross-validated performance of TabICLv2 ——
        try:
            dfcv = C["dfs"]["CV_Repeated"]
            row = dfcv[dfcv["Model"] == "TabICLv2"].iloc[0].to_dict()

            def _f(key, default=float("nan")):
                try:
                    return float(row.get(key, default))
                except Exception:
                    return default

            performance = {
                "AUC":          _f("AUC"),
                "AUC_95CI":     str(row.get("AUC_95CI", "")),
                "Sensitivity":  _f("Sensitivity"),
                "Specificity":  _f("Specificity"),
                "PPV":          _f("PPV"),
                "NPV":          _f("NPV"),
                "F1":           _f("F1_binary", _f("F1")),
                "Balanced_Acc": _f("Balanced_Acc"),
                "Brier":        _f("Brier_Score"),
                "ECE":          _f("ECE"),
            }
        except Exception as e:
            print(f"      [note] could not read CV metrics table ({e}).")

        # —— risk tertiles + observed PMD rate per tertile (cache OOF probs are
        #     already calibrated; cut at 33rd/67th percentile -> reproduces Fig 5G) ——
        probs = C.get("probs", {}).get("CV_Repeated", {}).get("TabICLv2")
        y_cache = C.get("y")
        if probs is not None:
            p_oof = np.asarray(probs)[:, 1]
            tertiles = [float(np.quantile(p_oof, 1 / 3)),
                        float(np.quantile(p_oof, 2 / 3))]
            if y_cache is not None:
                y_cache = np.asarray(y_cache).astype(int)
                bins = np.digitize(p_oof, tertiles)          # 0 / 1 / 2
                tertile_rates = [float(y_cache[bins == b].mean()) if (bins == b).any()
                                 else float("nan") for b in (0, 1, 2)]
                tertile_n = [int((bins == b).sum()) for b in (0, 1, 2)]
            else:
                tertile_rates, tertile_n = [float("nan")] * 3, [0, 0, 0]
        else:
            tertile_rates, tertile_n = [float("nan")] * 3, [0, 0, 0]

        print(f"      ✓ calibration = {calib['type']} | threshold = {threshold:.3f}")
        print(f"      ✓ AUC = {performance.get('AUC_95CI') or performance.get('AUC')}")
        print(f"      ✓ tertile cut-points = "
              f"{[round(t, 3) for t in tertiles] if tertiles else 'n/a'} | "
              f"observed PMD per tertile = {[round(r, 3) for r in tertile_rates]}")

    except FileNotFoundError:
        print(f"      [note] cache {CACHE_FILE} not found; performance panel will be hidden "
              f"and tertiles fall back to in-sample.")
        tertile_rates, tertile_n = [float("nan")] * 3, [0, 0, 0]
    except Exception as e:
        print(f"      [note] cache read failed ({type(e).__name__}: {e}); using fallbacks.")
        tertile_rates, tertile_n = [float("nan")] * 3, [0, 0, 0]

    # Fallback tertiles if no cache OOF probabilities were available
    if tertiles is None:
        p_cal_in = apply_calib(p_insample, calib)
        tertiles = [float(np.quantile(p_cal_in, 1 / 3)),
                    float(np.quantile(p_cal_in, 2 / 3))]

    print("[5/6] Packaging ...")
    bundle = {
        "feat_names":    feat_names,          # feature order used to build the input vector
        "X_train":       X,                   # 105 rows (app refits at start-up)
        "y_train":       y,
        "tabicl_kwargs": TABICL_KWARGS,       # TabICLv2 configuration
        "calib":         calib,               # apply to the raw probability
        "tertiles":      tertiles,            # [low|intermediate cut, intermediate|high cut]
        "tertile_rates": tertile_rates,       # observed PMD rate within each tertile (dev cohort)
        "tertile_n":     tertile_n,           # n per tertile (dev cohort)
        "feat_stats":    feat_stats,          # min/p25/median/p75/max/mean per feature
        "performance":   performance,         # CV metrics for TabICLv2 (may be empty)
        "threshold":     threshold,           # operating threshold (Gmean-optimised)
        "n_train":       int(len(df)),
        "pos_rate":      float(y.mean()),
        "model_name":    "TabICLv2",
        "seed":          SEED,
        "versions":      collect_versions(),
    }
    joblib.dump(bundle, OUT_BUNDLE)
    print(f"\n✅ Done → {OUT_BUNDLE}  ({Path(OUT_BUNDLE).stat().st_size/1024:.0f} KB)\n")

    print("[6/6] Pin these versions in requirements.txt (torch / tabicl / scikit-learn):")
    print(json.dumps(bundle["versions"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
