# -*- coding: utf-8 -*-
"""
app.py — Postoperative Motor Deficit (PMD) Risk Predictor after Intracranial
         Aneurysm Clipping.  Web decision-support tool built on TabICLv2.

Run locally:   python -m streamlit run app.py   (use the env where tabicl is installed)
Deploy:        see README.md (GitHub -> Streamlit Community Cloud)

Requires model_bundle.joblib (produced by export_model.py) in the same folder.

Outputs:
  - calibrated probability gauge + risk-tertile stratum
  - risk-strata reference table (development-cohort event rates, Fig 5G)
  - per-patient exact Shapley contribution plot (Fig 6C analogue)
  - patient-vs-cohort profile
  - measured interpretation + methodological details
Design: restrained, typographic, journal-grade. Research use only.
"""

import math
from itertools import product

import numpy as np
import joblib
import streamlit as st

BUNDLE_PATH = "model_bundle.joblib"

# ── design tokens ───────────────────────────────────────────────────────────
INK   = "#1b1b1b"
MUTED = "#5b6770"
FAINT = "#8a949c"
RULE  = "#e2e7ea"
ACCENT = "#8a1c2e"          # restrained deep crimson (journal accent)
LOW    = "#1b7837"          # colour-blind-aware green
INT    = "#b8860b"          # amber
HIGH   = "#b2182b"          # red
SHAP_UP   = "#b2182b"       # increases risk
SHAP_DOWN = "#2166ac"       # decreases risk

st.set_page_config(
    page_title="Postoperative Motor Deficit Risk Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── global styling ──────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1060px; }}
html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: {INK};
}}

.lx-title  {{ font-family:'Source Serif 4', Georgia, serif; font-size:30px; font-weight:700;
             letter-spacing:-0.2px; line-height:1.18; color:{INK}; margin:0; }}
.lx-rule   {{ height:2px; width:62px; background:{ACCENT}; border:none; margin:14px 0 0 0; }}
.lx-sub    {{ font-size:14.5px; color:{MUTED}; line-height:1.55; margin-top:12px; max-width:820px; }}
.lx-fact   {{ display:inline-block; font-size:12.5px; color:{MUTED}; margin:14px 26px 0 0; }}
.lx-fact b {{ color:{INK}; font-weight:600; }}
.lx-eyebrow{{ font-size:11px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase;
             color:{FAINT}; margin:30px 0 12px 0; }}
.lx-note   {{ font-size:12.5px; color:{MUTED}; line-height:1.5; }}
.lx-card   {{ border:1px solid {RULE}; border-radius:8px; padding:20px 22px; background:#ffffff; }}

table.lx-tbl {{ width:100%; border-collapse:collapse; font-size:13px; }}
table.lx-tbl th {{ text-align:left; color:{FAINT}; font-weight:600; font-size:11px;
                   letter-spacing:0.6px; text-transform:uppercase; padding:8px 10px;
                   border-bottom:1px solid {RULE}; }}
table.lx-tbl td {{ padding:9px 10px; border-bottom:1px solid {RULE}; color:{INK}; }}

.stButton > button {{
    background:{ACCENT}; color:#ffffff; border:none; border-radius:6px;
    font-weight:600; font-size:15px; height:46px; letter-spacing:0.2px;
}}
.stButton > button:hover {{ background:#71101f; color:#ffffff; }}
hr {{ border-color:{RULE}; }}
</style>
""", unsafe_allow_html=True)


# ── calibration ─────────────────────────────────────────────────────────────
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


# ── load bundle and refit TabICLv2 once per server boot ─────────────────────
@st.cache_resource(show_spinner="Loading model (first start downloads the TabICL checkpoint, ~30–60 s)…")
def load_model():
    import os
    import random
    bundle = joblib.load(BUNDLE_PATH)
    seed = int(bundle.get("seed", 42))
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    from tabicl import TabICLClassifier
    model = TabICLClassifier(**bundle["tabicl_kwargs"])
    model.fit(bundle["X_train"], bundle["y_train"])
    return model, bundle


def predict_calibrated(model, calib, X):
    """Raw model probability -> calibrated positive-class probability."""
    raw = model.predict_proba(np.atleast_2d(X).astype(float))[:, 1]
    return apply_calib(raw, calib)


# ── exact Shapley values (single cohort-median baseline) ────────────────────
def exact_shapley(model, calib, x_row, baseline_row):
    """
    Exact Shapley decomposition of the calibrated probability for one patient,
    relative to a single cohort-median baseline. n features -> 2^n coalitions,
    evaluated in a single batched forward pass. Returns (base_value, phi[]).
    base + sum(phi) == calibrated risk for this patient.
    """
    n = len(x_row)
    masks = np.array(list(product([0, 1], repeat=n)), dtype=int)        # (2^n, n)
    M = np.where(masks == 1, np.asarray(x_row, float)[None, :],
                 np.asarray(baseline_row, float)[None, :]).astype(float)
    cal = predict_calibrated(model, calib, M)                           # (2^n,)
    val = {tuple(int(b) for b in masks[i]): float(cal[i]) for i in range(len(masks))}

    fact = math.factorial
    phi = np.zeros(n)
    for j in range(n):
        tot = 0.0
        for m, fv in val.items():
            if m[j] == 1:
                continue
            k = int(sum(m))
            w = fact(k) * fact(n - k - 1) / fact(n)
            m_with = list(m); m_with[j] = 1
            tot += w * (val[tuple(m_with)] - fv)
        phi[j] = tot
    base = val[tuple([0] * n)]
    return float(base), phi


# ── feature presentation (English labels / units / help) ────────────────────
FEATURE_UI = {
    "Age": dict(label="Age", unit="years", kind="int",
                help="Patient age in years."),
    "Gender": dict(label="Sex", kind="binary",
                   options=[("Female", 0), ("Male", 1)]),
    "Hunt_Hess_grade": dict(label="Hunt–Hess grade", kind="grade",
                            help="0 for unruptured aneurysms; ruptured aneurysms graded within "
                                 "the range observed in the development cohort."),
    "Aneurysm_rupture": dict(label="Aneurysm rupture", kind="binary",
                             options=[("No", 0), ("Yes", 1)]),
    "NLA_on_CT": dict(label="New low-attenuation area on postoperative CT", kind="binary",
                      options=[("No", 0), ("Yes", 1)]),
    "temporary_clipping_duration": dict(label="Temporary clipping duration", unit="min", kind="float",
                                        help="Enter 0 if no temporary clipping was performed."),
    "MEP_change_time": dict(label="MEP deterioration duration", unit="min", kind="float",
                            help="Interval of MEP amplitude reduction >50% until recovery >50% of baseline. "
                                 "Enter 0 if no MEP deterioration occurred."),
    "SEP_change_time": dict(label="SEP deterioration duration", unit="min", kind="float",
                            help="SEP deterioration duration. Enter 0 if no SEP deterioration occurred."),
    "MEP_recovery_time": dict(label="MEP recovery time", unit="min", kind="float",
                              help="Interval from the corrective manoeuvre to MEP recovery >50% of baseline. "
                                   "Enter 0 if not applicable."),
    "SEP_recovery_time": dict(label="SEP recovery time", unit="min", kind="float",
                              help="Interval from the corrective manoeuvre to SEP recovery >50% of baseline. "
                                   "Enter 0 if not applicable."),
}

CLINICAL_ORDER = ["Age", "Gender", "Hunt_Hess_grade", "Aneurysm_rupture",
                  "NLA_on_CT", "temporary_clipping_duration"]
NEURO_ORDER    = ["MEP_change_time", "MEP_recovery_time",
                  "SEP_change_time", "SEP_recovery_time"]


def _short_label(name):
    ui = FEATURE_UI.get(name, {"label": name})
    lab = ui["label"]
    if ui.get("unit"):
        lab += f" ({ui['unit']})"
    return lab


def render_widget(name, stats):
    ui = FEATURE_UI.get(name, {"label": name, "kind": "float", "unit": ""})
    s = stats.get(name, {})
    kind = ui["kind"]
    rng = ""
    if {"min", "max"} <= set(s):
        rng = (f"Development range: {s['min']:.0f}–{s['max']:.0f}." if kind in ("int", "grade")
               else f"Development range: {s['min']:.1f}–{s['max']:.1f} {ui.get('unit','')}.")
    helptext = (ui.get("help", "") + (" " + rng if rng else "")).strip()

    if kind == "binary":
        labels = [o[0] for o in ui["options"]]
        mapping = dict(ui["options"])
        choice = st.radio(ui["label"], labels, horizontal=True, index=0, help=ui.get("help"))
        return mapping[choice]

    if kind == "grade":
        # data-driven ordinal slider: only grades actually present in the cohort
        gmin = int(np.floor(s.get("min", 0)))
        gmax = int(np.ceil(s.get("max", 3)))
        grades = list(range(gmin, gmax + 1)) or [0]
        default = int(round(s.get("median", grades[0])))
        if default not in grades:
            default = grades[0]
        return st.select_slider(ui["label"], options=grades, value=default, help=helptext)

    if kind == "int":
        lo = int(np.floor(s.get("min", 0))); hi = int(np.ceil(s.get("max", 120)))
        return st.number_input(_short_label(name), min_value=lo, max_value=hi,
                               value=int(round(s.get("median", lo))), step=1, help=helptext)

    hi = float(s.get("max", 60.0))
    return st.number_input(_short_label(name), min_value=0.0, max_value=max(hi * 2, hi, 1.0),
                           value=float(s.get("median", 0.0)), step=0.1, format="%.1f", help=helptext)


# ── plots ───────────────────────────────────────────────────────────────────
def make_gauge(p, tertiles):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    lo, hi = tertiles[0] * 100, tertiles[1] * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "font": {"size": 44, "color": INK}},
        title={"text": "Estimated probability of postoperative motor deficit",
               "font": {"size": 13, "color": MUTED}},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%", "tickwidth": 1,
                        "tickcolor": "#b8c0c6", "tickfont": {"size": 10.5, "color": MUTED}},
               "bar": {"color": "#33414a", "thickness": 0.30},
               "bgcolor": "white", "borderwidth": 0,
               "steps": [{"range": [0, lo], "color": "#e4efe7"},
                         {"range": [lo, hi], "color": "#f5edd6"},
                         {"range": [hi, 100], "color": "#f6e1e4"}]}))
    fig.update_layout(height=270, margin=dict(l=24, r=24, t=46, b=14),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font={"family": "-apple-system, Segoe UI, Roboto, sans-serif"})
    return fig


def make_shapley_plot(phi, names, base, final):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    order = np.argsort(np.abs(phi))                  # ascending -> largest at top (reversed axis)
    labels = [_short_label(names[i]) for i in order]
    vals = [float(phi[i]) * 100 for i in order]
    colors = [SHAP_UP if v > 0 else SHAP_DOWN for v in vals]
    text = [f"{v:+.1f}" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors, text=text, textposition="outside",
        textfont=dict(size=10.5, color=MUTED),
        hovertemplate="%{y}: %{x:+.1f} pp<extra></extra>"))
    pad = max(2.0, 0.18 * (max(abs(v) for v in vals) if vals else 1))
    xr = (min(vals + [0]) - pad, max(vals + [0]) + pad)
    fig.add_vline(x=0, line_width=1, line_color="#9aa3a9")
    fig.update_layout(
        height=max(260, 30 * len(labels) + 70),
        margin=dict(l=20, r=30, t=10, b=34),
        plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=list(xr), showgrid=True, gridcolor="#f0f2f4", zeroline=False,
                   title=dict(text="Contribution to predicted risk (percentage points)",
                              font=dict(size=11, color=MUTED)),
                   tickfont=dict(size=10.5, color=MUTED)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11.5, color=INK)),
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif"))
    return fig


def make_cohort_plot(inputs, stats):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    feats = ["Age", "Hunt_Hess_grade", "temporary_clipping_duration",
             "MEP_change_time", "MEP_recovery_time", "SEP_change_time", "SEP_recovery_time"]
    feats = [f for f in feats if f in stats and {"p25", "p75", "median"} <= set(stats[f])]
    labels = [_short_label(f) for f in feats]
    fig = go.Figure()
    for i, f in enumerate(feats):
        s = stats[f]
        fig.add_trace(go.Scatter(
            x=[s["p25"], s["p75"]], y=[labels[i], labels[i]], mode="lines",
            line=dict(color="#c9d2d8", width=11), showlegend=(i == 0),
            name="Cohort IQR (25th–75th)",
            hovertemplate=f"IQR {s['p25']:.1f}–{s['p75']:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[stats[f]["median"] for f in feats], y=labels, mode="markers",
        marker=dict(color="white", size=13, symbol="line-ns", line=dict(width=2.5, color="#33414a")),
        name="Cohort median", hovertemplate="Median %{x:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[inputs[f] for f in feats], y=labels, mode="markers",
        marker=dict(color=ACCENT, size=13, symbol="diamond", line=dict(width=1.5, color="white")),
        name="This patient", hovertemplate="Patient %{x:.1f}<extra></extra>"))
    fig.update_layout(
        height=330, margin=dict(l=20, r=20, t=18, b=36), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f0f2f4", zeroline=False,
                   title=dict(text="Value", font=dict(size=11, color=MUTED)),
                   tickfont=dict(size=10.5, color=MUTED)),
        yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=11.5, color=INK)),
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif"))
    return fig


def render_strata_table(tertiles, rates, ns, active_idx):
    t1, t2 = tertiles[0] * 100, tertiles[1] * 100
    rows = [
        ("Low",          f"&lt; {t1:.1f}%",            rates[0], ns[0], LOW),
        ("Intermediate", f"{t1:.1f}% – {t2:.1f}%",     rates[1], ns[1], INT),
        ("High",         f"&ge; {t2:.1f}%",            rates[2], ns[2], HIGH),
    ]
    body = ""
    for i, (name, rng, rate, n, col) in enumerate(rows):
        rate_s = f"{rate*100:.1f}%" if rate == rate else "—"
        n_s = f"{n}" if n else "—"
        hi = "background:#f6f3ee;" if i == active_idx else ""
        mark = (f"<span style='color:{col};font-weight:700;'>{name}</span>"
                + (" &nbsp;&larr; this patient" if i == active_idx else ""))
        body += (f"<tr style='{hi}'><td>{mark}</td><td>{rng}</td>"
                 f"<td><b>{rate_s}</b></td><td>{n_s}</td></tr>")
    return f"""
    <div class="lx-card" style="padding:14px 18px;">
      <table class="lx-tbl">
        <tr><th>Risk stratum</th><th>Predicted probability</th>
            <th>Observed deficit rate</th><th>Patients (n)</th></tr>
        {body}
      </table>
      <div class="lx-note" style="margin-top:8px;">
        Strata are model-derived risk tertiles from the development cohort; observed deficit
        rate is the proportion with a postoperative motor deficit within each tertile.
      </div>
    </div>"""


# ════════════════════════════════════════════════════════════════════════════
# App body
# ════════════════════════════════════════════════════════════════════════════
try:
    MODEL, B = load_model()
except FileNotFoundError:
    st.error("model_bundle.joblib not found. Run `python export_model.py` first and place the "
             "file in the same directory as app.py (or in the same repository).")
    st.stop()
except Exception as e:
    st.error(f"Model failed to load: {type(e).__name__}: {e}")
    st.stop()

feat_names = B["feat_names"]
stats      = B["feat_stats"]
calib      = B["calib"]
tertiles   = B["tertiles"]
t_rates    = B.get("tertile_rates", [float("nan")] * 3)
t_ns       = B.get("tertile_n", [0, 0, 0])
perf       = B.get("performance", {}) or {}
prevalence = B.get("pos_rate", float("nan"))
n_train    = B.get("n_train", "—")
baseline_row = np.array([stats[n]["median"] for n in feat_names], dtype=float)

_auc = perf.get("AUC")
auc_disp = perf.get("AUC_95CI") or (f"{_auc:.2f}" if isinstance(_auc, float) and _auc == _auc else None)

# ── header ──────────────────────────────────────────────────────────────────
facts = [f'<span class="lx-fact">Model&nbsp;&nbsp;<b>{B.get("model_name","TabICLv2")}</b></span>']
if auc_disp:
    facts.append(f'<span class="lx-fact">Discrimination&nbsp;&nbsp;<b>AUC {auc_disp}</b></span>')
facts.append(f'<span class="lx-fact">Development cohort&nbsp;&nbsp;<b>n = {n_train}</b></span>')
facts.append('<span class="lx-fact">Internal validation&nbsp;&nbsp;<b>repeated 10×10-fold CV</b></span>')

st.markdown(f"""
<div class="lx-title">Postoperative Motor Deficit Risk after Intracranial Aneurysm Clipping</div>
<hr class="lx-rule"/>
<div class="lx-sub">An internally validated machine-learning tool estimating the probability of
postoperative motor deficit in patients with reversible intraoperative motor-evoked potential (MEP)
and/or somatosensory-evoked potential (SEP) deterioration, integrating intraoperative
electrophysiological recovery dynamics with clinical and operative variables.</div>
<div>{''.join(facts)}</div>
""", unsafe_allow_html=True)

# ── input ───────────────────────────────────────────────────────────────────
st.markdown('<div class="lx-eyebrow">Patient data</div>', unsafe_allow_html=True)
inputs = {}
with st.container(border=True):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Clinical and operative variables**")
        for name in CLINICAL_ORDER:
            if name in feat_names:
                inputs[name] = render_widget(name, stats)
    with c2:
        st.markdown("**Intraoperative neurophysiology**")
        for name in NEURO_ORDER:
            if name in feat_names:
                inputs[name] = render_widget(name, stats)
    for name in feat_names:           # robustness: anything not placed above
        if name not in inputs:
            inputs[name] = render_widget(name, stats)

_, bcol, _ = st.columns([1, 1.4, 1])
with bcol:
    go_pred = st.button("Estimate risk", type="primary", use_container_width=True)

# ── result ──────────────────────────────────────────────────────────────────
if go_pred:
    x = np.array([float(inputs[n]) for n in feat_names], dtype=float)
    p = float(predict_calibrated(MODEL, calib, x)[0])

    if p < tertiles[0]:
        stratum, scolor, sidx = "Low", LOW, 0
    elif p < tertiles[1]:
        stratum, scolor, sidx = "Intermediate", INT, 1
    else:
        stratum, scolor, sidx = "High", HIGH, 2
    obs_rate = t_rates[sidx] if sidx < len(t_rates) else float("nan")

    # out-of-range (extrapolation) check for continuous inputs
    oor = []
    for name in feat_names:
        ui = FEATURE_UI.get(name, {})
        if ui.get("kind") in ("int", "float") and {"min", "max"} <= set(stats.get(name, {})):
            v = inputs[name]; s = stats[name]
            if v < s["min"] or v > s["max"]:
                oor.append(f"{_short_label(name)} = {v} (range {s['min']:.1f}–{s['max']:.1f})")
    if oor:
        st.warning("One or more inputs lie outside the development range; the estimate is an "
                   "extrapolation and should be interpreted with additional caution: "
                   + "; ".join(oor) + ".")

    st.markdown('<div class="lx-eyebrow">Result</div>', unsafe_allow_html=True)
    gcol, rcol = st.columns([1.1, 1], gap="large")
    with gcol:
        fig = make_gauge(p, tertiles)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div class="lx-note">Shaded bands denote model-derived risk tertiles '
            f'(cut-points {tertiles[0]*100:.1f}% and {tertiles[1]*100:.1f}%). '
            f'Cohort outcome prevalence {prevalence*100:.1f}%.</div>', unsafe_allow_html=True)
    with rcol:
        rate_line = (f"Observed deficit rate in this stratum (development cohort): "
                     f"<b>{obs_rate*100:.1f}%</b>." if obs_rate == obs_rate else "")
        st.markdown(f"""
        <div class="lx-card" style="border-left:4px solid {scolor};">
            <div style="font-size:11px; font-weight:700; letter-spacing:1.3px;
                 text-transform:uppercase; color:{FAINT};">Risk stratum</div>
            <div style="font-family:'Source Serif 4',Georgia,serif; font-size:30px;
                 font-weight:700; color:{scolor}; margin:4px 0 14px 0;">{stratum}</div>
            <div style="display:flex; justify-content:space-between; align-items:baseline;
                 border-top:1px solid {RULE}; padding-top:12px;">
                <span style="color:{MUTED}; font-size:13px;">Estimated probability</span>
                <span style="font-weight:700; font-size:18px; color:{INK};">{p*100:.1f}%</span>
            </div>
            <div style="margin-top:12px; color:{MUTED}; font-size:12.5px; line-height:1.55;">
                {rate_line}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # risk-strata reference table (Fig 5G analogue)
    st.markdown('<div class="lx-eyebrow">Risk strata (development cohort)</div>', unsafe_allow_html=True)
    st.markdown(render_strata_table(tertiles, t_rates, t_ns, sidx), unsafe_allow_html=True)

    # per-patient exact Shapley contributions (Fig 6C analogue)
    st.markdown('<div class="lx-eyebrow">Individual variable contributions</div>', unsafe_allow_html=True)
    base = None
    try:
        with st.spinner("Computing individual variable contributions…"):
            base, phi = exact_shapley(MODEL, calib, x, baseline_row)
        sfig = make_shapley_plot(phi, feat_names, base, p)
        st.markdown(
            f'<div class="lx-note" style="margin-bottom:4px;">'
            f'Baseline risk for a cohort-median patient <b>{base*100:.1f}%</b> '
            f'&rarr; estimate for this patient <b>{p*100:.1f}%</b>. '
            f'Bars are exact Shapley contributions (percentage points); '
            f'<span style="color:{SHAP_UP};font-weight:600;">red increases</span> and '
            f'<span style="color:{SHAP_DOWN};font-weight:600;">blue decreases</span> predicted risk.'
            f'</div>', unsafe_allow_html=True)
        if sfig is not None:
            st.plotly_chart(sfig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.markdown(f'<div class="lx-note">Individual contributions unavailable ({type(e).__name__}).</div>',
                    unsafe_allow_html=True)

    # patient vs cohort
    st.markdown('<div class="lx-eyebrow">Patient profile versus development cohort</div>',
                unsafe_allow_html=True)
    cfig = make_cohort_plot(inputs, stats)
    if cfig is not None:
        st.plotly_chart(cfig, use_container_width=True, config={"displayModeBar": False})

    # interpretation (measured, aligned with the manuscript)
    st.markdown('<div class="lx-eyebrow">Interpretation</div>', unsafe_allow_html=True)
    if stratum == "High":
        body = ("The estimated probability falls in the high-risk tertile of the development cohort. "
                "In this study, longer MEP recovery time was the strongest model contributor, and the "
                "highest predicted risk occurred when delayed recovery coexisted with prolonged "
                "electrophysiological deterioration. A high estimate supports continued attention to "
                "signal stability and to modifiable surgical, vascular, haemodynamic, anaesthetic, and "
                "physiological contributors before the warning event is considered resolved, and may "
                "justify closer early postoperative neurological surveillance.")
    elif stratum == "Intermediate":
        body = ("The estimated probability falls in the intermediate-risk tertile. The estimate should be "
                "interpreted together with the speed of electrophysiological recovery after corrective "
                "manoeuvres and the overall clinical context, rather than in isolation. Continued attention "
                "to signal stability and to reversible contributors remains appropriate.")
    else:
        body = ("The estimated probability falls in the low-risk tertile. In this cohort, rapid MEP recovery "
                "was associated with low predicted risk even in the presence of longer deterioration "
                "duration. A low estimate does not remove the need for routine postoperative neurological "
                "observation and vigilance for any new deficit.")
    st.markdown(f'<div class="lx-card"><div class="lx-note" style="font-size:13.5px; color:{INK};">'
                f'{body}</div></div>', unsafe_allow_html=True)

    # technical details
    with st.expander("Methodological details and input summary"):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Probability derivation**")
            calib_lab = {"platt": "Platt scaling", "beta": "Beta calibration",
                         "temperature": "temperature scaling", "none": "none"}.get(
                            calib.get("type", "none"), calib.get("type", "none"))
            thr = B.get("threshold", float("nan"))
            lines = [
                f"- Calibrated probability ({calib_lab}): `{p:.4f}`",
                f"- Operating threshold (Gmean-optimised): `{thr:.4f}`",
                f"- Classification at threshold: `{'Any deficit' if p >= thr else 'No deficit'}`",
            ]
            if base is not None:
                lines.append(f"- Shapley baseline (cohort-median patient): `{base*100:.1f}%`")
            st.markdown("\n".join(lines))
        with d2:
            st.markdown("**Cross-validated performance (TabICLv2)**")
            if perf:
                def fmt(k):
                    v = perf.get(k)
                    return f"{v:.3f}" if isinstance(v, float) and v == v else "—"
                st.markdown(
                    f"- AUC: **{perf.get('AUC_95CI') or fmt('AUC')}**\n"
                    f"- Sensitivity / Specificity: **{fmt('Sensitivity')} / {fmt('Specificity')}**\n"
                    f"- PPV / NPV: **{fmt('PPV')} / {fmt('NPV')}**\n"
                    f"- Brier / ECE: {fmt('Brier')} / {fmt('ECE')}")
            else:
                st.markdown("_Performance metrics unavailable (pipeline cache not found at export)._")
        st.markdown("**Input summary**")
        st.dataframe(
            {"Variable": [_short_label(n) for n in feat_names],
             "Value": [inputs[n] for n in feat_names]},
            hide_index=True, use_container_width=True)

# ── disclaimer / footer ─────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(f"""
<div class="lx-note">
<b style="color:{INK};">Intended use.</b> This tool was developed from a single-centre, retrospective,
event-enriched cohort (n={n_train}) and is a research prototype for risk visualisation. It has not
undergone prospective external validation. Its output is an adjunct to—and not a substitute for—surgical
judgement, intraoperative vascular inspection, haemodynamic optimisation, and postoperative neurological
surveillance. It must not be used as a stand-alone basis for clinical decisions.
</div>
""", unsafe_allow_html=True)
st.caption(f"Model: {B.get('model_name','TabICLv2')} · probability calibration applied · "
           f"internally validated by repeated stratified cross-validation · research use only")
