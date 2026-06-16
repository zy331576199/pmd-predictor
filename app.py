# -*- coding: utf-8 -*-
"""
app.py — Postoperative Motor Deficit (PMD) Risk Predictor after Intracranial
         Aneurysm Clipping.  Web decision-support tool built on TabICLv2.

Run locally:   python -m streamlit run app.py   (env where tabicl is installed)
Deploy:        see README.md (GitHub -> Streamlit Community Cloud)

Requires:
  - model_bundle.joblib  (from export_model.py)  in the same folder
  - .streamlit/config.toml  (locks a light, print-grade theme)

Design: restrained, typographic, journal-grade, light theme. Research use only.
"""

import numpy as np
import joblib
import streamlit as st

BUNDLE_PATH = "model_bundle.joblib"

# ── design tokens ───────────────────────────────────────────────────────────
INK   = "#1b1b1b"
MUTED = "#5b6770"
FAINT = "#8a949c"
RULE  = "#e4e8ec"
PAPER = "#ffffff"
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

# ── global styling (the app also ships .streamlit/config.toml to force light) ─
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');

#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none; }}
[data-testid="stAppViewContainer"], .stApp {{ background: {PAPER}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3.5rem; max-width: 1060px; }}
html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: {INK};
}}

.lx-title  {{ font-family:'Source Serif 4', Georgia, serif; font-size:31px; font-weight:700;
             letter-spacing:-0.2px; line-height:1.16; color:{INK}; margin:0; }}
.lx-rule   {{ height:2px; width:64px; background:{ACCENT}; border:none; margin:15px 0 0 0; }}
.lx-sub    {{ font-size:14.5px; color:{MUTED}; line-height:1.55; margin-top:13px; max-width:830px; }}
.lx-facts  {{ display:flex; flex-wrap:wrap; gap:10px 28px; margin-top:18px; padding-top:14px;
             border-top:1px solid {RULE}; }}
.lx-fact   {{ font-size:12.5px; color:{MUTED}; }}
.lx-fact b {{ color:{INK}; font-weight:600; }}
.lx-eyebrow{{ font-size:11px; font-weight:700; letter-spacing:1.6px; text-transform:uppercase;
             color:{FAINT}; margin:32px 0 12px 0; }}
.lx-note   {{ font-size:12.5px; color:{MUTED}; line-height:1.55; }}
.lx-card   {{ background:{PAPER}; border:1px solid {RULE}; border-radius:10px; padding:20px 22px;
             box-shadow:0 1px 2px rgba(0,0,0,0.04), 0 8px 22px rgba(20,24,28,0.045); }}

table.lx-tbl {{ width:100%; border-collapse:collapse; font-size:13px; }}
table.lx-tbl th {{ text-align:left; color:{FAINT}; font-weight:700; font-size:10.5px;
                   letter-spacing:0.6px; text-transform:uppercase; padding:9px 12px;
                   border-bottom:1.5px solid {RULE}; }}
table.lx-tbl td {{ padding:10px 12px; border-bottom:1px solid {RULE}; color:{INK}; }}

div[data-testid="stContainer"] {{ border-radius:10px; }}
.stButton > button {{
    background:{ACCENT}; color:#ffffff; border:none; border-radius:7px;
    font-weight:600; font-size:15px; height:48px; letter-spacing:0.2px;
    box-shadow:0 2px 8px rgba(138,28,46,0.22);
}}
.stButton > button:hover {{ background:#71101f; color:#ffffff; box-shadow:0 4px 12px rgba(138,28,46,0.30); }}
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
    raw = model.predict_proba(np.atleast_2d(X).astype(float))[:, 1]
    return apply_calib(raw, calib)


# ── Shapley values: interventional, development-cohort background ───────────
# Antithetic permutation sampling. Every patient in X_bg is used as a reference
# (each once with a random feature order and once with the reverse order), so the
# base value equals the cohort mean prediction E[f(X)] exactly, every feature gets
# a distribution-based contribution (no feature is forced to zero just because it
# equals a single reference), and additivity holds exactly: base + Σφ == f(x).
def shapley_sampling(model, calib, x_row, X_bg, seed=42):
    n = len(x_row)
    x_row = np.asarray(x_row, dtype=float)
    X_bg = np.asarray(X_bg, dtype=float)
    nb = len(X_bg)
    rng = np.random.default_rng(seed)

    perms = []                                   # (background_index, feature_order)
    for i in range(nb):
        pi = rng.permutation(n)
        perms.append((i, pi))
        perms.append((i, pi[::-1].copy()))       # antithetic partner (variance reduction)
    m = len(perms)                               # = 2 * nb

    # Build all coalition states (background -> patient, one feature at a time)
    M = np.empty((m * (n + 1), n), dtype=float)
    for idx, (bi, pi) in enumerate(perms):
        state = X_bg[bi].copy()
        r0 = idx * (n + 1)
        M[r0] = state
        for k in range(n):
            state[pi[k]] = x_row[pi[k]]
            M[r0 + k + 1] = state

    cal = predict_calibrated(model, calib, M).reshape(m, n + 1)
    base = float(cal[:, 0].mean())               # == E[f(X)] over the cohort
    phi = np.zeros(n)
    for idx, (bi, pi) in enumerate(perms):
        phi[pi] += cal[idx, 1:] - cal[idx, :-1]  # telescoping marginals per feature
    phi /= m
    return base, phi                             # base + phi.sum() == f(x_row)



# ── feature presentation ────────────────────────────────────────────────────
FEATURE_UI = {
    "Age": dict(label="Age", short="Age", unit="years", kind="int",
                help="Patient age in years."),
    "Gender": dict(label="Sex", short="Sex", kind="binary",
                   options=[("Female", 0), ("Male", 1)]),
    "Hunt_Hess_grade": dict(label="Hunt–Hess grade", short="Hunt–Hess grade", kind="grade",
                            help="0 for unruptured aneurysms; ruptured aneurysms graded within "
                                 "the range observed in the development cohort."),
    "Aneurysm_rupture": dict(label="Aneurysm rupture", short="Rupture", kind="binary",
                             options=[("No", 0), ("Yes", 1)]),
    "NLA_on_CT": dict(label="New low-attenuation area on postoperative CT", short="NLA on CT",
                      kind="binary", options=[("No", 0), ("Yes", 1)]),
    "temporary_clipping_duration": dict(label="Temporary clipping duration", short="Temporary clipping",
                                        unit="min", kind="float",
                                        help="Enter 0 if no temporary clipping was performed."),
    "MEP_change_time": dict(label="MEP deterioration duration", short="MEP deterioration",
                            unit="min", kind="float",
                            help="Interval of MEP amplitude reduction >50% until recovery >50% of baseline. "
                                 "Enter 0 if no MEP deterioration occurred."),
    "SEP_change_time": dict(label="SEP deterioration duration", short="SEP deterioration",
                            unit="min", kind="float",
                            help="SEP deterioration duration. Enter 0 if no SEP deterioration occurred."),
    "MEP_recovery_time": dict(label="MEP recovery time", short="MEP recovery", unit="min", kind="float",
                              help="Interval from the corrective manoeuvre to MEP recovery >50% of baseline. "
                                   "Enter 0 if not applicable."),
    "SEP_recovery_time": dict(label="SEP recovery time", short="SEP recovery", unit="min", kind="float",
                              help="Interval from the corrective manoeuvre to SEP recovery >50% of baseline. "
                                   "Enter 0 if not applicable."),
}

CLINICAL_ORDER = ["Age", "Gender", "Hunt_Hess_grade", "Aneurysm_rupture",
                  "NLA_on_CT", "temporary_clipping_duration"]
NEURO_ORDER    = ["MEP_change_time", "MEP_recovery_time",
                  "SEP_change_time", "SEP_recovery_time"]


def _long_label(name):
    ui = FEATURE_UI.get(name, {"label": name})
    lab = ui["label"]
    if ui.get("unit"):
        lab += f" ({ui['unit']})"
    return lab


def _short_label(name):
    ui = FEATURE_UI.get(name, {"short": name})
    lab = ui.get("short", ui.get("label", name))
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
        gmin = int(np.floor(s.get("min", 0))); gmax = int(np.ceil(s.get("max", 3)))
        grades = list(range(gmin, gmax + 1)) or [0]
        default = int(round(s.get("median", grades[0])))
        if default not in grades:
            default = grades[0]
        return st.select_slider(ui["label"], options=grades, value=default, help=helptext)
    if kind == "int":
        lo = int(np.floor(s.get("min", 0))); hi = int(np.ceil(s.get("max", 120)))
        return st.number_input(_long_label(name), min_value=lo, max_value=hi,
                               value=int(round(s.get("median", lo))), step=1, help=helptext)
    hi = float(s.get("max", 60.0))
    return st.number_input(_long_label(name), min_value=0.0, max_value=max(hi * 2, hi, 1.0),
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
        number={"suffix": "%", "font": {"size": 46, "color": INK, "family": "Source Serif 4, Georgia, serif"}},
        title={"text": "Estimated probability of postoperative motor deficit",
               "font": {"size": 13, "color": MUTED}},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%", "tickwidth": 1,
                        "tickcolor": "#c2cad0", "tickfont": {"size": 10.5, "color": MUTED}},
               "bar": {"color": "#33414a", "thickness": 0.30},
               "bgcolor": PAPER, "borderwidth": 0,
               "steps": [{"range": [0, lo], "color": "#e3efe7"},
                         {"range": [lo, hi], "color": "#f5edd6"},
                         {"range": [hi, 100], "color": "#f6dfe2"}]}))
    fig.update_layout(height=280, margin=dict(l=26, r=26, t=48, b=12),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font={"family": "-apple-system, Segoe UI, Roboto, sans-serif"})
    return fig


def make_shapley_plot(phi, names):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    order = np.argsort(np.abs(phi))                  # ascending -> largest at top (reversed y)
    labels = [_short_label(names[i]) for i in order]
    vals = [float(phi[i]) * 100 for i in order]
    colors = [SHAP_UP if v > 0 else SHAP_DOWN for v in vals]
    text = [f"{v:+.1f}" if abs(v) >= 0.1 else "" for v in vals]    # suppress near-zero clutter
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=text, textposition="outside", textfont=dict(size=10.5, color=MUTED),
        cliponaxis=False, hovertemplate="%{y}: %{x:+.2f} pp<extra></extra>"))
    span = max((abs(v) for v in vals), default=1.0)
    pad = max(1.5, 0.22 * span)
    fig.add_vline(x=0, line_width=1, line_color="#aab2b8")
    fig.update_layout(
        height=max(280, 30 * len(labels) + 64),
        margin=dict(l=10, r=34, t=8, b=34),
        plot_bgcolor=PAPER, paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[min(vals + [0]) - pad, max(vals + [0]) + pad],
                   showgrid=True, gridcolor="#eef1f3", zeroline=False,
                   title=dict(text="Contribution to predicted risk (percentage points)",
                              font=dict(size=11, color=MUTED)),
                   tickfont=dict(size=10.5, color=MUTED)),
        yaxis=dict(showgrid=False, automargin=True, tickfont=dict(size=11.5, color=INK)),
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
            line=dict(color="#cdd6dc", width=11), showlegend=(i == 0),
            name="Cohort IQR (25th–75th)",
            hovertemplate=f"IQR {s['p25']:.1f}–{s['p75']:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[stats[f]["median"] for f in feats], y=labels, mode="markers",
        marker=dict(color=PAPER, size=13, symbol="line-ns", line=dict(width=2.5, color="#33414a")),
        name="Cohort median", hovertemplate="Median %{x:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[inputs[f] for f in feats], y=labels, mode="markers",
        marker=dict(color=ACCENT, size=13, symbol="diamond", line=dict(width=1.5, color=PAPER)),
        name="This patient", hovertemplate="Patient %{x:.1f}<extra></extra>"))
    fig.update_layout(
        height=330, margin=dict(l=10, r=20, t=18, b=36), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor=PAPER, paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#eef1f3", zeroline=False,
                   title=dict(text="Value", font=dict(size=11, color=MUTED)),
                   tickfont=dict(size=10.5, color=MUTED)),
        yaxis=dict(autorange="reversed", showgrid=False, automargin=True,
                   tickfont=dict(size=11.5, color=INK)),
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif"))
    return fig


def render_strata_table(tertiles, rates, ns, active_idx):
    t1, t2 = tertiles[0] * 100, tertiles[1] * 100
    rows = [
        ("Low",          f"&lt; {t1:.1f}%",         rates[0], ns[0], LOW),
        ("Intermediate", f"{t1:.1f}% – {t2:.1f}%",  rates[1], ns[1], INT),
        ("High",         f"&ge; {t2:.1f}%",         rates[2], ns[2], HIGH),
    ]
    body = ""
    for i, (name, rng, rate, n, col) in enumerate(rows):
        rate_s = f"{rate*100:.1f}%" if rate == rate else "—"
        n_s = f"{n}" if n else "—"
        hi = "background:#faf6ef;" if i == active_idx else ""
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
      <div class="lx-note" style="margin-top:9px;">
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
X_bg       = np.asarray(B["X_train"], dtype=float)     # development-cohort SHAP background
shap_seed  = int(B.get("seed", 42))

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
<div class="lx-facts">{''.join(facts)}</div>
""", unsafe_allow_html=True)

# ── input ───────────────────────────────────────────────────────────────────
st.markdown('<div class="lx-eyebrow">Patient data</div>', unsafe_allow_html=True)
inputs = {}
with st.container(border=True):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Clinical and operative variables**")
        inputs["Age"] = render_widget("Age", stats)
        inputs["Gender"] = render_widget("Gender", stats)
        # Aneurysm rupture is asked first because Hunt–Hess grade depends on it:
        #   unruptured -> grade is 0 by definition; ruptured -> grade 1–3 (cohort range).
        inputs["Aneurysm_rupture"] = render_widget("Aneurysm_rupture", stats)
        _hh_max = int(np.ceil(stats.get("Hunt_Hess_grade", {}).get("max", 3)))
        if inputs["Aneurysm_rupture"] == 0:
            st.selectbox("Hunt–Hess grade", options=[0], index=0, disabled=True, key="hh_unrup",
                         help="Unruptured aneurysm: Hunt–Hess grade is 0 by definition.")
            inputs["Hunt_Hess_grade"] = 0
        else:
            _rg = list(range(1, max(_hh_max, 1) + 1))
            inputs["Hunt_Hess_grade"] = st.select_slider(
                "Hunt–Hess grade", options=_rg, value=_rg[0], key="hh_rup",
                help="Ruptured aneurysm: Hunt–Hess grade (range observed in the development cohort).")
        inputs["NLA_on_CT"] = render_widget("NLA_on_CT", stats)
        inputs["temporary_clipping_duration"] = render_widget("temporary_clipping_duration", stats)
    with c2:
        st.markdown("**Intraoperative neurophysiology**")
        for name in NEURO_ORDER:
            if name in feat_names:
                inputs[name] = render_widget(name, stats)
    for name in feat_names:
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
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False, "responsive": True})
        st.markdown(
            f'<div class="lx-note">Shaded bands denote model-derived risk tertiles '
            f'(cut-points {tertiles[0]*100:.1f}% and {tertiles[1]*100:.1f}%). '
            f'Cohort outcome prevalence {prevalence*100:.1f}%.</div>', unsafe_allow_html=True)
    with rcol:
        rate_line = (f"Observed deficit rate in this stratum (development cohort): "
                     f"<b>{obs_rate*100:.1f}%</b>." if obs_rate == obs_rate else "")
        st.markdown(f"""
        <div class="lx-card" style="border-top:3px solid {scolor};">
            <div style="font-size:11px; font-weight:700; letter-spacing:1.3px;
                 text-transform:uppercase; color:{FAINT};">Risk stratum</div>
            <div style="font-family:'Source Serif 4',Georgia,serif; font-size:32px;
                 font-weight:700; color:{scolor}; margin:4px 0 16px 0;">{stratum}</div>
            <div style="display:flex; justify-content:space-between; align-items:baseline;
                 border-top:1px solid {RULE}; padding-top:13px;">
                <span style="color:{MUTED}; font-size:13px;">Estimated probability</span>
                <span style="font-weight:700; font-size:19px; color:{INK};">{p*100:.1f}%</span>
            </div>
            <div style="margin-top:12px; color:{MUTED}; font-size:12.5px; line-height:1.55;">
                {rate_line}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="lx-eyebrow">Risk strata (development cohort)</div>', unsafe_allow_html=True)
    st.markdown(render_strata_table(tertiles, t_rates, t_ns, sidx), unsafe_allow_html=True)

    st.markdown('<div class="lx-eyebrow">Individual variable contributions</div>', unsafe_allow_html=True)
    base = None
    try:
        with st.spinner("Computing individual variable contributions…"):
            base, phi = shapley_sampling(MODEL, calib, x, X_bg, seed=shap_seed)
        _desc = np.argsort(np.abs(phi))[::-1]
        _bits = []
        for _i in _desc:
            _v = float(phi[_i]) * 100
            if abs(_v) < 0.1:
                continue
            _bits.append(f"{_short_label(feat_names[_i])} {_v:+.1f}")
            if len(_bits) >= 4:
                break
        _top = "; ".join(_bits) if _bits else "all variables near the reference"
        st.markdown(
            f'<div class="lx-note" style="margin-bottom:4px;">'
            f'Reference: the mean prediction across the development cohort (base value '
            f'<b>{base*100:.1f}%</b>) &rarr; this patient <b>{p*100:.1f}%</b>. '
            f'Bars are Shapley contributions (percentage points); '
            f'<span style="color:{SHAP_UP};font-weight:600;">red increases</span> and '
            f'<span style="color:{SHAP_DOWN};font-weight:600;">blue decreases</span> predicted risk. '
            f'<br>Largest contributions (pp): {_top}.'
            f'</div>', unsafe_allow_html=True)
        sfig = make_shapley_plot(phi, feat_names)
        if sfig is not None:
            st.plotly_chart(sfig, use_container_width=True,
                            config={"displayModeBar": False, "responsive": True})
    except Exception as e:
        st.markdown(f'<div class="lx-note">Individual contributions unavailable ({type(e).__name__}).</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="lx-eyebrow">Patient profile versus development cohort</div>',
                unsafe_allow_html=True)
    cfig = make_cohort_plot(inputs, stats)
    if cfig is not None:
        st.plotly_chart(cfig, use_container_width=True,
                        config={"displayModeBar": False, "responsive": True})

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
                lines.append(f"- Shapley base value (cohort mean prediction): `{base*100:.1f}%`")
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
            {"Variable": [_long_label(n) for n in feat_names],
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
