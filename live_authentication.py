"""Real-time authentication command center.

Streams simulated fingerprint verification attempts (a Poisson arrival process),
scores them against a live decision threshold and shows rolling operating
metrics, a live Bayesian posterior, and statistical anomaly alerts.

No fingerprint hardware is read: every score is generated from the same
Gaussian models that drive the rest of the project.
"""
from datetime import datetime
import base64
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.theme import ACCENT, GENUINE, IMPOSTOR, INK, LINE, MUTED, PANEL, TEXT, WARN
from utils.ui import page_header

# Reference score models (same as data/dataset_generator.py)
GEN_MU, GEN_SD = 0.82, 0.09
IMP_MU, IMP_SD = 0.42, 0.13

MAX_HISTORY = 1000
TICK_SECONDS = 1.0
MAX_EVENTS_PER_TICK = 25
ROLLING_WINDOW = 40

SCENARIOS = {
    "Normal operations": "Baseline genuine and impostor score models.",
    "Spoofing attack": "Impostor scores shift upward and impostor traffic surges.",
    "Sensor degradation": "Genuine scores decay exponentially as the sensor wears out.",
}
TARGET_FAR = {"10%": 0.10, "5%": 0.05, "1%": 0.01, "0.1%": 0.001}

LIVE_CSS = f"""
<style>
.scan-card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:18px; padding:18px 16px 14px; text-align:center; }}
.scan-card .verdict {{ font-size:1.4rem; font-weight:800; margin-top:8px; }}
.scan-card .sub {{ color:{MUTED}; font-size:.85rem; margin-top:2px; }}
.state-idle {{ --c:{ACCENT}; }} .state-ok {{ --c:{GENUINE}; }} .state-bad {{ --c:{IMPOSTOR}; }}
.scan-card .verdict {{ color:var(--c); }}
.beam-wrap {{ position:relative; width:190px; max-width:100%; margin:0 auto; overflow:hidden; }}
.beam-wrap img {{ width:100%; display:block; opacity:.9; }}
.beam {{ position:absolute; left:0; right:0; top:0; height:3px; background:var(--c); box-shadow:0 0 14px var(--c); border-radius:2px; }}
.running .beam {{ animation:sweep 1s linear infinite; }}
.paused .beam {{ display:none; }}
@keyframes sweep {{ from {{ top:2%; }} to {{ top:97%; }} }}
@media (prefers-reduced-motion: reduce) {{ .running .beam {{ animation:none; top:50%; }} }}
.alert-banner {{ border:1px solid {WARN}; background:rgba(251,191,36,.10); color:#FDE68A; border-radius:12px;
  padding:10px 14px; margin:0 0 8px; font-size:.93rem; }}
.alert-banner b {{ color:{WARN}; }}
.ok-banner {{ border:1px solid #1d5e57; background:rgba(45,212,191,.07); color:#99F0E4; border-radius:12px;
  padding:10px 14px; margin:0 0 8px; font-size:.93rem; }}
.quick-read {{ background:{PANEL}; border:1px solid {LINE}; border-left:4px solid {ACCENT}; border-radius:12px;
  padding:12px 14px; margin:8px 0 10px; color:{TEXT}; line-height:1.55; }}
.quick-read b {{ color:{ACCENT}; display:block; margin-bottom:3px; }}
.evlog {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; padding:6px 10px; }}
.ev {{ display:grid; grid-template-columns:70px 60px 60px 110px 1fr; gap:8px; padding:7px 4px;
  border-bottom:1px solid {LINE}; font-size:.88rem; font-variant-numeric:tabular-nums; align-items:center; }}
.ev:last-child {{ border-bottom:none; }}
.ev .t {{ color:{MUTED}; }}
.tag {{ border-radius:999px; padding:2px 10px; font-size:.75rem; font-weight:700; text-align:center; }}
.tag.acc {{ background:rgba(45,212,191,.15); color:{GENUINE}; }}
.tag.rej {{ background:rgba(141,162,195,.15); color:{MUTED}; }}
.flag {{ font-size:.78rem; font-weight:700; }}
.flag.fa {{ color:{IMPOSTOR}; }} .flag.fr {{ color:{WARN}; }}
</style>
"""


def _html(markup):
    """Render trusted HTML (st.html when available, st.markdown otherwise)."""
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
def _initialize_state():
    defaults = {
        "live_running": False,
        "live_attempts": [],
        "live_sequence": 0,
        "live_last_tick": 0.0,
        "live_rng": np.random.default_rng(),
        "live_totals": {"TP": 0, "FN": 0, "FP": 0, "TN": 0},
        "live_alerts": [],
        "live_active_alerts": [],
        "live_scenario": "Normal operations",
        "live_rate": 3,
        "live_genuine_mix": 80,
        "live_window": 100,
        "live_user": "U001",
        "live_target_far": "1%",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "live_threshold" not in st.session_state:
        from utils.ui import active_threshold
        st.session_state.live_threshold = float(min(max(active_threshold(), 0.05), 0.99))


def _reset_stream():
    st.session_state.live_attempts = []
    st.session_state.live_sequence = 0
    st.session_state.live_totals = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    st.session_state.live_alerts = []
    st.session_state.live_active_alerts = []


def _auto_tune():
    """Set the threshold that gives the chosen false-acceptance rate under the impostor model."""
    far = TARGET_FAR[st.session_state.live_target_far]
    st.session_state.live_threshold = float(np.clip(round(stats.norm.ppf(1 - far, IMP_MU, IMP_SD), 2), 0.05, 0.99))


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def _draw_attempt(rng, scenario, sequence, genuine_rate):
    """Draw (actual_class, score) for one attempt under the active scenario."""
    if scenario == "Spoofing attack":
        genuine_rate = min(genuine_rate, 0.35)
    is_genuine = bool(rng.random() < genuine_rate)
    if is_genuine:
        mu, sd = GEN_MU, GEN_SD
        if scenario == "Sensor degradation":
            decay = 1 - np.exp(-sequence / 150.0)  # exponential signal decay
            mu, sd = GEN_MU - 0.22 * decay, GEN_SD + 0.04 * decay
        return "Genuine", float(np.clip(rng.normal(mu, sd), 0.05, 1.0))
    mu, sd = (0.66, 0.11) if scenario == "Spoofing attack" else (IMP_MU, IMP_SD)
    return "Impostor", float(np.clip(rng.normal(mu, sd), 0.03, 0.97))


def _record_attempt(threshold, scenario, genuine_rate, user):
    rng = st.session_state.live_rng
    st.session_state.live_sequence += 1
    actual, score = _draw_attempt(rng, scenario, st.session_state.live_sequence, genuine_rate)
    accepted = score >= threshold
    outcome = {
        ("Genuine", True): "TP", ("Genuine", False): "FN",
        ("Impostor", True): "FP", ("Impostor", False): "TN",
    }[(actual, accepted)]
    st.session_state.live_totals[outcome] += 1
    st.session_state.live_attempts.append({
        "Attempt": st.session_state.live_sequence,
        "Time": datetime.now().astimezone().strftime("%H:%M:%S"),
        "User": user,
        "Actual": actual,
        "Score": round(score, 4),
        "Threshold": round(float(threshold), 4),
        "Decision": "Accepted" if accepted else "Rejected",
        "Outcome": outcome,
    })
    del st.session_state.live_attempts[:-MAX_HISTORY]


def _expected_rates(threshold):
    return 1 - stats.norm.cdf(threshold, GEN_MU, GEN_SD), 1 - stats.norm.cdf(threshold, IMP_MU, IMP_SD)


def _update_alerts(frame, threshold):
    """Raise alerts when recent behaviour is statistically unlikely under the baseline model."""
    exp_tar, exp_far = _expected_rates(threshold)
    recent = frame.tail(ROLLING_WINDOW)
    active, messages = [], {}

    impostors = recent[recent.Actual == "Impostor"]
    false_accepts = int((impostors.Outcome == "FP").sum())
    if len(impostors) >= 6 and false_accepts >= 2:
        lam = max(exp_far, 0.002) * len(impostors)
        p_tail = float(stats.poisson.sf(false_accepts - 1, lam))  # P(X >= k), X ~ Poisson(lambda)
        if p_tail < 0.01:
            active.append("burst")
            messages["burst"] = ("Possible spoofing burst",
                                 f"{false_accepts} false accepts in the last {len(recent)} attempts. "
                                 f"Poisson tail P(X ≥ {false_accepts}) {'< 0.0001' if p_tail < 1e-4 else '= ' + format(p_tail, '.4f')} at λ = {lam:.2f}.")

    genuines = recent[recent.Actual == "Genuine"]
    if len(genuines) >= 10:
        tar_now = float((genuines.Outcome == "TP").mean())
        if tar_now < exp_tar - 0.15:
            active.append("drift")
            messages["drift"] = ("Possible sensor drift",
                                 f"Genuine acceptance is {tar_now:.0%} against {exp_tar:.0%} expected at this threshold.")

    for key in active:
        if key not in st.session_state.live_active_alerts:  # newly raised only
            title, detail = messages[key]
            st.session_state.live_alerts.append({"Time": datetime.now().astimezone().strftime("%H:%M:%S"),
                                                 "Alert": title, "Detail": detail})
    st.session_state.live_active_alerts = active
    return {k: messages[k] for k in active}


# ----------------------------------------------------------------------------
# Visual components
# ----------------------------------------------------------------------------
def _fingerprint_data_uri(color):
    """Fingerprint ridge artwork as an image data URI (inline SVG is stripped by Streamlit's HTML sanitizer)."""
    ridges = []
    for i, radius in enumerate(range(8, 100, 7)):
        dash = f"{34 + (i * 13) % 46} {5 + (i * 7) % 9}"
        ridges.append(f'<ellipse cx="100" cy="122" rx="{radius * 0.86:.1f}" ry="{radius * 1.08:.1f}" fill="none" '
                      f'stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-dasharray="{dash}" '
                      f'transform="rotate({(i * 11) % 40 - 20} 100 122)"/>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 244">{"".join(ridges)}</svg>'
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def _scanner_markup(last, running):
    """Animated fingerprint scanner whose colour reflects the latest decision."""
    if last is None:
        state, color, verdict, sub = "state-idle", ACCENT, "Ready", "Start the stream to scan"
    else:
        accepted = last["Decision"] == "Accepted"
        state, color = ("state-ok", GENUINE) if accepted else ("state-bad", IMPOSTOR)
        verdict = f"{'Accepted' if accepted else 'Rejected'} · {last['Score']:.3f}"
        sub = f"Attempt #{last['Attempt']} · {last['User']} · {last['Actual'].lower()}"
    mode = "running" if running else "paused"
    return (f'<div class="scan-card {state} {mode}"><div class="beam-wrap">'
            f'<img alt="Fingerprint scanner" src="{_fingerprint_data_uri(color)}" style="filter:drop-shadow(0 0 8px {color})"/>'
            f'<div class="beam"></div></div>'
            f'<div class="verdict">{verdict}</div><div class="sub">{sub}</div></div>')


def _event_log_markup(frame, count=8):
    rows = []
    for row in frame.iloc[::-1].head(count).itertuples():
        tag = '<span class="tag acc">ACCEPT</span>' if row.Decision == "Accepted" else '<span class="tag rej">REJECT</span>'
        flag = {"FP": '<span class="flag fa">False accept: impostor got in</span>',
                "FN": '<span class="flag fr">False reject: genuine user blocked</span>'}.get(row.Outcome, "")
        rows.append(f'<div class="ev"><span class="t">{row.Time}</span><span>{row.User}</span>'
                    f'<span>{row.Score:.3f}</span>{tag}{flag}</div>')
    return f'<div class="evlog">{"".join(rows)}</div>'


def _style_figure(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Plus Jakarta Sans, sans-serif"), legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
        title=dict(font=dict(size=15)), hoverlabel=dict(bgcolor=INK),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


def _stream_chart(frame, threshold):
    view = frame.tail(150)
    fig = go.Figure()
    fig.add_hrect(y0=threshold, y1=1, fillcolor=GENUINE, opacity=0.07, line_width=0)
    for label, color in (("Genuine", GENUINE), ("Impostor", IMPOSTOR)):
        part = view[view.Actual == label]
        fig.add_trace(go.Scatter(x=part.Attempt, y=part.Score, mode="markers", name=label,
                                 marker=dict(color=color, size=7, opacity=0.85)))
    errors = view[view.Outcome.isin(["FP", "FN"])]
    fig.add_trace(go.Scatter(x=errors.Attempt, y=errors.Score, mode="markers", name="Wrong decision",
                             marker=dict(symbol="circle-open", size=14, color=WARN, line=dict(width=2))))
    fig.add_hline(y=threshold, line=dict(color=TEXT, dash="dash", width=1.5),
                  annotation_text=f"threshold {threshold:.2f}", annotation_font_color=MUTED)
    fig.update_yaxes(range=[0, 1], title="Similarity score")
    fig.update_xaxes(title="Attempt")
    fig.update_layout(title="Live score stream (last 150 attempts)")
    return _style_figure(fig, 340)


def _rolling_chart(frame):
    view = frame.tail(300).reset_index(drop=True)
    genuine, accepted = view.Actual.eq("Genuine"), view.Decision.eq("Accepted")
    roll = lambda s: s.astype(int).rolling(ROLLING_WINDOW, min_periods=10).sum()
    tar = roll(genuine & accepted) / roll(genuine).replace(0, np.nan)
    far = roll(~genuine & accepted) / roll(~genuine).replace(0, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.Attempt, y=tar, name="TAR", line=dict(color=GENUINE, width=2.5)))
    fig.add_trace(go.Scatter(x=view.Attempt, y=1 - tar, name="FRR", line=dict(color=WARN, width=2)))
    fig.add_trace(go.Scatter(x=view.Attempt, y=far, name="FAR", line=dict(color=IMPOSTOR, width=2.5)))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_xaxes(title="Attempt")
    fig.update_layout(title=f"Rolling TAR / FRR / FAR ({ROLLING_WINDOW}-attempt window)")
    return _style_figure(fig, 300)


def _histogram_chart(window, threshold):
    fig = go.Figure()
    for label, color in (("Genuine", GENUINE), ("Impostor", IMPOSTOR)):
        scores = window.loc[window.Actual == label, "Score"]
        fig.add_trace(go.Histogram(x=scores, name=label, marker_color=color, opacity=0.65,
                                   xbins=dict(start=0, end=1, size=0.04), histnorm="probability density"))
    x = np.linspace(0, 1, 200)
    fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, GEN_MU, GEN_SD), name="Genuine model",
                             line=dict(color=GENUINE, dash="dot", width=1.5)))
    fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, IMP_MU, IMP_SD), name="Impostor model",
                             line=dict(color=IMPOSTOR, dash="dot", width=1.5)))
    fig.add_vline(x=threshold, line=dict(color=TEXT, dash="dash", width=1.5))
    fig.update_layout(barmode="overlay", title="Live score distribution vs baseline model")
    fig.update_xaxes(range=[0, 1], title="Similarity score")
    return _style_figure(fig, 300)


def _confusion_chart(tp, fn, fp, tn):
    z = [[tp, fn], [fp, tn]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Accepted", "Rejected"], y=["Genuine", "Impostor"], showscale=False,
        colorscale=[[0, PANEL], [1, "#1f8f8a"]], text=[[str(v) for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=22, color=TEXT), xgap=3, ygap=3,
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Confusion matrix (window)")
    return _style_figure(fig, 340)


# ----------------------------------------------------------------------------
# Live view (re-runs every second while the stream is active)
# ----------------------------------------------------------------------------
def _live_view(threshold, scenario, genuine_rate, rate, window_size, user):
    running = st.session_state.live_running
    now = time.monotonic()
    if running and now - st.session_state.live_last_tick >= 0.9 * TICK_SECONDS:
        st.session_state.live_last_tick = now
        arrivals = int(min(st.session_state.live_rng.poisson(rate * TICK_SECONDS), MAX_EVENTS_PER_TICK))
        for _ in range(arrivals):  # Poisson arrival process
            _record_attempt(threshold, scenario, genuine_rate, user)

    attempts = st.session_state.live_attempts
    scanner_col, metric_col = st.columns([1, 2.6])
    if not attempts:
        with scanner_col:
            _html(_scanner_markup(None, running))
        with metric_col:
            st.info("Press **Start stream** to begin receiving simulated fingerprint attempts.")
        return

    frame = pd.DataFrame(attempts)
    window = frame.tail(window_size)
    counts = window.Outcome.value_counts()
    tp, fn, fp, tn = (int(counts.get(k, 0)) for k in ("TP", "FN", "FP", "TN"))
    genuine_n, impostor_n = tp + fn, fp + tn
    exp_tar, exp_far = _expected_rates(threshold)
    active_alerts = _update_alerts(frame, threshold)

    with scanner_col:
        _html(_scanner_markup(attempts[-1], running))
    with metric_col:
        row1 = st.columns(4)
        tar = tp / genuine_n if genuine_n else None
        far = fp / impostor_n if impostor_n else None
        row1[0].metric("TAR", f"{tar:.1%}" if tar is not None else "n/a", f"{(tar - exp_tar) * 100:+.1f} pts" if tar is not None else None,
                       help="True acceptance rate. The change shown is against the baseline model at this threshold.")
        row1[1].metric("FAR", f"{far:.1%}" if far is not None else "n/a",
                       f"{(far - exp_far) * 100:+.1f} pts" if far is not None else None, delta_color="inverse",
                       help="False acceptance rate: impostors let in. The change shown is against the baseline model.")
        row1[2].metric("FRR", f"{fn / genuine_n:.1%}" if genuine_n else "n/a")
        row1[3].metric("Accuracy", f"{(tp + tn) / len(window):.1%}")
        row2 = st.columns(4)
        row2[0].metric("Posterior", f"{tp / (tp + fp):.1%}" if tp + fp else "n/a",
                       help="Live Bayesian posterior P(Genuine | Accepted): the share of accepted attempts that were genuine.")
        row2[1].metric("Attempts", f"{st.session_state.live_sequence:,}")
        row2[2].metric("Throughput", f"{rate} / sec" if running else "paused")
        row2[3].metric("Alerts", len(active_alerts))
        if genuine_n < 8 or impostor_n < 5:
            quick_read = (
                f"There are {genuine_n} genuine and {impostor_n} impostor attempts in this window. "
                "That is a small sample, so FAR and FRR can change a lot as more attempts arrive."
            )
        else:
            quick_read = (
                f"Of {genuine_n} genuine attempts, {tp} were accepted and {fn} were rejected "
                f"(TAR {tp / genuine_n:.1%}). Of {impostor_n} impostor attempts, {fp} were mistakenly accepted "
                f"(FAR {fp / impostor_n:.1%})."
            )
        st.markdown(f'<div class="quick-read"><b>Quick read</b>{quick_read}</div>', unsafe_allow_html=True)
        if active_alerts:
            for title, detail in active_alerts.values():
                st.markdown(f'<div class="alert-banner"><b>{title}.</b> {detail}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-banner">No statistical anomalies in the recent attempts.</div>',
                        unsafe_allow_html=True)

    chart_left, chart_right = st.columns([2.2, 1])
    chart_left.plotly_chart(_stream_chart(frame, threshold), width="stretch", key="live_stream_chart")
    chart_right.plotly_chart(_confusion_chart(tp, fn, fp, tn), width="stretch", key="live_confusion_chart")
    lower_left, lower_right = st.columns(2)
    lower_left.plotly_chart(_rolling_chart(frame), width="stretch", key="live_rolling_chart")
    lower_right.plotly_chart(_histogram_chart(window, threshold), width="stretch", key="live_hist_chart")

    log_col, alert_col = st.columns([1.5, 1])
    with log_col:
        st.markdown("#### Event feed")
        _html(_event_log_markup(frame))
    with alert_col:
        st.markdown("#### Alert history")
        if st.session_state.live_alerts:
            st.dataframe(pd.DataFrame(st.session_state.live_alerts[-6:][::-1]), hide_index=True,
                         width="stretch", height=250)
        else:
            st.caption("Alerts appear here when false accepts or score drift become statistically unusual.")


def render_live_authentication():
    """Render the real-time simulated authentication command center."""
    _initialize_state()
    st.markdown(LIVE_CSS, unsafe_allow_html=True)
    page_header("Thresholds & decisions · Real time", "Live Command Center",
                "Simulated attempts arrive as a Poisson process and are scored against your threshold in real time, with a live Bayesian "
                "posterior and Poisson-tail anomaly alerts. No fingerprint scanner is read and no real person is authenticated.", icon="⚡")

    buttons = st.columns([1.2, 1.2, 1.2, 3])
    if buttons[0].button("Start stream", type="primary", width="stretch"):
        st.session_state.live_running = True
        st.session_state.live_last_tick = 0.0
    if buttons[1].button("Pause", width="stretch"):
        st.session_state.live_running = False
    if buttons[2].button("Reset", width="stretch"):
        _reset_stream()

    row1 = st.columns(4)
    scenario = row1[0].selectbox("Scenario", list(SCENARIOS), key="live_scenario", help="Choose the behaviour to simulate.")
    rate = row1[1].slider("Attempts per second", 1, 10, key="live_rate")
    genuine_rate = row1[2].slider("Genuine share", 0, 100, key="live_genuine_mix", format="%d%%") / 100.0
    window_size = row1[3].select_slider("Metrics window", [50, 100, 250, 500], key="live_window")
    row2 = st.columns([2, 1, 1, 1])
    threshold = float(row2[0].slider("Decision threshold", 0.05, 0.99, step=0.01, key="live_threshold"))
    row2[1].selectbox("Target FAR", list(TARGET_FAR), key="live_target_far")
    row2[2].button("Auto-tune threshold", on_click=_auto_tune, width="stretch",
                   help="Sets the threshold that gives the target false-acceptance rate under the impostor model.")
    user = row2[3].text_input("Claimed user ID", key="live_user", max_chars=32).strip() or "U001"
    st.caption(f"Scenario: {SCENARIOS[scenario]}")

    running = st.session_state.live_running
    st.markdown(f"**Stream status:** {'● running' if running else '○ paused'}")

    @st.fragment(run_every=TICK_SECONDS if running else None)
    def live_view():
        _live_view(threshold, scenario, genuine_rate, rate, window_size, user)

    live_view()

    if st.session_state.live_attempts and not running:
        frame = pd.DataFrame(st.session_state.live_attempts)
        summary = {"attempts": st.session_state.live_sequence, "lifetime_counts": st.session_state.live_totals,
                   "scenario": scenario, "threshold": threshold, "alerts": st.session_state.live_alerts}
        export = st.columns(2)
        export[0].download_button("Download attempts (CSV)", frame.to_csv(index=False).encode("utf-8"),
                                  "biosecure_live_attempts.csv", "text/csv", width="stretch")
        export[1].download_button("Download session summary (JSON)", json.dumps(summary, indent=2).encode("utf-8"),
                                  "biosecure_live_summary.json", "application/json", width="stretch")
    elif st.session_state.live_attempts:
        st.caption("Pause the stream to export the attempts and session summary.")
