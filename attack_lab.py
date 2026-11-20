"""Attack & Lockout Lab: spoofing strength, lockout policy and brute-force horizon, with Monte Carlo checks."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.advanced import (attempts_to_breach, fit_gaussians, lockout_analysis, lockout_monte_carlo, spoof_far)
from utils.calculations import performance_metrics, split_scores
from utils.theme import ACCENT, CYAN, GENUINE, IMPOSTOR, MUTED, TEXT, WARN, style_plotly
from utils.ui import active_threshold, callout, kpi_row, page_header, require_dataset


def render_attack_lab():
    page_header("Advanced · Security", "Attack & Lockout Lab",
                "Stress-test the decision threshold. See how presentation attacks (spoofs) raise the false accept rate, "
                "how lockout policies trade security against genuine-user convenience, and how many guesses a brute-force attacker needs.")
    if not require_dataset():
        return
    df = st.session_state.dataset
    genuine, impostor = split_scores(df)
    gen_p, imp_p = fit_gaussians(genuine, impostor)

    threshold = st.slider("Decision threshold", 0.05, 0.99, float(round(active_threshold(), 2)), 0.01, key="attack_threshold")
    m = performance_metrics(df, threshold)
    far0 = max(float(m["FAR"]), 1e-4)
    far_model = spoof_far(threshold, gen_p, imp_p, 0.0)
    kpi_row([("Threshold", f"{threshold:.2f}", "", "cyan"),
             ("Observed FAR", f"{m['FAR']:.2%}", f"{m['FP']} of {m['FP'] + m['TN']} impostor attempts", "bad"),
             ("Model FAR", f"{far_model:.2%}", "Gaussian fit to impostor scores", ""),
             ("Observed FRR", f"{m['FRR']:.2%}", f"{m['FN']} of {m['FN'] + m['TP']} genuine attempts", "warn")])

    tab_spoof, tab_lock, tab_brute = st.tabs(["Spoofing attack", "Lockout policy", "Brute-force horizon"])

    with tab_spoof:
        st.markdown("**Attack quality** `q` interpolates between a zero-effort impostor (q = 0) and a perfect copy of a genuine fingerprint (q = 1).")
        q_sel = st.slider("Attack quality q", 0.0, 1.0, 0.4, 0.05)
        qs = np.linspace(0, 1, 101)
        fig = go.Figure()
        for t, color in ((threshold, ACCENT), (min(threshold + 0.1, 0.99), CYAN), (min(threshold + 0.2, 0.99), GENUINE)):
            fig.add_trace(go.Scatter(x=qs, y=[spoof_far(t, gen_p, imp_p, q) for q in qs], name=f"threshold {t:.2f}",
                                     line=dict(color=color, width=2.6)))
        fig.add_vline(x=q_sel, line=dict(color=TEXT, dash="dash"))
        fig.update_xaxes(title="Attack quality q"); fig.update_yaxes(title="False accept rate", tickformat=".0%", range=[0, 1.02])
        st.plotly_chart(style_plotly(fig, 360, "FAR rises as attacks get closer to genuine"), width="stretch", key="attack_far")
        far_q = spoof_far(threshold, gen_p, imp_p, q_sel)
        target = st.select_slider("FAR budget under attack", [0.10, 0.05, 0.01, 0.001], 0.01, format_func=lambda v: f"{v:.1%}")
        grid = np.linspace(0.05, 0.99, 400)
        need = next((t for t in grid if spoof_far(t, gen_p, imp_p, q_sel) <= target), None)
        if need is None:
            callout("Threshold cannot fix this", f"At q = {q_sel:.2f} no threshold below 0.99 keeps FAR under {target:.1%}. "
                    "A better sensor or liveness detection is needed, not a stricter threshold.", "risk")
        else:
            frr_need = float((genuine < need).mean())
            kpi_row([("FAR at q", f"{far_q:.1%}", f"at threshold {threshold:.2f}", "bad" if far_q > target else "good"),
                     ("Threshold needed", f"{need:.2f}", f"to hold FAR ≤ {target:.1%}", "cyan"),
                     ("FRR cost", f"{frr_need:.1%}", "genuine users rejected at that threshold", "warn")])
            callout("Security vs convenience",
                    f"Holding FAR under {target:.1%} against attack quality {q_sel:.2f} needs a threshold of about {need:.2f}, "
                    f"which would reject roughly {frr_need:.0%} of genuine attempts. That cost is why liveness detection is usually preferred to ever-higher thresholds.",
                    "caution")

    with tab_lock:
        c1, c2 = st.columns(2)
        sessions = c1.slider("Attack windows (retries after each lockout)", 1, 50, 5)
        max_try = c2.slider("Tries allowed before lockout", 1, 10, 3)
        ms = np.arange(1, 11)
        p_attack, p_lock, p_multi = lockout_analysis(far0, float(m["FRR"]), ms, sessions)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ms, y=p_attack, name="Attacker succeeds in one window", line=dict(color=IMPOSTOR, width=2.6)))
        fig.add_trace(go.Scatter(x=ms, y=p_multi, name=f"Attacker succeeds over {sessions} windows", line=dict(color=IMPOSTOR, dash="dot", width=2.2)))
        fig.add_trace(go.Scatter(x=ms, y=p_lock, name="Genuine user locked out", line=dict(color=WARN, width=2.6)))
        fig.add_vline(x=max_try, line=dict(color=TEXT, dash="dash"))
        fig.update_xaxes(title="Tries allowed before lockout", dtick=1); fig.update_yaxes(title="Probability", tickformat=".1%")
        st.plotly_chart(style_plotly(fig, 360, "Lockout trade-off: attacker success vs genuine lockout"), width="stretch", key="attack_lock")
        i = max_try - 1
        mc_attack, mc_lock = lockout_monte_carlo(far0, float(m["FRR"]), max_try)
        kpi_row([("Attacker, one window", f"{p_attack[i]:.2%}", f"formula 1 − (1 − FAR)^{max_try}", "bad"),
                 ("Monte Carlo check", f"{mc_attack:.2%}", "20,000 simulated attackers", ""),
                 ("Genuine lockout", f"{p_lock[i]:.3%}", f"formula FRR^{max_try}", "warn"),
                 ("Monte Carlo check", f"{mc_lock:.3%}", "20,000 simulated genuine users", "")])
        st.caption("Each attempt is treated as independent. Real attackers may repeat the same artefact, which makes attempts correlated and the true risk higher.")

    with tab_brute:
        n95, mean_wait = attempts_to_breach(far0)
        rate = st.slider("Attacker attempts per hour", 1, 3600, 60)
        kpi_row([("Mean attempts to first breach", f"{mean_wait:,.0f}", "geometric: 1 / FAR", "bad"),
                 ("Attempts for 95% breach chance", f"{n95:,.0f}", "ln(0.05) / ln(1 − FAR)", "warn"),
                 ("Time to 95% chance", f"{n95 / rate:,.1f} h", f"at {rate} attempts/hour", "cyan")])
        n = np.arange(1, int(min(max(n95 * 2, 20), 5000)) + 1)
        fig = go.Figure(go.Scatter(x=n, y=1 - (1 - far0) ** n, line=dict(color=IMPOSTOR, width=2.6), fill="tozeroy",
                                   fillcolor="rgba(255,113,133,.12)"))
        fig.add_hline(y=0.95, line=dict(color=MUTED, dash="dash"))
        fig.update_xaxes(title="Number of attempts"); fig.update_yaxes(title="P(at least one false accept)", tickformat=".0%", range=[0, 1.02])
        st.plotly_chart(style_plotly(fig, 340, "Brute-force success probability", False), width="stretch", key="attack_brute")
        st.caption("Without rate-limiting, even a small FAR is eventually broken. Lockouts, delays and liveness detection all work by capping the attempt count.")
