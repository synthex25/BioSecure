"""Noise & Signal Decay: Gaussian sensor noise and Exponential signal decay (template ageing, component lifetime)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.advanced import decidability, eer_point
from utils.calculations import split_scores
from utils.insights import quick_read
from utils.theme import ACCENT, CYAN, GENUINE, IMPOSTOR, MUTED, TEXT, WARN, style_plotly
from utils.threshold_methods import gaussian_eer, gaussian_params
from utils.ui import active_threshold, concept_chips, kpi_row, page_header, require_dataset, theory


# -----------------------------------------------------------------------------
# Gaussian sensor noise
# -----------------------------------------------------------------------------
def _noisy_params(p, sigma_n):
    return {"mg": p["mg"], "mi": p["mi"], "sg": float(np.hypot(p["sg"], sigma_n)), "si": float(np.hypot(p["si"], sigma_n))}


def _tab_noise(genuine, impostor, p, t):
    l, r = st.columns([1.1, 1])
    with l:
        theory("Additive Gaussian noise model",
               "A real sensor adds random measurement noise ε to the ideal matching score. If ε is Gaussian and independent of the "
               "score, variances add while the means stay put — the two classes spread out and overlap more.",
               [r"X_{\text{obs}} = X + \varepsilon,\qquad \varepsilon\sim\mathcal N(0,\sigma_n^2)",
                r"X\mid G\sim\mathcal N(\mu_G,\sigma_G^2)\;\Rightarrow\;X_{\text{obs}}\mid G\sim\mathcal N(\mu_G,\;\sigma_G^2+\sigma_n^2)",
                r"d'(\sigma_n)=\frac{|\mu_G-\mu_I|}{\sqrt{\tfrac12(\sigma_G^2+\sigma_I^2)+\sigma_n^2}}"])
    with r:
        sigma_n = st.slider("Sensor noise σₙ", 0.0, 0.20, 0.05, 0.005, key="nd_sigma", format="%.3f")
        mc = st.toggle("Verify by Monte Carlo (add noise to the actual scores)", value=True, key="nd_mc")
        q = _noisy_params(p, sigma_n)
        t_eer0, eer0 = gaussian_eer(p)
        t_eer1, eer1 = gaussian_eer(q)
        far0, frr0 = stats.norm.sf(t, p["mi"], p["si"]), stats.norm.cdf(t, p["mg"], p["sg"])
        far1, frr1 = stats.norm.sf(t, q["mi"], q["si"]), stats.norm.cdf(t, q["mg"], q["sg"])
        kpi_row([("EER clean → noisy", f"{eer1:.2%}", f"was {eer0:.2%}", "warn"),
                 (f"FAR at t={t:.2f}", f"{far1:.2%}", f"was {far0:.2%}", "bad"),
                 (f"FRR at t={t:.2f}", f"{frr1:.2%}", f"was {frr0:.2%}", "warn")])

    x = np.linspace(0, 1, 500)
    fig = go.Figure()
    for (mu, sd, sdn, name, color) in ((p["mg"], p["sg"], q["sg"], "Genuine", GENUINE), (p["mi"], p["si"], q["si"], "Impostor", IMPOSTOR)):
        fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, mu, sd), name=f"{name} · clean", line=dict(color=color, width=2.6)))
        fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, mu, sdn), name=f"{name} · noisy", line=dict(color=color, width=2, dash="dash")))
    fig.add_vline(x=t, line=dict(color=TEXT, dash="dot"), annotation_text=f"t = {t:.2f}", annotation_font_color=TEXT)
    fig.update_xaxes(title="Score", range=[0, 1]); fig.update_yaxes(title="Density")
    st.plotly_chart(style_plotly(fig, 330, f"Score densities before and after noise σₙ = {sigma_n:.3f}"), width="stretch", key="nd_noise_pdf")

    levels = np.linspace(0, 0.20, 21)
    model_eer = [gaussian_eer(_noisy_params(p, s))[1] for s in levels]
    model_dp = [abs(p["mg"] - p["mi"]) / np.sqrt(0.5 * (p["sg"] ** 2 + p["si"] ** 2) + s ** 2) for s in levels]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=levels, y=model_eer, name="EER · Gaussian theory", line=dict(color=ACCENT, width=2.6)))
    if mc:
        rng = np.random.default_rng(17)
        sim = []
        for s in levels:
            g = np.clip(genuine + rng.normal(0, s, len(genuine)), 0, 1)
            i = np.clip(impostor + rng.normal(0, s, len(impostor)), 0, 1)
            sim.append(eer_point(g, i)[1])
        fig.add_trace(go.Scatter(x=levels, y=sim, name="EER · Monte Carlo on data", mode="markers+lines",
                                 line=dict(color=CYAN, width=1.5, dash="dot"), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=levels, y=model_dp, name="d′ (right axis)", yaxis="y2", line=dict(color=WARN, width=2)))
    fig.add_vline(x=sigma_n, line=dict(color=MUTED, dash="dot"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="d′", showgrid=False))
    fig.update_xaxes(title="Noise σₙ"); fig.update_yaxes(title="Equal error rate", tickformat=".0%")
    st.plotly_chart(style_plotly(fig, 340, "More noise → lower decidability → higher EER"), width="stretch", key="nd_noise_curve")
    quick_read(f"Noise of σₙ = {sigma_n:.3f} lowers d′ from {decidability(genuine, impostor):.2f} to "
               f"{model_dp[int(round(sigma_n / 0.01))]:.2f} and moves the EER from {eer0:.2%} to {eer1:.2%}. "
               "The theory curve and the simulation agree closely; small gaps come from clipping scores to [0, 1] and from non-normal tails.",
               tone="caution" if eer1 > 2 * eer0 else "info")


# -----------------------------------------------------------------------------
# Exponential signal decay / template ageing
# -----------------------------------------------------------------------------
def _tab_decay(df, p, t):
    l, r = st.columns([1.1, 1])
    with l:
        theory("Exponential signal decay (template ageing)",
               "As an enrolled fingerprint template ages (skin changes, wear, sensor drift) the genuine match signal fades toward the "
               "impostor noise floor. A first-order decay model gives an exponential curve with rate λ and half-life ln 2 / λ.",
               [r"\mu_G(d)=\mu_I+(\mu_G(0)-\mu_I)\,e^{-\lambda d},\qquad t_{1/2}=\frac{\ln 2}{\lambda}",
                r"\mathrm{FRR}(d)=\Phi\!\left(\frac{t-\mu_G(d)}{\sigma_G}\right)",
                r"d^*=\frac1\lambda\ln\frac{\mu_G(0)-\mu_I}{\,t-\sigma_G\,\Phi^{-1}(L)-\mu_I\,}\quad(\text{FRR reaches limit }L)"])
    with r:
        half_life = st.slider("Signal half-life (days)", 30, 2000, 1095, 15, key="nd_half")
        frr_limit = st.select_slider("Acceptable FRR limit L", [0.02, 0.05, 0.10, 0.15, 0.20, 0.30], value=0.15, key="nd_limit",
                                     format_func=lambda v: f"{v:.0%}")
        horizon = st.select_slider("Horizon (days)", [180, 365, 730, 1095], value=730, key="nd_horizon")
    lam = np.log(2) / half_life
    days = np.arange(0, horizon + 1)
    mu_d = p["mi"] + (p["mg"] - p["mi"]) * np.exp(-lam * days)
    frr_fixed = stats.norm.cdf(t, mu_d, p["sg"])
    target = t - p["sg"] * stats.norm.ppf(frr_limit) - p["mi"]
    if target <= 0:          # even a fully decayed signal keeps FRR below L
        d_star = np.inf
    elif p["mg"] - p["mi"] <= target:   # FRR already above L on day 0
        d_star = 0.0
    else:
        d_star = float(np.log((p["mg"] - p["mi"]) / target) / lam)
    # Adaptive alternative: lower the threshold to hold FRR at L; FAR then rises.
    t_adapt = mu_d + p["sg"] * stats.norm.ppf(frr_limit)
    far_adapt = stats.norm.sf(t_adapt, p["mi"], p["si"])
    far_fixed = stats.norm.sf(t, p["mi"], p["si"])
    initial_frr = frr_fixed[0]

    kpi_row([("Decay rate λ", f"{lam:.5f}/day", f"half-life {half_life} days", "cyan"),
             ("Genuine mean at horizon", f"{mu_d[-1]:.3f}", f"from {p['mg']:.3f}", "warn"),
             ("FRR at horizon (fixed t)", f"{frr_fixed[-1]:.1%}", f"from {initial_frr:.1%}", "bad"),
             ("Re-enrol after", "already exceeded" if initial_frr > frr_limit else ("never" if not np.isfinite(d_star) else f"{d_star:.0f} days"),
              f"FRR limit {frr_limit:.0%} at t = {t:.2f}", "good")])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=frr_fixed, name="FRR · fixed threshold", line=dict(color=WARN, width=2.6)))
    fig.add_trace(go.Scatter(x=days, y=far_adapt, name="FAR · adaptive threshold (holds FRR = L)", line=dict(color=IMPOSTOR, width=2.4, dash="dash")))
    fig.add_trace(go.Scatter(x=days, y=np.full_like(days, far_fixed, dtype=float), name="FAR · fixed threshold", line=dict(color=IMPOSTOR, width=1.4, dash="dot")))
    fig.add_hline(y=frr_limit, line=dict(color=MUTED, dash="dot"), annotation_text=f"L = {frr_limit:.0%}", annotation_font_color=MUTED)
    if np.isfinite(d_star) and 0 < d_star <= horizon:
        fig.add_vline(x=d_star, line=dict(color=GENUINE, dash="dash"), annotation_text=f"re-enrol ≈ day {d_star:.0f}", annotation_font_color=GENUINE)
    fig.update_xaxes(title="Template age (days)"); fig.update_yaxes(title="Error rate", tickformat=".0%")
    st.plotly_chart(style_plotly(fig, 350, "Ageing trade-off: re-enrol, or loosen the threshold and accept more impostors"),
                    width="stretch", key="nd_decay_curve")

    st.markdown("##### Estimating λ from data")
    st.caption("Linearise the model: ln(μ_G(d) − μ_I) = ln(μ_G(0) − μ_I) − λd, then fit a straight line by least squares. "
               "The default dataset has no ageing (λ = 0); turn on *template ageing* on the Biometric Data page to see it recovered.")
    sim = st.toggle("Use a simulated ageing study with the half-life above (instead of the shared dataset)", value=True, key="nd_sim")
    if sim:
        rng = np.random.default_rng(23)
        age = rng.integers(0, horizon + 1, 600)
        base = rng.normal(p["mg"], p["sg"], 600)
        score = p["mi"] + (base - p["mi"]) * np.exp(-lam * age)
        gen_age, gen_score = age, score
    else:
        gen = df[df.Class == "Genuine"]
        gen_age, gen_score = gen.Template_Age_Days.to_numpy(float), gen.Similarity_Score.to_numpy(float)
    bins = np.linspace(0, gen_age.max() if len(gen_age) else 1, 11)
    idx = np.clip(np.digitize(gen_age, bins) - 1, 0, len(bins) - 2)
    centers, means = [], []
    for k in range(len(bins) - 1):
        m = idx == k
        if m.sum() >= 5:
            centers.append((bins[k] + bins[k + 1]) / 2); means.append(gen_score[m].mean())
    centers, means = np.array(centers), np.array(means)
    ok = means - p["mi"] > 1e-3
    if ok.sum() >= 3:
        fit = stats.linregress(centers[ok], np.log(means[ok] - p["mi"]))
        lam_hat = -fit.slope
        ci = 1.96 * fit.stderr
        a, b = st.columns([1.4, 1])
        with a:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=gen_age, y=gen_score, mode="markers", marker=dict(color=GENUINE, size=4, opacity=.35), name="Genuine attempts"))
            fig.add_trace(go.Scatter(x=centers, y=means, mode="markers", marker=dict(color=TEXT, size=9), name="Binned mean"))
            fit_curve = p["mi"] + np.exp(fit.intercept) * np.exp(-lam_hat * days[: int(gen_age.max()) + 1])
            fig.add_trace(go.Scatter(x=days[: int(gen_age.max()) + 1], y=fit_curve, name="Fitted exponential", line=dict(color=ACCENT, width=2.6)))
            fig.update_xaxes(title="Template age (days)"); fig.update_yaxes(title="Genuine score")
            st.plotly_chart(style_plotly(fig, 320, "Exponential decay fitted by log-linear regression"), width="stretch", key="nd_fit")
        with b:
            kpi_row([("Estimated λ", f"{lam_hat:.5f}", f"95% CI ± {ci:.5f}", "cyan"),
                     ("Estimated half-life", f"{np.log(2) / lam_hat:.0f} d" if lam_hat > 1e-6 else "no decay", f"true {half_life} d" if sim else "dataset", "good"),
                     ("R² of log-fit", f"{fit.rvalue ** 2:.3f}", f"p = {fit.pvalue:.2g}", "")])
            quick_read("A slope indistinguishable from zero (wide CI around 0) means no ageing is detectable — expected for the default dataset."
                       if not sim and abs(lam_hat) < ci else
                       f"The regression recovers λ ≈ {lam_hat:.5f}/day. Hypothesis test H₀: λ = 0 gives p = {fit.pvalue:.2g}.")
    else:
        st.info("Not enough age bins with a mean above the impostor level to fit the decay.")


def _tab_lifetime():
    theory("Exponential lifetime and the memoryless property",
           "The time until a random failure event (a sensor fault, a breach, the next impostor attempt) is often modelled as Exponential. "
           "It is the only continuous distribution without memory: having already survived s days does not change the chance of surviving t more.",
           [r"T\sim\mathrm{Exp}(\lambda):\; f(t)=\lambda e^{-\lambda t},\; P(T>t)=e^{-\lambda t},\; E[T]=\tfrac1\lambda,\; \operatorname{Var}(T)=\tfrac1{\lambda^2}",
            r"P(T>s+t\mid T>s)=\frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}=e^{-\lambda t}=P(T>t)"])
    c = st.columns(3)
    mttf = c[0].slider("Mean time to sensor failure (days)", 30, 1000, 250, 10, key="nd_mttf")
    s = c[1].slider("Already survived s (days)", 0, 500, 100, 10, key="nd_s")
    tt = c[2].slider("Extra time t (days)", 10, 500, 90, 10, key="nd_t")
    lam = 1 / mttf
    rng = np.random.default_rng(5)
    life = rng.exponential(mttf, 200_000)
    survivors = life[life > s]
    cond_sim = (survivors > s + tt).mean() if len(survivors) else np.nan
    kpi_row([("P(T > t)", f"{np.exp(-lam * tt):.4f}", "unconditional, theory", "cyan"),
             ("P(T > s+t | T > s)", f"{cond_sim:.4f}", f"simulated on {len(survivors):,} survivors", "good"),
             ("Median lifetime", f"{np.log(2) * mttf:.0f} d", "ln 2 / λ", ""),
             ("P(fail within 1 year)", f"{1 - np.exp(-lam * 365):.1%}", "1 − e^(−365λ)", "warn")])
    grid = np.linspace(0, 3 * mttf, 400)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=life[:20000], histnorm="probability density", marker_color=MUTED, opacity=.5, name="Simulated lifetimes",
                               xbins=dict(start=0, end=3 * mttf, size=mttf / 12)))
    fig.add_trace(go.Scatter(x=grid, y=lam * np.exp(-lam * grid), name="λe^(−λt)", line=dict(color=ACCENT, width=2.6)))
    fig.add_vrect(x0=s, x1=s + tt, fillcolor=CYAN, opacity=.12, line_width=0, annotation_text="next t days", annotation_font_color=CYAN)
    fig.update_xaxes(title="Days until failure"); fig.update_yaxes(title="Density")
    st.plotly_chart(style_plotly(fig, 320, "Exponential lifetime density"), width="stretch", key="nd_life")
    quick_read("The simulated conditional survival matches the unconditional one: an old sensor is, under this model, as reliable as a new one. "
               "Real hardware usually wears out (an increasing hazard), which a Weibull model captures — the Exponential is the constant-hazard baseline.")


def render_noise_decay():
    page_header("Statistical analysis · Continuous models", "Noise & Signal Decay",
                "Gaussian distributions model measurement noise; Exponential distributions model signal decay and time-to-failure. "
                "See how each degrades recognition and how the threshold must respond.", icon="📡")
    concept_chips("Gaussian noise", "Sum of independent normals", "Decidability d′", "Exponential decay", "Half-life",
                  "Log-linear regression", "Memoryless property")
    if not require_dataset("Create the shared demo dataset — the noise and decay models start from its fitted Gaussians."):
        return
    df = st.session_state.dataset
    genuine, impostor = split_scores(df)
    p = gaussian_params(genuine, impostor)
    t = active_threshold()
    tabs = st.tabs(["〰️ Gaussian sensor noise", "📉 Exponential signal decay", "⏳ Exponential lifetime"])
    with tabs[0]:
        _tab_noise(genuine, impostor, p, t)
    with tabs[1]:
        _tab_decay(df, p, t)
    with tabs[2]:
        _tab_lifetime()
