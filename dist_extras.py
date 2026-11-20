"""Extra distribution panels shared by pages: goodness of fit, DET curve, Poisson-process checks."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.advanced import eer_point, far_frr_curve
from utils.insights import quick_read
from utils.theme import ACCENT, CYAN, GENUINE, IMPOSTOR, MUTED, TEXT, WARN, style_plotly
from utils.ui import kpi_row, theory


# -----------------------------------------------------------------------------
# Goodness of fit
# -----------------------------------------------------------------------------
def _anderson_normal(x):
    """Anderson–Darling statistic and 5% critical value, compatible with old and new SciPy APIs."""
    try:
        ad = stats.anderson(x, "norm", method="interpolate")   # SciPy >= 1.17
    except TypeError:
        ad = stats.anderson(x, "norm")                         # older SciPy
    crit5 = np.nan
    levels, crits = getattr(ad, "significance_level", None), getattr(ad, "critical_values", None)
    if levels is not None and crits is not None:
        levels = [float(v) for v in np.atleast_1d(levels)]
        if 5.0 in levels:
            crit5 = float(np.atleast_1d(crits)[levels.index(5.0)])
    if not np.isfinite(crit5):   # Stephens (1974): 5% point for a normal with estimated mean and SD
        n = len(x)
        crit5 = 0.752 / (1 + 0.75 / n + 2.25 / n ** 2)
    return ad, crit5


def gof_table(genuine, impostor):
    rows = []
    for name, x in (("Genuine", genuine), ("Impostor", impostor)):
        mu, sd = np.mean(x), np.std(x, ddof=1)
        sw = stats.shapiro(x[:5000])
        ks = stats.kstest(x, stats.norm(mu, sd).cdf)
        ad, crit5 = _anderson_normal(x)
        xc = np.clip(x, 1e-4, 1 - 1e-4)
        a, b, _, _ = stats.beta.fit(xc, floc=0, fscale=1)
        aic_norm = 2 * 2 - 2 * np.sum(stats.norm.logpdf(x, mu, sd))
        aic_beta = 2 * 2 - 2 * np.sum(stats.beta.logpdf(xc, a, b))
        rows.append({"Class": name, "n": len(x), "Shapiro–Wilk W": sw.statistic, "Shapiro p": sw.pvalue, "KS D": ks.statistic,
                     "KS p": ks.pvalue, "Anderson A²": ad.statistic, "A² 5% critical": crit5,
                     "AIC Normal": aic_norm, "AIC Beta": aic_beta, "Better (AIC)": "Normal" if aic_norm < aic_beta else "Beta"})
    return pd.DataFrame(rows)


def render_gof_panel(genuine, impostor):
    st.subheader("Goodness of fit · is the Gaussian model justified?")
    theory("Testing the normality assumption",
           "Every Gaussian formula in this project (Bayes thresholds, model FAR, confidence intervals) assumes the class scores are roughly normal. "
           "A Q-Q plot compares sample quantiles with normal quantiles — a straight line means a good fit. Formal tests set "
           "H₀: the scores come from a normal distribution.",
           [r"\text{KS: } D=\sup_x\,|F_n(x)-\Phi(\tfrac{x-\hat\mu}{\hat\sigma})|,\qquad \mathrm{AIC}=2k-2\ln\hat L"])
    cols = st.columns(2)
    for col, (name, x, color) in zip(cols, (("Genuine", genuine, GENUINE), ("Impostor", impostor, IMPOSTOR))):
        (osm, osr), (slope, intercept, r) = stats.probplot(x, dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color=color, size=5, opacity=.7), name="Sample quantiles"))
        fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[intercept + slope * osm.min(), intercept + slope * osm.max()],
                                 line=dict(color=TEXT, dash="dash"), name=f"Normal line (r = {r:.3f})"))
        fig.update_xaxes(title="Theoretical normal quantile"); fig.update_yaxes(title="Observed score")
        col.plotly_chart(style_plotly(fig, 310, f"Normal Q-Q plot · {name}"), width="stretch", key=f"qq_{name}")
    table = gof_table(genuine, impostor)
    st.dataframe(table.style.format({"Shapiro–Wilk W": "{:.4f}", "Shapiro p": "{:.4f}", "KS D": "{:.4f}", "KS p": "{:.4f}",
                                     "Anderson A²": "{:.3f}", "A² 5% critical": "{:.3f}", "AIC Normal": "{:.1f}", "AIC Beta": "{:.1f}"}),
                 width="stretch", hide_index=True)
    worst = table.loc[table["Shapiro p"].idxmin()]
    msg = (f"Normality is not rejected for either class at α = 0.05 (smallest Shapiro p = {worst['Shapiro p']:.3f}), so the Gaussian model is a reasonable working assumption."
           if worst["Shapiro p"] >= 0.05 else
           f"Normality is rejected for the {worst['Class'].lower()} scores (Shapiro p = {worst['Shapiro p']:.2g}). The data are clipped to [0, 1], which bends the "
           "Q-Q plot at the ends; treat Gaussian tail predictions (very small FAR) with care and prefer empirical or Beta-based estimates there.")
    quick_read(msg + " The KS p-value is approximate because μ and σ are estimated from the same data (the Lilliefors correction would make it stricter).",
               tone="good" if worst["Shapiro p"] >= 0.05 else "caution")


# -----------------------------------------------------------------------------
# DET curve
# -----------------------------------------------------------------------------
DET_TICKS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]


def render_det_curve(genuine, impostor, threshold):
    st.subheader("DET curve (probit axes)")
    st.caption("A Detection Error Trade-off curve plots FRR against FAR on normal-deviate (probit) axes. If both score distributions are Gaussian "
               "the curve becomes a straight line with slope σ_I/σ_G — another visual check of the Gaussian model, and the standard way biometric systems are compared.")
    t, far, frr = far_frr_curve(genuine, impostor, 2001)
    floor = 0.5 / max(len(impostor), len(genuine))
    ok = (far > 0) & (frr > 0) & (far < 1) & (frr < 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stats.norm.ppf(far[ok]), y=stats.norm.ppf(frr[ok]), name="Empirical DET", line=dict(color=ACCENT, width=2.8)))
    mg, sg, mi, si = np.mean(genuine), np.std(genuine, ddof=1), np.mean(impostor), np.std(impostor, ddof=1)
    tm = np.linspace(0, 1, 600)
    mfar, mfrr = stats.norm.sf(tm, mi, si), stats.norm.cdf(tm, mg, sg)
    okm = (mfar > 1e-4) & (mfrr > 1e-4) & (mfar < .5) & (mfrr < .5)
    fig.add_trace(go.Scatter(x=stats.norm.ppf(mfar[okm]), y=stats.norm.ppf(mfrr[okm]), name="Gaussian model (straight line)",
                             line=dict(color=CYAN, width=2, dash="dash")))
    eer_t, eer = eer_point(genuine, impostor)
    fig.add_trace(go.Scatter(x=[stats.norm.ppf(max(eer, floor))], y=[stats.norm.ppf(max(eer, floor))], mode="markers",
                             marker=dict(color=WARN, size=12, symbol="diamond"), name=f"EER {eer:.2%}"))
    a_far, a_frr = (impostor >= threshold).mean(), (genuine < threshold).mean()
    fig.add_trace(go.Scatter(x=[stats.norm.ppf(np.clip(a_far, floor, 1 - floor))], y=[stats.norm.ppf(np.clip(a_frr, floor, 1 - floor))],
                             mode="markers", marker=dict(color=TEXT, size=11), name=f"Active t = {threshold:.2f}"))
    lim = stats.norm.ppf([0.0008, 0.5])
    fig.add_trace(go.Scatter(x=lim, y=lim, line=dict(color=MUTED, dash="dot", width=1), name="FAR = FRR"))
    ticks = dict(tickvals=stats.norm.ppf(DET_TICKS), ticktext=[f"{v:.1%}" if v < 0.01 else f"{v:.0%}" for v in DET_TICKS])
    fig.update_xaxes(title="False Acceptance Rate", range=list(lim), **ticks)
    fig.update_yaxes(title="False Rejection Rate", range=list(lim), **ticks)
    st.plotly_chart(style_plotly(fig, 420, "Detection Error Trade-off"), width="stretch", key="det_curve")


# -----------------------------------------------------------------------------
# Poisson process check on timestamps
# -----------------------------------------------------------------------------
def _merge_bins(observed, expected, min_expected=5):
    obs, exp = list(observed), list(expected)
    while len(exp) > 2 and exp[-1] < min_expected:
        exp[-2] += exp.pop(); obs[-2] += obs.pop()
    while len(exp) > 2 and exp[0] < min_expected:
        exp[1] += exp.pop(0); obs[1] += obs.pop(0)
    return np.array(obs, float), np.array(exp, float)


def render_poisson_process(df, threshold):
    if "Timestamp_s" not in df.columns:
        st.info("This dataset has no timestamps. Re-create it on the Biometric Data page to add Poisson-process arrival times.")
        return
    theory("Attempts as a Poisson process",
           "If attempts arrive independently at a constant average rate λ, the number in any one-minute window is Poisson(λ) and the gaps "
           "between attempts are Exponential(λ). Both claims can be tested on the timestamps.",
           [r"N(\text{1 min})\sim\mathrm{Poisson}(\lambda),\quad P(N=k)=\frac{e^{-\lambda}\lambda^k}{k!},\qquad W\sim\mathrm{Exp}(\lambda),\quad E[W]=\tfrac1\lambda",
            r"\chi^2=\sum_k\frac{(O_k-E_k)^2}{E_k},\qquad \text{index of dispersion } D=\frac{s^2}{\bar N}\approx 1"])
    ts = np.sort(df["Timestamp_s"].to_numpy(float))
    window = st.select_slider("Counting window", [30, 60, 120, 300], value=60, key="pp_window", format_func=lambda v: f"{v} s")
    edges = np.arange(0, ts.max() + window, window)
    counts = np.histogram(ts, edges)[0][:-1] if len(edges) > 2 else np.histogram(ts, edges)[0]
    lam = counts.mean()
    kmax = int(counts.max())
    k = np.arange(0, kmax + 1)
    observed = np.bincount(counts, minlength=kmax + 1)
    expected = stats.poisson.pmf(k, lam) * len(counts)
    expected[-1] += stats.poisson.sf(kmax, lam) * len(counts)
    o, e = _merge_bins(observed, expected)
    chi2 = float(((o - e) ** 2 / e).sum())
    dof = max(len(o) - 2, 1)
    p_chi = float(stats.chi2.sf(chi2, dof))
    gaps = np.diff(ts)
    ks = stats.kstest(gaps, stats.expon(0, gaps.mean()).cdf)
    disp = counts.var(ddof=1) / lam if lam > 0 else np.nan

    kpi_row([("λ̂ per window", f"{lam:.3f}", f"{len(counts)} windows of {window}s", "cyan"),
             ("Dispersion s²/N̄", f"{disp:.3f}", "≈ 1 for Poisson", "good" if 0.8 < disp < 1.2 else "warn"),
             ("χ² goodness of fit", f"{chi2:.2f}", f"df = {dof}, p = {p_chi:.3f}", "good" if p_chi > 0.05 else "bad"),
             ("KS · gaps ~ Exp", f"D = {ks.statistic:.3f}", f"p = {ks.pvalue:.3f}", "good" if ks.pvalue > 0.05 else "bad")])
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=k, y=observed, name="Observed windows", marker_color=ACCENT, opacity=.8))
        fig.add_trace(go.Scatter(x=k, y=stats.poisson.pmf(k, lam) * len(counts), name=f"Poisson(λ = {lam:.2f})", mode="lines+markers",
                                 line=dict(color=TEXT, width=2)))
        fig.update_xaxes(title=f"Attempts per {window}s window"); fig.update_yaxes(title="Number of windows")
        st.plotly_chart(style_plotly(fig, 310, "Counts per window vs Poisson"), width="stretch", key="pp_counts")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=gaps, histnorm="probability density", marker_color=CYAN, opacity=.6, name="Observed gaps", nbinsx=40))
        g = np.linspace(0, np.percentile(gaps, 99.5), 300)
        fig.add_trace(go.Scatter(x=g, y=stats.expon.pdf(g, scale=gaps.mean()), name=f"Exp(mean {gaps.mean():.1f}s)", line=dict(color=TEXT, width=2)))
        fig.update_xaxes(title="Seconds between attempts"); fig.update_yaxes(title="Density")
        st.plotly_chart(style_plotly(fig, 310, "Inter-arrival times vs Exponential"), width="stretch", key="pp_gaps")

    st.markdown("##### From arrivals to breaches: thinning and the geometric distribution")
    impostor = df.loc[df.Class == "Impostor", "Similarity_Score"].to_numpy(float)
    far = float((impostor >= threshold).mean())
    pi = float((df.Class == "Impostor").mean())
    rate_per_min = len(ts) / (ts.max() / 60)
    lam_breach_hour = rate_per_min * 60 * pi * far
    far_eff = far if far > 0 else 0.5 / len(impostor)
    theory("Thinning a Poisson process",
           "Each arrival is independently an impostor (probability π_I) who is falsely accepted (probability FAR). Keeping only those arrivals "
           "gives another Poisson process with a smaller rate. The number of impostor tries until the first false accept is Geometric.",
           [r"\lambda_{\text{breach}}=\lambda\,\pi_I\,\mathrm{FAR},\qquad P(\text{no breach in }T)=e^{-\lambda_{\text{breach}}T}",
            r"K\sim\mathrm{Geometric}(\mathrm{FAR}):\;P(K=k)=(1-\mathrm{FAR})^{k-1}\mathrm{FAR},\quad E[K]=\tfrac1{\mathrm{FAR}}"])
    kpi_row([("Arrival rate λ", f"{rate_per_min:.2f}/min", "all attempts", "cyan"),
             ("Breach rate", f"{lam_breach_hour:.3f}/hour", f"π_I = {pi:.0%}, FAR = {far:.2%}", "bad"),
             ("P(≥1 breach in 8 h)", f"{1 - np.exp(-lam_breach_hour * 8):.1%}", "1 − e^(−8λ_breach)", "warn"),
             ("Mean impostor tries to first breach", f"{1 / far_eff:,.0f}", "geometric mean 1/FAR", "")])
    kk = np.arange(1, int(min(max(5 / far_eff, 20), 2000)))
    fig = go.Figure(go.Scatter(x=kk, y=1 - (1 - far_eff) ** kk, line=dict(color=IMPOSTOR, width=2.6), fill="tozeroy",
                               fillcolor="rgba(255,113,133,.12)", name="P(K ≤ k)"))
    fig.update_xaxes(title="Impostor attempts k"); fig.update_yaxes(title="P(first breach by attempt k)", tickformat=".0%", range=[0, 1])
    st.plotly_chart(style_plotly(fig, 280, "Geometric CDF of attempts until the first false accept", legend_below=False),
                    width="stretch", key="pp_geom")
    quick_read(f"The counts pass the Poisson check (χ² p = {p_chi:.2f}, dispersion {disp:.2f}) and the gaps look Exponential (KS p = {ks.pvalue:.2f}), "
               "as expected for a simulated Poisson process. On real logs, dispersion well above 1 would signal bursts — e.g. a scripted attack."
               if p_chi > 0.05 and ks.pvalue > 0.05 else
               f"At least one check fails (χ² p = {p_chi:.3f}, KS p = {ks.pvalue:.3f}); arrivals may be bursty or the rate may change over time.",
               tone="good" if p_chi > 0.05 and ks.pvalue > 0.05 else "caution")
