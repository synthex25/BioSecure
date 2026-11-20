"""Threshold Lab: compare threshold-selection rules (empirical, Bayes, Neyman-Pearson) with their theory."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.calculations import split_scores
from utils.insights import quick_read
from utils.theme import ACCENT, CYAN, GENUINE, IMPOSTOR, MUTED, TEXT, WARN, style_plotly
from utils.threshold_methods import (bayes_gaussian_threshold, bayes_threshold_vs_prior, bootstrap_thresholds,
                                     gaussian_eer, model_rates, posterior_curve, risk_curves, sensitivity,
                                     threshold_candidates, threshold_table)
from utils.ui import active_threshold, concept_chips, kpi_row, page_header, require_dataset, set_threshold, theory

FAMILY_COLORS = {"Empirical": CYAN, "Gaussian model": ACCENT, "Heuristic": WARN}


def _controls(df):
    prior_default = float((df.Class == "Impostor").mean() * 100)
    with st.container(border=True):
        st.markdown("**Decision context** · these assumptions drive the Bayes, cost and target rules below")
        c = st.columns(5)
        impostor_pct = c[0].slider("Impostor prior π_I (%)", 0.5, 95.0, round(prior_default, 1), 0.5, key="tl_prior",
                                   help="Share of attempts expected to be impostors in deployment. The dataset share is the default.")
        c_fa = c[1].number_input("Cost of false accept C_FA", 0.1, 1000.0, 5.0, 0.5, key="tl_cfa")
        c_fr = c[2].number_input("Cost of false reject C_FR", 0.1, 1000.0, 1.0, 0.5, key="tl_cfr")
        alpha = c[3].select_slider("Target FAR α", [0.001, 0.005, 0.01, 0.02, 0.05, 0.10], value=0.01, key="tl_alpha",
                                   format_func=lambda v: f"{v:.1%}")
        beta = c[4].select_slider("Target FRR β", [0.01, 0.02, 0.05, 0.10, 0.20], value=0.05, key="tl_beta",
                                  format_func=lambda v: f"{v:.0%}")
    return 1 - impostor_pct / 100, c_fa, c_fr, alpha, beta


def _methods_chart(genuine, impostor, table, active):
    fig = go.Figure()
    for label, scores, color in (("Genuine", genuine, GENUINE), ("Impostor", impostor, IMPOSTOR)):
        fig.add_trace(go.Histogram(x=scores, name=label, histnorm="probability density", marker_color=color,
                                   opacity=.5, xbins=dict(start=0, end=1, size=.025)))
    for k, row in table.reset_index(drop=True).iterrows():
        fig.add_trace(go.Scatter(x=[row.Threshold, row.Threshold], y=[0, 1], yaxis="y2", mode="lines",
                                 line=dict(color=FAMILY_COLORS[row.Family], width=1.4, dash="dot"),
                                 name=row.Method, showlegend=False,
                                 hovertemplate=f"<b>{row.Method}</b><br>t = {row.Threshold:.4f}<br>FAR {row.FAR:.2%} · FRR {row.FRR:.2%}<extra></extra>"))
    fig.add_vline(x=active, line=dict(color=TEXT, width=2.4), annotation_text=f"active {active:.3f}",
                  annotation_font_color=TEXT, annotation_position="top left")
    fig.update_layout(barmode="overlay", yaxis2=dict(overlaying="y", range=[0, 1], visible=False))
    fig.update_xaxes(range=[0.3, 1], title="Similarity score")
    fig.update_yaxes(title="Density")
    return style_plotly(fig, 380, "Where each rule puts the threshold (hover a dotted line)")


def _tab_compare(df, genuine, impostor, table, prior_g, active):
    st.caption("Every row is a different answer to *“where should the cut-off go?”*. Empirical rules use the observed scores; "
               "Gaussian-model rules use the fitted normal distributions and closed-form formulas. Rates are measured on the full dataset.")
    show = table.copy()
    styled = show.style.format({"Threshold": "{:.4f}", "FAR": "{:.2%}", "FRR": "{:.2%}", "TAR": "{:.2%}", "Error rate": "{:.2%}",
                                "Expected cost": "{:.4f}", "P(G | accept)": "{:.2%}", "HTER": "{:.2%}",
                                "Model FAR": "{:.2%}", "Model FRR": "{:.2%}"}) \
        .highlight_min(subset=["Expected cost"], color="rgba(53,211,154,.25)") \
        .highlight_min(subset=["Error rate"], color="rgba(94,200,242,.22)")
    st.dataframe(styled, width="stretch", hide_index=True, height=490)
    st.plotly_chart(_methods_chart(genuine, impostor, table, active), width="stretch", key="tl_methods_chart")

    best_cost = table.loc[table["Expected cost"].idxmin()]
    quick_read(f"With these assumptions the cheapest rule is “{best_cost.Method}” at t = {best_cost.Threshold:.3f} "
               f"(FAR {best_cost.FAR:.2%}, FRR {best_cost.FRR:.2%}). Rules that ignore costs, such as the EER, sit lower and accept more impostors. "
               "Green = lowest expected cost · blue = lowest error rate.", tone="good")
    c1, c2 = st.columns([2, 1])
    pick = c1.selectbox("Apply a rule across the whole app", table.Method.tolist(),
                        index=int(table["Expected cost"].idxmin()), key="tl_apply_pick")
    chosen = table.loc[table.Method == pick].iloc[0]
    c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c2.button(f"Apply t = {chosen.Threshold:.4f}", type="primary", width="stretch", key="tl_apply_btn"):
        set_threshold(chosen.Threshold, pick)
        st.session_state["_flash"] = f"{pick} applied · threshold {chosen.Threshold:.4f}"
        st.rerun()
    st.download_button("Download comparison (CSV)", table.to_csv(index=False).encode(), "threshold_methods.csv", "text/csv")


def _tab_bayes(genuine, impostor, p, prior_g, c_fa, c_fr):
    t_err, _ = bayes_gaussian_threshold(p, prior_g)
    t_risk, coef = bayes_gaussian_threshold(p, prior_g, c_fa, c_fr)
    left, right = st.columns([1, 1])
    with left:
        theory("Bayes decision rule",
               "Accept an attempt when the *weighted* genuine likelihood beats the impostor one. With equal costs this "
               "minimises the probability of error; with costs it minimises the expected loss (Bayes risk).",
               [r"\text{accept} \iff \pi_G\,C_{FR}\,f_G(x) \;\ge\; \pi_I\,C_{FA}\,f_I(x)",
                r"\Lambda(x)=\frac{f_G(x)}{f_I(x)} \;\ge\; \eta=\frac{\pi_I\,C_{FA}}{\pi_G\,C_{FR}}",
                r"R(t)=\pi_G C_{FR}\,\mathrm{FRR}(t)+\pi_I C_{FA}\,\mathrm{FAR}(t)"])
    with right:
        theory("Closed form for Gaussian classes",
               "With $f_G=\\mathcal N(\\mu_G,\\sigma_G^2)$ and $f_I=\\mathcal N(\\mu_I,\\sigma_I^2)$ the log of the rule is a quadratic "
               "$ax^2+bx+c=0$; the root between the means is the threshold.",
               [rf"a=\tfrac{{1}}{{2\sigma_I^2}}-\tfrac{{1}}{{2\sigma_G^2}}={coef['a']:.3f},\quad b=\tfrac{{\mu_G}}{{\sigma_G^2}}-\tfrac{{\mu_I}}{{\sigma_I^2}}={coef['b']:.3f}",
                rf"c=\ln\tfrac{{\pi_G C_{{FR}}\sigma_I}}{{\pi_I C_{{FA}}\sigma_G}}-\tfrac{{\mu_G^2}}{{2\sigma_G^2}}+\tfrac{{\mu_I^2}}{{2\sigma_I^2}}={coef['c']:.3f}",
                rf"t^*_{{\text{{risk}}}}={t_risk:.4f}\qquad t^*_{{\text{{error}}}}={t_err:.4f}"])

    x = np.linspace(0, 1, 600)
    fg, fi = stats.norm.pdf(x, p["mg"], p["sg"]), stats.norm.pdf(x, p["mi"], p["si"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=prior_g * c_fr * fg, name="π_G·C_FR·f_G(x)", line=dict(color=GENUINE, width=2.6), fill="tozeroy",
                             fillcolor="rgba(53,211,154,.12)"))
    fig.add_trace(go.Scatter(x=x, y=(1 - prior_g) * c_fa * fi, name="π_I·C_FA·f_I(x)", line=dict(color=IMPOSTOR, width=2.6), fill="tozeroy",
                             fillcolor="rgba(255,113,133,.12)"))
    fig.add_vline(x=t_risk, line=dict(color=ACCENT, dash="dash", width=2), annotation_text=f"Bayes t* = {t_risk:.3f}", annotation_font_color=ACCENT)
    fig.update_xaxes(title="Similarity score x"); fig.update_yaxes(title="Weighted density")
    st.plotly_chart(style_plotly(fig, 330, "The threshold is where the weighted likelihoods cross"), width="stretch", key="tl_bayes_cross")

    a, b = st.columns(2)
    with a:
        t, emp, mod = risk_curves(genuine, impostor, p, prior_g, c_fa, c_fr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=emp, name="Empirical risk", line=dict(color=CYAN, width=2.4)))
        fig.add_trace(go.Scatter(x=t, y=mod, name="Gaussian-model risk", line=dict(color=ACCENT, width=2.4, dash="dash")))
        j = int(np.argmin(emp))
        fig.add_trace(go.Scatter(x=[t[j]], y=[emp[j]], mode="markers", marker=dict(color=TEXT, size=10), name=f"min at {t[j]:.3f}"))
        fig.update_xaxes(title="Threshold t", range=[0.3, 1]); fig.update_yaxes(title="Expected cost per attempt")
        st.plotly_chart(style_plotly(fig, 320, "Bayes risk R(t)"), width="stretch", key="tl_risk")
    with b:
        priors = np.linspace(0.01, 0.95, 80)
        fig = go.Figure()
        for ratio, color in ((1, CYAN), (5, ACCENT), (25, IMPOSTOR)):
            fig.add_trace(go.Scatter(x=priors, y=bayes_threshold_vs_prior(p, priors, ratio), name=f"C_FA/C_FR = {ratio}",
                                     line=dict(color=color, width=2.4)))
        fig.add_vline(x=1 - prior_g, line=dict(color=MUTED, dash="dot"))
        fig.update_xaxes(title="Impostor prior π_I", tickformat=".0%"); fig.update_yaxes(title="Bayes threshold t*")
        st.plotly_chart(style_plotly(fig, 320, "More impostors or costlier breaches → stricter threshold"), width="stretch", key="tl_prior_sweep")

    quick_read(f"Raising the impostor prior or the false-accept cost raises η, so the likelihood ratio must be larger before accepting — "
               f"the threshold moves right. Here η = {(1 - prior_g) * c_fa / (prior_g * c_fr):.2f} gives t* = {t_risk:.3f}.")


def _tab_targets(genuine, impostor, p, alpha, beta, table):
    t_eer, eer_model = gaussian_eer(p)
    l, r = st.columns(2)
    with l:
        theory("Neyman–Pearson criterion",
               "Fix the security budget first: keep FAR at or below α, then make FRR as small as possible. By the Neyman–Pearson lemma, "
               "thresholding the likelihood ratio (here, the score) is the most powerful test at level α.",
               [r"\max_t\; \mathrm{TAR}(t)\quad \text{s.t.}\quad \mathrm{FAR}(t)=P(X\ge t\mid I)\le\alpha",
                rf"t_\alpha=\mu_I+\sigma_I\,z_{{1-\alpha}}={p['mi']:.3f}+{p['si']:.3f}\times{stats.norm.ppf(1 - alpha):.3f}"])
    with r:
        theory("Equal error rate in closed form",
               "Under the Gaussian model FAR and FRR are equal where the two standardised distances match.",
               [r"\frac{t-\mu_I}{\sigma_I}=\frac{\mu_G-t}{\sigma_G}\;\Rightarrow\; t_{EER}=\frac{\mu_I\sigma_G+\mu_G\sigma_I}{\sigma_I+\sigma_G}",
                rf"t_{{EER}}={t_eer:.4f},\qquad \mathrm{{EER}}=\Phi\!\left(\frac{{\mu_I-\mu_G}}{{\sigma_I+\sigma_G}}\right)={eer_model:.2%}"])
    t = np.linspace(0.3, 1, 400)
    emp_far = np.array([(impostor >= v).mean() for v in t]); emp_frr = np.array([(genuine < v).mean() for v in t])
    mfar, mfrr = model_rates(t, p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=np.maximum(emp_far, 1e-4), name="FAR empirical", line=dict(color=IMPOSTOR, width=2.4, shape="hv")))
    fig.add_trace(go.Scatter(x=t, y=np.maximum(mfar, 1e-4), name="FAR Gaussian", line=dict(color=IMPOSTOR, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=np.maximum(emp_frr, 1e-4), name="FRR empirical", line=dict(color=WARN, width=2.4, shape="hv")))
    fig.add_trace(go.Scatter(x=t, y=np.maximum(mfrr, 1e-4), name="FRR Gaussian", line=dict(color=WARN, dash="dash")))
    fig.add_hline(y=alpha, line=dict(color=ACCENT, dash="dot"), annotation_text=f"α = {alpha:.1%}", annotation_font_color=ACCENT)
    fig.add_hline(y=beta, line=dict(color=CYAN, dash="dot"), annotation_text=f"β = {beta:.0%}", annotation_font_color=CYAN,
                  annotation_position="bottom right")
    fig.update_yaxes(type="log", title="Error rate (log scale)", range=[-4, 0]); fig.update_xaxes(title="Threshold")
    st.plotly_chart(style_plotly(fig, 360, "Empirical vs Gaussian-model error curves (log scale shows the rare-event tail)"),
                    width="stretch", key="tl_np_curves")
    np_e = table[table.Method.str.startswith("Neyman–Pearson (")].iloc[0]
    np_m = table[table.Method.str.startswith("Neyman–Pearson model")].iloc[0]
    kpi_row([("NP threshold · empirical", f"{np_e.Threshold:.4f}", f"FAR {np_e.FAR:.2%} · FRR {np_e.FRR:.2%}", "cyan"),
             ("NP threshold · Gaussian", f"{np_m.Threshold:.4f}", f"model FAR {np_m['Model FAR']:.2%}", ""),
             ("Gaussian EER", f"{eer_model:.2%}", f"at t = {t_eer:.4f}", "warn"),
             ("Test resolution", f"{1 / len(impostor):.2%}", f"1 / {len(impostor)} impostor attempts", "bad")])
    quick_read("At small α the empirical curve becomes a staircase: with only a few hundred impostor attempts, FAR can only move in steps of "
               f"1/{len(impostor)}. The Gaussian model extrapolates into the tail smoothly, but only as well as the normal assumption holds — "
               "compare the two curves below 1% before trusting either.", tone="caution")


def _tab_sensitivity(df, genuine, impostor, p, prior_g, c_fa, c_fr, alpha, active):
    s = sensitivity(active, p, genuine, impostor)
    step = 0.01
    theory("How fast do the errors change?",
           "Differentiating the error rates gives the class densities at the threshold, so a small move Δt changes the rates by roughly "
           "the density × Δt. Where densities are high, the threshold is fragile.",
           [r"\frac{d\,\mathrm{FAR}}{dt}=-f_I(t),\qquad \frac{d\,\mathrm{FRR}}{dt}=f_G(t)",
            rf"\Delta\mathrm{{FAR}}\approx-{s['kde_fi']:.3f}\times{step}={-s['kde_fi'] * step:+.2%},\qquad "
            rf"\Delta\mathrm{{FRR}}\approx{s['kde_fg']:.3f}\times{step}={s['kde_fg'] * step:+.2%}"])
    kpi_row([("Active threshold", f"{active:.4f}", "used by every page", "cyan"),
             ("f_I(t) · smoothed", f"{s['kde_fi']:.3f}", f"Gaussian model {s['model_fi']:.3f}", "bad"),
             ("f_G(t) · smoothed", f"{s['kde_fg']:.3f}", f"Gaussian model {s['model_fg']:.3f}", "good"),
             ("+0.01 on t", f"{-s['kde_fi'] * step:+.2%} FAR", f"{s['kde_fg'] * step:+.2%} FRR", "warn")])

    st.markdown("##### Threshold stability · user-level bootstrap")
    st.caption("Resample whole users with replacement, re-select the threshold each time, and look at the spread. "
               "A rule whose threshold jumps around between resamples will not transfer well to new users.")
    n_boot = st.select_slider("Bootstrap resamples", [100, 200, 300, 500], value=200, key="tl_nboot")
    boots = bootstrap_thresholds(df, prior_g, c_fa, c_fr, alpha, n_boot=n_boot)
    fig = go.Figure()
    rows = []
    for (name, values), color in zip(boots.items(), (CYAN, ACCENT, IMPOSTOR)):
        fig.add_trace(go.Violin(x=values, name=name, line_color=color, orientation="h", box_visible=True, meanline_visible=True,
                                points=False, fillcolor=color, opacity=.45))
        lo, hi = np.percentile(values, [2.5, 97.5])
        rows.append({"Rule": name, "Mean t": values.mean(), "SD": values.std(ddof=1), "95% interval": f"{lo:.4f} – {hi:.4f}",
                     "Width": hi - lo})
    fig.add_vline(x=active, line=dict(color=TEXT, dash="dot"))
    fig.update_xaxes(title="Selected threshold"); fig.update_layout(showlegend=False)
    st.plotly_chart(style_plotly(fig, 300, "Bootstrap distribution of the chosen threshold"), width="stretch", key="tl_boot")
    st.dataframe(pd.DataFrame(rows).style.format({"Mean t": "{:.4f}", "SD": "{:.4f}", "Width": "{:.4f}"}),
                 width="stretch", hide_index=True)
    widest = max(rows, key=lambda r: r["Width"])
    quick_read(f"“{widest['Rule']}” is the least stable rule here (95% interval width {widest['Width']:.3f}). Tail-based rules such as "
               "Neyman–Pearson depend on a handful of extreme impostor scores, so they wobble most; rules that use the whole distribution are steadier.",
               tone="caution")


def _tab_table(genuine, impostor, prior_g, c_fa, c_fr, active):
    c = st.columns(3)
    start = c[0].number_input("From", 0.0, 0.95, 0.40, 0.05, key="tl_from")
    stop = c[1].number_input("To", 0.05, 1.0, 0.90, 0.05, key="tl_to")
    step = c[2].select_slider("Step", [0.01, 0.02, 0.025, 0.05, 0.1], value=0.05, key="tl_step")
    if stop <= start:
        st.warning("“To” must be greater than “From”.")
        return
    tab = threshold_table(genuine, impostor, prior_g, c_fa, c_fr, start, stop, step)
    nearest = int((tab.Threshold - active).abs().idxmin())

    def mark(row):
        return ["background-color: rgba(167,139,250,.28)" if row.name == nearest else "" for _ in row]

    st.dataframe(tab.style.apply(mark, axis=1).format({"Threshold": "{:.3f}", "FAR": "{:.2%}", "FRR": "{:.2%}", "TAR": "{:.2%}",
                                                       "Error rate": "{:.2%}", "Expected cost": "{:.4f}", "P(G | accept)": "{:.2%}",
                                                       "HTER": "{:.2%}", "Model FAR": "{:.2%}", "Model FRR": "{:.2%}"}),
                 width="stretch", hide_index=True, height=min(38 + 35 * len(tab), 560))
    st.caption(f"Highlighted row: closest to the active threshold {active:.3f}. P(G | accept) uses the prior chosen above, "
               "so it shows how trustworthy an accept is when impostors are rarer or more common than in the dataset.")
    st.download_button("Download threshold table (CSV)", tab.to_csv(index=False).encode(), "threshold_table.csv", "text/csv")


def render_threshold_lab():
    page_header("Decision theory", "Threshold Lab",
                "Thirteen ways to choose the acceptance threshold — from the equal error rate to Bayes-optimal and Neyman–Pearson rules — "
                "compared side by side, with the formulas behind them.", icon="🧪")
    concept_chips("Bayes decision rule", "Likelihood ratio", "Bayes risk", "Neyman–Pearson", "Gaussian closed forms",
                  "Bootstrap stability", "Sensitivity dFAR/dt")
    if not require_dataset("Create the shared demo dataset to compare threshold-selection rules."):
        return
    df = st.session_state.dataset
    genuine, impostor = split_scores(df)
    prior_g, c_fa, c_fr, alpha, beta = _controls(df)
    table, p = threshold_candidates(genuine, impostor, prior_g, c_fa, c_fr, alpha, beta)
    active = active_threshold()
    risk_row = table.loc[table.Method == "Bayes minimum risk"].iloc[0]
    eer_row = table.loc[table.Method == "Equal Error Rate"].iloc[0]
    kpi_row([("Active threshold", f"{active:.4f}", st.session_state.get("active_threshold_profile") or "default", "cyan"),
             ("EER threshold", f"{eer_row.Threshold:.4f}", f"EER {eer_row.FAR:.2%}", "warn"),
             ("Bayes min-risk", f"{risk_row.Threshold:.4f}", f"cost {risk_row['Expected cost']:.4f}", "good"),
             ("Spread of rules", f"{table.Threshold.min():.2f}–{table.Threshold.max():.2f}", f"{len(table)} rules compared", "")])

    tabs = st.tabs(["📋 Compare rules", "🎲 Bayes decision", "🎯 Neyman–Pearson & EER", "📐 Sensitivity & stability", "🔢 Threshold table"])
    with tabs[0]:
        _tab_compare(df, genuine, impostor, table, prior_g, active)
    with tabs[1]:
        _tab_bayes(genuine, impostor, p, prior_g, c_fa, c_fr)
    with tabs[2]:
        _tab_targets(genuine, impostor, p, alpha, beta, table)
    with tabs[3]:
        _tab_sensitivity(df, genuine, impostor, p, prior_g, c_fa, c_fr, alpha, active)
    with tabs[4]:
        _tab_table(genuine, impostor, prior_g, c_fa, c_fr, active)
