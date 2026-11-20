"""Statistical inference panels: error-rate confidence intervals, exact tests, non-parametric tests, power and sample size."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.insights import quick_read
from utils.theme import ACCENT, CYAN, GENUINE, IMPOSTOR, MUTED, TEXT, WARN, style_plotly
from utils.ui import kpi_row, theory


# -----------------------------------------------------------------------------
# Binomial proportion intervals
# -----------------------------------------------------------------------------
def wald_ci(k, n, conf=0.95):
    p = k / n
    z = stats.norm.ppf(0.5 + conf / 2)
    m = z * np.sqrt(p * (1 - p) / n)
    return max(0.0, p - m), min(1.0, p + m)


def wilson_ci(k, n, conf=0.95):
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def clopper_pearson_ci(k, n, conf=0.95):
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def proportion_intervals(k, n, conf=0.95):
    return {"Wald (normal)": wald_ci(k, n, conf), "Wilson score": wilson_ci(k, n, conf), "Clopper–Pearson (exact)": clopper_pearson_ci(k, n, conf)}


def render_error_rate_inference(df, threshold, conf=0.95):
    genuine = df.loc[df.Class == "Genuine", "Similarity_Score"].to_numpy(float)
    impostor = df.loc[df.Class == "Impostor", "Similarity_Score"].to_numpy(float)
    k_fa, n_i = int((impostor >= threshold).sum()), len(impostor)
    k_fr, n_g = int((genuine < threshold).sum()), len(genuine)
    theory("Confidence intervals for FAR and FRR",
           "FAR and FRR are binomial proportions: k errors out of n independent attempts. The textbook Wald interval breaks down when errors are rare "
           "(it can even go below zero), which is exactly the biometric case. Wilson and Clopper–Pearson intervals stay valid.",
           [r"\text{Wald: }\hat p\pm z\sqrt{\tfrac{\hat p(1-\hat p)}{n}}\qquad \text{Wilson: }\frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}",
            r"\text{Clopper–Pearson: }\big[\,B_{\alpha/2}(k,\,n-k+1),\;B_{1-\alpha/2}(k+1,\,n-k)\,\big]"])
    rows = []
    for rate, k, n in (("FAR", k_fa, n_i), ("FRR", k_fr, n_g)):
        for method, (lo, hi) in proportion_intervals(k, n, conf).items():
            rows.append({"Rate": rate, "Errors k / n": f"{k} / {n}", "Estimate": k / n, "Method": method, "Lower": lo, "Upper": hi, "Width": hi - lo})
    table = pd.DataFrame(rows)
    fig = go.Figure()
    for j, r in table.iterrows():
        color = IMPOSTOR if r.Rate == "FAR" else WARN
        label = f"{r.Rate} · {r.Method}"
        fig.add_trace(go.Scatter(x=[r.Estimate], y=[label], mode="markers", marker=dict(color=color, size=11), showlegend=False,
                                 error_x=dict(type="data", symmetric=False, array=[r.Upper - r.Estimate], arrayminus=[r.Estimate - r.Lower],
                                              color=color, thickness=3, width=7),
                                 hovertemplate=f"{label}<br>{r.Lower:.3%} – {r.Upper:.3%}<extra></extra>"))
    fig.update_xaxes(title="Error rate", tickformat=".1%", rangemode="tozero"); fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_plotly(fig, 320, f"{conf:.0%} intervals at threshold {threshold:.3f}", legend_below=False), width="stretch", key="inf_forest")
    st.dataframe(table.style.format({"Estimate": "{:.3%}", "Lower": "{:.3%}", "Upper": "{:.3%}", "Width": "{:.3%}"}), width="stretch", hide_index=True)

    st.markdown("##### Exact one-sided binomial tests against a requirement")
    c = st.columns(2)
    far_req = c[0].select_slider("Requirement: FAR below", [0.005, 0.01, 0.02, 0.05, 0.10], value=0.05, key="inf_far_req", format_func=lambda v: f"{v:.1%}")
    frr_req = c[1].select_slider("Requirement: FRR below", [0.02, 0.05, 0.10, 0.15, 0.20], value=0.10, key="inf_frr_req", format_func=lambda v: f"{v:.0%}")
    t_far = stats.binomtest(k_fa, n_i, far_req, alternative="less")
    t_frr = stats.binomtest(k_fr, n_g, frr_req, alternative="less")
    st.latex(r"H_0:\ p\ge p_0\quad\text{vs}\quad H_1:\ p<p_0,\qquad \text{p-value}=P(K\le k\mid K\sim\mathrm{Bin}(n,p_0))=\sum_{j=0}^{k}\binom{n}{j}p_0^{\,j}(1-p_0)^{n-j}")
    kpi_row([(f"FAR < {far_req:.1%}?", "Shown ✓" if t_far.pvalue < 0.05 else "Not shown", f"k = {k_fa}/{n_i}, p = {t_far.pvalue:.4f}",
              "good" if t_far.pvalue < 0.05 else "bad"),
             (f"FRR < {frr_req:.0%}?", "Shown ✓" if t_frr.pvalue < 0.05 else "Not shown", f"k = {k_fr}/{n_g}, p = {t_frr.pvalue:.4f}",
              "good" if t_frr.pvalue < 0.05 else "bad"),
             ("Rule of 3 · 95% upper FAR if k = 0", f"{3 / n_i:.2%}", f"3 / {n_i}", "cyan"),
             ("Attempts to prove FAR < req. with 0 errors", f"{int(np.ceil(np.log(0.05) / np.log(1 - far_req))):,}", "n = ln 0.05 / ln(1 − p₀) ≈ 3/p₀", "")])

    st.markdown("##### How many test attempts are enough?")
    theory("Sample-size rules for biometric testing",
           "**Rule of 3:** with zero errors in n trials, the 95% upper bound on the error rate is about 3/n. "
           "**Rule of 30 (Doddington):** to be 90% confident the true rate is within ±30% of the observed one, you need at least 30 errors. "
           "**Margin of error:** for a target half-width E, the normal approximation gives the formula below.",
           [r"n \approx \frac{3}{p_0}\qquad\qquad n_{30}\approx\frac{30}{p}\qquad\qquad n=\frac{z^2\,p(1-p)}{E^2}"])
    c = st.columns(2)
    p_plan = c[0].select_slider("Expected error rate p", [0.001, 0.005, 0.01, 0.02, 0.05, 0.10], value=0.01, key="inf_p_plan", format_func=lambda v: f"{v:.1%}")
    e_rel = c[1].select_slider("Desired relative margin (± % of p)", [0.1, 0.2, 0.3, 0.5], value=0.3, key="inf_e_rel", format_func=lambda v: f"±{v:.0%}")
    z = stats.norm.ppf(0.975)
    n_margin = int(np.ceil(z * z * p_plan * (1 - p_plan) / (e_rel * p_plan) ** 2))
    kpi_row([("Rule of 3", f"{int(np.ceil(3 / p_plan)):,}", "attempts with zero errors", "cyan"),
             ("Rule of 30", f"{int(np.ceil(30 / p_plan)):,}", "attempts for ~30 errors", "warn"),
             (f"Margin ±{e_rel:.0%} (95%)", f"{n_margin:,}", "independent attempts", "good"),
             ("This dataset", f"{n_i:,} impostor", f"{n_g:,} genuine attempts", "")])
    quick_read(f"With {n_i} impostor attempts the smallest non-zero FAR we can observe is {1 / n_i:.2%}, and even zero errors only bounds FAR below {3 / n_i:.2%}. "
               "Claims about very low FAR therefore need far larger tests — and because attempts from the same user are correlated, the effective "
               "sample size is smaller still (see the user-level bootstrap tab).", tone="caution")


# -----------------------------------------------------------------------------
# Non-parametric and variance tests
# -----------------------------------------------------------------------------
def render_nonparametric(df):
    genuine = df.loc[df.Class == "Genuine", "Similarity_Score"].to_numpy(float)
    impostor = df.loc[df.Class == "Impostor", "Similarity_Score"].to_numpy(float)
    mw = stats.mannwhitneyu(genuine, impostor, alternative="greater")
    auc = mw.statistic / (len(genuine) * len(impostor))
    ks2 = stats.ks_2samp(genuine, impostor)
    lev = stats.levene(genuine, impostor, center="median")
    vg, vi = np.var(genuine, ddof=1), np.var(impostor, ddof=1)
    f = vi / vg
    p_f = 2 * min(stats.f.sf(f, len(impostor) - 1, len(genuine) - 1), stats.f.cdf(f, len(impostor) - 1, len(genuine) - 1))
    t_pool = stats.ttest_ind(genuine, impostor, equal_var=True)
    t_welch = stats.ttest_ind(genuine, impostor, equal_var=False)

    rows = [
        {"Test": "Welch t-test", "H₀": "μ_G = μ_I (variances may differ)", "Statistic": t_welch.statistic, "p-value": t_welch.pvalue, "Assumes": "≈ normal means (CLT)"},
        {"Test": "Pooled (Student) t-test", "H₀": "μ_G = μ_I, σ_G = σ_I", "Statistic": t_pool.statistic, "p-value": t_pool.pvalue, "Assumes": "equal variances"},
        {"Test": "Mann–Whitney U (one-sided)", "H₀": "P(X_G > X_I) = ½", "Statistic": mw.statistic, "p-value": mw.pvalue, "Assumes": "nothing about shape"},
        {"Test": "Two-sample Kolmogorov–Smirnov", "H₀": "F_G = F_I", "Statistic": ks2.statistic, "p-value": ks2.pvalue, "Assumes": "continuous scores"},
        {"Test": "Levene (median-centred)", "H₀": "σ²_G = σ²_I", "Statistic": lev.statistic, "p-value": lev.pvalue, "Assumes": "robust to non-normality"},
        {"Test": "F-test for variances", "H₀": "σ²_G = σ²_I", "Statistic": f, "p-value": p_f, "Assumes": "normality (sensitive)"},
    ]
    st.dataframe(pd.DataFrame(rows).style.format({"Statistic": "{:.4f}", "p-value": "{:.3e}"}), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        theory("Mann–Whitney U equals the AUC",
               "The rank-sum statistic counts how often a genuine score beats an impostor score. Divided by the number of pairs it is "
               "exactly the area under the ROC curve — a test and a performance metric in one.",
               [r"\mathrm{AUC}=\frac{U}{n_G\,n_I}=P(X_G>X_I)", rf"=\frac{{{mw.statistic:,.0f}}}{{{len(genuine)}\times{len(impostor)}}}={auc:.4f}"])
    with c2:
        theory("Why Welch rather than the pooled t-test",
               "The pooled t-test assumes equal variances. Levene's test checks that assumption; when it is rejected, Welch's "
               "unequal-variance t-test (used on the first tab) is the safe choice.",
               [rf"s_I^2/s_G^2={f:.2f},\qquad \text{{Levene }}p={lev.pvalue:.2e}"])
    quick_read(f"All location tests agree that genuine scores are higher. Levene's test {'rejects' if lev.pvalue < 0.05 else 'does not reject'} equal variances "
               f"(p = {lev.pvalue:.2g}), {'which justifies using Welch’s test' if lev.pvalue < 0.05 else 'so pooled and Welch t-tests give similar answers'}. "
               "The rank-based Mann–Whitney test needs no normality assumption at all.", tone="good")

    st.markdown("##### Is FAR the same for two groups of users? (two-proportion z-test)")
    users = sorted(df.User_ID.unique())
    half = len(users) // 2
    t = st.slider("Threshold for the comparison", 0.3, 0.95, 0.60, 0.01, key="inf_2p_t")
    grp = np.where(df.User_ID.isin(users[:half]), "A", "B")
    imp = df.Class == "Impostor"
    k1, n1 = int(((df.Similarity_Score >= t) & imp & (grp == "A")).sum()), int((imp & (grp == "A")).sum())
    k2, n2 = int(((df.Similarity_Score >= t) & imp & (grp == "B")).sum()), int((imp & (grp == "B")).sum())
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    zstat = (p1 - p2) / se if se > 0 else 0.0
    pz = 2 * stats.norm.sf(abs(zstat))
    st.latex(rf"z=\frac{{\hat p_A-\hat p_B}}{{\sqrt{{\hat p(1-\hat p)(\frac1{{n_A}}+\frac1{{n_B}})}}}}=\frac{{{p1:.4f}-{p2:.4f}}}{{{se:.4f}}}={zstat:.3f},\qquad p={pz:.3f}")
    st.caption(f"Group A = users {users[0]}–{users[half - 1]} ({k1}/{n1} false accepts) · Group B = users {users[half]}–{users[-1]} ({k2}/{n2}). "
               f"{'No evidence that FAR differs between the groups.' if pz >= 0.05 else 'FAR differs between the groups — check per-user risk.'}")


# -----------------------------------------------------------------------------
# Errors, power and sample size
# -----------------------------------------------------------------------------
def t_test_power(d, n, alpha=0.05):
    df_ = 2 * n - 2
    ncp = d * np.sqrt(n / 2)
    tcrit = stats.t.ppf(1 - alpha / 2, df_)
    return stats.nct.sf(tcrit, df_, ncp) + stats.nct.cdf(-tcrit, df_, ncp)


def render_power(df):
    genuine = df.loc[df.Class == "Genuine", "Similarity_Score"].to_numpy(float)
    impostor = df.loc[df.Class == "Impostor", "Similarity_Score"].to_numpy(float)
    st.markdown("##### Hypothesis-test errors are biometric errors")
    st.dataframe(pd.DataFrame([
        {"": "H₀ true (the user IS genuine)", "Accept the claim": "✅ Correct — true accept (TAR)", "Reject the claim": "❌ Type I error = false reject (FRR = α)"},
        {"": "H₀ false (the user is an impostor)", "Accept the claim": "❌ Type II error = false accept (FAR = β)", "Reject the claim": "✅ Correct — true reject (power = 1 − FAR)"},
    ]), width="stretch", hide_index=True)
    st.caption("Treating “this person is who they claim to be” as H₀, the threshold is the critical value of a one-sample test on the score. "
               "Lowering α (fewer false rejects) necessarily raises β (more false accepts) for a fixed sample — the same trade-off as moving the threshold.")

    pooled = np.sqrt((np.var(genuine, ddof=1) + np.var(impostor, ddof=1)) / 2)
    d_obs = (genuine.mean() - impostor.mean()) / pooled
    theory("Power of a two-sample t-test",
           "Power is the probability of detecting a real difference. It grows with the effect size d and the sample size n, and is computed from the "
           "non-central t distribution.",
           [r"\text{power}=P\big(|T|>t_{1-\alpha/2,\,\nu}\big),\quad T\sim t_\nu(\delta),\quad \delta=d\sqrt{n/2},\quad \nu=2n-2"])
    alpha = st.select_slider("Significance level α", [0.01, 0.05, 0.10], value=0.05, key="inf_pow_alpha")
    ns = np.arange(2, 121)
    fig = go.Figure()
    for d, name, color in ((0.2, "small d = 0.2", MUTED), (0.5, "medium d = 0.5", CYAN), (0.8, "large d = 0.8", ACCENT)):
        fig.add_trace(go.Scatter(x=ns, y=[t_test_power(d, n, alpha) for n in ns], name=name, line=dict(color=color, width=2.4)))
    fig.add_trace(go.Scatter(x=ns, y=[t_test_power(d_obs, n, alpha) for n in ns], name=f"observed d = {d_obs:.2f}", line=dict(color=GENUINE, width=3)))
    fig.add_hline(y=0.8, line=dict(color=WARN, dash="dot"), annotation_text="80% power", annotation_font_color=WARN)
    fig.update_xaxes(title="Attempts per class n"); fig.update_yaxes(title="Power", tickformat=".0%", range=[0, 1.02])
    st.plotly_chart(style_plotly(fig, 340, "Power curves"), width="stretch", key="inf_power")
    need = {d: next((n for n in ns if t_test_power(d, n, alpha) >= 0.8), None) for d in (0.2, 0.5, 0.8)}
    kpi_row([("n for 80% power · d = 0.5", f"{need[0.5]}" if need[0.5] else "> 120", "per class", "cyan"),
             ("n for 80% power · d = 0.8", f"{need[0.8]}" if need[0.8] else "> 120", "per class", ""),
             ("Observed effect d", f"{d_obs:.2f}", "very large separation", "good"),
             ("Power with current data", f"{t_test_power(d_obs, min(len(genuine), len(impostor)), alpha):.4f}", f"n = {min(len(genuine), len(impostor))}", "good")])
    quick_read("Because genuine and impostor means are many standard deviations apart, even a handful of attempts would detect the difference. "
               "The hard statistical problem in biometrics is not *whether* the classes differ but how precisely the rare error rates are estimated.")
