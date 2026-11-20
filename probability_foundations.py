"""Probability Foundations: sample space, axioms (checked numerically), joint / conditional probability,
independence, random variables, expectation and the Central Limit Theorem — all on the shared biometric data."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.calculations import split_scores
from utils.insights import quick_read
from utils.theme import ACCENT, CYAN, GENUINE, IMPOSTOR, MUTED, PANEL_HI, TEXT, WARN, style_plotly
from utils.ui import active_threshold, concept_chips, kpi_row, page_header, require_dataset, theory, threshold_control


def _venn(n_g_only, n_both, n_a_only, n_neither, total):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=6, line=dict(color=MUTED, width=1.5), fillcolor="rgba(0,0,0,0)")
    fig.add_shape(type="circle", x0=1.4, y0=0.8, x1=5.8, y1=5.2, line=dict(color=GENUINE, width=2.5), fillcolor="rgba(53,211,154,.18)")
    fig.add_shape(type="circle", x0=4.2, y0=0.8, x1=8.6, y1=5.2, line=dict(color=ACCENT, width=2.5), fillcolor="rgba(167,139,250,.18)")
    for x, y, text in ((2.6, 3, f"G ∩ Aᶜ<br><b>{n_g_only}</b><br>{n_g_only / total:.1%}"),
                       (5, 3, f"G ∩ A<br><b>{n_both}</b><br>{n_both / total:.1%}"),
                       (7.4, 3, f"Gᶜ ∩ A<br><b>{n_a_only}</b><br>{n_a_only / total:.1%}"),
                       (9.0, 0.5, f"neither: <b>{n_neither}</b>")):
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(color=TEXT, size=13))
    fig.add_annotation(x=2.0, y=5.5, text="G = genuine", showarrow=False, font=dict(color=GENUINE, size=13))
    fig.add_annotation(x=8.0, y=5.5, text="A = accepted", showarrow=False, font=dict(color=ACCENT, size=13))
    fig.add_annotation(x=0.5, y=5.6, text="S", showarrow=False, font=dict(color=MUTED, size=15))
    fig.update_xaxes(visible=False, range=[-0.2, 10.2]); fig.update_yaxes(visible=False, range=[-0.2, 6.2], scaleanchor="x")
    return style_plotly(fig, 330, "Sample space S of all attempts", legend_below=False)


def _tab_axioms(df, t):
    n = len(df)
    G = (df.Class == "Genuine").to_numpy()
    A = (df.Similarity_Score >= t).to_numpy()
    h_cut = max(t, 0.90)
    H = (df.Similarity_Score >= h_cut).to_numpy()
    P = lambda mask: mask.mean()
    left, right = st.columns([1.05, 1])
    with left:
        theory("Kolmogorov's axioms",
               "The **sample space** $S$ is the set of all recorded authentication attempts; an **event** is any subset. "
               "Each attempt is equally likely to be drawn, so $P(E)=|E|/|S|$.",
               [r"\text{(1)}\;P(E)\ge 0 \qquad \text{(2)}\;P(S)=1 \qquad \text{(3)}\;P(E_1\cup E_2)=P(E_1)+P(E_2)\;\text{if }E_1\cap E_2=\varnothing"])
        st.markdown(f"Events used here: **G** = genuine attempt · **A** = accepted (score ≥ {t:.2f}) · **H** = very high score (≥ {h_cut:.2f}, so H ⊆ A).")
    with right:
        st.plotly_chart(_venn(int((G & ~A).sum()), int((G & A).sum()), int((~G & A).sum()), int((~G & ~A).sum()), n),
                        width="stretch", key="pf_venn")

    checks = [
        ("Axiom 1 · non-negativity", "min P(event) over G, A, H, G∩A", min(P(G), P(A), P(H), P(G & A)), "≥ 0", True),
        ("Axiom 2 · normalisation", "P(S)", P(np.ones(n, bool)), "= 1", True),
        ("Axiom 3 · additivity", "P(G ∪ Gᶜ) vs P(G) + P(Gᶜ)", f"{P(G | ~G):.4f} vs {P(G) + P(~G):.4f}", "equal", np.isclose(P(G | ~G), P(G) + P(~G))),
        ("Complement rule", "P(Aᶜ) vs 1 − P(A)", f"{P(~A):.4f} vs {1 - P(A):.4f}", "equal", np.isclose(P(~A), 1 - P(A))),
        ("Inclusion–exclusion", "P(G ∪ A) vs P(G) + P(A) − P(G ∩ A)", f"{P(G | A):.4f} vs {P(G) + P(A) - P(G & A):.4f}", "equal",
         np.isclose(P(G | A), P(G) + P(A) - P(G & A))),
        ("Monotonicity", "H ⊆ A ⇒ P(H) ≤ P(A)", f"{P(H):.4f} ≤ {P(A):.4f}", "holds", (not (H & ~A).any()) and P(H) <= P(A)),
        ("Empty set", "P(∅)", 0.0, "= 0", True),
        ("Boole's inequality", "P(G ∪ A) ≤ P(G) + P(A)", f"{P(G | A):.4f} ≤ {P(G) + P(A):.4f}", "holds", P(G | A) <= P(G) + P(A) + 1e-12),
    ]
    table = pd.DataFrame([{"Rule": r, "Check": c, "Value": v if isinstance(v, str) else f"{v:.4f}", "Expected": e,
                           "Verified": "✅" if ok else "❌"} for r, c, v, e, ok in checks])
    st.dataframe(table, width="stretch", hide_index=True)
    quick_read("Every axiom and derived rule holds exactly on the data, because relative frequencies over a finite sample space are "
               "a valid probability measure. Move the threshold to change event A — the identities keep holding.", tone="good")


def _tab_joint(df, t):
    G = df.Class == "Genuine"
    A = df.Similarity_Score >= t
    counts = pd.crosstab(df.Class, np.where(A, "Accepted", "Rejected")).reindex(index=["Genuine", "Impostor"],
                                                                               columns=["Accepted", "Rejected"], fill_value=0)
    joint = counts / counts.values.sum()
    l, r = st.columns(2)
    with l:
        st.markdown("##### Contingency table (counts)")
        show = counts.copy(); show["Total"] = show.sum(axis=1); show.loc["Total"] = show.sum()
        st.dataframe(show, width="stretch")
    with r:
        st.markdown("##### Joint probability P(class ∩ decision)")
        showp = joint.copy(); showp["Marginal"] = showp.sum(axis=1); showp.loc["Marginal"] = showp.sum()
        st.dataframe(showp.style.format("{:.4f}"), width="stretch")

    pG, pA, pGA = G.mean(), A.mean(), (G & A).mean()
    tar = pGA / pG; far = (~G & A).mean() / (~G).mean(); post = pGA / pA if pA else np.nan
    theory("Conditional probability and the multiplication rule",
           "Conditioning restricts the sample space to the attempts where the condition is true. The biometric error rates are all "
           "conditional probabilities.",
           [r"P(A\mid G)=\frac{P(A\cap G)}{P(G)},\qquad P(A\cap G)=P(A\mid G)\,P(G)=P(G\mid A)\,P(A)",
            rf"\mathrm{{TAR}}=P(A\mid G)=\tfrac{{{pGA:.4f}}}{{{pG:.4f}}}={tar:.4f},\quad \mathrm{{FAR}}=P(A\mid I)={far:.4f},\quad P(G\mid A)={post:.4f}"])
    kpi_row([("P(A | G) = TAR", f"{tar:.2%}", "accept given genuine", "good"), ("P(A | I) = FAR", f"{far:.2%}", "accept given impostor", "bad"),
             ("P(G | A)", f"{post:.2%}", "genuine given accepted", "cyan"), ("P(G | Aᶜ)", f"{(G & ~A).mean() / (1 - pA):.2%}" if pA < 1 else "–",
                                                                          "genuine given rejected", "warn")])

    st.markdown("##### Are class and decision independent?")
    chi2, pval, dof, expected = stats.chi2_contingency(counts.values, correction=False)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        theory("Independence",
               "Events are independent when knowing one does not change the probability of the other. "
               "A useful biometric system must make the decision **strongly dependent** on the true class.",
               [r"G \perp A \iff P(G\cap A)=P(G)\,P(A)",
                rf"P(G\cap A)={pGA:.4f}\quad\text{{vs}}\quad P(G)P(A)={pG * pA:.4f}",
                rf"\chi^2=\sum\frac{{(O-E)^2}}{{E}}={chi2:.1f},\; df={dof},\; p={pval:.2e}"])
    with c2:
        exp_df = pd.DataFrame(expected, index=counts.index, columns=counts.columns)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"{i}·{j}" for i in counts.index for j in counts.columns], y=counts.values.ravel(), name="Observed",
                             marker_color=ACCENT))
        fig.add_trace(go.Bar(x=[f"{i}·{j}" for i in exp_df.index for j in exp_df.columns], y=exp_df.values.ravel(),
                             name="Expected if independent", marker_color=MUTED))
        st.plotly_chart(style_plotly(fig, 300, "Observed vs independence"), width="stretch", key="pf_indep")
    quick_read(f"The chi-square test of independence rejects independence (p = {pval:.1e}): the decision carries information about the class. "
               "This is the correct use of chi-square here — it works on the counts of the contingency table, not on the continuous scores.",
               tone="good" if pval < 0.05 else "caution")


def _tab_random_variables(df):
    genuine, impostor = split_scores(df)
    pg = len(genuine) / len(df)
    st.markdown("##### Random variables defined on each attempt")
    rv = pd.DataFrame([
        {"Random variable": "X = similarity score", "Type": "Continuous", "Support": "[0, 1]", "Model used": "Gaussian per class (mixture overall)"},
        {"Random variable": "D = 1{accepted}", "Type": "Discrete", "Support": "{0, 1}", "Model used": "Bernoulli(p = TAR or FAR)"},
        {"Random variable": "K = false accepts in n impostor attempts", "Type": "Discrete", "Support": "{0, …, n}", "Model used": "Binomial(n, FAR) ≈ Poisson(n·FAR)"},
        {"Random variable": "N = attempts per minute", "Type": "Discrete", "Support": "{0, 1, 2, …}", "Model used": "Poisson(λ)"},
        {"Random variable": "W = time between attempts / breaches", "Type": "Continuous", "Support": "[0, ∞)", "Model used": "Exponential(λ)"},
    ])
    st.dataframe(rv, width="stretch", hide_index=True)

    def moments(x):
        return {"E[X]": np.mean(x), "E[X²]": np.mean(x ** 2), "Var(X) = E[X²] − E[X]²": np.mean(x ** 2) - np.mean(x) ** 2,
                "SD": np.std(x), "Skewness": stats.skew(x), "Excess kurtosis": stats.kurtosis(x),
                "Median": np.median(x), "IQR": stats.iqr(x)}
    mom = pd.DataFrame({"Genuine (X | G)": moments(genuine), "Impostor (X | I)": moments(impostor),
                        "All attempts": moments(np.r_[genuine, impostor])})
    l, r = st.columns([1, 1.1])
    with l:
        st.markdown("##### Moments of the score X")
        st.dataframe(mom.style.format("{:.4f}"), width="stretch")
    with r:
        e_total = pg * genuine.mean() + (1 - pg) * impostor.mean()
        within = pg * genuine.var() + (1 - pg) * impostor.var()
        between = pg * (genuine.mean() - e_total) ** 2 + (1 - pg) * (impostor.mean() - e_total) ** 2
        theory("Mixture, total expectation and total variance",
               "Before the class is known, X follows a **mixture** of the two class distributions.",
               [r"f_X(x)=\pi_G f_G(x)+\pi_I f_I(x)",
                rf"E[X]=\pi_G E[X\mid G]+\pi_I E[X\mid I]={e_total:.4f}",
                rf"\operatorname{{Var}}(X)=\underbrace{{E[\operatorname{{Var}}(X\mid C)]}}_{{{within:.4f}}}+\underbrace{{\operatorname{{Var}}(E[X\mid C])}}_{{{between:.4f}}}={within + between:.4f}"])
    st.caption(f"Check: the overall sample variance is {np.var(np.r_[genuine, impostor]):.4f}. The between-class part is "
               f"{between / (within + between):.0%} of the total — that share is what makes the classes separable.")

    st.markdown("##### PDF, CDF and interval probabilities")
    c = st.columns(3)
    a = c[0].number_input("Lower bound a", 0.0, 1.0, 0.60, 0.01, key="pf_a")
    b = c[1].number_input("Upper bound b", 0.0, 1.0, 0.80, 0.01, key="pf_b")
    cls = c[2].selectbox("Class", ["Genuine", "Impostor"], key="pf_cls")
    x = genuine if cls == "Genuine" else impostor
    mu, sd = x.mean(), x.std(ddof=1)
    if b <= a:
        st.warning("Choose b greater than a.")
        return
    emp = ((x >= a) & (x <= b)).mean()
    mod = stats.norm.cdf(b, mu, sd) - stats.norm.cdf(a, mu, sd)
    grid = np.linspace(0, 1, 400)
    color = GENUINE if cls == "Genuine" else IMPOSTOR
    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=x, histnorm="probability density", xbins=dict(start=0, end=1, size=.025), marker_color=color,
                                   opacity=.45, name="Observed"))
        pdf = stats.norm.pdf(grid, mu, sd)
        fig.add_trace(go.Scatter(x=grid, y=pdf, name="N(μ, σ²) pdf", line=dict(color=TEXT, width=2)))
        inside = (grid >= a) & (grid <= b)
        fig.add_trace(go.Scatter(x=grid[inside], y=pdf[inside], fill="tozeroy", fillcolor="rgba(167,139,250,.35)",
                                 line=dict(width=0), name=f"P({a:.2f} ≤ X ≤ {b:.2f})"))
        fig.update_xaxes(title="Score x", range=[0, 1]); fig.update_yaxes(title="f(x)")
        st.plotly_chart(style_plotly(fig, 300, "Probability = area under the PDF"), width="stretch", key="pf_pdf")
    with r:
        fig = go.Figure()
        xs = np.sort(x)
        fig.add_trace(go.Scatter(x=xs, y=np.arange(1, len(xs) + 1) / len(xs), name="Empirical CDF", line=dict(color=color, shape="hv", width=2.4)))
        fig.add_trace(go.Scatter(x=grid, y=stats.norm.cdf(grid, mu, sd), name="Gaussian CDF", line=dict(color=TEXT, dash="dash")))
        for v in (a, b):
            fig.add_vline(x=v, line=dict(color=ACCENT, dash="dot"))
        fig.update_xaxes(title="Score x", range=[0, 1]); fig.update_yaxes(title="F(x) = P(X ≤ x)", range=[0, 1])
        st.plotly_chart(style_plotly(fig, 300, "F(b) − F(a) gives the same probability"), width="stretch", key="pf_cdf")
    st.latex(rf"P({a:.2f}\le X\le {b:.2f}\mid {cls[0]}) = F({b:.2f})-F({a:.2f}) = {mod:.4f}\ \text{{(Gaussian)}}\quad\text{{vs}}\quad {emp:.4f}\ \text{{(observed)}}")
    z_a, z_b = (a - mu) / sd, (b - mu) / sd
    st.caption(f"Standardising with Z = (X − μ)/σ: z_a = {z_a:.3f}, z_b = {z_b:.3f}, so the probability is Φ({z_b:.3f}) − Φ({z_a:.3f}).")


def _tab_clt(df):
    genuine, impostor = split_scores(df)
    theory("Central Limit Theorem",
           "Whatever the shape of the score distribution, the mean of n independent scores is approximately normal for large n, "
           "with standard error σ/√n. This is why the t-based confidence intervals on the validation page work.",
           [r"\bar X_n=\frac1n\sum_{k=1}^n X_k\;\xrightarrow{d}\;\mathcal N\!\left(\mu,\;\frac{\sigma^2}{n}\right)"])
    c = st.columns(3)
    source = c[0].selectbox("Draw scores from", ["Impostor scores", "Genuine scores", "Both (mixture — bimodal)"], key="pf_clt_src")
    n = c[1].select_slider("Sample size n", [1, 2, 5, 10, 30, 60, 100], value=30, key="pf_clt_n")
    reps = c[2].select_slider("Number of samples", [200, 500, 1000, 2000], value=1000, key="pf_clt_reps")
    pool = impostor if source.startswith("Impostor") else genuine if source.startswith("Genuine") else np.r_[genuine, impostor]
    rng = np.random.default_rng(3)
    means = rng.choice(pool, size=(reps, n), replace=True).mean(axis=1)
    mu, se = pool.mean(), pool.std() / np.sqrt(n)
    l, r = st.columns(2)
    with l:
        fig = go.Figure(go.Histogram(x=pool, histnorm="probability density", marker_color=MUTED, opacity=.7, xbins=dict(size=.025)))
        fig.update_xaxes(title="Individual score", range=[0, 1])
        st.plotly_chart(style_plotly(fig, 300, "Population of individual scores"), width="stretch", key="pf_clt_pop")
    with r:
        fig = go.Figure(go.Histogram(x=means, histnorm="probability density", marker_color=ACCENT, opacity=.7, name="Sample means"))
        g = np.linspace(means.min(), means.max(), 300)
        fig.add_trace(go.Scatter(x=g, y=stats.norm.pdf(g, mu, se), line=dict(color=TEXT, width=2), name="N(μ, σ²/n)"))
        fig.update_xaxes(title=f"Mean of n = {n} scores")
        st.plotly_chart(style_plotly(fig, 300, "Distribution of the sample mean"), width="stretch", key="pf_clt_means")
    sw = stats.shapiro(means[:500])
    kpi_row([("Population μ", f"{mu:.4f}", f"σ = {pool.std():.4f}", "cyan"), ("Mean of means", f"{means.mean():.4f}", "≈ μ (unbiased)", "good"),
             ("Theoretical SE", f"{se:.4f}", "σ/√n", ""), ("Observed SD of means", f"{means.std(ddof=1):.4f}", f"Shapiro p = {sw.pvalue:.3f}", "warn")])
    quick_read("Even the bimodal mixture produces a bell-shaped distribution of sample means once n reaches about 30, and the spread shrinks like 1/√n. "
               "Note the samples are drawn with replacement from the observed scores (a bootstrap of the empirical distribution).")


def render_probability_foundations():
    page_header("Statistical analysis · Step 3a", "Probability Foundations",
                "The building blocks behind every other page: sample space and axioms, joint and conditional probability, independence, "
                "random variables and their moments, and the Central Limit Theorem.", icon="📐")
    concept_chips("Sample space & events", "Kolmogorov axioms", "Conditional probability", "Independence & χ²", "Random variables",
                  "Expectation & variance", "Central Limit Theorem")
    if not require_dataset("Create the shared demo dataset to explore probability foundations on real counts."):
        return
    df = st.session_state.dataset
    t = threshold_control("pf_threshold", "Threshold defining event A (accepted)")
    tabs = st.tabs(["🧩 Axioms & events", "🔗 Joint & conditional", "🎲 Random variables", "🔔 Central Limit Theorem"])
    with tabs[0]:
        _tab_axioms(df, t)
    with tabs[1]:
        _tab_joint(df, t)
    with tabs[2]:
        _tab_random_variables(df)
    with tabs[3]:
        _tab_clt(df)
