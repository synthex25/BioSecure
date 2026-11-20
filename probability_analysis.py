"""Step 3b: conditional probability, Bayes' theorem and Bayesian classification of biometric scores."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.calculations import probability_analysis, split_scores
from utils.insights import quick_read
from utils.theme import ACCENT, CYAN, GENUINE, IMPOSTOR, MUTED, TEXT, WARN, style_plotly
from utils.threshold_methods import gaussian_params, kde_posterior_curve, posterior_curve
from utils.ui import concept_chips, kpi_row, page_header, require_dataset, theory, threshold_control


def _bars(labels, values, colors, title, height=300):
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.4f}" for v in values], textposition="outside"))
    fig.update_yaxes(range=[0, 1.12], title="Probability")
    return style_plotly(fig, height, title, legend_below=False)


def _tab_steps(df, threshold, r):
    total, gc, ic = r["total"], r["genuine_count"], r["impostor_count"]
    pg, pi, eg, ei = r["prior_genuine"], r["prior_impostor"], r["p_e_given_genuine"], r["p_e_given_impostor"]
    num, den = r["numerator"], r["denominator"]
    post_g = r["posterior_genuine"] if den > 0 else pg
    post_i = 1 - post_g if den > 0 else pi

    kpi_row([("Attempts", f"{total:,}", f"{gc} genuine · {ic} impostor", "cyan"), ("Prior P(G)", f"{pg:.4f}", "before seeing the score", ""),
             ("Evidence P(E)", f"{den:.4f}", f"E = score ≥ {threshold:.2f}", "warn"), ("Posterior P(G | E)", f"{post_g:.4f}", "after the evidence", "good")])

    c1, c2 = st.columns(2)
    with c1:
        theory("① Prior probability", "Class proportions before looking at any score.",
               [rf"P(G)=\frac{{{gc}}}{{{total}}}={pg:.4f},\qquad P(I)=\frac{{{ic}}}{{{total}}}={pi:.4f}"])
        theory("② Conditional probability of the evidence", f"Evidence E: the score passes the threshold {threshold:.2f}.",
               [rf"P(E\mid G)=\frac{{{r['genuine_above']}}}{{{gc}}}={eg:.4f}\;(\mathrm{{TAR}})",
                rf"P(E\mid I)=\frac{{{r['impostor_above']}}}{{{ic}}}={ei:.4f}\;(\mathrm{{FAR}})"])
    with c2:
        theory("③ Law of total probability", "E can happen through either class, so its probability is a weighted sum.",
               [r"P(E)=P(E\mid G)P(G)+P(E\mid I)P(I)", rf"={eg:.4f}\times{pg:.4f}+{ei:.4f}\times{pi:.4f}={den:.6f}"])
        theory("④ Bayes' theorem", "Reverse the conditioning: from P(E | G) to P(G | E).",
               [r"P(G\mid E)=\frac{P(E\mid G)\,P(G)}{P(E)}", rf"=\frac{{{num:.6f}}}{{{den:.6f}}}={post_g:.4f}" if den > 0 else r"P(E)=0\Rightarrow\text{no update}"])

    lr = eg / ei if ei > 0 else np.inf
    inf_tex = r"\infty"
    lr_tex = f"{lr:.3f}" if np.isfinite(lr) else inf_tex
    post_odds_tex = f"{lr * pg / pi:.3f}" if np.isfinite(lr) else inf_tex
    theory("⑤ Likelihood ratio and odds form",
           "Bayes' theorem is cleanest in odds: the evidence multiplies the prior odds by the likelihood ratio.",
           [r"\underbrace{\frac{P(G\mid E)}{P(I\mid E)}}_{\text{posterior odds}}=\underbrace{\frac{P(E\mid G)}{P(E\mid I)}}_{\text{LR}}\times\underbrace{\frac{P(G)}{P(I)}}_{\text{prior odds}}",
            rf"\mathrm{{LR}}={lr_tex},\quad \text{{prior odds}}={pg / pi:.3f},\quad \text{{posterior odds}}={post_odds_tex}"])
    a, b = st.columns(2)
    a.plotly_chart(_bars(["P(E | G)", "P(E | I)"], [eg, ei], [GENUINE, IMPOSTOR], "Conditional probability of passing"), width="stretch", key="pa_cond")
    b.plotly_chart(_bars(["Prior G", "Posterior G", "Prior I", "Posterior I"], [pg, post_g, pi, post_i],
                         [MUTED, GENUINE, MUTED, IMPOSTOR], "Prior → posterior"), width="stretch", key="pa_post")
    if den == 0:
        quick_read(f"At threshold {threshold:.2f} no attempt passes, so P(E) = 0 and the prior cannot be updated. Lower the threshold.", tone="caution")
    else:
        quick_read(f"Passing the threshold multiplies the odds of “genuine” by {lr:.1f}: the probability rises from {pg:.1%} to {post_g:.1%}. "
                   "This posterior depends on the class mix of the dataset — see the base-rate tab for what happens when impostors are rare." if np.isfinite(lr) else
                   f"No impostor passes at {threshold:.2f}, so the likelihood ratio is infinite in this sample and P(G | E) = 1. "
                   "With more data a small but non-zero FAR would appear.")
    with st.expander("Probability axioms behind these numbers"):
        st.markdown("- **Non-negativity:** every probability above is ≥ 0.\n- **Normalisation:** P(G) + P(I) = 1 and P(G | E) + P(I | E) = 1.\n"
                    "- **Additivity:** G and I are disjoint, which is what allows the law of total probability.\n\n"
                    "The *Probability Foundations* page verifies each axiom numerically on the dataset.")


def _tab_classifier(df, threshold):
    genuine, impostor = split_scores(df)
    p = gaussian_params(genuine, impostor)
    prior_default = len(genuine) / len(df)
    c = st.columns([1, 1, 1])
    prior_g = c[0].slider("Prior P(Genuine)", 0.01, 0.99, float(round(prior_default, 2)), 0.01, key="pa_cls_prior")
    model = c[1].radio("Likelihood model", ["Gaussian", "Kernel density (KDE)"], horizontal=True, key="pa_cls_model")
    score = c[2].slider("Score to classify", 0.0, 1.0, 0.68, 0.01, key="pa_cls_score")
    theory("Bayesian classification of a single score",
           "Instead of a binary pass/fail event, use the *exact* score x as evidence. Densities replace probabilities in Bayes' theorem, "
           "and the maximum-a-posteriori (MAP) rule assigns the class with the larger posterior.",
           [r"P(G\mid X=x)=\frac{\pi_G\,f_G(x)}{\pi_G\,f_G(x)+\pi_I\,f_I(x)},\qquad \hat c(x)=\begin{cases}G & P(G\mid x)\ge \tfrac12\\ I & \text{otherwise}\end{cases}"])
    x = np.linspace(0, 1, 501)
    post = posterior_curve(x, p, prior_g) if model == "Gaussian" else kde_posterior_curve(x, genuine, impostor, prior_g)
    post_at = float(np.interp(score, x, post))
    fg = stats.norm.pdf(score, p["mg"], p["sg"]) if model == "Gaussian" else float(stats.gaussian_kde(genuine)(score)[0])
    fi = stats.norm.pdf(score, p["mi"], p["si"]) if model == "Gaussian" else float(stats.gaussian_kde(impostor)(score)[0])
    boundary_idx = np.where(np.diff(np.sign(post - 0.5)) != 0)[0]
    boundary = float(x[boundary_idx[len(boundary_idx) // 2]]) if len(boundary_idx) else np.nan

    l, r = st.columns([1.5, 1])
    with l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=post, name="P(G | x)", line=dict(color=GENUINE, width=3)))
        fig.add_trace(go.Scatter(x=x, y=1 - post, name="P(I | x)", line=dict(color=IMPOSTOR, width=3)))
        fig.add_hline(y=0.5, line=dict(color=MUTED, dash="dot"))
        fig.add_vline(x=score, line=dict(color=TEXT, dash="dash"), annotation_text=f"x = {score:.2f}", annotation_font_color=TEXT)
        if np.isfinite(boundary):
            fig.add_vline(x=boundary, line=dict(color=ACCENT, width=2), annotation_text=f"MAP boundary {boundary:.3f}",
                          annotation_font_color=ACCENT, annotation_position="bottom right")
        fig.update_xaxes(title="Similarity score x", range=[0.2, 1]); fig.update_yaxes(title="Posterior probability", range=[0, 1])
        st.plotly_chart(style_plotly(fig, 340, "Posterior probability curve"), width="stretch", key="pa_post_curve")
    with r:
        verdict = "GENUINE" if post_at >= 0.5 else "IMPOSTOR"
        color = GENUINE if post_at >= 0.5 else IMPOSTOR
        st.markdown(f'<div class="verdict" style="--vc:{color}"><span>MAP decision for x = {score:.2f}</span><b>{verdict}</b>'
                    f'<em>P(G | x) = {post_at:.4f}</em></div>', unsafe_allow_html=True)
        st.latex(rf"\Lambda(x)=\frac{{f_G(x)}}{{f_I(x)}}=\frac{{{fg:.3f}}}{{{fi:.3f}}}={fg / fi if fi > 0 else np.inf:.3g}")
        st.caption(f"Log-likelihood ratio ln Λ = {np.log(fg / fi) if fi > 0 and fg > 0 else float('inf'):.2f}. "
                   f"The current operating threshold is {threshold:.3f}; the MAP boundary is where P(G | x) = ½.")

    # Classify every attempt with the MAP rule and evaluate calibration
    scores = df.Similarity_Score.to_numpy(float)
    truth = (df.Class == "Genuine").to_numpy()
    probs = np.interp(scores, x, post)
    pred = probs >= 0.5
    tp, fn, fp, tn = int((pred & truth).sum()), int((~pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & ~truth).sum())
    brier = float(np.mean((probs - truth) ** 2))
    eps = 1e-9
    logloss = float(-np.mean(truth * np.log(probs + eps) + (~truth) * np.log(1 - probs + eps)))
    kpi_row([("MAP accuracy", f"{(tp + tn) / len(df):.2%}", "all attempts", "good"), ("FAR (MAP)", f"{fp / max(fp + tn, 1):.2%}", f"{fp} impostors accepted", "bad"),
             ("FRR (MAP)", f"{fn / max(tp + fn, 1):.2%}", f"{fn} genuine rejected", "warn"), ("Brier score", f"{brier:.4f}", f"log-loss {logloss:.3f}", "cyan")])

    a, b = st.columns(2)
    with a:
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(probs, bins) - 1, 0, 9)
        rel = pd.DataFrame({"bin": idx, "p": probs, "y": truth}).groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=MUTED, dash="dot"), name="Perfect calibration"))
        fig.add_trace(go.Scatter(x=rel.pred, y=rel.obs, mode="markers+lines", marker=dict(size=np.clip(rel.n / 6, 6, 22), color=ACCENT),
                                 line=dict(color=ACCENT), name="Observed", text=rel.n, hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<br>n = %{text}<extra></extra>"))
        fig.update_xaxes(title="Predicted P(G | x)", range=[0, 1]); fig.update_yaxes(title="Observed genuine share", range=[0, 1])
        st.plotly_chart(style_plotly(fig, 320, "Reliability diagram — are the posteriors honest?"), width="stretch", key="pa_rel")
    with b:
        fig = go.Figure(go.Heatmap(z=[[tp, fn], [fp, tn]], x=["Predicted genuine", "Predicted impostor"], y=["Genuine", "Impostor"],
                                   colorscale=[[0, "#1E2245"], [1, ACCENT]], text=[[tp, fn], [fp, tn]], texttemplate="%{text}",
                                   textfont=dict(size=20), showscale=False))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_plotly(fig, 320, "MAP classifier confusion matrix", legend_below=False), width="stretch", key="pa_cm")
    quick_read("Points near the diagonal mean the posterior probabilities can be read literally (“90% means right 9 times in 10”). "
               "The Gaussian model is slightly over-confident in the tails because the real scores are clipped to [0, 1]; the KDE follows the data more closely.")


def _tab_base_rate(df, threshold):
    r = probability_analysis(df, threshold)
    tar, far = r["p_e_given_genuine"], r["p_e_given_impostor"]
    far_eff = max(far, 0.5 / max(r["impostor_count"], 1))
    theory("Base-rate effect",
           "The same TAR and FAR can give very different answers to “how sure am I that an accepted person is genuine?”, depending on how common impostors are. "
           "When impostors are common, even a small FAR produces many false accepts.",
           [r"P(G\mid A)=\frac{\mathrm{TAR}\,\pi_G}{\mathrm{TAR}\,\pi_G+\mathrm{FAR}\,(1-\pi_G)}"])
    priors = np.linspace(0.001, 0.999, 400)
    post = tar * priors / (tar * priors + far_eff * (1 - priors))
    post_rej = (1 - tar) * priors / ((1 - tar) * priors + (1 - far_eff) * (1 - priors))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=1 - priors, y=post, name="P(G | accepted)", line=dict(color=GENUINE, width=3)))
    fig.add_trace(go.Scatter(x=1 - priors, y=post_rej, name="P(G | rejected)", line=dict(color=WARN, width=2.4)))
    fig.add_vline(x=r["prior_impostor"], line=dict(color=MUTED, dash="dot"), annotation_text="dataset mix", annotation_font_color=MUTED)
    fig.update_xaxes(title="Impostor share of attempts π_I", tickformat=".0%"); fig.update_yaxes(title="Posterior P(Genuine | decision)", tickformat=".0%")
    st.plotly_chart(style_plotly(fig, 330, f"Same sensor (TAR {tar:.1%}, FAR {far:.2%}), different populations"), width="stretch", key="pa_base")
    rows = []
    for share in (0.01, 0.05, 0.20, 0.50, 0.80, 0.95):
        pg = 1 - share
        rows.append({"Impostor share": share, "P(G | accepted)": tar * pg / (tar * pg + far_eff * share),
                     "False accepts per 10,000 attempts": far * share * 10000, "Genuine rejected per 10,000": (1 - tar) * pg * 10000})
    st.dataframe(pd.DataFrame(rows).style.format({"Impostor share": "{:.0%}", "P(G | accepted)": "{:.3%}",
                                                  "False accepts per 10,000 attempts": "{:.1f}", "Genuine rejected per 10,000": "{:.1f}"}),
                 width="stretch", hide_index=True)
    if far == 0:
        st.caption(f"No impostor passed in the sample, so FAR is replaced by a half-count correction ({far_eff:.3%}) to avoid an infinite likelihood ratio.")

    st.markdown("##### Sequential updating across repeated attempts")
    st.caption("When a user presents several fingers (or retries), each result updates the posterior. Assuming independent attempts, "
               "yesterday's posterior becomes today's prior.")
    start_prior = st.slider("Starting P(G)", 0.01, 0.99, 0.50, 0.01, key="pa_seq_prior")
    defaults = ["Accept", "Reject", "Accept", "—", "—", "—"]
    cols = st.columns(6)
    seq = [o for k, (col, d) in enumerate(zip(cols, defaults))
           for o in [col.selectbox(f"Attempt {k + 1}", ["—", "Accept", "Reject"], index=["—", "Accept", "Reject"].index(d), key=f"pa_seq_{k}")]
           if o != "—"]
    post_seq = [start_prior]
    for outcome in seq:
        prev = post_seq[-1]
        lg, li = (tar, far_eff) if outcome == "Accept" else (1 - tar, 1 - far_eff)
        post_seq.append(lg * prev / (lg * prev + li * (1 - prev)))
    fig = go.Figure(go.Scatter(x=list(range(len(post_seq))), y=post_seq, mode="lines+markers+text", text=[f"{v:.3f}" for v in post_seq],
                               textposition="top center", line=dict(color=CYAN, width=3), marker=dict(size=10)))
    fig.update_xaxes(tickvals=list(range(len(post_seq))), ticktext=["prior"] + [f"{k + 1}. {o}" for k, o in enumerate(seq)])
    fig.update_yaxes(range=[0, 1.1], title="P(Genuine)")
    st.plotly_chart(style_plotly(fig, 300, "Posterior after each attempt", legend_below=False), width="stretch", key="pa_seq_chart")
    st.latex(r"P_k(G)=\frac{L_k(G)\,P_{k-1}(G)}{L_k(G)\,P_{k-1}(G)+L_k(I)\,(1-P_{k-1}(G))}")


def render_probability_analysis():
    page_header("Statistical analysis · Step 3b", "Bayes' Theorem & Classification",
                "Turn a score into a probability: priors, conditional probability, the law of total probability, Bayes' theorem in "
                "probability and odds form, and a Bayesian classifier that labels each biometric pattern.", icon="🎲")
    concept_chips("Prior & posterior", "Conditional probability", "Total probability", "Bayes' theorem", "Likelihood ratio",
                  "MAP classification", "Calibration", "Base-rate effect")
    if not require_dataset("Create the shared demo dataset to run the Bayesian analysis."):
        return
    df = st.session_state.dataset
    threshold = threshold_control("pa_threshold", "Authentication threshold (defines evidence E)")
    r = probability_analysis(df, threshold)
    tabs = st.tabs(["🧮 Bayes step by step", "🤖 Bayesian classifier", "⚖️ Base rates & sequential updates"])
    with tabs[0]:
        _tab_steps(df, threshold, r)
    with tabs[1]:
        _tab_classifier(df, threshold)
    with tabs[2]:
        _tab_base_rate(df, threshold)
