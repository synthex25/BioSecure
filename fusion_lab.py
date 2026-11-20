"""Multimodal Fusion Lab: combine fingerprint with a simulated face matcher and compare fusion rules."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.advanced import (FUSION_METHODS, far_frr_curve, fusion_table, fuse, roc_auc, simulate_face_scores)
from utils.theme import ACCENT, CYAN, GENUINE, IMPOSTOR, LINE, MUTED, TEXT, WARN, style_plotly
from utils.ui import callout, kpi_row, page_header, require_dataset

PALETTE = {"Fingerprint only": ACCENT, "Face only": CYAN, "Sum rule": GENUINE, "Weighted sum": "#7BE0B6",
           "Product rule": WARN, "Min rule": "#F59E8B", "Max rule": "#C084FC", "Logistic regression": IMPOSTOR}


def render_fusion_lab():
    page_header("Advanced · Multimodal", "Multimodal Fusion Lab",
                "Combine the fingerprint matcher with a simulated face matcher and test whether score-level fusion "
                "lowers the error rates. Rules are compared by AUC, equal error rate, decidability d′ and the "
                "FRR you must accept to hold a strict FAR budget.")
    if not require_dataset():
        return
    df = st.session_state.dataset

    c1, c2, c3, c4 = st.columns(4)
    corr = c1.slider("Sensor correlation ρ", 0.0, 0.9, 0.25, 0.05,
                     help="How strongly the two sensors tend to fail together. Fusion helps most when ρ is low.")
    quality = c2.slider("Face matcher quality", 0.4, 1.3, 1.0, 0.05,
                        help="Scales the separation between genuine and impostor face scores.")
    weight = c3.slider("Fingerprint weight (weighted sum)", 0.0, 1.0, 0.6, 0.05)
    seed = c4.number_input("Face-score seed", 0, 99999, 2026)

    face = simulate_face_scores(df, corr, quality, int(seed))
    table, scores, train_users = fusion_table(df, face, weight)
    st.session_state.fusion_summary = table.round(5).to_dict("records")
    st.session_state.fusion_params = {"correlation": corr, "quality": quality, "weight_fp": weight}

    best = table.sort_values("EER").iloc[0]
    base = table[table.Method == "Fingerprint only"].iloc[0]
    gain = (base.EER - best.EER) / base.EER if base.EER > 0 else 0
    kpi_row([
        ("Best rule", best.Method, f"EER {best.EER:.2%}", "good"),
        ("Fingerprint EER", f"{base.EER:.2%}", "single-modality baseline", ""),
        ("Relative EER gain", f"{gain:+.0%}", "best fusion vs fingerprint only", "good" if gain > 0 else "warn"),
        ("Best AUC", f"{table.AUC.max():.4f}", "area under ROC", "cyan"),
    ])
    callout("Quick read",
            f"{best.Method} gives the lowest equal error rate ({best.EER:.2%}) against {base.EER:.2%} for fingerprint alone. "
            f"At ρ = {corr:.2f} the two sensors make partly independent mistakes, so combining them removes some errors. "
            "The logistic rule learns its weights on half of the users and is scored on the other half, so its numbers are not flattered by training data.",
            "good" if gain > 0 else "caution")

    tab_cmp, tab_roc, tab_space, tab_theory = st.tabs(["Comparison", "ROC & DET", "Score space", "How fusion works"])

    with tab_cmp:
        fmt = {"AUC": "{:.4f}", "EER": "{:.2%}", "d'": "{:.2f}", "FRR @ FAR 1%": "{:.1%}", "FRR @ FAR 0.1%": "{:.1%}"}
        styled = (table.style.format(fmt)
                  .highlight_min(subset=["EER", "FRR @ FAR 1%", "FRR @ FAR 0.1%"], color="#1d5e4a")
                  .highlight_max(subset=["AUC", "d'"], color="#1d5e4a"))
        st.dataframe(styled, hide_index=True, width="stretch")
        fig = go.Figure(go.Bar(x=table.Method, y=table.EER * 100, marker_color=[PALETTE[m] for m in table.Method],
                               text=[f"{v:.2%}" for v in table.EER], textposition="outside"))
        fig.update_yaxes(title="Equal error rate (%)")
        st.plotly_chart(style_plotly(fig, 330, "Equal error rate by fusion rule (lower is better)", False),
                        width="stretch", key="fusion_eer_bar")
        st.caption("The \"d′ for logistic regression\" is computed on the model's logit, because the probability output saturates near 0 and 1.")

    is_gen = (df.Class == "Genuine").to_numpy()
    with tab_roc:
        picks = st.multiselect("Rules to plot", FUSION_METHODS, ["Fingerprint only", "Face only", "Sum rule", "Logistic regression"])
        left, right = st.columns(2)
        roc, det = go.Figure(), go.Figure()
        for m in picks:
            s = scores[m]
            t, far, frr = far_frr_curve(s[is_gen], s[~is_gen], 801)
            roc.add_trace(go.Scatter(x=far, y=1 - frr, name=f"{m} (AUC {roc_auc(s[is_gen], s[~is_gen]):.3f})",
                                     line=dict(color=PALETTE[m], width=2.4)))
            det.add_trace(go.Scatter(x=np.clip(far, 1e-3, 1), y=np.clip(frr, 1e-3, 1), name=m,
                                     line=dict(color=PALETTE[m], width=2.4), showlegend=False))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=MUTED, dash="dash"), name="Chance"))
        roc.update_xaxes(title="False accept rate", range=[0, 1]); roc.update_yaxes(title="True accept rate", range=[0, 1.01])
        det.update_xaxes(title="FAR (log)", type="log", range=[-3, 0]); det.update_yaxes(title="FRR (log)", type="log", range=[-3, 0])
        left.plotly_chart(style_plotly(roc, 380, "ROC curves"), width="stretch", key="fusion_roc")
        right.plotly_chart(style_plotly(det, 380, "DET curves (closer to the origin is better)"), width="stretch", key="fusion_det")
        st.caption("DET curves use log axes, which makes differences at very low error rates visible. Floors at 0.1% reflect the finite sample size.")

    with tab_space:
        fp = df.Similarity_Score.to_numpy()
        sample = np.random.default_rng(1).choice(len(df), size=min(len(df), 900), replace=False)
        fig = go.Figure()
        for label, color in (("Genuine", GENUINE), ("Impostor", IMPOSTOR)):
            mask = (df.Class.to_numpy()[sample] == label)
            fig.add_trace(go.Scatter(x=fp[sample][mask], y=face[sample][mask], mode="markers", name=label,
                                     marker=dict(color=color, size=6, opacity=.7)))
        # Decision boundary of the sum rule at its EER threshold
        s_sum = scores["Sum rule"]
        from utils.advanced import eer_point
        t_eer, _ = eer_point(s_sum[is_gen], s_sum[~is_gen])
        xs = np.linspace(0, 1, 50)
        fig.add_trace(go.Scatter(x=xs, y=2 * t_eer - xs, mode="lines", name=f"Sum-rule boundary (t = {t_eer:.2f})",
                                 line=dict(color=TEXT, dash="dash")))
        fig.update_xaxes(title="Fingerprint score", range=[0, 1]); fig.update_yaxes(title="Face score", range=[0, 1])
        st.plotly_chart(style_plotly(fig, 440, "Two-sensor score space"), width="stretch", key="fusion_space")
        st.caption("A single-sensor threshold is a vertical or horizontal line. Fusion draws a slanted boundary, so a weak score from one sensor can be rescued by a strong score from the other.")
        _, w = fuse(df, face, "Logistic regression", cv_users=train_users)
        st.markdown(f"**Learned logistic weights** — intercept {w[0]:+.2f}, fingerprint {w[1]:+.2f}, face {w[2]:+.2f}. "
                    f"A larger weight means that sensor is trusted more.")

    with tab_theory:
        st.markdown("""
**Score-level fusion** merges the similarity scores before a single accept/reject decision is made.

| Rule | Idea | Typical strength |
|---|---|---|
| Sum / weighted sum | average the two scores, optionally trusting one sensor more | simple, robust when both are reasonable |
| Product (geometric mean) | both sensors must agree; one very low score drags the result down | good when errors are independent |
| Min / Max | accept only if the weaker (min) or stronger (max) sensor passes | min = stricter, max = more permissive |
| Logistic regression | learns the best linear weights from labelled data | best when training data is available |

**Why it works.** If the sensors' errors are not perfectly correlated, a genuine user rejected by one matcher is often
still scored well by the other, and an impostor who fools one rarely fools both. The gain shrinks as ρ → 1.

**Honest limits.** The face scores are simulated and aligned to the fingerprint rows for demonstration. Real systems must also
normalise scores from different matchers (min-max, z-score) and verify that both samples really belong to the same person.
""")
