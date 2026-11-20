"""Multi-Threshold Optimizer: sweep 0.00–1.00, compute FAR / FRR / TAR / Accuracy / HTER at every threshold,
recommend the minimum-HTER threshold, and apply it to the authentication system only after explicit confirmation."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.calculations import multi_threshold_optimization, performance_metrics
from utils.insights import quick_read
from utils.nav import REGISTRY
from utils.theme import ACCENT, CYAN, GENUINE, IMPOSTOR, MUTED, TEXT, WARN, style_plotly
from utils.ui import active_threshold, concept_chips, kpi_row, page_header, require_dataset, set_threshold, theory

APPLY_LABEL = "Minimum HTER (multi-threshold optimizer)"


def _rates_at(df, t):
    """Exact rates at an arbitrary threshold (the active one may lie between grid points)."""
    m = performance_metrics(df, t)
    far, frr = m["FAR"], m["FRR"]
    return {"FAR": far, "FRR": frr, "TAR": m["TAR"], "Accuracy": (m["TP"] + m["TN"]) / len(df), "HTER": (far + frr) / 2}


def _confirm_body(best, current_t, current):
    """Confirmation step: nothing changes unless the user presses Confirm."""
    st.markdown(f"This will change the operating threshold used by **Try Authentication** and every other page "
                f"from **{current_t:.4f}** to **{best['Threshold']:.2f}**.")
    rows = [("FAR (impostors accepted)", current["FAR"], best["FAR"]), ("FRR (genuine rejected)", current["FRR"], best["FRR"]),
            ("TAR", current["TAR"], best["TAR"]), ("Accuracy", current["Accuracy"], best["Accuracy"]), ("HTER", current["HTER"], best["HTER"])]
    st.table({"Metric": [r[0] for r in rows], "Current": [f"{r[1]:.2%}" for r in rows], "Recommended": [f"{r[2]:.2%}" for r in rows],
              "Change": [f"{(r[2] - r[1]) * 100:+.2f} pp" for r in rows]})
    if st.session_state.get("threshold_policy"):
        st.warning("A manual-review (three-zone) policy is active. Applying replaces it with this single threshold.")
    c1, c2 = st.columns(2)
    if c1.button("✅ Confirm and apply", type="primary", width="stretch", key="mto_confirm"):
        set_threshold(best["Threshold"], APPLY_LABEL)
        st.session_state["mto_applied"] = float(best["Threshold"])
        st.session_state["_flash"] = f"Threshold {best['Threshold']:.2f} applied to the authentication system"
        st.rerun()
    if c2.button("Cancel", width="stretch", key="mto_cancel"):
        st.session_state["_flash_info"] = "No change made — the threshold was not applied."
        st.rerun()


def _confirm_apply(best, current_t, current):
    """Open the confirmation dialog (st.dialog). Falls back to an inline confirmation box if dialogs are unavailable."""
    dialog = getattr(st, "dialog", None)
    if dialog is not None:
        dialog("Apply the recommended threshold?")(_confirm_body)(best, current_t, current)
    else:
        with st.container(border=True):
            _confirm_body(best, current_t, current)


def _xy(series):
    """Plain Python list for Plotly. Plotly >= 6 encodes pandas/NumPy columns as base64 'bdata' typed arrays, which only render
    when the chart library bundled with Streamlit can decode them; lists serialise as ordinary JSON arrays everywhere."""
    return [float(v) for v in series]


def _metric_chart(results, column, color, best, current_t, title, height=300):
    """One metric vs threshold, with the recommended threshold highlighted (dashed line + star + label)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_xy(results.Threshold), y=_xy(results[column]), mode="lines", name=column,
                             line=dict(color=color, width=2.8), hovertemplate="t = %{x:.2f}<br>" + column + " %{y:.2%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[float(best["Threshold"])], y=[float(best[column])], mode="markers+text", name="Recommended",
                             text=[f"{best[column]:.2%} @ {best['Threshold']:.2f}"], textposition="top center",
                             textfont=dict(color=GENUINE), marker=dict(color=GENUINE, size=14, symbol="star", line=dict(color=TEXT, width=1))))
    fig.add_vline(x=float(best["Threshold"]), line=dict(color=GENUINE, dash="dash", width=2))
    fig.add_vline(x=float(current_t), line=dict(color=MUTED, dash="dot", width=1.5))
    fig.update_xaxes(title="Threshold", range=[0, 1], dtick=0.2)
    fig.update_yaxes(title=column, tickformat=".0%", range=[0, float(results[column].max()) * 1.15 + 1e-9])
    style_plotly(fig, height, title, legend_below=False)
    fig.update_layout(showlegend=False, margin=dict(l=10, r=14, t=46, b=10))
    return fig


def _error_chart(results, best_t, current_t):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_xy(results.Threshold), y=_xy(results.FAR), name="FAR", line=dict(color=IMPOSTOR, width=2.4)))
    fig.add_trace(go.Scatter(x=_xy(results.Threshold), y=_xy(results.FRR), name="FRR", line=dict(color=WARN, width=2.4)))
    fig.add_trace(go.Scatter(x=_xy(results.Threshold), y=_xy(results.HTER), name="HTER = (FAR + FRR) / 2", line=dict(color=ACCENT, width=3.2)))
    best = results.loc[results.Recommended].iloc[0]
    fig.add_trace(go.Scatter(x=[float(best.Threshold)], y=[float(best.HTER)], mode="markers+text", text=[f"min HTER {best.HTER:.2%}"],
                             textposition="top right", marker=dict(color=GENUINE, size=14, symbol="star", line=dict(color=TEXT, width=1)),
                             name="Recommended", textfont=dict(color=GENUINE)))
    fig.add_vline(x=best_t, line=dict(color=GENUINE, dash="dash", width=2), annotation_text=f"recommended {best_t:.2f}",
                  annotation_font_color=GENUINE, annotation_position="top left")
    fig.add_vline(x=current_t, line=dict(color=MUTED, dash="dot"), annotation_text=f"current {current_t:.2f}",
                  annotation_font_color=MUTED, annotation_position="bottom right")
    fig.update_xaxes(title="Threshold", range=[0, 1], dtick=0.1); fig.update_yaxes(title="Error rate", tickformat=".0%")
    return style_plotly(fig, 380, "Error rates across all thresholds")


def _performance_chart(results, best_t):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_xy(results.Threshold), y=_xy(results.TAR), name="TAR", line=dict(color=GENUINE, width=2.6)))
    fig.add_trace(go.Scatter(x=_xy(results.Threshold), y=_xy(results.Accuracy), name="Accuracy", line=dict(color=CYAN, width=2.6)))
    fig.add_vline(x=best_t, line=dict(color=GENUINE, dash="dash", width=2))
    fig.update_xaxes(title="Threshold", range=[0, 1], dtick=0.1); fig.update_yaxes(title="Rate", tickformat=".0%", range=[0, 1.02])
    return style_plotly(fig, 330, "TAR and accuracy")


def _hter_zoom_chart(results, best_t, window=0.12):
    part = results[(results.Threshold >= best_t - window) & (results.Threshold <= best_t + window)]
    colors = [GENUINE if r else "rgba(167,139,250,.55)" for r in part.Recommended]
    fig = go.Figure(go.Bar(x=_xy(part.Threshold), y=_xy(part.HTER), marker_color=colors,
                           hovertemplate="t = %{x:.2f}<br>HTER %{y:.2%}<extra></extra>"))
    fig.update_xaxes(title="Threshold", dtick=0.02); fig.update_yaxes(title="HTER", tickformat=".1%")
    return style_plotly(fig, 330, f"HTER near the optimum (±{window:.2f})", legend_below=False)


def render_threshold_optimizer():
    page_header("Thresholds & decisions · Optimisation", "Multi-Threshold Optimizer",
                "Evaluate every threshold from 0.00 to 1.00, compute FAR, FRR, TAR, accuracy and HTER at each one, and find the "
                "threshold with the lowest half total error rate. You confirm before it is applied to authentication.", icon="📊")
    concept_chips("Threshold sweep", "FAR / FRR / TAR", "Accuracy", "HTER", "Grid search", "Operating point")
    if not require_dataset("No dataset in this session yet, so there is nothing to optimise. The FAR, FRR and HTER vs threshold "
                           "graphs appear here as soon as a dataset exists (a browser refresh starts a new, empty session)."):
        return
    info = st.session_state.pop("_flash_info", None)
    if info:
        st.info(info)

    df = st.session_state.dataset
    c1, c2 = st.columns([1, 2])
    step = c1.selectbox("Threshold step", [0.01, 0.005, 0.02, 0.05], index=0, format_func=lambda v: f"{v:g}", key="mto_step",
                        help="0.01 evaluates 101 thresholds (0.00, 0.01, …, 1.00).")
    results, best = multi_threshold_optimization(df, 0.0, 1.0, step)
    best_t = float(best["Threshold"])
    current_t = active_threshold()
    current = _rates_at(df, current_t)
    c2.markdown(f"<div style='padding-top:30px;color:#A9B3D6'>Evaluated <b>{len(results)}</b> thresholds from 0.00 to 1.00 on "
                f"<b>{len(df):,}</b> attempts.</div>", unsafe_allow_html=True)

    kpi_row([("Recommended threshold", f"{best_t:.2f}", "minimum HTER", "good"),
             ("HTER", f"{best['HTER']:.2%}", f"current {current['HTER']:.2%}", ""),
             ("FAR", f"{best['FAR']:.2%}", f"current {current['FAR']:.2%}", "bad"),
             ("FRR", f"{best['FRR']:.2%}", f"current {current['FRR']:.2%}", "warn"),
             ("TAR", f"{best['TAR']:.2%}", f"current {current['TAR']:.2%}", "good"),
             ("Accuracy", f"{best['Accuracy']:.2%}", f"current {current['Accuracy']:.2%}", "cyan")])
    if len(best["tied_thresholds"]) > 1:
        st.caption(f"{len(best['tied_thresholds'])} thresholds share the minimum HTER: "
                   f"{', '.join(f'{v:.2f}' for v in best['tied_thresholds'])}. The tie-break picked {best_t:.2f}.")


    # ----- Threshold performance graphs (always visible, directly under the KPIs) -----
    st.markdown('<div class="section-title">Threshold performance graphs</div>', unsafe_allow_html=True)
    st.caption(f"{len(results)} thresholds evaluated. Green star and dashed line: recommended threshold {best_t:.2f}. "
               f"Grey dotted line: current threshold {current_t:.2f}. Hover any curve for exact values; drag to zoom.")
    st.plotly_chart(_error_chart(results, best_t, current_t), width="stretch", key="mto_err")
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(_metric_chart(results, "FAR", IMPOSTOR, best, current_t, "FAR vs Threshold"), width="stretch", key="mto_far")
    g2.plotly_chart(_metric_chart(results, "FRR", WARN, best, current_t, "FRR vs Threshold"), width="stretch", key="mto_frr")
    g3.plotly_chart(_metric_chart(results, "HTER", ACCENT, best, current_t, "HTER vs Threshold"), width="stretch", key="mto_hter")
    l, r = st.columns(2)
    l.plotly_chart(_performance_chart(results, best_t), width="stretch", key="mto_perf")
    r.plotly_chart(_hter_zoom_chart(results, best_t), width="stretch", key="mto_zoom")
    max_acc = results.loc[results.Accuracy.idxmax()]
    quick_read(f"HTER is lowest at t = {best_t:.2f} ({best['HTER']:.2%}), close to where the FAR and FRR curves cross. "
               f"Accuracy peaks at t = {max_acc.Threshold:.2f} ({max_acc.Accuracy:.2%}); the two agree here because the dataset is balanced, "
               "but with far fewer impostors than genuine users, accuracy would favour a lax threshold while HTER would not.", tone="good")
    with st.expander("How the recommendation is chosen (formulas)"):
        theory("How the recommendation is chosen",
               "Every grid threshold is scored with the half total error rate: the plain average of the two error rates. Unlike accuracy, HTER "
               "does not depend on how many genuine vs impostor attempts the dataset happens to contain. Ties are broken by the most balanced point "
               "(smallest |FAR − FRR|), then by the higher (more secure) threshold.",
               [r"\mathrm{HTER}(t)=\frac{\mathrm{FAR}(t)+\mathrm{FRR}(t)}{2},\qquad t^*=\arg\min_{t\in\{0.00,\,0.01,\,\dots,\,1.00\}}\mathrm{HTER}(t)",
                r"\mathrm{Accuracy}(t)=\frac{TP+TN}{TP+TN+FP+FN},\qquad \mathrm{TAR}=1-\mathrm{FRR}"])

    # ----- Apply (with confirmation) -----
    already = abs(current_t - best_t) < 1e-9
    with st.container(border=True):
        a, b = st.columns([2.2, 1])
        if already:
            a.success(f"The recommended threshold {best_t:.2f} is already active in the authentication system.")
        else:
            gain = (current["HTER"] - best["HTER"]) * 100
            a.markdown(f"**Apply to authentication?** The current threshold is **{current_t:.4f}** "
                       f"({st.session_state.get('active_threshold_profile') or 'default'}). Switching to **{best_t:.2f}** "
                       f"changes HTER by **{-gain:+.2f} percentage points**. You will be asked to confirm.")
        if b.button("Apply recommended threshold…", type="primary", width="stretch", disabled=already, key="mto_apply"):
            _confirm_apply(best, current_t, current)
        if st.session_state.get("mto_applied") is not None and already and REGISTRY.get("auth") is not None:
            st.page_link(REGISTRY["auth"], label="Open Try Authentication to test the new threshold →", icon="🔐")

    # ----- Comparison table -----
    st.markdown('<div class="section-title">Comparison table</div>', unsafe_allow_html=True)
    fmt = {"Threshold": "{:.3f}" if step < 0.01 else "{:.2f}", "FAR": "{:.2%}", "FRR": "{:.2%}", "TAR": "{:.2%}",
           "Accuracy": "{:.2%}", "HTER": "{:.2%}"}
    st.markdown("**⭐ Recommended threshold (minimum HTER)**")
    pinned = results.loc[results.Recommended].drop(columns=["Recommended"])
    st.dataframe(pinned.style.format(fmt).set_properties(**{"background-color": "rgba(53,211,154,.30)", "font-weight": "700"}),
                 width="stretch", hide_index=True)
    view = st.radio("Show", ["All thresholds", "±0.10 around the recommendation", "10 lowest HTER"], horizontal=True, key="mto_view")
    table = results.drop(columns=["Recommended"]).copy()
    if view.startswith("±"):
        table = table[(table.Threshold >= best_t - 0.10 - 1e-9) & (table.Threshold <= best_t + 0.10 + 1e-9)]
    elif view.startswith("10"):
        table = table.sort_values(["HTER", "Threshold"]).head(10)
    nearest_current = float(results.Threshold.iloc[int((results.Threshold - current_t).abs().idxmin())])

    def highlight(row):
        if np.isclose(row.Threshold, best_t):
            return ["background-color: rgba(53,211,154,.30); font-weight: 700"] * len(row)
        if np.isclose(row.Threshold, nearest_current):
            return ["background-color: rgba(167,139,250,.22)"] * len(row)
        return [""] * len(row)

    st.dataframe(table.style.apply(highlight, axis=1).format(fmt),
                 width="stretch", hide_index=True, height=min(40 + 35 * len(table), 560))
    st.caption(f"Green row: recommended threshold {best_t:.2f}. Purple row: grid point nearest the current threshold {current_t:.4f}. "
               "TP/FN/FP/TN are attempt counts at each threshold.")
    st.download_button("Download all results (CSV)", results.to_csv(index=False).encode(), "multi_threshold_optimization.csv",
                       "text/csv", key="mto_csv")
