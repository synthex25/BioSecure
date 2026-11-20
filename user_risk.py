"""User Risk Profiling: per-user error rates and goat / lamb / sheep tagging."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.advanced import user_profiles
from utils.theme import ACCENT, CYAN, GENUINE, IMPOSTOR, MUTED, TEXT, WARN, style_plotly
from utils.ui import active_threshold, callout, kpi_row, page_header, require_dataset

PROFILE_COLORS = {"Sheep": GENUINE, "Goat": WARN, "Lamb": CYAN, "Goat + Lamb": IMPOSTOR}


def render_user_risk():
    page_header("Advanced · Per-user", "User Risk Profiling",
                "Averages hide individuals. This page breaks results down by user and flags the people a single global "
                "threshold serves badly — the \"biometric zoo\" of Doddington et al.")
    if not require_dataset():
        return
    df = st.session_state.dataset
    threshold = st.slider("Decision threshold", 0.05, 0.99, float(round(active_threshold(), 2)), 0.01, key="risk_threshold")
    prof = user_profiles(df, threshold)
    counts = prof.Profile.value_counts()
    kpi_row([("Sheep", int(counts.get("Sheep", 0)), "well served by the threshold", "good"),
             ("Goats", int(counts.get("Goat", 0)) + int(counts.get("Goat + Lamb", 0)), "genuine attempts often rejected", "warn"),
             ("Lambs", int(counts.get("Lamb", 0)) + int(counts.get("Goat + Lamb", 0)), "easy for impostors to imitate", "cyan"),
             ("Worst user FRR", f"{prof.FRR.max():.0%}", prof.loc[prof.FRR.idxmax(), "User"], "bad")])
    callout("Quick read",
            f"Across {len(prof)} users, FRR ranges from {prof.FRR.min():.0%} to {prof.FRR.max():.0%} and FAR from {prof.FAR.min():.0%} to {prof.FAR.max():.0%}. "
            "A single global threshold gives every user the same rule, but their score distributions differ, so some users carry most of the errors.",
            "info")

    tab_map, tab_table, tab_user = st.tabs(["Risk map", "User table", "User drill-down"])
    with tab_map:
        rng = np.random.default_rng(3)  # small display-only jitter: many users share identical discrete rates
        jx, jy = rng.normal(0, .004, len(prof)), rng.normal(0, .004, len(prof))
        fig = go.Figure()
        for name, color in PROFILE_COLORS.items():
            part = prof[prof.Profile == name]
            if part.empty:
                continue
            fig.add_trace(go.Scatter(x=part.FAR + jx[part.index], y=part.FRR + jy[part.index], mode="markers+text", name=name, text=part.User, textposition="top center",
                                     textfont=dict(size=10, color=MUTED),
                                     marker=dict(color=color, size=12 + 30 * part["Risk score"], opacity=.85, line=dict(color=TEXT, width=1)),
                                     hovertemplate="%{text}<br>FAR %{x:.1%}<br>FRR %{y:.1%}<extra></extra>"))
        fig.update_xaxes(title="Per-user false accept rate", tickformat=".0%"); fig.update_yaxes(title="Per-user false reject rate", tickformat=".0%")
        st.plotly_chart(style_plotly(fig, 440, "Each bubble is a user; bigger = higher risk score"), width="stretch", key="risk_map")
        st.caption("Goat = weakest genuine matching (or FRR ≥ 20%). Lamb = most exposed to impostors (or FAR ≥ 10%). Bubble positions carry a tiny display jitter so users with identical rates stay visible. Wolves cannot be identified because impostor attempts are not tagged with an attacker ID.")

    with tab_table:
        st.dataframe(prof.style.format({"Genuine mean": "{:.3f}", "Genuine SD": "{:.3f}", "Impostor mean": "{:.3f}", "Impostor max": "{:.3f}",
                                         "FRR": "{:.0%}", "FAR": "{:.0%}", "Margin": "{:.3f}"}).background_gradient(subset=["Risk score"], cmap="magma"),
                     hide_index=True, width="stretch", height=420)
        st.download_button("Download user profiles (CSV)", prof.to_csv(index=False).encode(), "biosecure_user_profiles.csv", "text/csv")

    with tab_user:
        user = st.selectbox("User", prof.User.tolist(), key="risk_user")
        row = prof[prof.User == user].iloc[0]
        part = df[df.User_ID == user]
        fig = go.Figure()
        for label, color in (("Genuine", GENUINE), ("Impostor", IMPOSTOR)):
            fig.add_trace(go.Histogram(x=part.loc[part.Class == label, "Similarity_Score"], name=label, marker_color=color,
                                       opacity=.7, xbins=dict(start=0, end=1, size=.04)))
        fig.add_vline(x=threshold, line=dict(color=TEXT, dash="dash"), annotation_text=f"threshold {threshold:.2f}")
        fig.update_layout(barmode="overlay"); fig.update_xaxes(range=[0, 1], title="Similarity score"); fig.update_yaxes(title="Attempts")
        st.plotly_chart(style_plotly(fig, 330, f"{user} — score distributions"), width="stretch", key="risk_user_hist")
        g = part.loc[part.Class == "Genuine", "Similarity_Score"].to_numpy(); i = part.loc[part.Class == "Impostor", "Similarity_Score"].to_numpy()
        # Per-user threshold: midpoint of the two score means, weighted by spread (equal-error point of the user's own Gaussians)
        sg, si = (g.std(ddof=1) or 1e-6), (i.std(ddof=1) or 1e-6)
        t_user = float(np.clip((g.mean() * si + i.mean() * sg) / (sg + si), 0.05, 0.99))
        frr_u, far_u = float((g < t_user).mean()), float((i >= t_user).mean())
        kpi_row([("Profile", row.Profile, f"risk score {row['Risk score']:.2f}", ""),
                 ("FRR / FAR at global threshold", f"{row.FRR:.0%} / {row.FAR:.0%}", f"threshold {threshold:.2f}", "warn"),
                 ("User-specific threshold", f"{t_user:.2f}", "equal-error point of this user's scores", "cyan"),
                 ("FRR / FAR at that threshold", f"{frr_u:.0%} / {far_u:.0%}", "in-sample, optimistic", "good")])
        st.caption("Per-user thresholds are fitted and scored on the same attempts here, so they look better than they would on new data. Treat them as a diagnostic, not a recommendation.")
