"""Advanced operating-point selection and multi-threshold analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.calculations import cluster_bootstrap_rates
from utils.theme import ACCENT, GENUINE, IMPOSTOR, LINE, PANEL, TEXT, WARN
from utils.ui import active_threshold, require_dataset, set_threshold


TARGET_FARS = {
    "10% FAR": 0.10,
    "5% FAR": 0.05,
    "1% FAR": 0.01,
    "0.1% FAR": 0.001,
}


def _split_by_user(dataset):
    """Split whole users into development and test sets for honest threshold checks."""
    users = np.asarray(sorted(dataset["User_ID"].dropna().unique()))
    if len(users) < 2:
        return dataset.copy(), dataset.copy(), False

    shuffled = np.random.default_rng(7).permutation(users)
    dev_users = set(shuffled[: max(1, len(shuffled) // 2)])
    dev = dataset[dataset["User_ID"].isin(dev_users)].copy()
    test = dataset[~dataset["User_ID"].isin(dev_users)].copy()
    return dev, test, True


def _split_scores(dataset):
    genuine = dataset.loc[dataset["Class"] == "Genuine", "Similarity_Score"].to_numpy(float)
    impostor = dataset.loc[dataset["Class"] == "Impostor", "Similarity_Score"].to_numpy(float)
    return genuine, impostor


def _threshold_for_target_far(impostor_dev, target_far):
    """Choose the lowest score boundary that keeps development FAR at or below target."""
    ordered = np.sort(np.asarray(impostor_dev, dtype=float))[::-1]
    if not len(ordered):
        return 1.0

    allowed_false_accepts = int(np.floor(target_far * len(ordered)))
    if allowed_false_accepts >= len(ordered):
        return 0.0

    # The project scores are stored to four decimals. A small step above the
    # boundary excludes tied boundary scores while remaining score-scale aware.
    boundary = float(ordered[allowed_false_accepts])
    return min(boundary + 0.00001, 1.0)


def _rates(genuine, impostor, threshold):
    tar = float((genuine >= threshold).mean()) if len(genuine) else 0.0
    far = float((impostor >= threshold).mean()) if len(impostor) else 0.0
    return {"TAR": tar, "FRR": 1 - tar, "FAR": far, "TRR": 1 - far}


def _apply_threshold(label, threshold):
    """Apply a selected single threshold throughout the app."""
    set_threshold(threshold, label)


def _cost_at(genuine, impostor, reject_threshold, accept_threshold,
             impostor_prior, false_accept_cost, false_reject_cost, review_cost):
    """Expected per-attempt cost and operating rates for a three-zone policy."""
    pg = float((genuine < reject_threshold).mean()) if len(genuine) else 0.0
    rg = float(((genuine >= reject_threshold) & (genuine < accept_threshold)).mean()) if len(genuine) else 0.0
    ai = float((impostor >= accept_threshold).mean()) if len(impostor) else 0.0
    ri = float(((impostor >= reject_threshold) & (impostor < accept_threshold)).mean()) if len(impostor) else 0.0
    genuine_prior = 1 - impostor_prior
    cost = genuine_prior * (false_reject_cost * pg + review_cost * rg)
    cost += impostor_prior * (false_accept_cost * ai + review_cost * ri)
    return {
        "cost": float(cost),
        "hard_reject_rate": pg,
        "genuine_review_rate": rg,
        "accept_far": ai,
        "impostor_review_rate": ri,
        "review_rate": genuine_prior * rg + impostor_prior * ri,
    }


def _best_cost_policies(genuine_dev, impostor_dev, genuine_test, impostor_test,
                        impostor_prior, false_accept_cost, false_reject_cost, review_cost):
    """Search development-set thresholds and report costs on unseen users."""
    observed_scores = np.concatenate([genuine_dev, impostor_dev])
    if not len(observed_scores):
        grid = np.linspace(0, 1, 201)
    elif len(observed_scores) <= 1200:
        grid = np.unique(np.concatenate([
            np.array([0.0, 1.0]),
            observed_scores,
            np.minimum(observed_scores + 0.00001, 1.0),
        ]))
    else:
        grid = np.unique(np.concatenate([
            np.linspace(0, 1, 201),
            np.quantile(observed_scores, np.linspace(0, 1, 201)),
        ]))

    sorted_genuine = np.sort(genuine_dev)
    sorted_impostor = np.sort(impostor_dev)
    f_g = np.searchsorted(sorted_genuine, grid, side="left") / max(len(sorted_genuine), 1)
    f_i = np.searchsorted(sorted_impostor, grid, side="left") / max(len(sorted_impostor), 1)

    reject = grid[:, None]
    accept = grid[None, :]
    genuine_reject = f_g[:, None]
    genuine_review = f_g[None, :] - f_g[:, None]
    impostor_accept = 1 - f_i[None, :]
    impostor_review = f_i[None, :] - f_i[:, None]
    cost = ((1 - impostor_prior) * (false_reject_cost * genuine_reject + review_cost * genuine_review)
            + impostor_prior * (false_accept_cost * impostor_accept + review_cost * impostor_review))
    cost = np.where(reject <= accept, cost, np.inf)

    zone_index = np.unravel_index(np.argmin(cost), cost.shape)
    reject_threshold = float(grid[zone_index[0]])
    accept_threshold = float(grid[zone_index[1]])

    single_costs = np.diag(cost)
    single_index = int(np.argmin(single_costs))
    single_threshold = float(grid[single_index])

    zone_test = _cost_at(
        genuine_test, impostor_test, reject_threshold, accept_threshold,
        impostor_prior, false_accept_cost, false_reject_cost, review_cost,
    )
    single_test = _cost_at(
        genuine_test, impostor_test, single_threshold, single_threshold,
        impostor_prior, false_accept_cost, false_reject_cost, review_cost,
    )
    return {
        "reject_threshold": reject_threshold,
        "accept_threshold": accept_threshold,
        "single_threshold": single_threshold,
        "zone_test": zone_test,
        "single_test": single_test,
        "cost_reduction": (1 - zone_test["cost"] / single_test["cost"])
        if single_test["cost"] else 0.0,
        "grid": grid,
    }


def _comparison_chart(genuine_test, impostor_test, operating_points):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=genuine_test, name="Genuine", histnorm="probability density",
        marker_color=GENUINE, opacity=0.58, xbins=dict(start=0, end=1, size=0.035),
    ))
    fig.add_trace(go.Histogram(
        x=impostor_test, name="Impostor", histnorm="probability density",
        marker_color=IMPOSTOR, opacity=0.58, xbins=dict(start=0, end=1, size=0.035),
    ))
    palette = [ACCENT, WARN, "#C084FC", "#F472B6"]
    for (label, point), color in zip(operating_points.items(), palette):
        fig.add_vline(
            x=point["threshold"], line=dict(color=color, dash="dash", width=1.5),
            annotation_text=label, annotation_position="top", annotation_font_color=color,
        )
    fig.update_layout(
        barmode="overlay", title="Held-out score distributions and dev-selected thresholds",
        height=390, margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Plus Jakarta Sans, sans-serif"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
        hoverlabel=dict(bgcolor="#0B1220"),
    )
    fig.update_xaxes(range=[0, 1], title="Similarity score", gridcolor=LINE, zeroline=False)
    fig.update_yaxes(title="Density", gridcolor=LINE, zeroline=False)
    return fig


def _decision_diagram(genuine, impostor, accept_threshold, preview_score, reject_threshold=None):
    """Show score distributions alongside the active reject / review / accept regions."""
    fig = go.Figure()
    if reject_threshold is None:
        regions = [
            (0.0, accept_threshold, "Reject", IMPOSTOR),
            (accept_threshold, 1.0, "Accept", GENUINE),
        ]
    else:
        regions = [
            (0.0, reject_threshold, "Reject", IMPOSTOR),
            (reject_threshold, accept_threshold, "Manual review", WARN),
            (accept_threshold, 1.0, "Accept", GENUINE),
        ]

    for start, end, label, color in regions:
        if end > start:
            fig.add_vrect(
                x0=start, x1=end, fillcolor=color, opacity=0.09, line_width=0,
                layer="below",
            )
            fig.add_annotation(
                x=(start + end) / 2, y=0.98, xref="x", yref="paper", text=label,
                showarrow=False, font=dict(color=color, size=12),
            )

    fig.add_trace(go.Histogram(
        x=genuine, name="Genuine", histnorm="probability density",
        marker_color=GENUINE, opacity=0.58, xbins=dict(start=0, end=1, size=0.035),
    ))
    fig.add_trace(go.Histogram(
        x=impostor, name="Impostor", histnorm="probability density",
        marker_color=IMPOSTOR, opacity=0.58, xbins=dict(start=0, end=1, size=0.035),
    ))

    if reject_threshold is not None:
        fig.add_vline(
            x=reject_threshold, line=dict(color=WARN, dash="dash", width=2),
            annotation_text=f"Reject below {reject_threshold:.4f}",
            annotation_position="top left", annotation_font_color=WARN,
        )
    fig.add_vline(
        x=accept_threshold, line=dict(color=ACCENT, dash="dash", width=2),
        annotation_text=f"Accept at {accept_threshold:.4f}",
        annotation_position="top right", annotation_font_color=ACCENT,
    )
    fig.add_vline(
        x=preview_score, line=dict(color=TEXT, dash="dot", width=2),
        annotation_text=f"Example score {preview_score:.2f}",
        annotation_position="bottom", annotation_font_color=TEXT,
    )
    fig.update_layout(
        barmode="overlay", title="Diagrammatic threshold view · held-out scores",
        height=390, margin=dict(l=10, r=10, t=64, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Plus Jakarta Sans, sans-serif"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
        hoverlabel=dict(bgcolor="#0B1220"),
    )
    fig.update_xaxes(range=[0, 1], title="Similarity score", gridcolor=LINE, zeroline=False)
    fig.update_yaxes(title="Density", gridcolor=LINE, zeroline=False)
    return fig


def render_threshold_studio():
    """Render multiple target-FAR operating points and a review-zone policy."""
    st.markdown("""
    <div class="bio-hero">
      <div class="bio-kicker">A SIMPLE WAY TO CHOOSE</div>
      <h1>Choose a threshold</h1>
      <p>Pick what matters most. We’ll show how that choice performed before you apply it.</p>
    </div>
    """, unsafe_allow_html=True)

    if not require_dataset("Create the shared demo dataset to compare and apply threshold choices."):
        return
    dataset = st.session_state.dataset

    dev, test, user_split = _split_by_user(dataset)
    genuine_dev, impostor_dev = _split_scores(dev)
    genuine_test, impostor_test = _split_scores(test)
    if not len(genuine_dev) or not len(impostor_dev) or not len(genuine_test) or not len(impostor_test):
        st.warning("Both development and test groups need Genuine and Impostor scores to compare thresholds.")
        return

    if user_split:
        st.markdown(
            f'<div class="bio-overview"><b>Evaluation split</b><p>{dev.User_ID.nunique()} development users set each threshold; '
            f'{test.User_ID.nunique()} separate users measure test FAR, TAR, and FRR.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="bio-overview"><b>Evaluation split</b><p>Only one user is available, so development and test use the same records. '
            'Treat the rates as exploratory rather than held-out estimates.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">What matters most?</div>', unsafe_allow_html=True)

    operating_points = {}
    rows = []
    for label, target_far in TARGET_FARS.items():
        threshold = _threshold_for_target_far(impostor_dev, target_far)
        dev_rates = _rates(genuine_dev, impostor_dev, threshold)
        test_rates = _rates(genuine_test, impostor_test, threshold)
        operating_points[label] = {"threshold": threshold, **test_rates, "target_far": target_far}
        rows.append({
            "Target FAR": label,
            "Threshold (dev)": threshold,
            "Dev FAR": dev_rates["FAR"],
            "Test FAR": test_rates["FAR"],
            "Test TAR": test_rates["TAR"],
            "Test FRR": test_rates["FRR"],
            "Test false accepts": f"{int((impostor_test >= threshold).sum())} / {len(impostor_test)}",
        })

    quick_choices = [
        ("0.1% FAR", "Higher security", "Tighter impostor control; genuine users may face more rejections.", "#FF7185", "🔒"),
        ("1% FAR", "Balanced · recommended", "A practical starting point for balancing security and convenience.", "#A78BFA", "⚖️"),
        ("5% FAR", "Easier access", "More genuine attempts pass, with a higher chance of accepting an impostor.", "#35D39A", "↗"),
    ]
    choice_columns = st.columns(3)
    for column, (label, title, description, color, icon) in zip(choice_columns, quick_choices):
        point = operating_points[label]
        threshold = point["threshold"]
        with column:
            st.markdown(
                f'<div style="background:#171A33;border:1px solid #39436D;border-left:5px solid {color};'
                f'border-radius:14px;padding:16px 18px;min-height:142px;margin-bottom:12px">'
                f'<div style="color:{color};font-size:1.03rem;font-weight:800">{icon} {title}</div>'
                f'<div style="color:#AAB4D3;margin-top:8px;line-height:1.5">{description}</div>'
                f'<div style="color:#F4F5FF;margin-top:10px;font-weight:700">Target false accepts: {label}</div></div>',
                unsafe_allow_html=True,
            )
            st.metric("Observed test FAR", f"{point['FAR']:.2%}")
            st.caption(f"Genuine accepted: {point['TAR']:.1%} · Genuine not accepted: {point['FRR']:.1%} · T={threshold:.4f}")
            if st.button(f"Use {title.split(' ·')[0]}", key=f"quick_{label}", type="primary", width="stretch"):
                _apply_threshold(label, threshold)
                st.success(f"{title} is active at threshold {threshold:.5f}.")

    st.markdown('<div class="section-title">Fine-tune a custom threshold</div>', unsafe_allow_html=True)
    st.caption(
        "Adjust any cutoff from 0 to 1. Development rates show the data used while tuning; held-out rates estimate "
        "how the selected cutoff performs for separate users. Choosing based on held-out rates can overfit that set."
    )
    current_threshold = active_threshold()
    custom_threshold = st.slider(
        "Custom acceptance threshold", min_value=0.0, max_value=1.0,
        value=round(float(current_threshold), 2), step=0.01, format="%.2f",
        key="custom_threshold_slider",
        help="Scores at or above this value are accepted; lower scores are rejected.",
    )
    custom_dev_rates = _rates(genuine_dev, impostor_dev, custom_threshold)
    custom_test_rates = _rates(genuine_test, impostor_test, custom_threshold)
    try:
        custom_uncertainty = cluster_bootstrap_rates(test, custom_threshold, n_boot=500, random_state=42)
    except ValueError:
        custom_uncertainty = None
    custom_metrics = st.columns(5)
    custom_metrics[0].metric("Threshold", f"{custom_threshold:.2f}")
    custom_metrics[1].metric("Dev FAR", f"{custom_dev_rates['FAR']:.2%}")
    custom_metrics[2].metric("Held-out FAR", f"{custom_test_rates['FAR']:.2%}")
    custom_metrics[3].metric("Held-out TAR", f"{custom_test_rates['TAR']:.2%}")
    custom_metrics[4].metric("Held-out FRR", f"{custom_test_rates['FRR']:.2%}")
    if custom_uncertainty is not None:
        far_low, far_high = custom_uncertainty["FAR_CI"]
        tar_low, tar_high = custom_uncertainty["TAR_CI"]
        st.caption(
            f"95% user-level bootstrap intervals · FAR {far_low:.2%}–{far_high:.2%} · "
            f"TAR {tar_low:.2%}–{tar_high:.2%} · FRR {1 - tar_high:.2%}–{1 - tar_low:.2%} "
            f"({custom_uncertainty['resamples']} resamples across {custom_uncertainty['users']} held-out users)."
        )
    else:
        st.caption("User-level uncertainty intervals require at least two held-out users.")
    if st.button("Apply custom threshold", key="apply_custom_threshold", type="primary", width="stretch"):
        _apply_threshold("Custom threshold", custom_threshold)
        st.success(f"Custom threshold {custom_threshold:.2f} is active across the app.")

    active_profile = st.session_state.get("active_threshold_profile")
    if active_profile:
        if st.session_state.get("locked_threshold") is not None:
            st.caption(f"Active choice: **{active_profile}** · threshold {st.session_state.locked_threshold:.5f}")
        else:
            st.caption(f"Active choice: **{active_profile}** · review policy is on the Authentication page")

    st.markdown('<div class="section-title">Try a similarity score</div>', unsafe_allow_html=True)
    active_policy = st.session_state.get("threshold_policy")
    if active_policy and active_policy.get("mode") == "three_zone":
        reject_boundary = float(active_policy["reject_threshold"])
        accept_boundary = float(active_policy["accept_threshold"])
        preview_label = "Active review-zone policy"
    elif st.session_state.get("locked_threshold") is not None:
        reject_boundary = None
        accept_boundary = float(st.session_state.locked_threshold)
        preview_label = st.session_state.get("active_threshold_profile") or "Active threshold"
    else:
        reject_boundary = None
        accept_boundary = float(operating_points["1% FAR"]["threshold"])
        preview_label = "Balanced preview — apply a choice above to use it"

    preview_columns = st.columns([1, 1.2])
    preview_score = preview_columns[0].slider(
        "Example similarity score", 0.0, 1.0, 0.75, 0.01, key="threshold_score_preview",
        help="Move this score to see how the active threshold classifies it.",
    )
    preview_columns[1].metric("Threshold setting", preview_label)

    if reject_boundary is None:
        accepted = preview_score >= accept_boundary
        result_text = "ACCEPTED" if accepted else "REJECTED"
        result_color = GENUINE if accepted else IMPOSTOR
        left_pct = max(0.0, min(100.0, accept_boundary * 100))
        track = f"linear-gradient(90deg, {IMPOSTOR} 0%, {IMPOSTOR} {left_pct:.3f}%, {GENUINE} {left_pct:.3f}%, {GENUINE} 100%)"
        boundary_text = f"Scores below {accept_boundary:.4f} are rejected; scores at or above it are accepted."
    else:
        if preview_score < reject_boundary:
            result_text, result_color = "REJECTED", IMPOSTOR
        elif preview_score < accept_boundary:
            result_text, result_color = "MANUAL REVIEW", WARN
        else:
            result_text, result_color = "ACCEPTED", GENUINE
        reject_pct = max(0.0, min(100.0, reject_boundary * 100))
        accept_pct = max(reject_pct, min(100.0, accept_boundary * 100))
        track = (f"linear-gradient(90deg, {IMPOSTOR} 0%, {IMPOSTOR} {reject_pct:.3f}%, "
                 f"{WARN} {reject_pct:.3f}%, {WARN} {accept_pct:.3f}%, "
                 f"{GENUINE} {accept_pct:.3f}%, {GENUINE} 100%)")
        boundary_text = f"Below {reject_boundary:.4f}: reject · {reject_boundary:.4f}–{accept_boundary:.4f}: review · {accept_boundary:.4f} or above: accept."

    st.markdown(
        f'<div style="background:{PANEL};border:1px solid {LINE};border-left:5px solid {result_color};'
        f'border-radius:14px;padding:14px 18px;margin:8px 0">'
        f'<div style="font-size:1.15rem;font-weight:800;color:{result_color}">Score {preview_score:.2f}: {result_text}</div>'
        f'<div style="height:14px;border-radius:99px;background:{track};margin:14px 0 8px"></div>'
        f'<div style="color:{TEXT};font-size:.92rem">{boundary_text}</div>'
        f'<div style="display:flex;justify-content:space-between;color:#AAB4D3;font-size:.78rem;margin-top:5px">'
        f'<span>0.00 · Reject</span><span>Threshold boundary</span><span>1.00 · Accept</span></div></div>',
        unsafe_allow_html=True,
    )

    if st.checkbox("Show diagrammatic threshold view", value=True, key="show_threshold_diagram"):
        st.caption("Shaded areas show the decision regions. The dotted line follows the example score slider above.")
        st.plotly_chart(
            _decision_diagram(
                genuine_test, impostor_test, accept_boundary, preview_score,
                reject_threshold=reject_boundary,
            ),
            width="stretch", key="threshold_decision_diagram",
        )

    with st.expander("Advanced: compare all FAR targets", expanded=False):
        st.caption("Thresholds are chosen using one group of users, then measured on separate users where possible.")
        threshold_table = pd.DataFrame(rows)
        st.dataframe(
            threshold_table.style.format({
                "Threshold (dev)": "{:.5f}", "Dev FAR": "{:.2%}", "Test FAR": "{:.2%}",
                "Test TAR": "{:.2%}", "Test FRR": "{:.2%}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption(
            f"Test resolution is 1 / {len(impostor_test):,} = {1 / len(impostor_test):.2%}. "
            "A very small target can show zero observed false accepts in this sample; that does not prove zero population risk."
        )
        advanced_label = st.selectbox("Choose a different target", list(TARGET_FARS), index=2)
        advanced_threshold = operating_points[advanced_label]["threshold"]
        if st.button("Use this target instead", key="apply_advanced_target"):
            set_threshold(advanced_threshold, advanced_label)
            st.success(f"{advanced_label} applied at threshold {advanced_threshold:.5f}.")
        st.plotly_chart(_comparison_chart(genuine_test, impostor_test, operating_points), width="stretch")

    with st.expander("Advanced: set up a manual review zone", expanded=False):
        st.caption("Scores in the review band go to a person instead of being automatically accepted or rejected. Costs and impostor share are assumptions you can change.")
        controls = st.columns(4)
        impostor_prior = controls[0].slider("Impostor attempt share", 0.1, 20.0, 2.0, 0.1, format="%.1f%%") / 100
        false_accept_cost = controls[1].number_input("False accept cost", min_value=1.0, max_value=10000.0, value=500.0, step=10.0)
        false_reject_cost = controls[2].number_input("False reject cost", min_value=0.0, max_value=1000.0, value=3.0, step=1.0)
        review_cost = controls[3].number_input("Manual review cost", min_value=0.0, max_value=1000.0, value=1.0, step=1.0)

        policy = _best_cost_policies(
            genuine_dev, impostor_dev, genuine_test, impostor_test,
            impostor_prior, false_accept_cost, false_reject_cost, review_cost,
        )
        policy_metrics = st.columns(4)
        policy_metrics[0].metric("Reject below", f"{policy['reject_threshold']:.4f}")
        policy_metrics[1].metric("Accept at / above", f"{policy['accept_threshold']:.4f}")
        policy_metrics[2].metric("Review share · test", f"{policy['zone_test']['review_rate']:.1%}")
        policy_metrics[3].metric("Test cost change", f"{policy['cost_reduction']:.1%}", help="Compared with the best single threshold on held-out users.")

        policy_table = pd.DataFrame([
            {
                "Policy": "Reject / review / accept",
                "Development thresholds": f"<{policy['reject_threshold']:.4f} · review · ≥{policy['accept_threshold']:.4f}",
                "Test expected cost": policy["zone_test"]["cost"],
                "Test accept FAR": policy["zone_test"]["accept_far"],
                "Test hard-reject FRR": policy["zone_test"]["hard_reject_rate"],
                "Test review share": policy["zone_test"]["review_rate"],
            },
            {
                "Policy": "Best single threshold",
                "Development thresholds": f"{policy['single_threshold']:.4f}",
                "Test expected cost": policy["single_test"]["cost"],
                "Test accept FAR": policy["single_test"]["accept_far"],
                "Test hard-reject FRR": policy["single_test"]["hard_reject_rate"],
                "Test review share": policy["single_test"]["review_rate"],
            },
        ])
        st.dataframe(
            policy_table.style.format({
                "Test expected cost": "{:.3f}", "Test accept FAR": "{:.2%}",
                "Test hard-reject FRR": "{:.2%}", "Test review share": "{:.2%}",
            }),
            width="stretch", hide_index=True,
        )
        if st.button("Use this review policy on Authentication", width="stretch"):
            st.session_state.threshold = float(policy["accept_threshold"])
            st.session_state.locked_threshold = None
            st.session_state.threshold_policy = {
                "mode": "three_zone",
                "reject_threshold": float(policy["reject_threshold"]),
                "accept_threshold": float(policy["accept_threshold"]),
            }
            st.session_state.active_threshold_profile = "Cost-aware three-zone"
            st.success("Review policy is active on the Authentication page.")
        st.caption("This review policy applies to the interactive Authentication page. Other result pages use the accept boundary as one threshold.")
        st.info(
            "Quality-aware thresholds need per-attempt image-quality data, which this dataset does not include. "
            "This simulation is educational and does not establish production biometric security."
        )
