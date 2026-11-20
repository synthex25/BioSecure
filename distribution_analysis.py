"""Step 4: distribution analysis of the shared biometric score dataset."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from utils.calculations import (
    effect_size_and_overlap,
    performance_metrics,
    roc_metrics,
    split_scores,
    threshold_sweep,
)
from utils.insights import quick_read
from utils.ui import active_threshold, concept_chips, page_header, require_dataset, set_threshold
from utils.dist_extras import render_det_curve, render_gof_panel
from utils.theme import ACCENT, GENUINE, IMPOSTOR, MUTED, TEXT, WARN


def _finish_chart(axis, title, xlabel, ylabel, legend=True, y_limits=None):
    """Apply consistent labels, title, legend, and spacing to one chart."""
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    if legend:
        axis.legend()
    axis.figure.tight_layout()
    return axis.figure


def _score_distribution_chart(genuine, impostor, threshold):
    """Plot both observed score distributions on the same histogram."""
    figure, axis = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 26)
    axis.hist(genuine, bins=bins, alpha=0.65, color=GENUINE, label="Genuine")
    axis.hist(impostor, bins=bins, alpha=0.65, color=IMPOSTOR, label="Impostor")
    axis.axvline(threshold, color=TEXT, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
    return _finish_chart(
        axis,
        "Distribution of Genuine and Impostor Similarity Scores",
        "Similarity Score",
        "Frequency",
    )


def _mean_sd_chart(genuine_mean, impostor_mean, genuine_sd, impostor_sd):
    """Compare class means and show their score spread as error bars."""
    figure, axis = plt.subplots(figsize=(8, 4.8))
    labels = ["Genuine", "Impostor"]
    means = [genuine_mean, impostor_mean]
    standard_deviations = [genuine_sd, impostor_sd]
    bars = axis.bar(
        labels,
        means,
        yerr=standard_deviations,
        capsize=8,
        color=[GENUINE, IMPOSTOR],
        alpha=0.85,
    )
    for bar, mean in zip(bars, means):
        axis.text(bar.get_x() + bar.get_width() / 2, min(mean + 0.04, 0.97), f"{mean:.4f}", ha="center")
    return _finish_chart(
        axis,
        "Mean and Standard Deviation Comparison",
        "Actual Class",
        "Similarity Score",
        legend=False,
        y_limits=(0, 1),
    )


def _normal_fit_chart(genuine, impostor, threshold, genuine_mean, impostor_mean, genuine_sd, impostor_sd):
    """Show observed density histograms and Gaussian curves fitted to them."""
    figure, axis = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 26)
    score_range = np.linspace(0, 1, 400)
    axis.hist(genuine, bins=bins, density=True, alpha=0.3, color=GENUINE, label="Genuine observed density")
    axis.hist(impostor, bins=bins, density=True, alpha=0.3, color=IMPOSTOR, label="Impostor observed density")

    if genuine_sd > 0:
        axis.plot(score_range, stats.norm.pdf(score_range, genuine_mean, genuine_sd), color=ACCENT, linewidth=2, label="Genuine Gaussian fit")
    if impostor_sd > 0:
        axis.plot(score_range, stats.norm.pdf(score_range, impostor_mean, impostor_sd), color=IMPOSTOR, linewidth=2, label="Impostor Gaussian fit")

    axis.axvline(threshold, color=TEXT, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
    return _finish_chart(
        axis,
        "Gaussian Fits to Genuine and Impostor Scores",
        "Similarity Score",
        "Probability Density",
    )


def _cdf_chart(genuine, impostor, threshold):
    """Create empirical CDFs directly from the observed score arrays."""
    figure, axis = plt.subplots(figsize=(10, 5))
    for scores, label, color in [
        (genuine, "Genuine", GENUINE),
        (impostor, "Impostor", IMPOSTOR),
    ]:
        ordered_scores = np.sort(scores)
        cumulative_probability = np.arange(1, len(ordered_scores) + 1) / len(ordered_scores)
        axis.step(ordered_scores, cumulative_probability, where="post", color=color, linewidth=2, label=label)
    axis.axvline(threshold, color=TEXT, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
    return _finish_chart(
        axis,
        "Cumulative Distribution Function of Similarity Scores",
        "Similarity Score",
        "Cumulative Probability",
        y_limits=(0, 1),
    )


def _density_chart(genuine, impostor, threshold):
    """Compare observed probability densities without generating any new scores."""
    figure, axis = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 26)
    axis.hist(genuine, bins=bins, density=True, histtype="step", linewidth=2, color=GENUINE, label="Genuine density")
    axis.hist(impostor, bins=bins, density=True, histtype="step", linewidth=2, color=IMPOSTOR, label="Impostor density")
    axis.axvline(threshold, color=TEXT, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
    return _finish_chart(
        axis,
        "Probability Density Comparison",
        "Similarity Score",
        "Probability Density",
    )


def _threshold_curve_chart(rows, current_threshold):
    """Plot all four biometric operating rates across the threshold grid."""
    figure, axis = plt.subplots(figsize=(10, 5))
    thresholds = [row["threshold"] for row in rows]
    for key, label, color in [
        ("TAR", "TAR", GENUINE),
        ("FRR", "FRR", WARN),
        ("TRR", "TRR", ACCENT),
        ("FAR", "FAR", IMPOSTOR),
    ]:
        axis.plot(thresholds, [row[key] for row in rows], label=label, color=color, linewidth=2)
    axis.axvline(current_threshold, color=TEXT, linestyle="--", linewidth=2, label=f"Current threshold = {current_threshold:.2f}")
    return _finish_chart(
        axis,
        "Threshold Performance Curve",
        "Authentication Threshold",
        "Rate",
        y_limits=(0, 1),
    )


def _roc_chart(fpr, tpr, auc):
    """Draw an ROC curve and its random-classifier reference line."""
    figure, axis = plt.subplots(figsize=(7, 5.5))
    axis.plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f"ROC curve (AUC = {auc:.4f})")
    axis.plot([0, 1], [0, 1], "--", color=MUTED, label="Random classifier reference")
    return _finish_chart(
        axis,
        "ROC Curve and AUC",
        "False Positive Rate",
        "True Positive Rate",
        y_limits=(0, 1),
    )


def _high_security_row(rows):
    """Pick the lowest-FAR row that still keeps TAR reasonably usable (>= 0.50)."""
    usable_rows = [row for row in rows if row["TAR"] >= 0.50]
    candidates = usable_rows if usable_rows else rows
    return min(candidates, key=lambda row: row["FAR"])


def _high_convenience_row(rows):
    """Pick the lowest-FRR row that still keeps FAR reasonably controlled (<= 0.50)."""
    usable_rows = [row for row in rows if row["FAR"] <= 0.50]
    candidates = usable_rows if usable_rows else rows
    return min(candidates, key=lambda row: row["FRR"])


def render_distribution_analysis():
    """Render the complete Step 4 distribution analysis using shared state."""
    page_header("Statistical analysis · Step 4", "Distribution Analysis",
                "Describe the genuine and impostor score distributions, test whether a Gaussian model fits, measure how far apart "
                "they are, and see how that separation becomes TAR, FAR, ROC and DET curves.", icon="📈")
    concept_chips("Histogram & PDF", "Empirical CDF", "Gaussian fit", "Q-Q plot & KS test", "Cohen's d & overlap", "ROC / AUC", "DET curve", "EER")
    if not require_dataset("Create the shared demo dataset to analyse score distributions."):
        return

    dataset = st.session_state.dataset
    threshold = active_threshold()
    genuine_scores, impostor_scores = split_scores(dataset)

    # The Step 2 generator always creates both classes; this keeps the page safe
    # if a different dataset is later supplied.
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        st.warning("The dataset needs at least one Genuine and one Impostor score for distribution analysis.")
        return

    st.caption("Quick guide: TAR = genuine attempts accepted · FRR = genuine attempts rejected · FAR = impostor attempts accepted · TRR = impostor attempts rejected.")

    genuine_count = len(genuine_scores)
    impostor_count = len(impostor_scores)
    total_attempts = genuine_count + impostor_count
    genuine_mean = float(np.mean(genuine_scores))
    impostor_mean = float(np.mean(impostor_scores))
    genuine_sd = float(np.std(genuine_scores, ddof=1)) if genuine_count > 1 else 0.0
    impostor_sd = float(np.std(impostor_scores, ddof=1)) if impostor_count > 1 else 0.0

    dist_tabs = st.tabs(['📊 Distributions', '🔔 Gaussian fit & CDF', '↔️ Separation', '📈 Thresholds, ROC & DET', '🔒 Conclusion & lock'])
    with dist_tabs[0]:
        st.subheader("1. Similarity Score Distribution")
        st.pyplot(_score_distribution_chart(genuine_scores, impostor_scores, threshold), width="stretch")

        st.subheader("2. Dataset Summary")
        summary_columns = st.columns(6)
        summary_columns[0].metric("Total Attempts", total_attempts)
        summary_columns[1].metric("Genuine Attempts", genuine_count)
        summary_columns[2].metric("Impostor Attempts", impostor_count)
        summary_columns[3].metric("Genuine Percentage", f"{genuine_count / total_attempts:.2%}")
        summary_columns[4].metric("Impostor Percentage", f"{impostor_count / total_attempts:.2%}")
        summary_columns[5].metric("Current Threshold", f"{threshold:.2f}")

        st.subheader("3. Descriptive Statistics")
        statistics_table = pd.DataFrame(
            {
                "Statistic": [
                    "Number of Samples",
                    "Mean",
                    "Standard Deviation",
                    "Median",
                    "Minimum",
                    "Maximum",
                    "25th Percentile",
                    "75th Percentile",
                ],
                "Genuine": [
                    genuine_count,
                    genuine_mean,
                    genuine_sd,
                    np.median(genuine_scores),
                    np.min(genuine_scores),
                    np.max(genuine_scores),
                    np.percentile(genuine_scores, 25),
                    np.percentile(genuine_scores, 75),
                ],
                "Impostor": [
                    impostor_count,
                    impostor_mean,
                    impostor_sd,
                    np.median(impostor_scores),
                    np.min(impostor_scores),
                    np.max(impostor_scores),
                    np.percentile(impostor_scores, 25),
                    np.percentile(impostor_scores, 75),
                ],
            }
        )
        st.dataframe(
            statistics_table.style.format({"Genuine": "{:.4f}", "Impostor": "{:.4f}"}),
            width="stretch",
            hide_index=True,
        )

        st.subheader("4. Mean and Standard Deviation Comparison")
        st.pyplot(_mean_sd_chart(genuine_mean, impostor_mean, genuine_sd, impostor_sd), width="stretch")
        st.caption("The bar height is the mean similarity score; the error bars show one standard deviation of score spread.")

    with dist_tabs[1]:
        st.subheader("5. Normal Distribution Fitting")
        st.caption("Gaussian curves are fitted approximations for comparison; they do not establish that the observed scores are normally distributed.")
        st.write(
            "Here the Gaussian / Normal distribution models the noise in biometric measurements: "
            "each class has a central expected score (its mean), and every measurement scatters "
            "around that mean because of sensor noise, skin/fingerprint variability, and other random "
            "factors. The score variability is therefore treated as additive Gaussian noise, "
            "so a genuine attempt can be written as `X = mu_genuine + epsilon` and an impostor "
            "attempt as `X = mu_impostor + epsilon`, where the noise term `epsilon` follows a "
            "normal distribution centered at zero with the fitted standard deviation."
        )
        st.pyplot(
            _normal_fit_chart(
                genuine_scores,
                impostor_scores,
                threshold,
                genuine_mean,
                impostor_mean,
                genuine_sd,
                impostor_sd,
            ),
            width="stretch",
        )

        st.subheader("6. Cumulative Distribution Function")
        st.pyplot(_cdf_chart(genuine_scores, impostor_scores, threshold), width="stretch")

        st.subheader("7. Probability Density Comparison")
        st.pyplot(_density_chart(genuine_scores, impostor_scores, threshold), width="stretch")
        st.caption("The density curves show where genuine and impostor scores are concentrated and how much the two distributions overlap.")

        render_gof_panel(genuine_scores, impostor_scores)

    with dist_tabs[2]:
        st.subheader("8. Distribution Separation")
        if genuine_count > 1 and impostor_count > 1:
            separation = effect_size_and_overlap(genuine_scores, impostor_scores)
            pooled_sd = separation["pooled_sd"]
            effect_size = abs(separation["effect_size"])
            estimated_overlap = separation["overlap"]
            estimated_separation = separation["separation"]
        else:
            pooled_sd = 0.0
            effect_size = 0.0
            estimated_overlap = 0.0
            estimated_separation = 0.0

        mean_difference = abs(genuine_mean - impostor_mean)
        separation_columns = st.columns(4)
        separation_columns[0].metric("Genuine Mean", f"{genuine_mean:.4f}")
        separation_columns[1].metric("Impostor Mean", f"{impostor_mean:.4f}")
        separation_columns[2].metric("Mean Difference", f"{mean_difference:.4f}")
        separation_columns[3].metric("Pooled Standard Deviation", f"{pooled_sd:.4f}")
        effect_columns = st.columns(3)
        effect_columns[0].metric("Cohen's d / Effect Size", f"{effect_size:.4f}")
        effect_columns[1].metric("Estimated Overlap", f"{estimated_overlap:.2%}")
        effect_columns[2].metric("Estimated Separation", f"{estimated_separation:.2%}")
        st.latex(r"s_p = \sqrt{\frac{(n_G - 1)s_G^2 + (n_I - 1)s_I^2}{n_G + n_I - 2}}, \qquad d = \frac{|\bar{x}_G - \bar{x}_I|}{s_p}")
        st.caption("Estimated overlap is an approximate value based on fitted Gaussian distributions, not an exact measurement of the real biometric system.")

    with dist_tabs[3]:
        st.subheader("9. Threshold Performance")
        performance = performance_metrics(dataset, threshold)
        count_columns = st.columns(4)
        count_columns[0].metric("Genuine scores >= threshold", performance["TP"])
        count_columns[1].metric("Genuine scores < threshold", performance["FN"])
        count_columns[2].metric("Impostor scores >= threshold", performance["FP"])
        count_columns[3].metric("Impostor scores < threshold", performance["TN"])
        rate_columns = st.columns(4)
        rate_columns[0].metric("TAR", f"{performance['TAR']:.2%}")
        rate_columns[1].metric("FRR", f"{performance['FRR']:.2%}")
        rate_columns[2].metric("TRR", f"{performance['TRR']:.2%}")
        rate_columns[3].metric("FAR", f"{performance['FAR']:.2%}")
        st.code(
            f"TAR = Genuine accepted / Total genuine = {performance['TP']} / {genuine_count} = {performance['TAR']:.4f}\n"
            f"FRR = Genuine rejected / Total genuine = {performance['FN']} / {genuine_count} = {performance['FRR']:.4f}\n"
            f"TRR = Impostor rejected / Total impostor = {performance['TN']} / {impostor_count} = {performance['TRR']:.4f}\n"
            f"FAR = Impostor accepted / Total impostor = {performance['FP']} / {impostor_count} = {performance['FAR']:.4f}"
        )

        st.subheader("10. Threshold Performance Curve")
        threshold_rows, eer = threshold_sweep(dataset)
        st.pyplot(_threshold_curve_chart(threshold_rows, threshold), width="stretch")
        st.caption("The curves show the trade-off between accepting genuine users and rejecting impostors as the authentication threshold changes.")

        st.subheader("11. ROC Curve and AUC")
        false_positive_rate, true_positive_rate, auc = roc_metrics(dataset)
        st.pyplot(_roc_chart(false_positive_rate, true_positive_rate, auc), width="stretch")
        eer_columns = st.columns(2)
        eer_columns[0].metric("Approximate EER", f"{eer['eer']:.2%}")
        eer_columns[1].metric("Approximate EER Threshold", f"{eer['threshold']:.2f}")
        st.caption("A higher AUC indicates better separation between genuine and impostor scores on the observed dataset. AUC does not prove production-level biometric security. The EER is approximate because the threshold grid is discrete.")
        quick_read(
            f"At threshold {threshold:.2f}, {performance['TAR']:.1%} of genuine attempts pass and "
            f"{performance['FAR']:.1%} of impostor attempts pass. The AUC is {auc:.3f}; closer to 1 means better score separation in this simulated sample.",
        )

        render_det_curve(genuine_scores, impostor_scores, threshold)

    with dist_tabs[4]:
        st.subheader("12. Interpretation and Conclusion")
        if effect_size >= 2:
            separation_description = "strong"
        elif effect_size >= 0.8:
            separation_description = "moderate"
        else:
            separation_description = "limited"

        st.write(
            f"On the observed dataset, the genuine mean is {genuine_mean:.4f} and the impostor mean is {impostor_mean:.4f}, "
            f"giving a mean separation of {mean_difference:.4f}. At the selected threshold of {threshold:.2f}, "
            f"the system accepts {performance['TAR']:.2%} of genuine attempts (TAR), rejects {performance['FRR']:.2%} of genuine attempts (FRR), "
            f"correctly rejects {performance['TRR']:.2%} of impostor attempts (TRR), and incorrectly accepts {performance['FAR']:.2%} of impostor attempts (FAR). "
            f"The fitted-Gaussian estimate gives {estimated_overlap:.2%} overlap and {estimated_separation:.2%} separation. "
            f"The ROC AUC is {auc:.4f}, and the approximate EER is {eer['eer']:.2%} at threshold {eer['threshold']:.2f}."
        )
        st.info(
            f"The genuine and impostor score distributions show {separation_description} statistical separation on the observed dataset. "
            "The threshold analysis demonstrates the trade-off between accepting genuine users and rejecting impostors. "
            "ROC-AUC and EER provide additional measures of classification performance. This is an academic prototype based on simulated biometric data and should not be interpreted as a production-security evaluation."
        )

        # =====================================================================
        # 13. Operating Threshold Selection (lock the final decision threshold)
        # =====================================================================
        st.subheader("13. Select and Lock the Operating Threshold")
        st.write(
            "The threshold can be freely adjusted while exploring probability and distribution "
            "behaviour. Once the analysis is complete, a single operating threshold should be "
            "locked in and used consistently for authentication and the final results, instead "
            "of an arbitrary value left over from exploration."
        )

        balanced_threshold = eer["threshold"]
        security_row = _high_security_row(threshold_rows)
        convenience_row = _high_convenience_row(threshold_rows)

        operating_point_table = pd.DataFrame(
            [
                {
                    "Strategy": "Balanced (Equal Error Rate)",
                    "Threshold": balanced_threshold,
                    "TAR": next(row["TAR"] for row in threshold_rows if row["threshold"] == balanced_threshold),
                    "FAR": next(row["FAR"] for row in threshold_rows if row["threshold"] == balanced_threshold),
                },
                {
                    "Strategy": "High Security (minimize FAR)",
                    "Threshold": security_row["threshold"],
                    "TAR": security_row["TAR"],
                    "FAR": security_row["FAR"],
                },
                {
                    "Strategy": "High Convenience (minimize FRR)",
                    "Threshold": convenience_row["threshold"],
                    "TAR": convenience_row["TAR"],
                    "FAR": convenience_row["FAR"],
                },
            ]
        )
        st.dataframe(
            operating_point_table.style.format({"Threshold": "{:.2f}", "TAR": "{:.2%}", "FAR": "{:.2%}"}),
            width="stretch",
            hide_index=True,
        )

        strategy_choice = st.radio(
            "Choose an operating-point strategy to lock",
            [
                "Balanced (Equal Error Rate)",
                "High Security (minimize FAR)",
                "High Convenience (minimize FRR)",
                "Use the current slider value",
            ],
            horizontal=False,
        )

        if strategy_choice == "Balanced (Equal Error Rate)":
            candidate_threshold = balanced_threshold
        elif strategy_choice == "High Security (minimize FAR)":
            candidate_threshold = security_row["threshold"]
        elif strategy_choice == "High Convenience (minimize FRR)":
            candidate_threshold = convenience_row["threshold"]
        else:
            candidate_threshold = threshold

        st.caption(f"Selected strategy would lock the threshold at {candidate_threshold:.2f}.")

        lock_columns = st.columns(2)
        with lock_columns[0]:
            if st.button("Lock this as the final operating threshold", type="primary", width="stretch"):
                set_threshold(candidate_threshold, strategy_choice)
                st.success(f"Operating threshold locked at {candidate_threshold:.2f}. Final Results will now use this fixed value.")
        with lock_columns[1]:
            if st.session_state.get("locked_threshold") is not None:
                if st.button("Unlock and go back to exploration mode", width="stretch"):
                    st.session_state.locked_threshold = None
                    st.session_state.active_threshold_profile = None
                    st.info("Threshold unlocked. Final Results will follow the live slider value again.")

        if st.session_state.get("locked_threshold") is not None:
            st.success(f"Current locked operating threshold: {st.session_state.locked_threshold:.2f}. This value is used on the Final Results page regardless of later slider changes.")
        else:
            st.warning("No operating threshold is locked yet. The Final Results page will use the current live threshold, which may change if it is adjusted elsewhere.")
