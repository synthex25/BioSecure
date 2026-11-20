"""Confidence intervals and hypothesis-testing page."""

import streamlit as st
import plotly.graph_objects as go

from config.settings import CONFIDENCE_LEVEL, SIGNIFICANCE_LEVEL
from utils.calculations import cluster_bootstrap_rates, confidence_interval, hypothesis_test, split_scores
from utils.insights import quick_read
from utils.inference import render_error_rate_inference, render_nonparametric, render_power
from utils.ui import active_threshold, concept_chips, page_header, require_dataset
from utils.theme import GENUINE, IMPOSTOR, LINE, PANEL, TEXT
from utils.visualizations import confidence_interval_plot


def _bootstrap_interval_chart(result):
    """Show user-level TAR and FAR bootstrap ranges on a shared 0–100% scale."""
    figure = go.Figure()
    for label, point_key, interval_key, color in (
        ("TAR", "TAR", "TAR_CI", GENUINE),
        ("FAR", "FAR", "FAR_CI", IMPOSTOR),
    ):
        point = result[point_key]
        low, high = result[interval_key]
        figure.add_trace(go.Scatter(
            x=[point], y=[label], mode="markers", name=label,
            marker=dict(color=color, size=13),
            error_x=dict(
                type="data", symmetric=False, array=[max(0.0, high - point)],
                arrayminus=[max(0.0, point - low)], color=color, thickness=3, width=8,
            ),
        ))
    figure.update_layout(
        title="95% user-level bootstrap ranges",
        height=250,
        margin=dict(l=20, r=20, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Plus Jakarta Sans, sans-serif"),
        showlegend=False,
    )
    figure.update_xaxes(range=[0, 1], tickformat=".0%", title="Rate", gridcolor=LINE, zeroline=False)
    figure.update_yaxes(title=None, gridcolor=LINE, zeroline=False)
    return figure


def render_statistical_validation():
    """Render the statistical validation analysis."""

    page_header("Statistical analysis · Step 6", "Inference & Hypothesis Tests",
                "Validate recognition accuracy with statistical inference: confidence intervals for mean scores and for the rare error rates, "
                "parametric and non-parametric tests, exact binomial tests against requirements, power and sample size.", icon="✅")
    concept_chips("t confidence interval", "Welch t-test", "Wilson & Clopper–Pearson", "Exact binomial test", "Rule of 3 / 30",
                  "Mann–Whitney = AUC", "Levene", "Type I / II & power", "Cluster bootstrap")
    if not require_dataset("Create the shared demo dataset to run the statistical tests."):
        return

    # ---------------------------------------------------------
    # 1. Split genuine and impostor similarity scores
    # ---------------------------------------------------------
    genuine, impostor = split_scores(st.session_state.dataset)

    # ---------------------------------------------------------
    # 2. Calculate confidence intervals
    # ---------------------------------------------------------
    gci = confidence_interval(genuine, CONFIDENCE_LEVEL)

    ici = confidence_interval(impostor, CONFIDENCE_LEVEL)

    # Calculate means
    genuine_mean = genuine.mean()
    impostor_mean = impostor.mean()

    # ---------------------------------------------------------
    # 3. Confidence Intervals
    # ---------------------------------------------------------
    val_tabs = st.tabs(['📏 Mean scores: CI & t-test', '🎯 Error-rate inference', '🔬 Non-parametric & variance', '⚡ Errors & power', '👥 User-level bootstrap'])
    with val_tabs[0]:
        st.subheader(
            f"{int(CONFIDENCE_LEVEL * 100)}% confidence intervals for mean similarity score"
        )

        c1, c2 = st.columns(2)

        c1.metric("Genuine mean CI", f"[{gci[0]:.4f}, {gci[1]:.4f}]")

        c2.metric("Impostor mean CI", f"[{ici[0]:.4f}, {ici[1]:.4f}]")

        # ---------------------------------------------------------
        # 4. Confidence Interval Visualization
        # ---------------------------------------------------------
        st.pyplot(
            confidence_interval_plot(genuine_mean, gci, impostor_mean, ici),
            width="stretch",
        )

        # ---------------------------------------------------------
        # 5. Calculation Details
        # ---------------------------------------------------------
        st.subheader("Calculation Details")

        st.write(
            "The confidence interval estimates the range in which the "
            "population mean similarity score is expected to lie."
        )

        st.code(f"""Genuine mean = {genuine_mean:.4f}
    Impostor mean = {impostor_mean:.4f}

    {int(CONFIDENCE_LEVEL * 100)}% Confidence Interval:

    Genuine CI = [{gci[0]:.4f}, {gci[1]:.4f}]
    Impostor CI = [{ici[0]:.4f}, {ici[1]:.4f}]""")

        # ---------------------------------------------------------
        # 6. Interpretation of Confidence Intervals
        # ---------------------------------------------------------
        if gci[0] > ici[1]:
            st.success(
                "The confidence intervals do not overlap. "
                "This indicates a clear separation between the "
                "genuine and impostor mean similarity scores."
            )
        else:
            st.info(
                "The confidence intervals overlap. "
                "The difference between the two mean scores should "
                "be interpreted together with the hypothesis test."
            )

        # ---------------------------------------------------------
        # 7. Welch Two-Sample t-test
        # ---------------------------------------------------------
        statistic, p_value = hypothesis_test(genuine, impostor)
        if p_value < SIGNIFICANCE_LEVEL:
            quick_read(
                f"The genuine and impostor average scores differ in this sample (p = {p_value:.3g}). "
                "This is evidence of a difference in averages; it does not prove that a real biometric system is secure.",
                tone="good",
            )
        else:
            quick_read(
                f"This sample does not give strong evidence that the two average scores differ (p = {p_value:.3g}). "
                "The result may be uncertain, especially when there are few samples.",
                tone="caution",
            )

        st.subheader("Welch two-sample t-test")

        st.write("H₀: genuine and impostor mean scores do not differ.")

        st.write("H₁: genuine and impostor mean scores differ.")

        c1, c2, c3 = st.columns(3)

        c1.metric("Test statistic", f"{statistic:.4f}")

        c2.metric("p-value", f"{p_value:.3e}")

        c3.metric("α", f"{SIGNIFICANCE_LEVEL:.2f}")

        # ---------------------------------------------------------
        # 8. Hypothesis Test Calculation
        # ---------------------------------------------------------
        st.subheader("Hypothesis Test Calculation")

        st.code(f"""H₀: μGenuine = μImpostor
    H₁: μGenuine ≠ μImpostor

    Welch t-statistic = {statistic:.4f}
    p-value = {p_value:.3e}
    α = {SIGNIFICANCE_LEVEL:.2f}

    Decision rule:
    If p-value < α → Reject H₀
    If p-value ≥ α → Fail to reject H₀""")

        # ---------------------------------------------------------
        # 9. Decision
        # ---------------------------------------------------------
        if p_value < SIGNIFICANCE_LEVEL:

            st.success(
                "Decision: Reject H₀; the observed score means " "differ significantly."
            )

        else:

            st.info(
                "Decision: Fail to reject H₀; the observed data "
                "do not provide sufficient evidence of a mean difference."
            )

        # ---------------------------------------------------------
        # 10. Chi-Square Explanation
        # ---------------------------------------------------------
        st.subheader("Where the chi-square test fits")

        st.caption(
            "Chi-square is not used to compare these mean scores, because similarity scores are continuous and the "
            "question here is about means. It is the right tool for counts, and the project uses it twice: the test of "
            "independence between true class and accept/reject decision (Probability Foundations → Joint & conditional), "
            "and the goodness-of-fit test of attempt counts against a Poisson model (Discrete & Rare Events → Poisson process check)."
        )

        # ---------------------------------------------------------
        # 11. Statistical Validation Conclusion
        # ---------------------------------------------------------
        st.subheader("Statistical Validation Conclusion")

        if p_value < SIGNIFICANCE_LEVEL:

            st.write(
                f"The {int(CONFIDENCE_LEVEL * 100)}% confidence intervals "
                f"show that the genuine and impostor mean similarity "
                f"scores are clearly separated. The Welch two-sample "
                f"t-test produced a p-value of {p_value:.3e}, which is "
                f"less than the significance level α = "
                f"{SIGNIFICANCE_LEVEL:.2f}. Therefore, H₀ is rejected "
                f"and the difference between the genuine and impostor "
                f"mean scores is statistically significant on the "
                f"observed dataset."
            )

        else:

            st.write(
                f"The {int(CONFIDENCE_LEVEL * 100)}% confidence intervals "
                f"were compared for the genuine and impostor scores. "
                f"The Welch two-sample t-test produced a p-value of "
                f"{p_value:.3e}, which is not less than α = "
                f"{SIGNIFICANCE_LEVEL:.2f}. Therefore, H₀ is not "
                f"rejected based on the observed dataset."
            )

    with val_tabs[1]:
        render_error_rate_inference(st.session_state.dataset, active_threshold())
    with val_tabs[2]:
        render_nonparametric(st.session_state.dataset)
    with val_tabs[3]:
        render_power(st.session_state.dataset)
    with val_tabs[4]:
        st.subheader("A more realistic range for repeated users")
        st.caption("This extra method is different from the mean-score confidence intervals above. It checks how TAR and FAR could vary when users, rather than individual attempts, are resampled.")
        with st.container():
            st.write(
                "Attempts from the same user may be alike. The bootstrap therefore samples whole users and keeps each selected user's attempts together. "
                "The colored bars show the central 95% of 1,000 resamples. Wider bars mean more user-to-user uncertainty."
            )
            bootstrap_threshold = active_threshold()
            try:
                bootstrap = cluster_bootstrap_rates(st.session_state.dataset, bootstrap_threshold, n_boot=1000)
            except ValueError as error:
                st.info(str(error))
            else:
                bootstrap_columns = st.columns(3)
                bootstrap_columns[0].metric(
                    "TAR · 95% user range",
                    f"{bootstrap['TAR_CI'][0]:.1%}–{bootstrap['TAR_CI'][1]:.1%}",
                    help=f"Observed TAR: {bootstrap['TAR']:.1%}",
                )
                bootstrap_columns[1].metric(
                    "FAR · 95% user range",
                    f"{bootstrap['FAR_CI'][0]:.1%}–{bootstrap['FAR_CI'][1]:.1%}",
                    help=f"Observed FAR: {bootstrap['FAR']:.1%}",
                )
                bootstrap_columns[2].metric("Users / resamples", f"{bootstrap['users']} / {bootstrap['resamples']:,}")
                quick_read(
                    f"After resampling whole users, TAR ranged from {bootstrap['TAR_CI'][0]:.1%} to {bootstrap['TAR_CI'][1]:.1%}, "
                    f"and FAR ranged from {bootstrap['FAR_CI'][0]:.1%} to {bootstrap['FAR_CI'][1]:.1%} in 95% of resamples. "
                    "These ranges show uncertainty from the users in this simulated dataset; they are not guarantees for new populations.",
                    tone="caution",
                    title="What this means",
                )
                st.plotly_chart(_bootstrap_interval_chart(bootstrap), width="stretch")
