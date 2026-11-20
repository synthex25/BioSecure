"""Consolidated evidence-based project results."""

import streamlit as st

from utils.calculations import (
    bernoulli_model,
    binomial_model,
    binomial_poisson_comparison,
    confidence_interval,
    effect_size_and_overlap,
    exponential_model,
    false_accept_gaps,
    hypothesis_test,
    performance_metrics,
    probability_analysis,
    roc_metrics,
    split_scores,
    threshold_sweep,
)
from utils.insights import quick_read

from utils.ui import page_header, require_dataset
from utils.visualizations import (
    probability_comparison,
    roc_curve,
)


def render_final_results():
    """Render the consolidated final results of the biometric project."""

    page_header("Results", "Final Results",
                "One consolidated view of every analysis at the active operating threshold — performance, Bayes, ROC/EER, "
                "rare-event models and statistical validation — with the project conclusion.", icon="🏁")
    if not require_dataset("Create the shared demo dataset to see the consolidated results."):
        return

    # ---------------------------------------------------------
    # Threshold resolution: prefer the locked operating threshold
    # (set on the Distribution Analysis page) over the live slider
    # value, so the final results report one fixed, justified
    # decision point instead of whatever the slider is currently at.
    # ---------------------------------------------------------

    locked_threshold = st.session_state.get("locked_threshold")
    threshold_policy = st.session_state.get("threshold_policy")
    using_locked_threshold = locked_threshold is not None
    threshold = locked_threshold if using_locked_threshold else st.session_state.threshold

    if threshold_policy and threshold_policy.get("mode") == "three_zone":
        threshold = float(threshold_policy["accept_threshold"])
        st.info(
            f"The Authentication page has a three-zone policy active. Final Results show the binary operating rates at its "
            f"accept boundary ({threshold:.4f}); the manual-review zone is reported on the Authentication page."
        )
    elif using_locked_threshold:
        profile = st.session_state.get("active_threshold_profile") or "chosen threshold"
        st.success(f"Reporting Final Results at the active operating threshold {threshold:.4f} ({profile}). "
                   "Change it from the sidebar, Threshold Studio or Threshold Lab.")
    else:
        st.info(f"Using the default threshold {threshold:.2f}. Choose a justified operating point from the sidebar presets, "
                "Threshold Studio or Threshold Lab before reporting.")

    # ---------------------------------------------------------
    # Dataset and calculations
    # ---------------------------------------------------------

    df = st.session_state.dataset

    genuine, impostor = split_scores(df)

    metrics = performance_metrics(df, threshold)

    separation = effect_size_and_overlap(genuine, impostor)

    test_statistic, p_value = hypothesis_test(genuine, impostor)

    genuine_ci = confidence_interval(genuine)

    impostor_ci = confidence_interval(impostor)

    probability = probability_analysis(df, threshold)

    _, eer = threshold_sweep(df)

    fpr, tpr, auc = roc_metrics(df)

    # ---------------------------------------------------------
    # Discrete / rare-event modeling (Step: Discrete & Rare-Event Modeling)
    # ---------------------------------------------------------

    rare_bernoulli = bernoulli_model(df, threshold)
    rare_impostor_binomial = binomial_model(
        rare_bernoulli["impostor_trials"], rare_bernoulli["p_impostor_accept"]
    )
    rare_poisson = binomial_poisson_comparison(
        rare_bernoulli["impostor_trials"], rare_bernoulli["p_impostor_accept"]
    )
    rare_exponential = exponential_model(false_accept_gaps(df, threshold))

    # ---------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------

    total = len(df)

    accuracy = (metrics["TP"] + metrics["TN"]) / total if total > 0 else 0

    quick_read(
        f"At threshold {threshold:.2f}, genuine acceptance is {metrics['TAR']:.1%} and impostor false acceptance is "
        f"{metrics['FAR']:.1%}. AUC is {auc:.3f}. The results summarize this simulated dataset; use the threshold page to decide whether security or convenience matters more.",
        tone="good" if metrics["FAR"] < 0.05 and metrics["TAR"] >= 0.8 else "caution",
        title="Summary in plain language",
    )

    # ---------------------------------------------------------
    # 1. Overall Dataset Results
    # ---------------------------------------------------------

    st.subheader("1. Overall Dataset Results")

    threshold_label = "Threshold (active)" if using_locked_threshold else "Threshold (default)"

    values = [
        ("Dataset size", len(df)),
        ("Genuine samples", len(genuine)),
        ("Impostor samples", len(impostor)),
        (threshold_label, f"{threshold:.2f}"),
        ("Genuine mean", f"{genuine.mean():.4f}"),
        ("Impostor mean", f"{impostor.mean():.4f}"),
        ("TAR", f"{metrics['TAR']:.2%}"),
        ("TRR", f"{metrics['TRR']:.2%}"),
        ("FAR", f"{metrics['FAR']:.2%}"),
        ("FRR", f"{metrics['FRR']:.2%}"),
        ("Accuracy", f"{accuracy:.2%}"),
        ("Effect size", f"{separation['effect_size']:.4f}"),
        ("Distribution overlap", f"{separation['overlap']:.2%}"),
        ("Distribution separation", f"{separation['separation']:.2%}"),
        ("Genuine 95% CI", f"[{genuine_ci[0]:.4f}, {genuine_ci[1]:.4f}]"),
        ("Impostor 95% CI", f"[{impostor_ci[0]:.4f}, {impostor_ci[1]:.4f}]"),
        ("p-value", f"{p_value:.3e}"),
    ]

    for index in range(0, len(values), 4):

        columns = st.columns(4)

        for column, (label, value) in zip(columns, values[index : index + 4]):
            column.metric(label, value)

    # ---------------------------------------------------------
    # 2. Confusion Matrix Counts
    # ---------------------------------------------------------

    st.subheader("2. Authentication Classification Results")

    count_columns = st.columns(4)

    count_columns[0].metric("True Positives (TP)", metrics["TP"])

    count_columns[1].metric("True Negatives (TN)", metrics["TN"])

    count_columns[2].metric("False Positives (FP)", metrics["FP"])

    count_columns[3].metric("False Negatives (FN)", metrics["FN"])

    st.caption(
        f"At threshold {threshold:.2f}, scores greater than or equal "
        "to the threshold are classified as accepted, while scores "
        "below the threshold are classified as rejected."
    )

    # ---------------------------------------------------------
    # 3. Probability and Bayes Summary
    # ---------------------------------------------------------

    st.subheader("3. Probability and Bayes Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("P(Genuine)", f"{probability['prior_genuine']:.4f}")

    c2.metric("P(Impostor)", f"{probability['prior_impostor']:.4f}")

    c3.metric(
        "Posterior P(Genuine | Evidence)", f"{probability['posterior_genuine']:.4f}"
    )

    st.write(
        f"Evidence is defined as a similarity score greater than "
        f"or equal to the selected threshold of {threshold:.2f}."
    )

    st.pyplot(
        probability_comparison(
            probability["prior_genuine"], probability["posterior_genuine"]
        ),
        width="stretch",
    )

    # ---------------------------------------------------------
    # 4. ROC and EER Summary
    # ---------------------------------------------------------

    st.subheader("4. ROC and EER Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("ROC AUC", f"{auc:.3f}")

    c2.metric("Approx. EER", f"{eer['eer']:.2%}")

    c3.metric("EER threshold", f"{eer['threshold']:.2f}")

    st.pyplot(roc_curve(fpr, tpr, auc), width="stretch")

    st.caption(
        "A higher AUC generally indicates better separation between "
        "genuine and impostor scores. The EER threshold represents "
        "an approximate operating point where false acceptance and "
        "false rejection rates are similar. This is independent of "
        "which threshold is currently locked or selected above."
    )

    # ---------------------------------------------------------
    # 5. Discrete & Rare-Event Modeling Summary
    # ---------------------------------------------------------

    st.subheader("5. Discrete & Rare-Event Modeling Summary")

    st.write(
        "Each authentication attempt is treated as a Bernoulli trial whose rare "
        "failure mode is a false acceptance of an impostor. The Bernoulli, Binomial, "
        "Poisson and Exponential models below are fitted using the threshold applied "
        "to this project's observed dataset."
    )

    rare_event_columns = st.columns(4)

    rare_event_columns[0].metric(
        "Bernoulli FAR",
        f"{rare_bernoulli['p_impostor_accept']:.4f}",
        help="P(Accept | Impostor): the per-attempt false-acceptance probability.",
    )

    rare_event_columns[1].metric(
        "Binomial expected breaches",
        f"{rare_impostor_binomial['mean']:.2f}",
        help=(
            f"Expected false acceptances out of {rare_bernoulli['impostor_trials']} impostor "
            "attempts under Binomial(n, FAR)."
        ),
    )

    rare_event_columns[2].metric(
        "Poisson λ = n × FAR",
        f"{rare_poisson['lambda']:.4f}",
        help="Rate parameter of the Poisson approximation to the rare-event Binomial.",
    )

    rare_event_columns[3].metric(
        "Exponential breach rate",
        (
            f"{rare_exponential['rate']:.5f}"
            if rare_exponential is not None
            else "Not fitted"
        ),
        help=(
            "Impulse-rate per attempt of the Exponential waiting-time model between "
            "successive false-acceptance events."
        ),
    )

    st.write(
        f"With {rare_bernoulli['impostor_trials']} impostor attempts at the current "
        f"threshold, the observed FAR of {rare_bernoulli['p_impostor_accept']:.4f} "
        f"leads to {rare_bernoulli['impostor_successes']} observed false acceptances, "
        f"while the Binomial model predicts "
        f"{rare_impostor_binomial['mean']:.2f} expected breaches and the Poisson "
        f"approximation with λ = {rare_poisson['lambda']:.3f} closely reproduces it "
        f"(max |Binomial − Poisson| = {rare_poisson['max_absolute_difference']:.5f})."
    )

    if rare_exponential is not None:
        st.write(
            f"The waiting time between successive false-acceptance events is modeled as an "
            f"Exponential distribution with rate λ = {rare_exponential['rate']:.5f} per attempt, "
            f"corresponding to a mean gap of {rare_exponential['mean_gap']:.2f} attempts between "
            f"breaches."
        )
    else:
        st.info(
            "Fewer than two false-acceptance events occur at this threshold, so no Exponential "
            "waiting-time model could be fitted. This is consistent with a very low breach rate "
            "at the current operating threshold."
        )

    # ---------------------------------------------------------
    # 6. Statistical Validation
    # ---------------------------------------------------------

    st.subheader("6. Statistical Validation")

    st.write(f"Genuine 95% CI: " f"[{genuine_ci[0]:.4f}, {genuine_ci[1]:.4f}]")

    st.write(f"Impostor 95% CI: " f"[{impostor_ci[0]:.4f}, {impostor_ci[1]:.4f}]")

    st.write(f"Welch test statistic: " f"{test_statistic:.4f}")

    st.write(f"p-value: " f"{p_value:.3e}")

    if p_value < 0.05:

        st.success(
            "Decision: Reject H₀. The genuine and impostor "
            "mean similarity scores differ significantly "
            "on the observed dataset."
        )

    else:

        st.info(
            "Decision: Fail to reject H₀. The observed data "
            "do not provide sufficient evidence of a significant "
            "difference between the two means."
        )

    # ---------------------------------------------------------
    # 7. Connection Between All Steps
    # ---------------------------------------------------------

    st.subheader("7. Connection Between Project Steps")

    st.write(f"""
        **Step 3 — Probability Analysis:** Prior probabilities,
        conditional probabilities, Bayes' theorem and likelihood
        ratios were calculated from the biometric attempts.

        **Step 4 — Distribution Analysis:** Genuine and impostor
        similarity-score distributions were analyzed using
        descriptive statistics, Gaussian fitting, CDF, PDF,
        distribution separation, ROC and EER analysis. The final
        operating threshold used below was selected and locked on
        this page.

        **Step 4b — Discrete & Rare-Event Modeling:** Each attempt
        was modeled as a Bernoulli trial, the number of false
        acceptances as a Binomial distribution, and the rare breach
        rate approximated by a Poisson distribution, with an
        Exponential waiting-time distribution between successive
        false-acceptance events.

        **Step 5 — Authentication:** The selected threshold of
        {threshold:.2f} was applied to classify authentication
        attempts as accepted or rejected. TP, TN, FP and FN were
        calculated.

        **Step 6 — Statistical Validation:** Confidence intervals
        and a Welch two-sample t-test were used to statistically
        evaluate the difference between genuine and impostor
        similarity scores.

        **Step 7 — Final Results:** The results from all previous
        stages are consolidated to evaluate the overall behavior
        of the biometric authentication prototype at a single,
        fixed operating threshold.
        """)

    # ---------------------------------------------------------
    # 8. Project Conclusion
    # ---------------------------------------------------------

    st.subheader("8. Project Conclusion")

    threshold_status_sentence = (
        f"This threshold was applied from the sidebar, Threshold Studio, Threshold Lab or Distribution Analysis as the "
        f"final operating point for the system."
        if using_locked_threshold
        else "This threshold was not explicitly locked and reflects the live "
        "slider value at the time of viewing this page."
    )

    st.write(f"""
        On the observed synthetic dataset, genuine and impostor
        similarity scores are statistically distinguishable.
        At the selected threshold of {threshold:.2f}, the system
        achieves a TAR of {metrics['TAR']:.2%}, a TRR of
        {metrics['TRR']:.2%}, a FAR of {metrics['FAR']:.2%}, and
        an FRR of {metrics['FRR']:.2%}. {threshold_status_sentence}

        The ROC AUC of {auc:.3f} and effect size of
        {separation['effect_size']:.4f} indicate strong separation
        between the two observed score classes.

        These results demonstrate the application of probability,
        statistical distributions, Bayesian analysis, hypothesis
        testing and threshold-based biometric authentication.
        They represent an academic prototype and do not certify
        production-level biometric security.
        """)

    # ---------------------------------------------------------
    # 9. Limitations and Future Improvements
    # ---------------------------------------------------------

    st.subheader("9. Limitations and Future Improvements")

    st.write("""
        • The dataset is synthetically generated.

        • Only one similarity-score modality is considered.

        • The dataset uses a fixed class balance.

        • The results are based on the observed dataset and may
          not generalize to real-world biometric systems.

        • Future work can use consented real-world benchmark data,
          cross-validation, ROC/DET analysis, calibrated operating
          thresholds and spoofing-resistance evaluation.
        """)

    # ---------------------------------------------------------
    # Final message
    # ---------------------------------------------------------

    st.success("BioSecure project analysis completed successfully.")
