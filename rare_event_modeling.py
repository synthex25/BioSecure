"""Discrete and continuous rare-event modeling: Bernoulli, Binomial, Poisson, Exponential.

This page fills the project-scope requirement to model rare authentication
events with Bernoulli, Binomial, and Poisson distributions, and to use an
Exponential distribution alongside the Gaussian to represent noise, decay,
and rare-event timing.
"""

import streamlit as st

from config.settings import DEFAULT_FUTURE_BATCH_SIZE
from utils.calculations import (
    bernoulli_model,
    binomial_model,
    binomial_poisson_comparison,
    exponential_model,
    false_accept_gaps,
    projected_breach_distribution,
)
from utils.insights import quick_read
from utils.ui import active_threshold, concept_chips, page_header, require_dataset
from utils.dist_extras import render_poisson_process
from utils.visualizations import (
    bernoulli_chart,
    binomial_chart,
    binomial_vs_poisson_chart,
    exponential_chart,
    exponential_survival_chart,
    poisson_chart,
)


def render_rare_event_modeling():
    """Render Bernoulli, Binomial, Poisson, and Exponential modeling of authentication events."""
    page_header("Statistical analysis · Rare events", "Discrete & Rare-Event Modeling",
                "A single attempt is a Bernoulli trial; counts of false accepts are Binomial and, because FAR is small, approximately Poisson; "
                "the waiting time between breaches is Exponential. The last tab checks the Poisson-process assumption on the attempt timestamps.",
                icon="🧮")
    concept_chips("Bernoulli", "Binomial", "Poisson approximation", "Exponential waiting time", "Geometric", "Poisson process", "χ² goodness of fit")
    if not require_dataset("Create the shared demo dataset to model rare authentication events."):
        return

    dataset = st.session_state.dataset
    threshold = active_threshold()

    st.caption(
        f"All calculations below use the threshold of {threshold:.2f} "
        f"({'locked operating threshold' if st.session_state.get('locked_threshold') else 'current live threshold'})."
    )

    # =====================================================================
    # 1. Bernoulli Trial Model
    # =====================================================================
    rare_tabs = st.tabs(['① Bernoulli', '② Binomial', '③ Poisson', '④ Exponential', '⑤ Poisson process check', 'Σ Summary'])
    with rare_tabs[0]:
        st.subheader("1. Bernoulli Trial Model")
        st.write(
            "Each authentication attempt can be treated as a single Bernoulli trial: "
            "the outcome is Accept (X = 1) or Reject (X = 0). The Bernoulli parameter "
            "p is estimated separately for genuine and impostor attempts as the "
            "observed acceptance rate at the current threshold."
        )
        bernoulli = bernoulli_model(dataset, threshold)
        st.latex(r"X \sim \text{Bernoulli}(p), \qquad P(X=1) = p, \quad P(X=0) = 1-p")

        bernoulli_columns = st.columns(4)
        bernoulli_columns[0].metric("P(Accept | Genuine)", f"{bernoulli['p_genuine_accept']:.4f}")
        bernoulli_columns[1].metric("P(Accept | Impostor)", f"{bernoulli['p_impostor_accept']:.4f}")
        bernoulli_columns[2].metric("Var(X) — Genuine", f"{bernoulli['variance_genuine']:.4f}")
        bernoulli_columns[3].metric("Var(X) — Impostor", f"{bernoulli['variance_impostor']:.4f}")

        st.code(
            f"p_genuine  = {bernoulli['genuine_successes']} / {bernoulli['genuine_trials']} = {bernoulli['p_genuine_accept']:.4f}\n"
            f"p_impostor = {bernoulli['impostor_successes']} / {bernoulli['impostor_trials']} = {bernoulli['p_impostor_accept']:.4f}\n\n"
            f"Var(X) = p(1-p)"
        )
        st.pyplot(bernoulli_chart(bernoulli["p_genuine_accept"], bernoulli["p_impostor_accept"]), width="stretch")
        st.caption(
            "The impostor acceptance probability is the False Acceptance Rate (FAR) expressed "
            "as a Bernoulli success probability — this is the 'rare event' probability carried "
            "forward into the Binomial and Poisson models below."
        )

        # =====================================================================
        # 2. Binomial Distribution
        # =====================================================================
    with rare_tabs[1]:
        st.subheader("2. Binomial Distribution")
        st.write(
            "Repeating the Bernoulli trial across many independent attempts gives a Binomial "
            "distribution for the total number of successes. Two cases are modeled: how many "
            "genuine attempts are accepted out of all genuine attempts, and — the rare-event "
            "case — how many impostor attempts are falsely accepted out of all impostor attempts."
        )
        st.latex(r"X \sim \text{Binomial}(n, p), \qquad P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}")

        genuine_binomial = binomial_model(bernoulli["genuine_trials"], bernoulli["p_genuine_accept"])
        impostor_binomial = binomial_model(bernoulli["impostor_trials"], bernoulli["p_impostor_accept"])

        binomial_tab_genuine, binomial_tab_impostor = st.tabs(["Genuine acceptances", "Impostor false acceptances (rare event)"])
        with binomial_tab_genuine:
            st.write(
                f"n = {genuine_binomial['n']} genuine attempts, p = {genuine_binomial['p']:.4f} "
                f"(per-attempt acceptance probability)."
            )
            gb_columns = st.columns(3)
            gb_columns[0].metric("Expected successes", f"{genuine_binomial['mean']:.2f}")
            gb_columns[1].metric("Standard deviation", f"{genuine_binomial['std']:.2f}")
            gb_columns[2].metric("Observed successes", bernoulli["genuine_successes"])
            st.pyplot(
                binomial_chart(genuine_binomial, observed_k=bernoulli["genuine_successes"], title="Binomial Distribution: Genuine Acceptances"),
                width="stretch",
            )

        with binomial_tab_impostor:
            st.write(
                f"n = {impostor_binomial['n']} impostor attempts, p = {impostor_binomial['p']:.4f} "
                "(FAR, a typically small / rare per-attempt probability)."
            )
            ib_columns = st.columns(3)
            ib_columns[0].metric("Expected false accepts", f"{impostor_binomial['mean']:.2f}")
            ib_columns[1].metric("Standard deviation", f"{impostor_binomial['std']:.2f}")
            ib_columns[2].metric("Observed false accepts", bernoulli["impostor_successes"])
            st.pyplot(
                binomial_chart(impostor_binomial, observed_k=bernoulli["impostor_successes"], title="Binomial Distribution: Impostor False Acceptances"),
                width="stretch",
            )
            st.caption(
                "This is the rare-event distribution the project scope refers to: false acceptance "
                "is the undesired, low-probability outcome the system is designed to keep rare."
            )

        # =====================================================================
        # 3. Poisson Approximation for Rare False-Acceptance Events
        # =====================================================================
    with rare_tabs[2]:
        st.subheader("3. Poisson Approximation for Rare False-Acceptance Events")
        st.write(
            "When n is large and p is small, the Binomial distribution is well approximated by a "
            "Poisson distribution with rate λ = n × p. This is the standard way rare authentication "
            "breach events are modeled, and it is compared here against the exact Binomial computed above."
        )
        st.latex(r"\lambda = np, \qquad X \sim \text{Poisson}(\lambda), \qquad P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}")

        comparison = binomial_poisson_comparison(bernoulli["impostor_trials"], bernoulli["p_impostor_accept"])
        poisson_columns = st.columns(3)
        poisson_columns[0].metric("λ = n × FAR", f"{comparison['lambda']:.4f}")
        poisson_columns[1].metric("Max |Binomial − Poisson|", f"{comparison['max_absolute_difference']:.5f}")
        poisson_columns[2].metric("Observed false accepts", bernoulli["impostor_successes"])

        st.pyplot(binomial_vs_poisson_chart(comparison), width="stretch")
        if comparison["max_absolute_difference"] < 0.02:
            st.success("The Poisson approximation closely matches the exact Binomial distribution on this dataset, confirming the rare-event approximation is appropriate here.")
        else:
            st.info("The Poisson approximation shows a visible difference from the exact Binomial here, typically because FAR is not small enough relative to n for the approximation to be tight.")

        st.markdown("**Projected breaches over a future batch of attempts**")
        future_attempts = st.number_input(
            "Future authentication attempts to project over",
            min_value=10,
            max_value=1_000_000,
            value=DEFAULT_FUTURE_BATCH_SIZE,
            step=100,
        )
        projection = projected_breach_distribution(bernoulli["p_impostor_accept"], int(future_attempts))
        quick_read(
            f"Using the observed FAR of {bernoulli['p_impostor_accept']:.2%}, the simple model expects about "
            f"{projection['mean']:.2f} false accepts in {int(future_attempts):,} impostor attempts. "
            "This is a projection from simulated data, not a guarantee about future attempts.",
            tone="caution" if projection["mean"] >= 1 else "info",
        )
        projection_columns = st.columns(2)
        projection_columns[0].metric("Expected breaches (λ)", f"{projection['mean']:.3f}")
        projection_columns[1].metric("Standard deviation", f"{projection['std']:.3f}")
        st.pyplot(
            poisson_chart(projection, title=f"Projected False-Acceptance Breaches over {int(future_attempts)} Future Attempts"),
            width="stretch",
        )
        st.caption(
            f"At the current FAR of {bernoulli['p_impostor_accept']:.4%}, roughly {projection['mean']:.2f} "
            f"false acceptances are expected per {int(future_attempts)} impostor attempts on this system."
        )

        # =====================================================================
        # 4. Exponential Waiting-Time Model
        # =====================================================================
    with rare_tabs[3]:
        st.subheader("4. Exponential Waiting-Time Model")
        st.write(
            "If false-acceptance breaches occur as rare, independent events at a roughly constant "
            "rate per attempt, the number of attempts between successive breaches follows an "
            "Exponential distribution. This models both the waiting time until the next breach and "
            "the exponential decay of the probability that no breach has yet occurred, alongside the "
            "Gaussian model already used for score noise."
        )
        st.latex(r"T \sim \text{Exponential}(\lambda), \qquad f(t) = \lambda e^{-\lambda t}, \qquad P(T > t) = e^{-\lambda t}")

        gaps = false_accept_gaps(dataset, threshold)
        exponential = exponential_model(gaps)

        if exponential is None:
            st.info(
                "Fewer than two false-acceptance events occur at this threshold, so there are no "
                "observed gaps to fit an Exponential distribution to. Lower the threshold on the "
                "Probability Analysis page to generate more false-acceptance events, or note this "
                "as evidence of a very low breach rate at the current operating threshold."
            )
        else:
            exponential_columns = st.columns(3)
            exponential_columns[0].metric("Observed breach events", len(gaps) + 1)
            exponential_columns[1].metric("Mean gap (attempts)", f"{exponential['mean_gap']:.2f}")
            exponential_columns[2].metric("Rate λ (per attempt)", f"{exponential['rate']:.5f}")

            st.code(
                f"Mean gap between breaches = {exponential['mean_gap']:.4f} attempts\n"
                f"Rate λ = 1 / mean gap = {exponential['rate']:.6f} breaches per attempt"
            )
            st.pyplot(exponential_chart(exponential), width="stretch")

            st.markdown("**Breach-free probability over future attempts**")
            st.pyplot(exponential_survival_chart(exponential["rate"]), width="stretch")

            query_attempts = st.slider("Check breach-free probability after this many attempts", 1, 500, 50)
            survival_probability = exponential["prob_no_breach_within"](query_attempts)
            st.metric(f"P(no breach within {query_attempts} attempts)", f"{survival_probability:.2%}")
            st.caption(
                "This probability decays exponentially as the number of attempts grows, which is why "
                "the Exponential distribution is the natural continuous-time counterpart to the "
                "discrete Bernoulli/Binomial/Poisson rare-event models above."
            )

        # =====================================================================
        # 5. Summary
        # =====================================================================
    with rare_tabs[4]:
        render_poisson_process(dataset, threshold)

    with rare_tabs[5]:
        st.subheader("5. Summary")
        st.write(
            f"On the observed dataset at threshold {threshold:.2f}, a single authentication attempt "
            f"is modeled as a Bernoulli trial with P(Accept | Impostor) = {bernoulli['p_impostor_accept']:.4f}. "
            f"Across all {bernoulli['impostor_trials']} impostor attempts, the Binomial distribution gives an "
            f"expected {impostor_binomial['mean']:.2f} false acceptances (observed: {bernoulli['impostor_successes']}), "
            f"which the Poisson approximation with λ = {comparison['lambda']:.3f} closely reproduces. "
        )
        if exponential is not None:
            st.write(
                f"The waiting time between successive false-acceptance events is modeled as Exponential "
                f"with rate λ = {exponential['rate']:.5f} per attempt, giving a mean gap of "
                f"{exponential['mean_gap']:.2f} attempts between breaches."
            )
        st.info(
            "Together with the Gaussian fitting already used for genuine/impostor score noise in "
            "Distribution Analysis, this page completes the discrete (Bernoulli, Binomial, Poisson) "
            "and continuous (Exponential) rare-event modeling required by the project scope."
        )
