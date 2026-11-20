"""Interactive threshold-based authentication demonstration."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils.calculations import performance_metrics, split_scores
from utils.theme import GENUINE, IMPOSTOR, TEXT, WARN
from utils.visualizations import confusion_matrix
from utils.ui import page_header, require_dataset


def _authentication_score_chart(dataset, threshold, policy=None):
    """Plot the existing genuine and impostor scores against their attempt numbers."""
    genuine_data = dataset[dataset["Class"] == "Genuine"]
    impostor_data = dataset[dataset["Class"] == "Impostor"]

    figure, axis = plt.subplots(figsize=(10, 5))

    axis.scatter(
        genuine_data["Sample_ID"],
        genuine_data["Similarity_Score"],
        color=GENUINE,
        marker="o",
        alpha=0.7,
        label="Genuine",
    )

    axis.scatter(
        impostor_data["Sample_ID"],
        impostor_data["Similarity_Score"],
        color=IMPOSTOR,
        marker="x",
        alpha=0.7,
        label="Impostor",
    )

    if policy:
        reject_threshold = policy["reject_threshold"]
        axis.axhspan(reject_threshold, threshold, color=WARN, alpha=0.10, label="Manual review zone")
        axis.axhline(reject_threshold, color=WARN, linestyle="--", linewidth=2,
                     label=f"Reject below {reject_threshold:.3f}")
        axis.axhline(threshold, color=TEXT, linestyle="--", linewidth=2,
                     label=f"Accept at / above {threshold:.3f}")
    else:
        axis.axhline(
            threshold,
            color=TEXT,
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {threshold:.2f}",
        )

    axis.set_title("Genuine and Impostor Authentication Scores")
    axis.set_xlabel("Attempt Number")
    axis.set_ylabel("Similarity Score")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    return figure


def render_authentication():
    """Render the interactive demonstration and dataset-based authentication results."""

    page_header("Thresholds & decisions · Step 5", "Try Authentication",
                "Pick a user, enter a query score and see the decision the active threshold (or review policy) makes — then the "
                "dataset-wide confusion matrix and error rates.", icon="🔐")
    if not require_dataset("Create the shared demo dataset to try authentication decisions."):
        return

    dataset = st.session_state.dataset

    # ---------------------------------------------------------
    # Determine the authentication threshold
    #
    # If a threshold has been locked in Distribution Analysis,
    # use that value.
    #
    # Otherwise, use the current live threshold.
    # ---------------------------------------------------------
    locked_threshold = st.session_state.get("locked_threshold")
    policy = st.session_state.get("threshold_policy")

    if policy and policy.get("mode") == "three_zone":
        threshold = float(policy["accept_threshold"])
        st.info(
            f"Cost-aware three-zone policy active: reject below {policy['reject_threshold']:.4f}, "
            f"review from {policy['reject_threshold']:.4f} to {threshold:.4f}, and accept at or above {threshold:.4f}."
        )
    elif locked_threshold is not None:
        threshold = float(locked_threshold)

        profile = st.session_state.get("active_threshold_profile")
        st.success(f"Using locked operating threshold: {threshold:.2f}" + (f" · set by {profile}" if profile else ""))

    else:
        threshold = float(st.session_state.threshold)

        st.warning(
            f"No operating threshold is locked. "
            f"Using live threshold: {threshold:.2f}"
        )

    # ---------------------------------------------------------
    # Interactive authentication demonstration
    # ---------------------------------------------------------
    users = sorted(dataset["User_ID"].unique())

    reference = st.selectbox(
        "Reference user / User ID",
        users,
    )

    score = st.slider(
        "Query fingerprint similarity score",
        0.0,
        1.0,
        0.75,
        0.01,
    )

    if policy and policy.get("mode") == "three_zone":
        if score < policy["reject_threshold"]:
            decision = "Rejected"
        elif score < threshold:
            decision = "Manual review"
        else:
            decision = "Accepted"
    else:
        decision = "Accepted" if score >= threshold else "Rejected"

    # ---------------------------------------------------------
    # Authentication summary metrics
    # ---------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Reference fingerprint",
        reference,
    )

    c2.metric(
        "Similarity score",
        f"{score:.2f}",
    )

    if policy and policy.get("mode") == "three_zone":
        c3.metric("Reject threshold", f"{policy['reject_threshold']:.3f}")
        c4.metric("Accept threshold", f"{threshold:.3f}")
    else:
        c3.metric("Authentication threshold", f"{threshold:.2f}")
        c4.metric("Difference from threshold", f"{score - threshold:+.2f}")

    # ---------------------------------------------------------
    # Authentication decision
    # ---------------------------------------------------------
    if decision == "Accepted":
        st.success("AUTHENTICATION ACCEPTED - Classification: Genuine")
    elif decision == "Manual review":
        st.warning("MANUAL REVIEW - Similarity score is inside the review zone")
    else:
        st.error("AUTHENTICATION REJECTED - Classification: Impostor")

    st.caption(
        "Similarity-score-based fingerprint authentication prototype. "
        "The score represents the similarity between the query fingerprint "
        "and the claimed/reference fingerprint; no fingerprint image "
        "extraction pipeline is implemented."
    )

    # ---------------------------------------------------------
    # Conceptual workflow
    # ---------------------------------------------------------
    st.subheader("Conceptual fingerprint-verification workflow")

    st.write(
        "Reference fingerprint -> preprocessing -> feature extraction -> "
        "query-template comparison -> similarity score -> "
        + ("reject / manual review / accept decision" if policy else "threshold decision")
        + " -> authentication outcome"
    )

    # ---------------------------------------------------------
    # Dataset authentication performance
    # ---------------------------------------------------------
    genuine_scores, impostor_scores = split_scores(dataset)

    metrics = performance_metrics(
        dataset,
        threshold,
    )

    total_attempts = len(dataset)

    accuracy = (metrics["TP"] + metrics["TN"]) / total_attempts if total_attempts else 0

    st.subheader("Dataset Authentication Performance")

    # ---------------------------------------------------------
    # Confusion-matrix counts
    # ---------------------------------------------------------
    if policy and policy.get("mode") == "three_zone":
        reject_threshold = policy["reject_threshold"]
        genuine_reject = int((genuine_scores < reject_threshold).sum())
        genuine_review = int(((genuine_scores >= reject_threshold) & (genuine_scores < threshold)).sum())
        genuine_accept = int((genuine_scores >= threshold).sum())
        impostor_reject = int((impostor_scores < reject_threshold).sum())
        impostor_review = int(((impostor_scores >= reject_threshold) & (impostor_scores < threshold)).sum())
        impostor_accept = int((impostor_scores >= threshold).sum())
        zone_table = pd.DataFrame([
            {"Actual class": "Genuine", "Reject": f"{genuine_reject} ({genuine_reject / len(genuine_scores):.1%})",
             "Manual review": f"{genuine_review} ({genuine_review / len(genuine_scores):.1%})",
             "Accept": f"{genuine_accept} ({genuine_accept / len(genuine_scores):.1%})"},
            {"Actual class": "Impostor", "Reject": f"{impostor_reject} ({impostor_reject / len(impostor_scores):.1%})",
             "Manual review": f"{impostor_review} ({impostor_review / len(impostor_scores):.1%})",
             "Accept": f"{impostor_accept} ({impostor_accept / len(impostor_scores):.1%})"},
        ])
        st.dataframe(zone_table, width="stretch", hide_index=True)
        zone_columns = st.columns(3)
        zone_columns[0].metric("Accept FAR", f"{impostor_accept / len(impostor_scores):.2%}")
        zone_columns[1].metric("Hard-reject FRR", f"{genuine_reject / len(genuine_scores):.2%}")
        total_review = genuine_review + impostor_review
        zone_columns[2].metric("Manual review share", f"{total_review / total_attempts:.2%}")
        st.caption("The review zone is counted separately. Other project pages use the accept threshold as their single binary cutoff.")
    else:
        count_columns = st.columns(4)
        count_columns[0].metric("True Positives (TP)", metrics["TP"])
        count_columns[1].metric("True Negatives (TN)", metrics["TN"])
        count_columns[2].metric("False Positives (FP)", metrics["FP"])
        count_columns[3].metric("False Negatives (FN)", metrics["FN"])

        rate_columns = st.columns(5)
        rate_columns[0].metric("TAR", f"{metrics['TAR']:.2%}")
        rate_columns[1].metric("FRR", f"{metrics['FRR']:.2%}")
        rate_columns[2].metric("TRR", f"{metrics['TRR']:.2%}")
        rate_columns[3].metric("FAR", f"{metrics['FAR']:.2%}")
        rate_columns[4].metric("Accuracy", f"{accuracy:.2%}")

        st.caption(
            f"TAR = {metrics['TP']} / {len(genuine_scores)}, "
            f"FRR = {metrics['FN']} / {len(genuine_scores)}, "
            f"TRR = {metrics['TN']} / {len(impostor_scores)}, "
            f"FAR = {metrics['FP']} / {len(impostor_scores)}."
        )

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------
    if not (policy and policy.get("mode") == "three_zone"):
        st.subheader("Confusion Matrix")
        st.pyplot(confusion_matrix(metrics), width="content")

    # ---------------------------------------------------------
    # Authentication Score Scatter Plot
    # ---------------------------------------------------------
    st.subheader("Authentication Score Scatter Plot")

    st.pyplot(
        _authentication_score_chart(
            dataset,
            threshold,
            policy,
        ),
        width="stretch",
    )

    # ---------------------------------------------------------
    # Dataset Authentication Results
    # ---------------------------------------------------------
    st.subheader("Dataset Authentication Results")

    result_table = dataset[
        [
            "Sample_ID",
            "User_ID",
            "Attempt_ID",
            "Class",
            "Similarity_Score",
        ]
    ].copy()

    if policy and policy.get("mode") == "three_zone":
        result_table["System Decision"] = result_table["Similarity_Score"].map(
            lambda value: "Rejected" if value < policy["reject_threshold"]
            else "Manual review" if value < threshold else "Accepted"
        )
    else:
        result_table["System Decision"] = result_table["Similarity_Score"].ge(threshold).map(
            {True: "Accepted", False: "Rejected"}
        )

    # Authentication result
    result_table["Authentication Result"] = [
        (
            "Manual Review"
            if decision == "Manual review"
            else "True Positive"
            if actual_class == "Genuine" and decision == "Accepted"
            else "False Negative"
            if actual_class == "Genuine"
            else "False Positive"
            if decision == "Accepted"
            else "True Negative"
        )
        for actual_class, decision in zip(
            result_table["Class"],
            result_table["System Decision"],
        )
    ]

    st.dataframe(
        result_table.head(25),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Showing the first 25 dataset attempts. "
        "Each attempt is classified using the active authentication policy."
    )

    # ---------------------------------------------------------
    # Automatic Interpretation
    # ---------------------------------------------------------
    st.subheader("Automatic Interpretation")

    st.info(
        (
            f"With reject threshold {policy['reject_threshold']:.4f} and accept threshold {threshold:.4f}, "
            f"the dataset sends {total_review / total_attempts:.2%} of attempts to manual review. "
            f"The accept zone's FAR is {impostor_accept / len(impostor_scores):.2%}; "
            f"the hard-reject FRR is {genuine_reject / len(genuine_scores):.2%}."
            if policy and policy.get("mode") == "three_zone"
            else f"At the current threshold of {threshold:.2f}, the observed dataset accepts {metrics['TAR']:.2%} "
            f"of genuine attempts and incorrectly accepts {metrics['FAR']:.2%} of impostor attempts. "
            f"It correctly rejects {metrics['TRR']:.2%} of impostor attempts, with an overall accuracy of {accuracy:.2%}."
        )
    )

    # ---------------------------------------------------------
    # Connection with Previous Analysis
    # ---------------------------------------------------------
    st.subheader("Connection with Previous Analysis")

    if policy and policy.get("mode") == "three_zone":
        st.write(
            "The reject and accept boundaries are selected from the development users in the Advanced Threshold Studio. "
            "Scores below the lower boundary are rejected, scores between boundaries are routed to manual review, "
            "and scores at or above the upper boundary are accepted."
        )
    else:
        st.write(
            "The authentication threshold is selected based on the "
            "similarity-score distributions analyzed in Step 4. "
            "Genuine scores above the threshold are accepted, while "
            "impostor scores below the threshold are rejected. "
            "The resulting TP, TN, FP and FN values are used to "
            "evaluate authentication performance."
        )
