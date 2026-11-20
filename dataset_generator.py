"""Synthetic biometric similarity-score dataset generation.

The base scores (genuine ~ N(0.82, 0.09) clipped to [0.40, 1], impostor ~ N(0.42, 0.13) clipped to [0.03, 0.90]) are drawn
exactly as in the original project, so the default dataset is unchanged. Optional realism controls are applied afterwards
from a separate random stream:

* user_variability  – SD of a per-user offset on genuine scores (some users are naturally harder to match: "goats")
* sensor_noise      – SD of additive Gaussian measurement noise on every score
* template_decay    – exponential ageing rate λ (per day): genuine scores decay toward the impostor mean as the
                      enrolled template gets older, s(d) = μ_I + (s − μ_I)·e^(−λd)
* arrival_rate      – attempts per minute; arrivals form a Poisson process, so the gaps between attempts are Exponential
"""
import numpy as np
import pandas as pd

IMPOSTOR_MEAN = 0.42


def generate_biometric_dataset(number_users=20, genuine_samples_per_user=20, impostor_samples_per_user=20, seed=42,
                               biometric_type="Fingerprint Verification", user_variability=0.0, sensor_noise=0.0,
                               template_decay=0.0, arrival_rate=6.0, max_template_age=180):
    """Return a simulated fingerprint-verification dataset bounded to [0, 1]."""
    rng, records, sample_id = np.random.default_rng(seed), [], 1
    for user_number in range(1, number_users + 1):
        user_id = f"U{user_number:03d}"
        for attempt, score in enumerate(np.clip(rng.normal(.82, .09, genuine_samples_per_user), .40, 1), 1):
            records.append({"Sample_ID": sample_id, "User_ID": user_id, "Attempt_ID": f"{user_id}-G{attempt:02d}", "Biometric_Type": biometric_type, "Actual_Class": "Genuine", "Class": "Genuine", "Similarity_Score": round(float(score), 4)}); sample_id += 1
        for attempt, score in enumerate(np.clip(rng.normal(.42, .13, impostor_samples_per_user), .03, .90), 1):
            records.append({"Sample_ID": sample_id, "User_ID": user_id, "Attempt_ID": f"{user_id}-I{attempt:02d}", "Biometric_Type": biometric_type, "Actual_Class": "Impostor", "Class": "Impostor", "Similarity_Score": round(float(score), 4)}); sample_id += 1
    df = pd.DataFrame(records)

    # Independent stream for the extra columns so the base scores never change.
    aux = np.random.default_rng(seed + 7919)
    n = len(df)
    gaps = aux.exponential(60.0 / max(float(arrival_rate), 1e-6), n)
    df["Timestamp_s"] = np.round(np.cumsum(gaps), 2)
    df["Template_Age_Days"] = aux.integers(0, int(max_template_age) + 1, n)

    genuine = (df["Class"] == "Genuine").to_numpy()
    scores = df["Similarity_Score"].to_numpy(float).copy()
    changed = False
    if user_variability > 0:
        offsets = dict(zip(df["User_ID"].unique(), aux.normal(0, user_variability, df["User_ID"].nunique())))
        scores[genuine] += df.loc[genuine, "User_ID"].map(offsets).to_numpy(float)
        changed = True
    if template_decay > 0:
        age = df["Template_Age_Days"].to_numpy(float)
        scores[genuine] = IMPOSTOR_MEAN + (scores[genuine] - IMPOSTOR_MEAN) * np.exp(-template_decay * age[genuine])
        changed = True
    if sensor_noise > 0:
        scores += aux.normal(0, sensor_noise, n)
        changed = True
    if changed:
        df["Similarity_Score"] = np.round(np.clip(scores, 0, 1), 4)
    return df
