"""Reusable probability, classification, and inference calculations."""
import numpy as np
from scipy import stats


def split_scores(df):
    return (df.loc[df["Class"] == "Genuine", "Similarity_Score"].to_numpy(float), df.loc[df["Class"] == "Impostor", "Similarity_Score"].to_numpy(float))


def probability_analysis(df, threshold):
    genuine, impostor = split_scores(df); total = len(genuine) + len(impostor)
    pg, pi = len(genuine) / total if total else 0, len(impostor) / total if total else 0
    eg, ei = (genuine >= threshold).mean() if len(genuine) else 0, (impostor >= threshold).mean() if len(impostor) else 0
    denominator = eg * pg + ei * pi
    return {"total": total, "genuine_count": len(genuine), "impostor_count": len(impostor), "prior_genuine": pg, "prior_impostor": pi, "p_e_given_genuine": eg, "p_e_given_impostor": ei, "genuine_above": int((genuine >= threshold).sum()), "impostor_above": int((impostor >= threshold).sum()), "numerator": eg * pg, "denominator": denominator, "posterior_genuine": eg * pg / denominator if denominator else 0}


def performance_metrics(df, threshold):
    genuine, impostor = split_scores(df)
    tp, fn = int((genuine >= threshold).sum()), int((genuine < threshold).sum())
    fp, tn = int((impostor >= threshold).sum()), int((impostor < threshold).sum())
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "TAR": tp / len(genuine) if len(genuine) else 0, "FRR": fn / len(genuine) if len(genuine) else 0, "FAR": fp / len(impostor) if len(impostor) else 0, "TRR": tn / len(impostor) if len(impostor) else 0}


def descriptive_statistics(scores):
    return {"Samples": len(scores), "Mean": np.mean(scores), "Standard Deviation": np.std(scores, ddof=1), "Median": np.median(scores), "Minimum": np.min(scores), "Maximum": np.max(scores)}


def effect_size_and_overlap(genuine, impostor):
    pooled = np.sqrt(((len(genuine)-1)*np.var(genuine, ddof=1)+(len(impostor)-1)*np.var(impostor, ddof=1))/(len(genuine)+len(impostor)-2))
    effect = (np.mean(genuine)-np.mean(impostor))/pooled if pooled else 0
    overlap = 2 * stats.norm.cdf(-abs(effect)/2)
    return {"mean_difference": np.mean(genuine)-np.mean(impostor), "pooled_sd": pooled, "effect_size": effect, "overlap": overlap, "separation": 1-overlap}


def confidence_interval(scores, confidence=.95):
    mean, sem = np.mean(scores), stats.sem(scores)
    margin = stats.t.ppf((1+confidence)/2, len(scores)-1) * sem
    return mean-margin, mean+margin


def cluster_bootstrap_rates(df, threshold, n_boot=1000, random_state=42):
    """Bootstrap TAR and FAR by resampling whole users, keeping their attempts together."""
    users = df["User_ID"].dropna().unique()
    if len(users) < 2:
        raise ValueError("At least two users are required for a user-level bootstrap.")

    accepted = df["Similarity_Score"].to_numpy(float) >= float(threshold)
    prepared = df[["User_ID", "Class"]].copy()
    prepared["Accepted"] = accepted

    per_user = {}
    for label, prefix in (("Genuine", "genuine"), ("Impostor", "impostor")):
        group = prepared.loc[prepared["Class"] == label].groupby("User_ID")["Accepted"].agg(["sum", "count"])
        per_user[f"{prefix}_accepted"] = group["sum"].reindex(users, fill_value=0).to_numpy(float)
        per_user[f"{prefix}_total"] = group["count"].reindex(users, fill_value=0).to_numpy(float)

    if per_user["genuine_total"].sum() == 0 or per_user["impostor_total"].sum() == 0:
        raise ValueError("Both genuine and impostor attempts are required.")

    rng = np.random.default_rng(random_state)
    draws = rng.integers(0, len(users), size=(int(n_boot), len(users)))

    def bootstrap_interval(success_key, total_key):
        successes = per_user[success_key][draws].sum(axis=1)
        totals = per_user[total_key][draws].sum(axis=1)
        rates = np.full(int(n_boot), np.nan, dtype=float)
        np.divide(successes, totals, out=rates, where=totals > 0)
        valid = rates[np.isfinite(rates)]
        if not len(valid):
            raise ValueError("The bootstrap could not form valid samples for this rate.")
        return np.nanpercentile(valid, [2.5, 97.5]), valid

    tar_ci, tar_samples = bootstrap_interval("genuine_accepted", "genuine_total")
    far_ci, far_samples = bootstrap_interval("impostor_accepted", "impostor_total")
    return {
        "users": int(len(users)),
        "resamples": int(n_boot),
        "TAR": float(per_user["genuine_accepted"].sum() / per_user["genuine_total"].sum()),
        "TAR_CI": tuple(float(value) for value in tar_ci),
        "TAR_samples": tar_samples,
        "FAR": float(per_user["impostor_accepted"].sum() / per_user["impostor_total"].sum()),
        "FAR_CI": tuple(float(value) for value in far_ci),
        "FAR_samples": far_samples,
    }


def hypothesis_test(genuine, impostor):
    return stats.ttest_ind(genuine, impostor, equal_var=False)


def threshold_sweep(df, thresholds=None):
    """Evaluate biometric operating rates over thresholds and estimate EER."""
    thresholds = np.linspace(0, 1, 101) if thresholds is None else np.asarray(thresholds)
    rows = []
    for threshold in thresholds:
        metrics = performance_metrics(df, threshold)
        rows.append({"threshold": threshold, **metrics})
    eer_index = int(np.argmin([abs(row["FAR"] - row["FRR"]) for row in rows]))
    eer = (rows[eer_index]["FAR"] + rows[eer_index]["FRR"]) / 2
    return rows, {"threshold": rows[eer_index]["threshold"], "eer": eer}


def multi_threshold_optimization(df, start=0.0, stop=1.0, step=0.01):
    """Evaluate every threshold on a regular grid and pick the one with minimum HTER.

    For each threshold t (scores >= t are accepted):
        FAR = FP / impostors      FRR = FN / genuine      TAR = 1 - FRR
        Accuracy = (TP + TN) / all attempts
        HTER = (FAR + FRR) / 2    (half total error rate; independent of the class mix)

    Ties on HTER are common because empirical error rates move in steps. They are broken by
    (1) the smallest |FAR - FRR| (the most balanced point), then (2) the higher threshold
    (the more secure choice). Returns (results DataFrame, recommended row as a dict).
    """
    import pandas as pd

    genuine, impostor = split_scores(df)
    if not len(genuine) or not len(impostor):
        raise ValueError("Both genuine and impostor scores are required.")
    count = int(round((stop - start) / step)) + 1
    thresholds = np.round(start + step * np.arange(count), 6)   # exact 0.00, 0.01, ... (no float drift)

    g_sorted, i_sorted = np.sort(genuine), np.sort(impostor)
    fn = np.searchsorted(g_sorted, thresholds, side="left")      # genuine scores below t
    tn = np.searchsorted(i_sorted, thresholds, side="left")      # impostor scores below t
    tp, fp = len(genuine) - fn, len(impostor) - tn
    far, frr = fp / len(impostor), fn / len(genuine)
    results = pd.DataFrame({
        "Threshold": thresholds, "FAR": far, "FRR": frr, "TAR": 1 - frr,
        "Accuracy": (tp + tn) / (len(genuine) + len(impostor)), "HTER": (far + frr) / 2,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    })
    order = np.lexsort((-results["Threshold"].to_numpy(), np.abs(far - frr), np.round(results["HTER"].to_numpy(), 12)))
    best_index = int(order[0])
    results["Recommended"] = False
    results.loc[best_index, "Recommended"] = True
    best = results.loc[best_index].to_dict()
    best["tied_thresholds"] = results.loc[np.isclose(results["HTER"], best["HTER"]), "Threshold"].tolist()
    return results, best


def roc_metrics(df):
    """Compute a score-based ROC curve and trapezoidal AUC without extra dependencies."""
    genuine, impostor = split_scores(df)
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[genuine, impostor]))[::-1], -np.inf]
    fpr = np.array([(impostor >= threshold).mean() for threshold in thresholds])
    tpr = np.array([(genuine >= threshold).mean() for threshold in thresholds])
    order = np.argsort(fpr)
    return fpr[order], tpr[order], float(np.trapezoid(tpr[order], fpr[order]))


# =============================================================================
# Discrete and continuous rare-event modeling: Bernoulli, Binomial, Poisson,
# Exponential. These treat a single authentication attempt as a Bernoulli
# trial, the count of successes across attempts as Binomial, the count of
# rare false-acceptance events as Poisson (the classical rare-event
# approximation to Binomial), and the waiting time between successive
# false-acceptance events as Exponential.
# =============================================================================

def bernoulli_model(df, threshold):
    """Model a single authentication attempt as a Bernoulli trial for each class."""
    genuine, impostor = split_scores(df)
    genuine_trials, impostor_trials = len(genuine), len(impostor)
    genuine_successes = int((genuine >= threshold).sum())
    impostor_successes = int((impostor >= threshold).sum())
    p_genuine_accept = genuine_successes / genuine_trials if genuine_trials else 0
    p_impostor_accept = impostor_successes / impostor_trials if impostor_trials else 0
    return {
        "genuine_trials": genuine_trials,
        "impostor_trials": impostor_trials,
        "genuine_successes": genuine_successes,
        "impostor_successes": impostor_successes,
        "p_genuine_accept": p_genuine_accept,
        "p_impostor_accept": p_impostor_accept,
        "variance_genuine": p_genuine_accept * (1 - p_genuine_accept),
        "variance_impostor": p_impostor_accept * (1 - p_impostor_accept),
    }


def binomial_model(n_trials, p, k_values=None):
    """Return the Binomial(n, p) PMF, mean, and standard deviation."""
    n_trials = int(n_trials)
    if k_values is None:
        k_values = np.arange(0, n_trials + 1)
    else:
        k_values = np.asarray(k_values)
    pmf = stats.binom.pmf(k_values, n_trials, p)
    return {
        "n": n_trials,
        "p": p,
        "k": k_values,
        "pmf": pmf,
        "mean": n_trials * p,
        "std": float(np.sqrt(n_trials * p * (1 - p))) if 0 < p < 1 else 0.0,
    }


def poisson_model(lam, k_max=None):
    """Return the Poisson(lambda) PMF, mean, and standard deviation."""
    lam = max(float(lam), 0.0)
    if k_max is None:
        k_max = int(max(10, lam + 4 * np.sqrt(lam) if lam > 0 else 10))
    k_values = np.arange(0, k_max + 1)
    pmf = stats.poisson.pmf(k_values, lam) if lam > 0 else np.r_[1.0, np.zeros(k_max)]
    return {
        "lambda": lam,
        "k": k_values,
        "pmf": pmf,
        "mean": lam,
        "std": float(np.sqrt(lam)),
    }


def binomial_poisson_comparison(n_trials, p):
    """Compare the exact Binomial(n, p) with its Poisson(lambda = n*p) approximation.

    This is the standard rare-event approximation: valid when n is large and
    p is small, which is exactly the false-acceptance-rate scenario in
    biometric authentication.
    """
    lam = n_trials * p
    binomial = binomial_model(n_trials, p)
    poisson = poisson_model(lam, k_max=int(binomial["k"][-1]) if len(binomial["k"]) else 10)
    max_absolute_difference = float(np.max(np.abs(binomial["pmf"][: len(poisson["pmf"])] - poisson["pmf"][: len(binomial["pmf"])])))
    return {"binomial": binomial, "poisson": poisson, "lambda": lam, "max_absolute_difference": max_absolute_difference}


def projected_breach_distribution(far, future_attempts):
    """Poisson-project the expected number of false-acceptance breaches over a future batch of attempts."""
    lam = far * future_attempts
    return poisson_model(lam)


def false_accept_gaps(df, threshold):
    """Return attempt-count gaps between successive impostor false-acceptance events.

    Attempts are ordered by Sample_ID to approximate a sequential attempt
    stream. A false-acceptance event occurs when an Impostor attempt scores
    at or above the threshold. The gap is the number of attempts between one
    such event and the next.
    """
    ordered = df.sort_values("Sample_ID")
    is_breach = (ordered["Class"] == "Impostor") & (ordered["Similarity_Score"] >= threshold)
    breach_sample_ids = ordered.loc[is_breach, "Sample_ID"].to_numpy(dtype=float)
    if len(breach_sample_ids) < 2:
        return np.array([])
    return np.diff(breach_sample_ids)


def exponential_model(gaps):
    """Fit an Exponential distribution to observed inter-breach attempt gaps."""
    if len(gaps) == 0:
        return None
    mean_gap = float(np.mean(gaps))
    rate = 1.0 / mean_gap if mean_gap > 0 else 0.0
    x = np.linspace(0, float(gaps.max()) * 1.2 if gaps.max() > 0 else 1.0, 300)
    pdf = stats.expon.pdf(x, scale=1 / rate) if rate > 0 else np.zeros_like(x)
    return {
        "rate": rate,
        "mean_gap": mean_gap,
        "gaps": gaps,
        "x": x,
        "pdf": pdf,
        "prob_no_breach_within": lambda n_attempts: float(np.exp(-rate * n_attempts)) if rate > 0 else 1.0,
    }
