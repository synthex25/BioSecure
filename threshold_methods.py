"""Threshold-selection theory: empirical criteria and closed-form Gaussian (Bayes / Neyman-Pearson) thresholds.

Notation used throughout
------------------------
X                  similarity score (a continuous random variable)
f_G, f_I           class-conditional densities of X for genuine (G) and impostor (I) attempts
pi_G, pi_I         prior probabilities of a genuine / impostor attempt (pi_G + pi_I = 1)
C_FA, C_FR         cost of a false accept / a false reject
FAR(t) = P(X >= t | I)      FRR(t) = P(X < t | G)
Bayes risk  R(t) = pi_G * C_FR * FRR(t) + pi_I * C_FA * FAR(t)
"""
import numpy as np
import pandas as pd
from scipy import stats

from utils.advanced import far_frr_curve

EPS = 1e-4


# -----------------------------------------------------------------------------
# Gaussian model of the two score distributions
# -----------------------------------------------------------------------------
def gaussian_params(genuine, impostor):
    """Maximum-likelihood style Gaussian fit (sample mean, sample SD) for each class."""
    return {"mg": float(np.mean(genuine)), "sg": float(np.std(genuine, ddof=1)),
            "mi": float(np.mean(impostor)), "si": float(np.std(impostor, ddof=1))}


def model_rates(t, p):
    """FAR / FRR predicted by the Gaussian model at threshold(s) t."""
    t = np.asarray(t, float)
    return stats.norm.sf(t, p["mi"], p["si"]), stats.norm.cdf(t, p["mg"], p["sg"])


def bayes_gaussian_threshold(p, prior_g, c_fa=1.0, c_fr=1.0):
    """Minimum-risk threshold for two Gaussian classes.

    Accept when pi_G C_FR f_G(x) >= pi_I C_FA f_I(x). Taking logs gives a quadratic a x^2 + b x + c = 0:
        a = 1/(2 s_I^2) - 1/(2 s_G^2)
        b = m_G/s_G^2 - m_I/s_I^2
        c = ln(pi_G C_FR s_I / (pi_I C_FA s_G)) - m_G^2/(2 s_G^2) + m_I^2/(2 s_I^2)
    The root between the two means (or nearest to it) is the decision boundary.
    """
    prior_g = float(np.clip(prior_g, 1e-6, 1 - 1e-6))
    mg, sg, mi, si = p["mg"], p["sg"], p["mi"], p["si"]
    k = np.log((prior_g * c_fr * si) / ((1 - prior_g) * c_fa * sg))
    a = 1 / (2 * si ** 2) - 1 / (2 * sg ** 2)
    b = mg / sg ** 2 - mi / si ** 2
    c = k - mg ** 2 / (2 * sg ** 2) + mi ** 2 / (2 * si ** 2)
    if abs(a) < 1e-12:
        roots = np.array([-c / b])
    else:
        disc = b * b - 4 * a * c
        roots = np.array([(-b + s * np.sqrt(disc)) / (2 * a) for s in (1, -1)]) if disc >= 0 else np.array([])
    mid = (mg + mi) / 2
    if len(roots):
        inside = roots[(roots >= min(mi, mg) - 3 * max(sg, si)) & (roots <= max(mi, mg) + 3 * max(sg, si))]
        pool = inside if len(inside) else roots
        t = float(pool[np.argmin(np.abs(pool - mid))])
    else:  # no crossing: one weighted density dominates everywhere
        t = 0.0 if k > 0 else 1.0
    return float(np.clip(t, 0, 1)), {"a": a, "b": b, "c": c, "k": k, "roots": roots.tolist()}


def gaussian_eer(p):
    """Closed-form EER under the Gaussian model: (t - m_I)/s_I = (m_G - t)/s_G."""
    t = (p["mi"] * p["sg"] + p["mg"] * p["si"]) / (p["si"] + p["sg"])
    return float(t), float(stats.norm.cdf((p["mi"] - p["mg"]) / (p["si"] + p["sg"])))


def neyman_pearson_gaussian(p, alpha):
    """Smallest threshold with model FAR <= alpha:  t = m_I + s_I * z_(1-alpha)."""
    return float(p["mi"] + p["si"] * stats.norm.ppf(1 - alpha))


def frr_target_gaussian(p, beta):
    """Largest threshold with model FRR <= beta:  t = m_G + s_G * z_beta."""
    return float(p["mg"] + p["sg"] * stats.norm.ppf(beta))


# -----------------------------------------------------------------------------
# Empirical criteria
# -----------------------------------------------------------------------------
def _grid(genuine, impostor, points=2001):
    return far_frr_curve(genuine, impostor, points)


def empirical_rates(genuine, impostor, t):
    far = float((np.asarray(impostor) >= t).mean())
    frr = float((np.asarray(genuine) < t).mean())
    return far, frr


def evaluate_threshold(genuine, impostor, t, prior_g, c_fa, c_fr, p=None):
    far, frr = empirical_rates(genuine, impostor, t)
    tar = 1 - frr
    pi = 1 - prior_g
    accept = tar * prior_g + far * pi
    row = {
        "Threshold": float(t), "FAR": far, "FRR": frr, "TAR": tar,
        "Error rate": prior_g * frr + pi * far,
        "Expected cost": prior_g * c_fr * frr + pi * c_fa * far,
        "P(G | accept)": tar * prior_g / accept if accept > 0 else np.nan,
        "HTER": (far + frr) / 2,
    }
    if p is not None:
        mfar, mfrr = model_rates(t, p)
        row["Model FAR"], row["Model FRR"] = float(mfar), float(mfrr)
    return row


def threshold_candidates(genuine, impostor, prior_g=0.5, c_fa=1.0, c_fr=1.0, alpha=0.01, beta=0.05):
    """Every threshold-selection rule the lab compares, with its criterion and the family it belongs to."""
    p = gaussian_params(genuine, impostor)
    t, far, frr = _grid(genuine, impostor)
    pi = 1 - prior_g
    out = []

    def add(method, family, criterion, thr):
        out.append({"Method": method, "Family": family, "Criterion": criterion, "Threshold": float(np.clip(thr, 0, 1))})

    i = int(np.argmin(np.abs(far - frr)))
    add("Equal Error Rate", "Empirical", "FAR = FRR", t[i])
    add("Youden's J", "Empirical", "max TAR − FAR", t[int(np.argmin(far + frr))])
    add("Minimum total error", "Empirical", "min π_G·FRR + π_I·FAR", t[int(np.argmin(prior_g * frr + pi * far))])
    add("Minimum expected cost", "Empirical", "min π_G·C_FR·FRR + π_I·C_FA·FAR",
        t[int(np.argmin(prior_g * c_fr * frr + pi * c_fa * far))])
    ok = far <= alpha
    add(f"Neyman–Pearson (FAR ≤ {alpha:.2%})", "Empirical", "lowest t with FAR ≤ α", t[int(np.argmax(ok))] if ok.any() else 1.0)
    okb = frr <= beta
    add(f"FRR target (FRR ≤ {beta:.0%})", "Empirical", "highest t with FRR ≤ β", t[np.where(okb)[0][-1]] if okb.any() else 0.0)
    add("Zero-FAR", "Empirical", "just above the highest impostor score", float(np.max(impostor)) + EPS)
    add("Midpoint of means", "Heuristic", "(μ_G + μ_I) / 2", (p["mg"] + p["mi"]) / 2)
    add("Gaussian EER", "Gaussian model", "(μ_I σ_G + μ_G σ_I) / (σ_I + σ_G)", gaussian_eer(p)[0])
    add("Bayes minimum error", "Gaussian model", "π_G f_G(x) = π_I f_I(x)", bayes_gaussian_threshold(p, prior_g)[0])
    add("Bayes minimum risk", "Gaussian model", "π_G C_FR f_G(x) = π_I C_FA f_I(x)", bayes_gaussian_threshold(p, prior_g, c_fa, c_fr)[0])
    add(f"Neyman–Pearson model (FAR ≤ {alpha:.2%})", "Gaussian model", "μ_I + σ_I·z₁₋α", neyman_pearson_gaussian(p, alpha))
    add(f"FRR target model (FRR ≤ {beta:.0%})", "Gaussian model", "μ_G + σ_G·z_β", frr_target_gaussian(p, beta))

    rows = [{**c, **{k: v for k, v in evaluate_threshold(genuine, impostor, c["Threshold"], prior_g, c_fa, c_fr, p).items()
                     if k != "Threshold"}} for c in out]
    return pd.DataFrame(rows), p


def risk_curves(genuine, impostor, p, prior_g, c_fa, c_fr, points=401):
    t, far, frr = _grid(genuine, impostor, points)
    mfar, mfrr = model_rates(t, p)
    pi = 1 - prior_g
    return t, prior_g * c_fr * frr + pi * c_fa * far, prior_g * c_fr * mfrr + pi * c_fa * mfar


def bayes_threshold_vs_prior(p, impostor_priors, cost_ratio=1.0):
    """How the Bayes threshold moves as the impostor prior changes (cost_ratio = C_FA / C_FR)."""
    return np.array([bayes_gaussian_threshold(p, 1 - q, cost_ratio, 1.0)[0] for q in impostor_priors])


def posterior_curve(x, p, prior_g):
    """P(G | X = x) under the Gaussian class-conditional model (Bayes' theorem with densities)."""
    num = prior_g * stats.norm.pdf(x, p["mg"], p["sg"])
    den = num + (1 - prior_g) * stats.norm.pdf(x, p["mi"], p["si"])
    return np.divide(num, den, out=np.full_like(np.asarray(x, float), np.nan), where=den > 0)


def kde_posterior_curve(x, genuine, impostor, prior_g):
    """Same posterior but with non-parametric (Gaussian kernel) density estimates."""
    fg, fi = stats.gaussian_kde(genuine)(x), stats.gaussian_kde(impostor)(x)
    num = prior_g * fg
    den = num + (1 - prior_g) * fi
    return np.divide(num, den, out=np.full_like(np.asarray(x, float), np.nan), where=den > 1e-12)


def sensitivity(t, p, genuine, impostor, bandwidth=0.03):
    """Local slope of the error curves: dFAR/dt = -f_I(t), dFRR/dt = f_G(t) (model and smoothed empirical)."""
    kg, ki = stats.gaussian_kde(genuine, bw_method=bandwidth / np.std(genuine, ddof=1)), \
        stats.gaussian_kde(impostor, bw_method=bandwidth / np.std(impostor, ddof=1))
    return {"model_fi": float(stats.norm.pdf(t, p["mi"], p["si"])), "model_fg": float(stats.norm.pdf(t, p["mg"], p["sg"])),
            "kde_fi": float(ki(t)[0]), "kde_fg": float(kg(t)[0])}


def bootstrap_thresholds(df, prior_g, c_fa, c_fr, alpha, n_boot=300, seed=11):
    """User-level bootstrap of the EER, Bayes-risk and Neyman-Pearson thresholds (how stable is each choice?)."""
    rng = np.random.default_rng(seed)
    users = df["User_ID"].unique()
    groups = {u: part for u, part in df.groupby("User_ID")}
    out = {"Equal Error Rate": [], "Bayes minimum risk": [], "Neyman–Pearson (empirical)": []}
    for _ in range(int(n_boot)):
        pick = rng.choice(users, len(users), replace=True)
        sample = pd.concat([groups[u] for u in pick], ignore_index=True)
        g = sample.loc[sample.Class == "Genuine", "Similarity_Score"].to_numpy(float)
        i = sample.loc[sample.Class == "Impostor", "Similarity_Score"].to_numpy(float)
        t, far, frr = _grid(g, i, 1001)
        out["Equal Error Rate"].append(t[int(np.argmin(np.abs(far - frr)))])
        out["Bayes minimum risk"].append(bayes_gaussian_threshold(gaussian_params(g, i), prior_g, c_fa, c_fr)[0])
        ok = far <= alpha
        out["Neyman–Pearson (empirical)"].append(t[int(np.argmax(ok))] if ok.any() else 1.0)
    return {k: np.asarray(v) for k, v in out.items()}


def threshold_table(genuine, impostor, prior_g, c_fa, c_fr, start=0.30, stop=0.95, step=0.05):
    p = gaussian_params(genuine, impostor)
    grid = np.round(np.arange(start, stop + step / 2, step), 4)
    return pd.DataFrame([evaluate_threshold(genuine, impostor, t, prior_g, c_fa, c_fr, p) for t in grid])
