"""Advanced analytics: multimodal fusion, DET/EER tools, per-user risk profiling, attack and lockout models."""
import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# Score-level tools shared by several pages
# ----------------------------------------------------------------------------
def far_frr_curve(genuine, impostor, points=401):
    """Return thresholds, FAR and FRR over [0, 1] using sorted-score lookups (fast, vectorised)."""
    genuine, impostor = np.sort(np.asarray(genuine, float)), np.sort(np.asarray(impostor, float))
    t = np.linspace(0, 1, points)
    far = 1 - np.searchsorted(impostor, t, side="left") / len(impostor)
    frr = np.searchsorted(genuine, t, side="left") / len(genuine)
    return t, far, frr


def eer_point(genuine, impostor, points=1001):
    """Equal error rate and the threshold where FAR and FRR cross."""
    t, far, frr = far_frr_curve(genuine, impostor, points)
    i = int(np.argmin(np.abs(far - frr)))
    return float(t[i]), float((far[i] + frr[i]) / 2)


def roc_auc(genuine, impostor):
    """AUC as P(genuine score > impostor score) (Mann-Whitney U), with ties counted as one half."""
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    ranks = stats.rankdata(np.r_[g, i])
    u = ranks[: len(g)].sum() - len(g) * (len(g) + 1) / 2
    return float(u / (len(g) * len(i)))


def decidability(genuine, impostor):
    """Daugman's decidability index d' (larger = better separation)."""
    mg, mi = np.mean(genuine), np.mean(impostor)
    vg, vi = np.var(genuine, ddof=1), np.var(impostor, ddof=1)
    return float(abs(mg - mi) / np.sqrt((vg + vi) / 2)) if (vg + vi) > 0 else 0.0


def frr_at_far(genuine, impostor, target_far):
    """Lowest FRR achievable when FAR must not exceed target_far (empirical)."""
    t, far, frr = far_frr_curve(genuine, impostor, 2001)
    ok = far <= target_far
    if not ok.any():
        return 1.0, 1.0
    i = int(np.argmax(ok))  # smallest threshold meeting the FAR budget
    return float(frr[i]), float(t[i])


def bootstrap_auc_ci(genuine, impostor, n_boot=400, seed=7):
    rng = np.random.default_rng(seed)
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    vals = [roc_auc(rng.choice(g, len(g)), rng.choice(i, len(i))) for _ in range(n_boot)]
    return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))


# ----------------------------------------------------------------------------
# Multimodal fusion (fingerprint + face)
# ----------------------------------------------------------------------------
FACE_GEN = (0.76, 0.11)
FACE_IMP = (0.40, 0.15)


def simulate_face_scores(df, correlation=0.25, quality=1.0, seed=2026):
    """Create a simulated second modality aligned row-by-row with the fingerprint data.

    Within each class the face score is correlated with the fingerprint score by `correlation`
    (a correlation of 0 means the two sensors fail independently, which is where fusion helps most).
    `quality` in (0, 1.5] scales the separation between face genuine/impostor means.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(len(df))
    scores = df["Similarity_Score"].to_numpy(float)
    classes = df["Class"].to_numpy()
    for label, (mu, sd) in (("Genuine", FACE_GEN), ("Impostor", FACE_IMP)):
        mask = classes == label
        if not mask.any():
            continue
        fp = scores[mask]
        z = (fp - fp.mean()) / (fp.std(ddof=1) or 1.0)
        eps = rng.standard_normal(mask.sum())
        mean = FACE_GEN[0] - (FACE_GEN[0] - FACE_IMP[0]) * (1 - quality) / 2 if label == "Genuine" else \
            FACE_IMP[0] + (FACE_GEN[0] - FACE_IMP[0]) * (1 - quality) / 2
        out[mask] = mean + sd * (correlation * z + np.sqrt(max(1 - correlation ** 2, 0)) * eps)
    return np.clip(out, 0.0, 1.0)


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def fit_logistic(X, y, l2=1e-2, iters=60):
    """Ridge logistic regression via Newton-Raphson (IRLS). Returns weight vector incl. intercept."""
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    reg = l2 * np.eye(Xb.shape[1]); reg[0, 0] = 0
    for _ in range(iters):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) + reg @ w
        hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def fuse(df, face, method, weight_fp=0.5, cv_users=None):
    """Return a fused score in [0, 1] for each row.

    Methods: Fingerprint only · Face only · Sum rule · Weighted sum · Product rule · Min rule · Max rule ·
    Logistic regression (learned weights; fit on `cv_users` when given, to avoid scoring on training users).
    """
    fp = df["Similarity_Score"].to_numpy(float)
    face = np.asarray(face, float)
    if method == "Fingerprint only":
        return fp, None
    if method == "Face only":
        return face, None
    if method == "Sum rule":
        return (fp + face) / 2, None
    if method == "Weighted sum":
        return weight_fp * fp + (1 - weight_fp) * face, None
    if method == "Product rule":
        return np.sqrt(fp * face), None
    if method == "Min rule":
        return np.minimum(fp, face), None
    if method == "Max rule":
        return np.maximum(fp, face), None
    if method == "Logistic regression":
        y = (df["Class"].to_numpy() == "Genuine").astype(float)
        X = np.c_[fp, face]
        train = np.ones(len(df), bool) if cv_users is None else df["User_ID"].isin(cv_users).to_numpy()
        w = fit_logistic(X[train], y[train])
        return _sigmoid(np.c_[np.ones(len(X)), X] @ w), w
    raise ValueError(method)


FUSION_METHODS = ["Fingerprint only", "Face only", "Sum rule", "Weighted sum", "Product rule",
                  "Min rule", "Max rule", "Logistic regression"]


def fusion_table(df, face, weight_fp=0.5, seed=11):
    """Compare every fusion rule: AUC, EER, d', FRR@FAR=1% and FRR@FAR=0.1%. Logistic fit uses a user-level split."""
    users = df["User_ID"].unique()
    rng = np.random.default_rng(seed)
    train_users = set(rng.choice(users, size=max(2, len(users) // 2), replace=False))
    test_mask = ~df["User_ID"].isin(train_users)
    rows, scores_by_method = [], {}
    is_gen = (df["Class"].to_numpy() == "Genuine")
    for method in FUSION_METHODS:
        s, w = fuse(df, face, method, weight_fp, cv_users=train_users)
        scores_by_method[method] = s
        evaluate = test_mask.to_numpy() if method == "Logistic regression" else np.ones(len(df), bool)
        g, i = s[is_gen & evaluate], s[~is_gen & evaluate]
        eer_t, eer = eer_point(g, i)
        # d' is only meaningful on a roughly linear score, so use the logit of the learned probability
        if method == "Logistic regression":
            dg, di = np.log(np.clip(g, 1e-9, 1 - 1e-9) / (1 - np.clip(g, 1e-9, 1 - 1e-9))), \
                     np.log(np.clip(i, 1e-9, 1 - 1e-9) / (1 - np.clip(i, 1e-9, 1 - 1e-9)))
        else:
            dg, di = g, i
        rows.append({
            "Method": method, "AUC": roc_auc(g, i), "EER": eer, "d'": decidability(dg, di),
            "FRR @ FAR 1%": frr_at_far(g, i, 0.01)[0], "FRR @ FAR 0.1%": frr_at_far(g, i, 0.001)[0],
            "Evaluated on": "held-out users" if method == "Logistic regression" else "all users",
        })
    return pd.DataFrame(rows), scores_by_method, train_users


# ----------------------------------------------------------------------------
# Per-user risk profile ("biometric zoo")
# ----------------------------------------------------------------------------
def user_profiles(df, threshold):
    """Per-user performance with goat / lamb / sheep tagging relative to the population."""
    rows = []
    for user, part in df.groupby("User_ID"):
        g = part.loc[part.Class == "Genuine", "Similarity_Score"].to_numpy(float)
        i = part.loc[part.Class == "Impostor", "Similarity_Score"].to_numpy(float)
        rows.append({
            "User": user,
            "Genuine mean": g.mean() if len(g) else np.nan,
            "Genuine SD": g.std(ddof=1) if len(g) > 1 else np.nan,
            "Impostor mean": i.mean() if len(i) else np.nan,
            "Impostor max": i.max() if len(i) else np.nan,
            "FRR": float((g < threshold).mean()) if len(g) else np.nan,
            "FAR": float((i >= threshold).mean()) if len(i) else np.nan,
            "Margin": (g.mean() - i.mean()) if len(g) and len(i) else np.nan,
            "Attempts": len(part),
        })
    out = pd.DataFrame(rows)
    # Population cut-offs: goats = weakest genuine matching, lambs = most exposed to impostors
    gm_cut = out["Genuine mean"].quantile(0.20)
    im_cut = out["Impostor mean"].quantile(0.80)
    def tag(r):
        goat = r["Genuine mean"] <= gm_cut or r["FRR"] >= 0.20
        lamb = r["Impostor mean"] >= im_cut or r["FAR"] >= 0.10
        if goat and lamb:
            return "Goat + Lamb"
        return "Goat" if goat else "Lamb" if lamb else "Sheep"
    out["Profile"] = out.apply(tag, axis=1)
    out["Risk score"] = (0.5 * out["FRR"].rank(pct=True) + 0.5 * out["FAR"].rank(pct=True) +
                         0.25 * (1 - out["Margin"].rank(pct=True))).round(3)
    return out.sort_values("Risk score", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Attack and lockout models
# ----------------------------------------------------------------------------
def fit_gaussians(genuine, impostor):
    return (float(np.mean(genuine)), float(np.std(genuine, ddof=1))), (float(np.mean(impostor)), float(np.std(impostor, ddof=1)))


def spoof_far(threshold, gen_params, imp_params, quality):
    """FAR when attack artefacts have quality q: scores are drawn from a model interpolated between
    the zero-effort impostor model (q = 0) and the genuine model (q = 1)."""
    (mg, sg), (mi, si) = gen_params, imp_params
    mu = mi + quality * (mg - mi)
    sd = si + quality * (sg - si)
    return float(stats.norm.sf(threshold, mu, sd))


def lockout_analysis(far, frr, max_tries, sessions=1):
    """Closed-form lockout trade-off for independent attempts.

    - attacker breaks in within one lockout window: 1 - (1 - FAR)^m
    - genuine user is locked out (fails all m tries): FRR^m
    - attacker success over several windows (re-tries after the lockout expires): 1 - (1 - p_window)^sessions
    """
    m = np.asarray(max_tries, float)
    p_attack = 1 - (1 - far) ** m
    p_lock = frr ** m
    p_multi = 1 - (1 - p_attack) ** sessions
    return p_attack, p_lock, p_multi


def lockout_monte_carlo(far, frr, max_tries, trials=20000, seed=5):
    """Simulate attacker and genuine sessions to verify the closed-form lockout probabilities."""
    rng = np.random.default_rng(seed)
    attack = (rng.random((trials, max_tries)) < far).any(axis=1).mean()
    genuine_lock = (rng.random((trials, max_tries)) >= (1 - frr)).all(axis=1).mean()
    return float(attack), float(genuine_lock)


def attempts_to_breach(far, confidence=0.95):
    """Attempts needed for a given probability of at least one false accept; mean waiting time (geometric)."""
    if far <= 0:
        return np.inf, np.inf
    return float(np.log(1 - confidence) / np.log(1 - far)), float(1 / far)
