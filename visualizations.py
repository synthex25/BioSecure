"""Matplotlib charts used throughout BioSecure."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from utils.theme import ACCENT, GENUINE, IMPOSTOR, MUTED, TEXT, WARN


def similarity_distribution(genuine, impostor, threshold, density=False):
    fig, ax = plt.subplots(figsize=(10, 4.8)); bins = np.linspace(0, 1, 26)
    ax.hist(genuine, bins=bins, alpha=.6, label="Genuine", density=density, color=GENUINE)
    ax.hist(impostor, bins=bins, alpha=.6, label="Impostor", density=density, color=IMPOSTOR)
    ax.axvline(threshold, color=TEXT, ls="--", label=f"Threshold = {threshold:.2f}"); ax.set(xlabel="Similarity score", ylabel="Density" if density else "Frequency", xlim=(0,1)); ax.legend(); ax.grid(axis="y", alpha=.2); return fig


def normal_pdf(genuine, impostor, threshold):
    x = np.linspace(0, 1, 300); fig, ax = plt.subplots(figsize=(10, 4.8))
    for scores, label, color in [(genuine,"Genuine",GENUINE),(impostor,"Impostor",IMPOSTOR)]: ax.plot(x, stats.norm.pdf(x, np.mean(scores), np.std(scores, ddof=1)), label=f"{label} Gaussian fit", color=color)
    ax.axvline(threshold, color=TEXT, ls="--", label=f"Threshold = {threshold:.2f}"); ax.set(xlabel="Similarity score", ylabel="Probability density", xlim=(0,1)); ax.legend(); ax.grid(alpha=.2); return fig


def empirical_cdf(genuine, impostor, threshold=None):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for scores, label, color in [(genuine,"Genuine",GENUINE),(impostor,"Impostor",IMPOSTOR)]:
        ordered = np.sort(scores); ax.step(ordered, np.arange(1,len(ordered)+1)/len(ordered), where="post", label=label, color=color)
    if threshold is not None: ax.axvline(threshold,color=TEXT,ls="--",label=f"Threshold = {threshold:.2f}")
    ax.set(xlabel="Similarity score", ylabel="Cumulative probability", xlim=(0,1), ylim=(0,1)); ax.legend(); ax.grid(alpha=.2); return fig


def mean_sd_comparison(genuine, impostor):
    fig, ax = plt.subplots(figsize=(8, 4.5)); means=[np.mean(genuine),np.mean(impostor)]; stds=[np.std(genuine,ddof=1),np.std(impostor,ddof=1)
    ]; bars=ax.bar(["Genuine","Impostor"],means,yerr=stds,capsize=8,color=[GENUINE,IMPOSTOR],alpha=.8)
    ax.set(ylabel="Similarity score",ylim=(0,1),title="Mean Similarity Score ± Standard Deviation"); ax.grid(axis="y",alpha=.2)
    for bar,value in zip(bars,means): ax.text(bar.get_x()+bar.get_width()/2,value+.04,f"{value:.3f}",ha="center")
    return fig


def boxplot_scores(genuine, impostor):
    fig, ax=plt.subplots(figsize=(8,4.5)); plot=ax.boxplot([genuine,impostor],tick_labels=["Genuine","Impostor"],patch_artist=True)
    for patch,color in zip(plot["boxes"],[GENUINE,IMPOSTOR]): patch.set_facecolor(color); patch.set_alpha(.65)
    ax.set(ylabel="Similarity score",ylim=(0,1),title="Fingerprint Similarity Score Spread"); ax.grid(axis="y",alpha=.2); return fig


def threshold_curve(rows, eer):
    fig,ax=plt.subplots(figsize=(10,4.8)); x=[r["threshold"] for r in rows]; ax.plot(x,[r["FAR"] for r in rows],label="FAR",color=IMPOSTOR,lw=2); ax.plot(x,[r["FRR"] for r in rows],label="FRR",color=WARN,lw=2)
    ax.scatter(eer["threshold"],eer["eer"],color=TEXT,zorder=3,label=f"Approx. EER = {eer['eer']:.2%}"); ax.axvline(eer["threshold"],color=MUTED,ls="--",alpha=.7); ax.set(xlabel="Authentication threshold",ylabel="Error rate",xlim=(0,1),ylim=(0,1),title="Threshold vs False Acceptance and False Rejection"); ax.legend(); ax.grid(alpha=.25); return fig


def roc_curve(fpr, tpr, auc):
    fig,ax=plt.subplots(figsize=(6.5,5.5)); ax.plot(fpr,tpr,lw=2,color=ACCENT,label=f"ROC (AUC = {auc:.3f})"); ax.plot([0,1],[0,1],"--",color=MUTED,label="Chance reference"); ax.set(xlabel="False Positive Rate",ylabel="True Positive Rate",xlim=(0,1),ylim=(0,1),title="Fingerprint Verification ROC Curve"); ax.legend(loc="lower right"); ax.grid(alpha=.25); return fig


def confidence_interval_plot(genuine_mean, genuine_ci, impostor_mean, impostor_ci):
    fig,ax=plt.subplots(figsize=(8,4.5)); means=np.array([genuine_mean,impostor_mean]); errors=np.array([(genuine_mean-genuine_ci[0],impostor_mean-impostor_ci[0]),(genuine_ci[1]-genuine_mean,impostor_ci[1]-impostor_mean)])
    ax.errorbar([0,1],means,yerr=errors,fmt="o",capsize=8,markersize=9,color=ACCENT); ax.set(xticks=[0,1],xticklabels=["Genuine","Impostor"],ylabel="Mean similarity score",ylim=(0,1),title="Mean ± 95% Confidence Interval"); ax.grid(axis="y",alpha=.25); return fig


def probability_comparison(prior, posterior):
    fig, ax = plt.subplots(figsize=(7,4)); labels=["Prior P(Genuine)","Posterior P(Genuine | E)"]; values=[prior,posterior]; bars=ax.bar(labels, values, color=[ACCENT,GENUINE]); ax.set_ylim(0,1); ax.set_ylabel("Probability"); ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,value+.03,f"{value:.3f}",ha="center")
    return fig


def class_probability_chart(genuine, impostor, title, ylabel="Probability"):
    fig, ax=plt.subplots(figsize=(7,4)); values=[genuine,impostor]; bars=ax.bar(["Genuine","Impostor"],values,color=[GENUINE,IMPOSTOR])
    ax.set(ylim=(0,1),ylabel=ylabel,title=title); ax.grid(axis="y",alpha=.2)
    for bar,value in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,value+.03,f"{value:.3f}",ha="center")
    return fig


def confusion_matrix(metrics):
    fig, ax = plt.subplots(figsize=(5,4)); matrix=np.array([[metrics["TP"],metrics["FN"]],[metrics["FP"],metrics["TN"]]]); image=ax.imshow(matrix,cmap="Purples")
    ax.set(xticks=[0,1],xticklabels=["Accepted","Rejected"],yticks=[0,1],yticklabels=["Genuine","Impostor"],xlabel="System decision",ylabel="Actual class")
    for i in range(2):
        for j in range(2): ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=14)
    fig.colorbar(image, ax=ax); return fig


# =============================================================================
# Discrete and continuous rare-event modeling charts:
# Bernoulli, Binomial, Poisson, Exponential.
# =============================================================================

def bernoulli_chart(p_genuine_accept, p_impostor_accept):
    """Bar chart of the Bernoulli success probability for each class."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["Genuine\nP(Accept)", "Impostor\nP(Accept)"]
    values = [p_genuine_accept, p_impostor_accept]
    bars = ax.bar(labels, values, color=[GENUINE, IMPOSTOR])
    ax.set(ylim=(0, 1), ylabel="P(X = 1)", title="Bernoulli Trial: Probability of Acceptance per Attempt")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, min(value + 0.03, 0.97), f"{value:.4f}", ha="center")
    return fig


def binomial_chart(binomial_result, observed_k=None, title="Binomial Distribution"):
    """Bar chart of a Binomial PMF, with the observed count and theoretical mean marked."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(binomial_result["k"], binomial_result["pmf"], color=GENUINE, alpha=0.75, width=0.9, label="Binomial PMF")
    ax.axvline(binomial_result["mean"], color=IMPOSTOR, linestyle=":", linewidth=2, label=f"Expected mean = {binomial_result['mean']:.2f}")
    if observed_k is not None:
        ax.axvline(observed_k, color=TEXT, linestyle="--", linewidth=2, label=f"Observed = {observed_k}")
    ax.set(xlabel="Number of successes (k)", ylabel="P(X = k)", title=title)
    ax.legend(); ax.grid(alpha=0.25)
    return fig


def poisson_chart(poisson_result, observed_k=None, title="Poisson Distribution"):
    """Bar chart of a Poisson PMF, with the observed count and lambda marked."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(poisson_result["k"], poisson_result["pmf"], color=ACCENT, alpha=0.75, width=0.9, label="Poisson PMF")
    ax.axvline(poisson_result["mean"], color=WARN, linestyle=":", linewidth=2, label=f"λ = {poisson_result['mean']:.2f}")
    if observed_k is not None:
        ax.axvline(observed_k, color=TEXT, linestyle="--", linewidth=2, label=f"Observed = {observed_k}")
    ax.set(xlabel="Number of rare events (k)", ylabel="P(X = k)", title=title)
    ax.legend(); ax.grid(alpha=0.25)
    return fig


def binomial_vs_poisson_chart(comparison):
    """Side-by-side bars comparing the exact Binomial PMF with its Poisson approximation."""
    binomial, poisson = comparison["binomial"], comparison["poisson"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(binomial["k"] - 0.15, binomial["pmf"], width=0.3, color=GENUINE, label="Binomial (exact)")
    ax.bar(poisson["k"][: len(binomial["k"])] + 0.15, poisson["pmf"][: len(binomial["k"])], width=0.3, color=ACCENT, label="Poisson (approximation)")
    ax.set(
        xlabel="Number of false-acceptance events (k)",
        ylabel="P(X = k)",
        title=f"Binomial vs Poisson Approximation (λ = {comparison['lambda']:.3f})",
    )
    ax.legend(); ax.grid(alpha=0.25)
    return fig


def exponential_chart(exponential_result):
    """Histogram of observed inter-breach gaps with the fitted Exponential PDF overlaid."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    gaps = exponential_result["gaps"]
    bin_count = max(5, min(20, len(gaps) // 2 if len(gaps) >= 10 else len(gaps)))
    ax.hist(gaps, bins=bin_count, density=True, alpha=0.5, color=IMPOSTOR, label="Observed inter-breach gaps")
    ax.plot(exponential_result["x"], exponential_result["pdf"], color=ACCENT, linewidth=2, label=f"Exponential fit (rate = {exponential_result['rate']:.4f} / attempt)")
    ax.set(
        xlabel="Number of attempts between successive false-acceptance events",
        ylabel="Density",
        title="Exponential Model of Inter-Breach Waiting Time",
    )
    ax.legend(); ax.grid(alpha=0.25)
    return fig


def exponential_survival_chart(rate, max_attempts=200):
    """Plot the probability of surviving (no breach) as attempts increase, and its exponential decay."""
    x = np.arange(0, max_attempts + 1)
    survival = np.exp(-rate * x)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, survival, color=ACCENT, linewidth=2, label="P(no breach within n attempts)")
    ax.fill_between(x, survival, alpha=0.15, color=ACCENT)
    ax.set(xlabel="Number of attempts (n)", ylabel="Survival probability", ylim=(0, 1), title="Exponential Decay of Breach-Free Probability")
    ax.legend(); ax.grid(alpha=0.25)
    return fig
