"""Report Center: one-click, self-contained HTML report (print to PDF) plus JSON / CSV exports."""
import base64
import io
import json
from datetime import datetime
from html import escape

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from config.settings import PROJECT_NAME, PROJECT_TITLE
from utils.advanced import decidability, eer_point, roc_auc
from utils.calculations import (cluster_bootstrap_rates, effect_size_and_overlap, hypothesis_test, performance_metrics,
                                probability_analysis, split_scores)
from utils.ui import active_threshold, kpi_row, page_header, require_dataset
from utils.visualizations import confusion_matrix, similarity_distribution, threshold_curve
from utils.calculations import threshold_sweep
from utils.dist_extras import gof_table
from utils.inference import clopper_pearson_ci, wilson_ci
from utils.threshold_methods import gaussian_eer, threshold_candidates


def _svg(fig):
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def compute_summary(df, threshold):
    genuine, impostor = split_scores(df)
    m = performance_metrics(df, threshold)
    prob = probability_analysis(df, threshold)
    effect = effect_size_and_overlap(genuine, impostor)
    test = hypothesis_test(genuine, impostor)
    eer_t, eer = eer_point(genuine, impostor)
    try:
        boot = cluster_bootstrap_rates(df, threshold, n_boot=500)
        tar_ci, far_ci = boot["TAR_CI"], boot["FAR_CI"]
    except ValueError:
        tar_ci = far_ci = (float("nan"), float("nan"))
    return {
        "threshold": threshold, "attempts": len(df), "users": int(df.User_ID.nunique()),
        "TAR": m["TAR"], "FAR": m["FAR"], "FRR": m["FRR"], "TRR": m["TRR"],
        "accuracy": (m["TP"] + m["TN"]) / len(df), "confusion": {k: m[k] for k in ("TP", "FN", "FP", "TN")},
        "AUC": roc_auc(genuine, impostor), "EER": eer, "EER_threshold": eer_t, "d_prime": decidability(genuine, impostor),
        "cohens_d": float(effect["effect_size"]), "overlap": float(effect["overlap"]),
        "welch_t": float(test.statistic), "welch_p": float(test.pvalue),
        "posterior_genuine_given_accept": prob["posterior_genuine"], "prior_genuine": prob["prior_genuine"],
        "TAR_CI95": tar_ci, "FAR_CI95": far_ci,
        "FAR_wilson95": wilson_ci(m["FP"], m["FP"] + m["TN"]), "FAR_clopper_pearson95": clopper_pearson_ci(m["FP"], m["FP"] + m["TN"]),
        "FRR_wilson95": wilson_ci(m["FN"], m["FN"] + m["TP"]), "FRR_clopper_pearson95": clopper_pearson_ci(m["FN"], m["FN"] + m["TP"]),
        "gaussian_fit": {"genuine_mean": float(np.mean(genuine)), "genuine_sd": float(np.std(genuine, ddof=1)),
                         "impostor_mean": float(np.mean(impostor)), "impostor_sd": float(np.std(impostor, ddof=1))},
        "gaussian_EER": gaussian_eer({"mg": float(np.mean(genuine)), "sg": float(np.std(genuine, ddof=1)),
                                      "mi": float(np.mean(impostor)), "si": float(np.std(impostor, ddof=1))})[1],
    }


def build_html_report(df, threshold, notes="", fusion=None):
    s = compute_summary(df, threshold)
    genuine, impostor = split_scores(df)
    rows, eer = threshold_sweep(df)
    figs = [("Score distributions", _svg(similarity_distribution(genuine, impostor, threshold))),
            ("Threshold trade-off", _svg(threshold_curve(rows, eer))),
            ("Confusion matrix", _svg(confusion_matrix(performance_metrics(df, threshold))))]
    cards = "".join(f'<div class="c"><div class="l">{escape(l)}</div><div class="v">{escape(v)}</div></div>' for l, v in [
        ("TAR", f"{s['TAR']:.1%}"), ("FAR", f"{s['FAR']:.1%}"), ("FRR", f"{s['FRR']:.1%}"), ("Accuracy", f"{s['accuracy']:.1%}"),
        ("AUC", f"{s['AUC']:.4f}"), ("EER", f"{s['EER']:.2%}"), ("d′", f"{s['d_prime']:.2f}"), ("Threshold", f"{threshold:.2f}")])
    fig_html = "".join(f"<section><h2>{escape(t)}</h2><div class='fig'>{svg}</div></section>" for t, svg in figs)
    fusion_html = ""
    if fusion:
        head = "".join(f"<th>{escape(k)}</th>" for k in fusion[0])
        body = "".join("<tr>" + "".join(f"<td>{escape(f'{v:.4f}' if isinstance(v, float) else str(v))}</td>" for v in r.values()) + "</tr>" for r in fusion)
        fusion_html = f"<section><h2>Multimodal fusion comparison</h2><table><tr>{head}</tr>{body}</table></section>"
    table, _ = threshold_candidates(genuine, impostor, len(genuine) / len(df), 1.0, 1.0, 0.01, 0.05)
    trow = "".join(f"<tr><td>{escape(r.Method)}</td><td>{escape(r.Family)}</td><td>{r.Threshold:.4f}</td><td>{r.FAR:.2%}</td>"
                   f"<td>{r.FRR:.2%}</td><td>{r['P(G | accept)']:.2%}</td></tr>" for _, r in table.iterrows())
    thr_html = ("<section><h2>Threshold selection rules</h2><p class='sub'>Equal costs, dataset class mix, α = 1%, β = 5%. "
                f"The active threshold is {threshold:.4f}.</p><table><tr><th>Rule</th><th>Family</th><th>Threshold</th><th>FAR</th><th>FRR</th>"
                f"<th>P(G | accept)</th></tr>{trow}</table></section>")
    ci2 = lambda c: f"{c[0]:.2%} – {c[1]:.2%}"
    ci_html = ("<section><h2>Error-rate confidence intervals (95%)</h2><table><tr><th>Rate</th><th>Estimate</th><th>Wilson</th><th>Clopper–Pearson</th>"
               f"<th>User-level bootstrap</th></tr><tr><td>FAR</td><td>{s['FAR']:.2%}</td><td>{ci2(s['FAR_wilson95'])}</td><td>{ci2(s['FAR_clopper_pearson95'])}</td>"
               f"<td>{ci2(s['FAR_CI95'])}</td></tr><tr><td>FRR</td><td>{s['FRR']:.2%}</td><td>{ci2(s['FRR_wilson95'])}</td><td>{ci2(s['FRR_clopper_pearson95'])}</td>"
               f"<td>{ci2((1 - s['TAR_CI95'][1], 1 - s['TAR_CI95'][0]))}</td></tr></table></section>")
    gof = gof_table(genuine, impostor)
    grow = "".join(f"<tr><td>{r.Class}</td><td>{r['Shapiro p']:.4f}</td><td>{r['KS p']:.4f}</td><td>{r['Anderson A²']:.3f}</td><td>{r['Better (AIC)']}</td></tr>"
                   for _, r in gof.iterrows())
    gof_html = ("<section><h2>Gaussian goodness of fit</h2><table><tr><th>Class</th><th>Shapiro p</th><th>KS p</th><th>Anderson A²</th><th>Better by AIC</th></tr>"
                f"{grow}</table><p class='sub'>Gaussian-model EER {s['gaussian_EER']:.2%} vs empirical EER {s['EER']:.2%}.</p></section>")
    note_html = f"<section><h2>Analyst notes</h2><p>{escape(notes).replace(chr(10), '<br>')}</p></section>" if notes.strip() else ""
    ci = lambda c: f"{c[0]:.1%} – {c[1]:.1%}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{escape(PROJECT_NAME)} report</title>
<style>body{{font-family:'Plus Jakarta Sans',system-ui,sans-serif;background:#0B0D1E;color:#F4F5FF;margin:0;padding:40px;line-height:1.6;
-webkit-print-color-adjust:exact;print-color-adjust:exact}}main{{max-width:980px;margin:auto}}h1{{font-size:2rem;margin:0}}
h2{{border-left:5px solid #A78BFA;padding-left:12px;margin-top:34px}}.sub{{color:#A9B3D6}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}
.c{{background:#151833;border:1px solid #2F3866;border-radius:14px;padding:14px 16px}}.l{{color:#A9B3D6;font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;font-weight:700}}
.v{{font-size:1.6rem;font-weight:800}}.fig{{background:#151833;border:1px solid #2F3866;border-radius:14px;padding:12px;text-align:center}}.fig svg{{max-width:100%;height:auto}}
table{{width:100%;border-collapse:collapse;background:#151833}}td,th{{border:1px solid #2F3866;padding:7px 10px;text-align:left;font-size:.88rem}}th{{background:#1E2245}}
.note{{background:#151833;border-left:4px solid #FFC857;padding:12px 16px;border-radius:10px;color:#A9B3D6}}@media print{{body{{padding:12px}}section{{break-inside:avoid}}}}</style></head><body><main>
<div class="sub">{escape(PROJECT_NAME)} · generated {datetime.now().strftime('%d %b %Y %H:%M')}</div><h1>{escape(PROJECT_TITLE)}</h1>
<p class="sub">{s['attempts']:,} simulated attempts from {s['users']} users, evaluated at threshold {threshold:.2f}.</p>
<div class="grid">{cards}</div>
<section><h2>Summary</h2><p>At threshold {threshold:.2f} the system accepts {s['TAR']:.1%} of genuine attempts (95% user-level interval {ci(s['TAR_CI95'])}) and lets in {s['FAR']:.1%} of impostor attempts (interval {ci(s['FAR_CI95'])}).
The equal error rate is {s['EER']:.2%} near threshold {s['EER_threshold']:.2f}, with AUC {s['AUC']:.4f}. Genuine and impostor means differ by Cohen's d = {s['cohens_d']:.2f}
(Welch p = {s['welch_p']:.2g}). Given an accepted attempt, the Bayesian posterior probability that it was genuine is {s['posterior_genuine_given_accept']:.1%}
(prior {s['prior_genuine']:.1%}).</p></section>{fig_html}{thr_html}{ci_html}{gof_html}{fusion_html}{note_html}
<section><h2>Limitations</h2><p class="note">Scores are simulated and the model is educational. Results do not establish production biometric security; real evaluation needs benchmark data, calibration and spoofing tests.</p></section>
</main></body></html>"""


def render_report_center():
    page_header("Results · Export", "Report Center",
                "Generate a self-contained report of the current dataset and decision threshold. Open the HTML in a browser and use Print → Save as PDF for a PDF copy.")
    if not require_dataset():
        return
    df = st.session_state.dataset
    threshold = active_threshold()
    st.caption(f"The report uses the active threshold **{threshold:.4f}** ({st.session_state.get('active_threshold_profile') or 'default'}). "
               "Change it from the sidebar or the threshold pages.")
    s = compute_summary(df, threshold)
    kpi_row([("TAR", f"{s['TAR']:.1%}", f"95% CI {s['TAR_CI95'][0]:.1%}–{s['TAR_CI95'][1]:.1%}", "good"),
             ("FAR", f"{s['FAR']:.1%}", f"95% CI {s['FAR_CI95'][0]:.1%}–{s['FAR_CI95'][1]:.1%}", "bad"),
             ("EER", f"{s['EER']:.2%}", f"near threshold {s['EER_threshold']:.2f}", "warn"),
             ("AUC", f"{s['AUC']:.4f}", f"d′ {s['d_prime']:.2f}", "cyan")])
    notes = st.text_area("Analyst notes (optional, added to the report)", placeholder="Findings, assumptions, recommended threshold…")
    fusion = st.session_state.get("fusion_summary")
    include_fusion = st.checkbox("Include the latest Fusion Lab comparison", value=bool(fusion), disabled=not fusion,
                                 help="Open the Fusion Lab page first to create a comparison.")
    html = build_html_report(df, threshold, notes, fusion if include_fusion else None)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download report (HTML)", html.encode("utf-8"), "biosecure_report.html", "text/html", type="primary", width="stretch")
    c2.download_button("Download metrics (JSON)", json.dumps(s, indent=2, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v)).encode(), "biosecure_metrics.json", "application/json", width="stretch")
    c3.download_button("Download dataset (CSV)", df.to_csv(index=False).encode(), "biosecure_dataset.csv", "text/csv", width="stretch")
    st.caption("The report is a single offline HTML file: open it in any browser, or use Print → Save as PDF.")
