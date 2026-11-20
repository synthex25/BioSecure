"""Dashboard (landing page) and biometric data pages."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config.settings import DEFAULT_THRESHOLD
from data.dataset_generator import generate_biometric_dataset
from utils.advanced import decidability, eer_point, far_frr_curve, roc_auc
from utils.calculations import performance_metrics, split_scores
from utils.insights import quick_read
from utils.nav import REGISTRY, go as go_to
from utils.theme import GENUINE, IMPOSTOR, TEXT, ACCENT, CYAN, WARN, MUTED, style_plotly
from utils.ui import active_threshold, feature_tile, kpi_row, page_header, concept_chips


def _create_dataset():
    st.session_state.dataset = generate_biometric_dataset()
    st.session_state.threshold = DEFAULT_THRESHOLD
    st.session_state.locked_threshold = None
    st.session_state.threshold_policy = None
    st.session_state.active_threshold_profile = None
    if "live_threshold" in st.session_state:
        st.session_state.live_threshold = DEFAULT_THRESHOLD


def render_dashboard():
    """Landing page: hero, live KPIs, charts, and quick actions."""
    st.markdown("""
    <div class="bio-hero"><div class="bio-kicker">MATHEMATICS MICRO PROJECT · TEAM 4</div>
    <h1>BioSecure — Probabilistic Biometric Authentication</h1>
    <p>Model fingerprint verification as a statistical decision problem: distributions, Bayes, thresholds, rare events, fusion and attacks — in one interactive workbench.</p>
    <div class="pill-row"><span class="pill">Bayes &amp; conditional probability</span><span class="pill">Gaussian · Poisson · Exponential</span>
    <span class="pill">Bayes-optimal thresholds</span><span class="pill">ROC · DET · EER</span><span class="pill">CIs &amp; hypothesis tests</span><span class="pill">Live simulation</span><span class="pill">Multimodal fusion</span></div></div>
    """, unsafe_allow_html=True)

    df = st.session_state.dataset
    if df is None:
        st.markdown('<div class="bio-cta"><b>Ready to begin?</b><p>Create the simulated dataset to unlock every analysis page, or jump straight into the live monitor — it needs no data.</p></div>', unsafe_allow_html=True)
        actions = st.columns(3)
        if actions[0].button("Create demo dataset", type="primary", width="stretch"):
            _create_dataset(); st.rerun()
        if actions[1].button("Choose a threshold", width="stretch"):
            go_to("threshold")
        if actions[2].button("Start live monitor", width="stretch"):
            go_to("live")
        st.caption("The demo dataset is simulated and stays in this browser session. No fingerprint sensor is required.")
        _workflow_and_tiles()
        return

    threshold = active_threshold()
    metrics = performance_metrics(df, threshold)
    genuine, impostor = split_scores(df)
    accuracy = (metrics["TP"] + metrics["TN"]) / len(df)
    _, eer = eer_point(genuine, impostor)
    kpi_row([
        ("Attempts", f"{len(df):,}", f"{df.User_ID.nunique()} users · threshold {threshold:.2f}", "cyan"),
        ("Accuracy", f"{accuracy:.1%}", "correct decisions", "good"),
        ("TAR", f"{metrics['TAR']:.1%}", "genuine accepted", "good"),
        ("FAR", f"{metrics['FAR']:.1%}", "impostors accepted", "bad"),
        ("FRR", f"{metrics['FRR']:.1%}", "genuine rejected", "warn"),
        ("EER", f"{eer:.2%}", f"AUC {roc_auc(genuine, impostor):.3f} · d′ {decidability(genuine, impostor):.2f}", ""),
    ])

    left, right = st.columns([1.5, 1])
    with left:
        fig = go.Figure()
        for label, scores, color in (("Genuine", genuine, GENUINE), ("Impostor", impostor, IMPOSTOR)):
            fig.add_trace(go.Histogram(x=scores, name=label, marker_color=color, opacity=.65, xbins=dict(start=0, end=1, size=.025)))
        fig.add_vrect(x0=threshold, x1=1, fillcolor=GENUINE, opacity=.06, line_width=0)
        fig.add_vline(x=threshold, line=dict(color=TEXT, dash="dash", width=1.6), annotation_text=f"threshold {threshold:.2f}", annotation_font_color=MUTED)
        fig.update_layout(barmode="overlay"); fig.update_xaxes(range=[0, 1], title="Similarity score"); fig.update_yaxes(title="Attempts")
        st.plotly_chart(style_plotly(fig, 360, "Genuine vs impostor scores"), width="stretch", key="dash_hist")
    with right:
        t, far, frr = far_frr_curve(genuine, impostor, 301)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=far, name="FAR", line=dict(color=IMPOSTOR, width=2.6)))
        fig.add_trace(go.Scatter(x=t, y=frr, name="FRR", line=dict(color=WARN, width=2.6)))
        fig.add_vline(x=threshold, line=dict(color=TEXT, dash="dash", width=1.4))
        fig.update_xaxes(range=[0, 1], title="Threshold"); fig.update_yaxes(tickformat=".0%", title="Error rate")
        st.plotly_chart(style_plotly(fig, 360, "Error trade-off"), width="stretch", key="dash_tradeoff")

    quick_read(f"At threshold {threshold:.2f}, {metrics['TP']} of {metrics['TP'] + metrics['FN']} genuine attempts are accepted and "
               f"{metrics['FP']} of {metrics['FP'] + metrics['TN']} impostor attempts get in. Move the threshold to trade convenience for security.",
               tone="good" if metrics["FAR"] < .05 else "caution")

    a, b, c, d = st.columns(4)
    if a.button("🧪 Threshold Lab", width="stretch"): go_to("tlab")
    if b.button("⚡ Live monitor", width="stretch"): go_to("live")
    if c.button("🧬 Fusion lab", width="stretch"): go_to("fusion")
    if d.button("📄 Build report", width="stretch"): go_to("report")
    _workflow_and_tiles()


SCOPE_COVERAGE = [
    ("Probability axioms", "Kolmogorov axioms, complement, inclusion–exclusion verified on the data", "found"),
    ("Conditional probability", "Contingency tables, TAR/FAR as P(A | class), independence χ² test", "found"),
    ("Bayes' theorem", "Prior → posterior, total probability, odds form, base-rate effect", "prob"),
    ("Random variables", "Discrete & continuous RVs, PDF/CDF, moments, total variance, CLT", "found"),
    ("Classifying patterns", "MAP Bayesian classifier, calibration, 13 threshold rules", "tlab"),
    ("Bernoulli & Binomial", "Single attempt as Bernoulli; error counts as Binomial", "rare"),
    ("Poisson rare events", "Poisson approximation, Poisson-process check, breach rate", "rare"),
    ("Gaussian noise", "Gaussian fits, Q-Q / KS tests, additive sensor-noise model", "noise"),
    ("Exponential decay", "Template ageing, half-life, fitted λ, waiting times", "noise"),
    ("Confidence intervals", "t-interval, Wilson, Clopper–Pearson, bootstrap", "valid"),
    ("Hypothesis testing", "Welch, Mann–Whitney, exact binomial, power analysis", "valid"),
    ("Secure AI biometrics", "Fusion, spoof attacks, lockout, live monitoring", "attack"),
]


def _workflow_and_tiles():
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    steps = [("1", "Generate data", "Create the simulated fingerprint dataset."), ("2", "Model", "Axioms, Bayes, distributions, noise and rare events."),
             ("3", "Decide", "Compare 13 threshold rules, add a review zone."), ("4", "Stress-test", "Live stream, attacks and lockout policies."),
             ("5", "Validate & report", "Intervals, hypothesis tests, exportable report.")]
    for column, (n, title, text) in zip(st.columns(5), steps):
        column.markdown(f'<div class="bio-step"><span class="n">STEP {n}</span><b>{title}</b><span>{text}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Project scope coverage</div>', unsafe_allow_html=True)
    st.caption("Every requirement in the project brief, and where it is demonstrated. Click a card's link to open that page.")
    for row in range(0, len(SCOPE_COVERAGE), 4):
        cols = st.columns(4)
        for col, (concept, how, key) in zip(cols, SCOPE_COVERAGE[row: row + 4]):
            with col:
                st.markdown(f'<div class="scope-card"><span class="tick">✓</span><b>{concept}</b><span>{how}</span></div>', unsafe_allow_html=True)
                page = REGISTRY.get(key)
                if page is not None:
                    st.page_link(page, label=f"{page.title} →")

    st.markdown('<div class="section-title">What is inside</div>', unsafe_allow_html=True)
    tiles = [("🧪", "Threshold Lab", "EER, Bayes min-risk, Neyman–Pearson and more — with formulas, risk curves and bootstrap stability."),
             ("📐", "Probability Foundations", "Axioms checked on data, joint & conditional tables, random variables, Central Limit Theorem."),
             ("📡", "Noise & Signal Decay", "Gaussian sensor noise and exponential template ageing — when to re-enrol."),
             ("⚡", "Live Command Center", "Poisson-arrival stream with Bayesian posterior and anomaly alerts."),
             ("🧬", "Fusion Lab", "Fingerprint + face score fusion compared by AUC, EER and DET."),
             ("📄", "Report Center", "Self-contained HTML report plus JSON and CSV exports.")]
    for row in (tiles[:3], tiles[3:]):
        for column, (icon, title, text) in zip(st.columns(3), row):
            column.markdown(feature_tile(icon, title, text), unsafe_allow_html=True)


def render_biometric_data():
    page_header("Overview · Step 2", "Biometric Data",
                "A simulated fingerprint-verification dataset: each row is one attempt, scored by how similar the query print is to the "
                "enrolled template (1.00 = identical). Every analysis page reads this one shared dataset.", icon="🗂️")
    concept_chips("Sample space", "Genuine vs impostor classes", "Random variable X = score", "Poisson arrival times", "Template age")
    with st.container(border=True):
        st.markdown("**Generate a dataset**")
        c1, c2, c3, c4 = st.columns(4)
        users = c1.number_input("Number of users", 5, 100, 20)
        genuine = c2.number_input("Genuine samples / user", 5, 100, 20)
        impostor = c3.number_input("Impostor samples / user", 5, 100, 20)
        seed = c4.number_input("Random seed", 0, 999999, 42)
        with st.expander("Realism options (all off by default, so the standard dataset is unchanged)"):
            r1, r2, r3, r4 = st.columns(4)
            user_var = r1.slider("User variability σ_u", 0.0, 0.10, 0.0, 0.005, help="Per-user offset on genuine scores — creates harder-to-match users ('goats').")
            noise = r2.slider("Sensor noise σₙ", 0.0, 0.10, 0.0, 0.005, help="Additive Gaussian measurement noise on every score.")
            decay = r3.slider("Template ageing λ (per day)", 0.0, 0.01, 0.0, 0.0005, format="%.4f",
                              help="Exponential decay of genuine scores toward the impostor mean as the template ages (0–180 days).")
            rate = r4.slider("Arrival rate (attempts/min)", 1.0, 30.0, 6.0, 0.5, help="Attempts arrive as a Poisson process with this rate.")
        if st.button("Generate biometric dataset", type="primary", width="stretch"):
            st.session_state.dataset = generate_biometric_dataset(int(users), int(genuine), int(impostor), int(seed), "Fingerprint Verification",
                                                                  user_variability=user_var, sensor_noise=noise, template_decay=decay, arrival_rate=rate)
            st.session_state.locked_threshold = None
            st.session_state.threshold_policy = None
            st.session_state.active_threshold_profile = None
            st.session_state["_flash"] = f"Generated {len(st.session_state.dataset):,} attempts — now the shared dataset"
            st.rerun()
    df = st.session_state.dataset
    if df is None:
        st.info("No dataset yet — press **Generate biometric dataset** above (the defaults reproduce the standard project data).")
        return
    threshold = active_threshold()
    g, i = split_scores(df)
    kpi_row([("Attempts", f"{len(df):,}", f"{df.User_ID.nunique()} users", "cyan"), ("Genuine", f"{len(g):,}", f"mean {g.mean():.3f} · SD {g.std(ddof=1):.3f}", "good"),
             ("Impostor", f"{len(i):,}", f"mean {i.mean():.3f} · SD {i.std(ddof=1):.3f}", "bad"),
             ("Mean gap", f"{g.mean() - i.mean():.3f}", f"d′ {decidability(g, i):.2f}", ""),
             ("Time span", f"{df.Timestamp_s.max() / 60:.0f} min" if "Timestamp_s" in df else "–", "Poisson arrivals", "warn")])
    quick_read(f"Genuine scores average {g.mean():.3f} and impostor scores {i.mean():.3f}. That gap — about {decidability(g, i):.1f} pooled standard deviations — "
               "is the signal every threshold, Bayes and hypothesis-testing page builds on.")
    tabs = st.tabs(["📊 Charts", "🧾 Data table", "📐 Summary statistics"])
    with tabs[0]:
        a, b = st.columns([1.4, 1])
        with a:
            fig = go.Figure()
            for label, scores, color in (("Genuine", g, GENUINE), ("Impostor", i, IMPOSTOR)):
                fig.add_trace(go.Histogram(x=scores, name=label, marker_color=color, opacity=.62, xbins=dict(start=0, end=1, size=.025)))
            fig.add_vline(x=threshold, line=dict(color=TEXT, dash="dash"), annotation_text=f"t = {threshold:.2f}", annotation_font_color=MUTED)
            fig.update_layout(barmode="overlay"); fig.update_xaxes(range=[0, 1], title="Similarity score"); fig.update_yaxes(title="Attempts")
            st.plotly_chart(style_plotly(fig, 340, "Score histogram"), width="stretch", key="data_hist")
        with b:
            fig = go.Figure()
            for label, scores, color in (("Genuine", g, GENUINE), ("Impostor", i, IMPOSTOR)):
                fig.add_trace(go.Violin(y=scores, name=label, line_color=color, fillcolor=color, opacity=.5, box_visible=True, meanline_visible=True, points=False))
            fig.add_hline(y=threshold, line=dict(color=TEXT, dash="dash"))
            fig.update_yaxes(range=[0, 1], title="Similarity score"); fig.update_layout(showlegend=False)
            st.plotly_chart(style_plotly(fig, 340, "Spread (violin + box)"), width="stretch", key="data_violin")
        fig = go.Figure()
        for label, color, symbol in (("Genuine", GENUINE, "circle"), ("Impostor", IMPOSTOR, "x")):
            part = df[df.Class == label]
            fig.add_trace(go.Scattergl(x=part.Sample_ID, y=part.Similarity_Score, mode="markers", name=label,
                                       marker=dict(color=color, size=5, symbol=symbol, opacity=.7), text=part.Attempt_ID,
                                       hovertemplate="%{text}<br>score %{y:.3f}<extra></extra>"))
        fig.add_hline(y=threshold, line=dict(color=TEXT, dash="dash"), annotation_text="threshold", annotation_font_color=MUTED)
        fig.update_xaxes(title="Attempt number"); fig.update_yaxes(title="Similarity score", range=[0, 1])
        st.plotly_chart(style_plotly(fig, 330, "Every attempt (hover for its ID)"), width="stretch", key="data_scatter")
        st.caption("Points above the dashed line are accepted. Red crosses above it are false accepts; green dots below it are false rejects.")
    with tabs[1]:
        st.dataframe(df, width="stretch", height=420, hide_index=True)
        st.download_button("Download dataset as CSV", df.to_csv(index=False).encode(), "biosecure_biometric_dataset.csv", "text/csv", width="stretch")
    with tabs[2]:
        st.dataframe(df.groupby("Class")["Similarity_Score"].describe().round(4), width="stretch")
        per_user = df.groupby(["User_ID", "Class"])["Similarity_Score"].mean().unstack().round(4)
        st.markdown("##### Mean score per user")
        st.dataframe(per_user, width="stretch", height=300)
