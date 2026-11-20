"""Shared UI building blocks: status bar, page header, KPI cards, badges, callouts."""

from html import escape

import streamlit as st

from config.settings import DEFAULT_THRESHOLD


def page_header(eyebrow, title, description="", icon=None):
    """Consistent page header used by every page."""
    desc = f"<p>{escape(description)}</p>" if description else ""
    mark = f'<div class="page-icon">{icon}</div>' if icon else ""
    st.markdown(
        f'<div class="page-head">{mark}<div><div class="eyebrow">{escape(eyebrow)}</div><h1>{escape(title)}</h1>{desc}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """Render KPI cards. items: iterable of (label, value, sub, tone) where tone in {'', good, bad, warn, cyan}."""
    cards = []
    for item in items:
        label, value, *rest = item
        sub = rest[0] if len(rest) > 0 else ""
        tone = rest[1] if len(rest) > 1 else ""
        sub_html = f'<div class="s">{escape(str(sub))}</div>' if sub else ""
        cards.append(f'<div class="kpi {tone}"><div class="l">{escape(str(label))}</div>'
                     f'<div class="v{" long" if len(str(value)) > 11 else ""}">{escape(str(value))}</div>{sub_html}</div>')
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def badge(text, tone="info"):
    return f'<span class="badge {tone}">{escape(text)}</span>'


def callout(title, body, tone="info"):
    """Accent-bordered note (tone: info | good | caution | risk)."""
    from utils.insights import quick_read
    quick_read(body, tone=tone, title=title)


def feature_tile(icon, title, text):
    return (f'<div class="feature-tile"><div class="ic">{icon}</div><b>{escape(title)}</b>'
            f'<span>{escape(text)}</span></div>')


def status_chips(page_title=None):
    """Return the (dataset, policy) status strings shown across the app."""
    dataset = st.session_state.get("dataset")
    data_status = "No dataset loaded" if dataset is None else f"Dataset ready · {len(dataset):,} attempts"
    policy = st.session_state.get("threshold_policy")
    if policy and policy.get("mode") == "three_zone":
        threshold_status = (f"Review policy · reject < {policy['reject_threshold']:.3f} · "
                            f"accept ≥ {policy['accept_threshold']:.3f}")
    else:
        profile = st.session_state.get("active_threshold_profile")
        threshold_status = f"Threshold {active_threshold():.3f}" + (f" · {profile}" if profile else "")
    return data_status, threshold_status


def render_project_status(page):
    """Show the dataset, threshold, and policy currently used by the app."""
    data_status, threshold_status = status_chips(page)
    chips = [(data_status, dataset_ok()), (threshold_status, True), (page, False)]
    markup = "".join(f'<span class="{"hl" if hl else ""}">{escape(text)}</span>' for text, hl in chips)
    st.markdown(f'<div class="project-status">{markup}</div>', unsafe_allow_html=True)


def dataset_ok():
    return st.session_state.get("dataset") is not None


def require_dataset(message="Create the shared demo dataset first — it powers this page."):
    """Show a friendly empty state with a one-click fix. Returns True when data is available."""
    if dataset_ok():
        return True
    from data.dataset_generator import generate_biometric_dataset
    st.markdown(f'<div class="bio-cta"><b>No dataset yet</b><p>{escape(message)}</p></div>', unsafe_allow_html=True)
    if st.button("Create demo dataset", type="primary", key="mk_ds_empty_state"):
        st.session_state.dataset = generate_biometric_dataset()
        st.session_state.threshold = DEFAULT_THRESHOLD
        st.session_state.locked_threshold = None
        st.session_state.threshold_policy = None
        st.rerun()
    return False


def active_threshold():
    """The single operating threshold every page uses (accept boundary when a review policy is active)."""
    policy = st.session_state.get("threshold_policy")
    if policy and policy.get("mode") == "three_zone":
        return float(policy["accept_threshold"])
    locked = st.session_state.get("locked_threshold")
    return float(locked if locked is not None else st.session_state.get("threshold", DEFAULT_THRESHOLD))


def set_threshold(value, label="Custom threshold"):
    """Apply one operating threshold across every page (clears any three-zone review policy)."""
    value = float(min(max(value, 0.0), 1.0))
    st.session_state.threshold = value
    st.session_state.locked_threshold = value
    st.session_state.threshold_policy = None
    st.session_state.active_threshold_profile = label
    if "live_threshold" in st.session_state:
        st.session_state.live_threshold = float(min(max(value, 0.05), 0.99))


def _threshold_slider_changed(key):
    set_threshold(st.session_state[key], "Manual slider")


def threshold_control(key="page_threshold", label="Operating threshold", help_text=None):
    """A slider bound to the global threshold: moving it on any page updates every page."""
    st.session_state[key] = round(active_threshold(), 2)
    st.slider(label, 0.0, 1.0, step=0.01, format="%.2f", key=key,
              on_change=_threshold_slider_changed, args=(key,),
              help=help_text or "Scores at or above the threshold are accepted. This control is shared by every page.")
    return active_threshold()


def concept_chips(*names):
    """Small row of syllabus-concept tags shown under a page header."""
    st.markdown('<div class="concepts">' + "".join(f"<span>{escape(n)}</span>" for n in names) + "</div>",
                unsafe_allow_html=True)


def theory(title, body_md=None, latex=None, tone="info"):
    """A theory card: short explanation plus optional formulas."""
    with st.container(border=True):
        st.markdown(f'<div class="theory-title {tone}">{escape(title)}</div>', unsafe_allow_html=True)
        if body_md:
            st.markdown(body_md)
        for formula in ([latex] if isinstance(latex, str) else (latex or [])):
            st.latex(formula)
