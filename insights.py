"""Small, plain-language summaries for analysis pages."""

from html import escape

import streamlit as st

from utils.theme import ACCENT, GENUINE, IMPOSTOR, WARN


_TONE_COLORS = {
    "info": ACCENT,
    "good": GENUINE,
    "caution": WARN,
    "risk": IMPOSTOR,
}


def quick_read(message, tone="info", title="Quick read"):
    """Render a compact summary that explains the main result without jargon."""
    color = _TONE_COLORS.get(tone, ACCENT)
    st.markdown(
        f'<div class="analysis-read" style="--read-color:{color}">'
        f'<b>{escape(title)}</b><p>{escape(str(message))}</p></div>',
        unsafe_allow_html=True,
    )
