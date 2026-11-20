"""Tiny registry so pages can link to each other when using st.navigation."""
import streamlit as st

REGISTRY = {}


def go(key):
    """Switch to a registered page (no-op if navigation has not been set up, e.g. in tests)."""
    page = REGISTRY.get(key)
    if page is not None:
        st.switch_page(page)
