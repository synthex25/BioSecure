"""Shared BioSecure design system: palette, global CSS, Matplotlib and Plotly styling."""
import matplotlib.pyplot as plt
import streamlit as st

# Palette (genuine/accepted = green, impostor/rejected = rose, alerts = amber)
INK = "#0B0D1E"
PANEL = "#151833"
PANEL_HI = "#1E2245"
LINE = "#2F3866"
TEXT = "#F4F5FF"
MUTED = "#A9B3D6"
GENUINE = "#35D39A"
IMPOSTOR = "#FF7185"
ACCENT = "#A78BFA"
CYAN = "#5EC8F2"
WARN = "#FFC857"

FONT_STACK = "'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif"

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root {{ --ink:{INK}; --panel:{PANEL}; --panel-hi:{PANEL_HI}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED};
  --genuine:{GENUINE}; --impostor:{IMPOSTOR}; --accent:{ACCENT}; --cyan:{CYAN}; --warn:{WARN}; }}
html, body, [class*="css"], .stApp {{ font-family: {FONT_STACK}; }}
.stApp {{ background:
  radial-gradient(900px 500px at 92% -8%, rgba(167,139,250,.22) 0%, transparent 60%),
  radial-gradient(800px 480px at -8% 8%, rgba(94,200,242,.12) 0%, transparent 55%),
  {INK}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.01em; }}
hr {{ border-color: {LINE} !important; opacity:.6; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #12153199 0%, {PANEL} 100%); border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{ border-radius: 10px; margin: 1px 0; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{ background: {PANEL_HI}; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
  background: linear-gradient(90deg, rgba(167,139,250,.28), rgba(94,200,242,.10)); border:1px solid {LINE}; }}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap:.2rem; }}
.nav-label {{ position:relative; z-index:3; font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; font-weight:800; margin:18px 0 10px 6px; }}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{ border-radius:10px; padding:5px 10px; margin:1px 0; border:1px solid transparent; }}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{ background:{PANEL_HI}; }}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {{
  background: linear-gradient(90deg, rgba(167,139,250,.30), rgba(94,200,242,.10)); border-color:{LINE}; }}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p {{ font-size:.92rem; font-weight:600; }}
.brand {{ display:flex; align-items:center; gap:12px; padding:6px 4px 10px; }}
.brand-mark {{ width:42px; height:42px; border-radius:13px; display:grid; place-items:center; font-size:22px;
  background: linear-gradient(135deg, {ACCENT}, {CYAN}); box-shadow: 0 6px 22px rgba(167,139,250,.35); }}
.brand-name {{ font-size:22px; font-weight:800; color:{TEXT}; line-height:1.05; }}
.brand-sub {{ font-size:12px; color:{MUTED}; letter-spacing:.04em; }}
.side-card {{ background:{PANEL_HI}; border:1px solid {LINE}; border-radius:14px; padding:12px 14px; margin:10px 0; }}
.side-card .k {{ font-size:11px; letter-spacing:.09em; color:{MUTED}; font-weight:700; text-transform:uppercase; }}
.side-card .v {{ font-size:15px; color:{TEXT}; font-weight:700; margin-top:2px; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; }}
.dot.on {{ background:{GENUINE}; box-shadow:0 0 10px {GENUINE}; }} .dot.off {{ background:{WARN}; }}

/* Legacy titles used by the analysis pages */
.sidebar-logo {{ font-size: 28px; font-weight: 800; color: {TEXT}; }}
.sidebar-subtitle {{ font-size: 14px; color: {MUTED}; line-height: 1.5; }}
.project-title {{ font-size: 40px; font-weight: 800; color: {TEXT}; margin: 0 0 2px; }}
.project-subtitle {{ font-size: 17px; color: {MUTED}; margin-bottom: 26px; }}
.project-status {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 18px; }}
.project-status span {{ background:rgba(21,24,51,.85); border:1px solid {LINE}; border-radius:999px; padding:6px 12px;
  color:{MUTED}; font-size:.78rem; font-weight:600; backdrop-filter: blur(6px); }}
.project-status span.hl {{ color:{TEXT}; border-color:{ACCENT}; background:rgba(167,139,250,.14); }}
.section-title {{ font-size: 28px; font-weight: 800; color: {TEXT}; margin: 8px 0 14px; padding-left:14px;
  border-left:5px solid {ACCENT}; border-image: linear-gradient(180deg,{ACCENT},{CYAN}) 1; letter-spacing:-.015em; }}

/* Cards */
.info-card, .bio-overview, .glass {{ background: linear-gradient(160deg, rgba(30,34,69,.78), rgba(21,24,51,.78));
  border: 1px solid {LINE}; border-radius: 16px; padding: 18px 20px; color: {MUTED}; line-height: 1.7;
  margin: 10px 0 18px; backdrop-filter: blur(8px); box-shadow: 0 10px 30px rgba(0,0,0,.25); }}
.bio-overview h3, .glass h3 {{ margin: 0 0 8px; font-size: 1.05rem; color: {TEXT}; }}
.bio-overview p, .glass p {{ margin: 0; }}
.analysis-read {{ background:rgba(21,24,51,.85); border:1px solid {LINE}; border-left:4px solid var(--read-color, {ACCENT});
  border-radius:12px; padding:14px 16px; margin:10px 0 18px; color:{TEXT}; line-height:1.6; }}
.analysis-read b {{ color:var(--read-color, {ACCENT}); display:block; margin-bottom:4px; }}
.analysis-read p {{ color:{MUTED}; margin:0; }}

/* KPI cards */
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(138px,1fr)); gap:14px; margin:6px 0 18px; }}
.kpi {{ position:relative; overflow:hidden; background:linear-gradient(160deg, rgba(30,34,69,.9), rgba(21,24,51,.9));
  border:1px solid {LINE}; border-radius:16px; padding:16px 18px 14px; box-shadow:0 10px 26px rgba(0,0,0,.22); }}
.kpi::before {{ content:""; position:absolute; left:0; top:0; right:0; height:3px; background:var(--c, {ACCENT}); opacity:.9; }}
.kpi .l {{ font-size:11.5px; letter-spacing:.09em; text-transform:uppercase; color:{MUTED}; font-weight:700; }}
.kpi .v {{ font-size:30px; font-weight:800; color:{TEXT}; margin-top:4px; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }}
.kpi .s {{ font-size:12.5px; color:{MUTED}; margin-top:2px; }}
.kpi .v.long {{ font-size:20px; line-height:1.25; margin-top:8px; }}
.kpi.good {{ --c:{GENUINE}; }} .kpi.bad {{ --c:{IMPOSTOR}; }} .kpi.warn {{ --c:{WARN}; }} .kpi.cyan {{ --c:{CYAN}; }}

/* Native metrics */
div[data-testid="stMetric"] {{ background: linear-gradient(160deg, rgba(30,34,69,.9), rgba(21,24,51,.9)); border: 1px solid {LINE};
  border-radius: 14px; padding: 14px 16px; box-shadow:0 8px 22px rgba(0,0,0,.2); }}
div[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.82rem; font-weight:600; }}
div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; font-weight: 800; font-size: 1.7rem; }}

/* Controls */
.stButton > button, .stDownloadButton > button {{ border-radius: 12px; border: 1px solid {LINE}; font-weight: 700;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease; background:{PANEL_HI}; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ transform: translateY(-1px); border-color:{ACCENT}; box-shadow:0 8px 20px rgba(167,139,250,.22); }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background: linear-gradient(135deg, {GENUINE}, #2bb9a0); color: {INK}; border: none; }}
button[data-baseweb="tab"] {{ font-weight:700; }}
div[data-testid="stExpander"] {{ border:1px solid {LINE}; border-radius:14px; background:rgba(21,24,51,.6); }}
div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:12px; overflow:hidden; }}

/* Hero */
.bio-hero {{ position:relative; overflow:hidden; background: linear-gradient(120deg, #2e2466 0%, #4b3a9a 48%, #1f6b86 100%);
  padding: 2.4rem 2.6rem; border-radius: 22px; color: white; margin: 0 0 1.4rem; border: 1px solid #6857af;
  box-shadow: 0 18px 50px rgba(75,58,154,.35); }}
.bio-hero::after {{ content:""; position:absolute; right:-60px; top:-60px; width:300px; height:300px; border-radius:50%;
  background: radial-gradient(circle, rgba(255,255,255,.18), transparent 65%); }}
.bio-hero h1 {{ color: white !important; font-size: 2.2rem !important; margin: 0 0 .4rem !important; font-weight:800; }}
.bio-hero p {{ margin: 0; color: #d6ecff; font-size: 1.05rem; max-width: 760px; }}
.bio-kicker {{ font-size: .78rem; letter-spacing: .12em; font-weight: 700; color: #D8CBFF; margin-bottom: .6rem; }}
.pill-row {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }}
.pill {{ background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25); color:#fff; border-radius:999px;
  padding:5px 12px; font-size:.78rem; font-weight:600; }}

/* Page header */
.page-head {{ margin:0 0 18px; }}
.page-head .eyebrow {{ font-size:.74rem; letter-spacing:.14em; color:{ACCENT}; font-weight:800; text-transform:uppercase; }}
.page-head h1 {{ font-size:2rem; margin:.15rem 0 .35rem !important; padding:0 !important; font-weight:800; letter-spacing:-.02em; line-height:1.15; }}
.page-head p {{ color:{MUTED}; margin:0; max-width:820px; line-height:1.6; }}

.bio-step {{ border:1px solid {LINE}; border-top:3px solid {GENUINE}; background:rgba(21,24,51,.85); border-radius:14px; padding: 1rem; min-height: 120px; }}
.bio-step b {{ color: {TEXT}; display: block; margin: .3rem 0; }}
.bio-step span {{ color: {MUTED}; font-size: .88rem; line-height: 1.45; }}
.bio-step .n {{ color:{GENUINE}; font-weight:800; font-size:.72rem; letter-spacing:.1em; }}
.bio-cta {{ background: linear-gradient(120deg, rgba(167,139,250,.16), rgba(94,200,242,.08)); border: 1px solid #493a76;
  border-radius: 16px; padding: 1.2rem 1.4rem; margin: 1rem 0; }}
.bio-cta b {{ color: {TEXT}; font-size: 1.05rem; }}
.bio-cta p {{ color: {MUTED}; margin: .35rem 0 0; }}
.feature-tile {{ background:rgba(21,24,51,.85); border:1px solid {LINE}; border-radius:16px; padding:16px 18px; height:100%; }}
.feature-tile .ic {{ font-size:22px; }} .feature-tile b {{ color:{TEXT}; display:block; margin:6px 0 4px; }}
.feature-tile span {{ color:{MUTED}; font-size:.88rem; line-height:1.5; }}
.badge {{ display:inline-block; border-radius:999px; padding:2px 10px; font-size:.72rem; font-weight:800; letter-spacing:.04em; }}
.badge.good {{ background:rgba(53,211,154,.15); color:{GENUINE}; }} .badge.bad {{ background:rgba(255,113,133,.15); color:{IMPOSTOR}; }}
.badge.warn {{ background:rgba(255,200,87,.15); color:{WARN}; }} .badge.info {{ background:rgba(167,139,250,.18); color:{ACCENT}; }}
/* Page header with icon */
.page-head {{ display:flex; gap:16px; align-items:flex-start; }}
.page-icon {{ flex:0 0 54px; height:54px; border-radius:16px; display:grid; place-items:center; font-size:26px;
  background:linear-gradient(135deg, rgba(167,139,250,.35), rgba(94,200,242,.18)); border:1px solid {LINE};
  box-shadow:0 8px 24px rgba(167,139,250,.18); margin-top:4px; }}
/* Concept chips under headers */
.concepts {{ display:flex; flex-wrap:wrap; gap:6px; margin:-6px 0 18px; }}
.concepts span {{ font-size:.74rem; font-weight:700; color:{CYAN}; background:rgba(94,200,242,.08); border:1px solid rgba(94,200,242,.28);
  border-radius:999px; padding:3px 10px; letter-spacing:.01em; }}
/* Theory cards (bordered containers) */
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{LINE} !important; border-radius:16px !important;
  background:linear-gradient(160deg, rgba(30,34,69,.55), rgba(21,24,51,.55)); }}
.theory-title {{ font-weight:800; font-size:1rem; color:{TEXT}; margin-bottom:2px; display:flex; align-items:center; gap:8px; }}
.theory-title::before {{ content:""; width:8px; height:8px; border-radius:50%; background:{ACCENT}; box-shadow:0 0 10px {ACCENT}; }}
.theory-title.good::before {{ background:{GENUINE}; box-shadow:0 0 10px {GENUINE}; }}
.theory-title.caution::before {{ background:{WARN}; box-shadow:0 0 10px {WARN}; }}
.katex {{ font-size:1.06em !important; }}
.katex-display {{ margin:.55em 0 !important; overflow-x:auto; overflow-y:hidden; padding:2px 0; }}
/* Verdict box */
.verdict {{ border:1px solid {LINE}; border-left:5px solid var(--vc); border-radius:14px; padding:14px 18px; background:rgba(21,24,51,.85); margin:8px 0 10px; }}
.verdict span {{ display:block; color:{MUTED}; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700; }}
.verdict b {{ display:block; color:var(--vc); font-size:1.9rem; font-weight:800; letter-spacing:-.01em; }}
.verdict em {{ color:{TEXT}; font-style:normal; font-variant-numeric:tabular-nums; }}
/* Scope coverage cards */
.scope-card {{ position:relative; background:rgba(21,24,51,.85); border:1px solid {LINE}; border-radius:14px; padding:14px 16px 12px 44px; min-height:124px; }}
.scope-card .tick {{ position:absolute; left:14px; top:14px; width:20px; height:20px; border-radius:50%; display:grid; place-items:center;
  background:rgba(53,211,154,.18); color:{GENUINE}; font-size:.75rem; font-weight:800; }}
.scope-card b {{ color:{TEXT}; display:block; margin-bottom:3px; }}
.scope-card span {{ color:{MUTED}; font-size:.84rem; line-height:1.45; }}
/* Pill-style tabs (Streamlit >= 1.50 react-aria markup, plus legacy baseweb fallback) */
div[data-testid="stTabs"] [role="tablist"], div[data-baseweb="tab-list"] {{ gap:6px; background:rgba(21,24,51,.7); padding:5px;
  border-radius:14px; border:1px solid {LINE}; flex-wrap:wrap; box-shadow:none; }}
div[data-testid="stTabs"] [role="tab"], button[data-baseweb="tab"] {{ border-radius:10px !important; padding:7px 14px !important;
  font-weight:700; border-bottom:none !important; transition:background .12s ease; }}
div[data-testid="stTabs"] [role="tab"]:hover {{ background:{PANEL_HI}; }}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] {{
  background:linear-gradient(135deg, rgba(167,139,250,.38), rgba(94,200,242,.16)); color:{TEXT} !important; }}
div[data-testid="stTabs"] [role="tab"] p {{ font-weight:700; }}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{ display:none; }}
div[data-testid="stTabs"] {{ margin-top:6px; }}
/* Sidebar threshold card */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {{ background:{PANEL_HI}; }}
section[data-testid="stSidebar"] .stButton > button {{ padding:2px 4px; font-size:.78rem; min-height:32px; }}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ font-size:.78rem; }}
/* Headings inside pages */
h3 {{ font-size:1.25rem !important; margin-top:.6rem !important; }}
h5 {{ color:{TEXT}; margin-top:1rem !important; font-weight:800 !important; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>
"""


def apply_theme():
    """Inject global CSS and dark matplotlib defaults (call once per script run)."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    plt.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
        "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "grid.color": LINE, "legend.facecolor": PANEL_HI, "legend.edgecolor": LINE,
        "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
    })


def style_plotly(fig, height=340, title=None, legend_below=True):
    """Apply the BioSecure look to a Plotly figure."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=46 if title else 24, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
        font=dict(color=TEXT, family=FONT_STACK), hoverlabel=dict(bgcolor=INK),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0) if legend_below else dict(),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig
