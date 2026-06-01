"""
🚦 Urban Traffic Volume Prediction App
========================================
Streamlit application for predicting traffic counts using
government traffic sensor data (2006–2025).

Built on an XGBoost model trained on 2,73,913 records
from 1,967 traffic stations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ─── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Volume Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom Styling ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
    /* ───────── GLOBAL ───────── */
    *, *::before, *::after { box-sizing: border-box; }

    .stApp {
        background: linear-gradient(145deg, #080618 0%, #0e0b2e 35%, #150f3a 65%, #1b1145 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide only branding — never hide the header or sidebar controls */
    #MainMenu, footer, .stDeployButton {
        display: none !important;
    }

    /* Keep header transparent so sidebar toggle remains clickable */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 3rem !important;
    }

    /* ── CRITICAL: Always show sidebar & its collapse/expand button ── */
    section[data-testid="stSidebar"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    /* The arrow button that re-opens a collapsed sidebar */
    button[data-testid="collapsedControl"],
    div[data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        pointer-events: auto !important;
    }

    /* Base body text */
    .stApp, .stApp p, .stApp span, .stApp div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ───────── SIDEBAR ───────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0a24 0%, #13103a 50%, #1a1550 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.12);
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #b0b0e0 !important;
        font-size: 0.88rem;
    }
    section[data-testid="stSidebar"] .stMarkdown strong {
        color: #d4d4ff !important;
    }

    /* Sidebar radio buttons — indigo accent */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        color: #c0c0e8 !important;
        transition: all 0.25s ease;
        border-radius: 8px;
        padding: 4px 8px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(99, 102, 241, 0.10);
        color: #e0e0ff !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] div[data-checked="true"] + label {
        color: #a5b4fc !important;
        font-weight: 600;
    }
    /* Radio dot accent */
    section[data-testid="stSidebar"] input[type="radio"]:checked + div {
        background-color: #6366f1 !important;
        border-color: #6366f1 !important;
    }

    /* ───────── HEADERS ───────── */
    h1 {
        color: #e8e8ff !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        font-size: 2.2rem !important;
    }
    h2 {
        color: #c5c5ff !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px;
    }
    h3 {
        color: #a5a5ff !important;
        font-weight: 600 !important;
    }
    p, li {
        color: #c0c0d8 !important;
        line-height: 1.65;
    }
    blockquote {
        border-left: 3px solid #6366f1 !important;
        background: rgba(99, 102, 241, 0.06);
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1.2rem;
    }

    /* ───────── METRIC CARDS ───────── */
    div[data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.04);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2),
                     inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2),
                     0 0 0 1px rgba(99, 102, 241, 0.15),
                     inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #8b8bcc !important;
        font-size: 0.78rem !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.9rem !important;
        letter-spacing: -0.5px;
    }

    /* ───────── BUTTONS ───────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 50%, #a78bfa 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
        filter: brightness(1.1);
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.99);
    }

    /* ───────── TABS ───────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(99, 102, 241, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 500;
        color: #8888bb !important;
        transition: all 0.25s ease;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.08);
        color: #c0c0ff !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        color: #a5b4fc !important;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.15);
    }
    /* Tab highlight bar */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #6366f1 !important;
        border-radius: 2px;
    }

    /* ───────── SELECT / INPUT ───────── */
    div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 10px !important;
        color: #d0d0f0 !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(99, 102, 241, 0.35) !important;
    }
    div[data-baseweb="select"] > div:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    /* Dropdown menu */
    ul[data-baseweb="menu"] {
        background: #1a1745 !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 10px !important;
    }
    li[data-baseweb="menu-item"] {
        color: #c0c0e8 !important;
    }
    li[data-baseweb="menu-item"]:hover {
        background: rgba(99, 102, 241, 0.15) !important;
    }

    /* ───────── SLIDER ───────── */
    div[data-baseweb="slider"] div[role="slider"] {
        background: #6366f1 !important;
        border-color: #818cf8 !important;
    }
    div[data-baseweb="slider"] div[data-testid="stTickBar"] > div {
        background: #6366f1 !important;
    }

    /* ───────── DATAFRAMES ───────── */
    .stDataFrame, div[data-testid="stDataFrame"] {
        border-radius: 14px !important;
        overflow: hidden;
        border: 1px solid rgba(99, 102, 241, 0.1) !important;
    }
    .stDataFrame iframe {
        border-radius: 14px !important;
    }

    /* ───────── CHART CONTAINERS ───────── */
    div[data-testid="stImage"],
    .stPlotlyChart,
    div.stPyplot {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(99, 102, 241, 0.08);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
    }

    /* ───────── ALERTS / SUCCESS ───────── */
    .stAlert {
        border-radius: 14px !important;
        border: 1px solid rgba(99, 102, 241, 0.12) !important;
        backdrop-filter: blur(8px);
    }

    /* ───────── DIVIDERS ───────── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg,
            transparent 0%,
            rgba(99, 102, 241, 0.25) 50%,
            transparent 100%) !important;
        margin: 1.5rem 0 !important;
    }

    /* ───────── EXPANDER ───────── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #a5b4fc !important;
        font-size: 0.95rem !important;
    }

    /* ───────── SCROLLBAR ───────── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.25);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.45);
    }

    /* ───────── ANIMATIONS (main content only) ───────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    /* Scope ONLY to the main block — NOT sidebar or outer wrappers */
    div[data-testid="stMainBlockContainer"] div[data-testid="stVerticalBlock"] > div {
        animation: fadeInUp 0.45s ease-out both;
    }
    div[data-testid="stMainBlockContainer"] div[data-testid="stVerticalBlock"] > div:nth-child(1) { animation-delay: 0s; }
    div[data-testid="stMainBlockContainer"] div[data-testid="stVerticalBlock"] > div:nth-child(2) { animation-delay: 0.05s; }
    div[data-testid="stMainBlockContainer"] div[data-testid="stVerticalBlock"] > div:nth-child(3) { animation-delay: 0.1s; }
    div[data-testid="stMainBlockContainer"] div[data-testid="stVerticalBlock"] > div:nth-child(4) { animation-delay: 0.15s; }
    div[data-testid="stMainBlockContainer"] div[data-testid="stVerticalBlock"] > div:nth-child(5) { animation-delay: 0.2s; }
    div[data-testid="stMainBlockContainer"] div[data-testid="stVerticalBlock"] > div:nth-child(6) { animation-delay: 0.25s; }
    /* Ensure sidebar children are never affected by fade */
    section[data-testid="stSidebar"] * {
        animation: none !important;
        opacity: 1 !important;
    }

    /* ───────── LABELS ───────── */
    label, .stSelectbox label, .stSlider label {
        color: #9898cc !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.3px;
    }

    /* ───────── SPINNER ───────── */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper: Determine base path ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Traffic_Data_Gov.csv")
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")


# ─── Data Loading (cached) ──────────────────────────────────────
@st.cache_data(show_spinner="📦 Loading traffic dataset …")
def load_raw_data():
    """Read the raw CSV once and cache it."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    return df


@st.cache_data(show_spinner="🧹 Cleaning data …")
def clean_data(df):
    """
    Replicate the notebook cleaning pipeline:
      1. Drop useless columns
      2. Remove bad rows (year 2026, SCHOOL HOLIDAYS)
      3. Fill missing values with median
      4. Keep only the columns we actually need
    """
    drop_cols = [
        "the_geom", "the_geom_webmercator", "cartodb_id",
        "record_id", "station_key", "data_start_date",
        "data_end_date", "data_duration", "data_quality_indicator",
        "publish", "md5", "updated_on",
        "traffic_direction_seq", "cardinal_direction_seq",
        "classification_type_seq", "period_seq",
    ]
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)

    # fill missing numeric columns
    for col in ("data_availability", "data_reliability"):
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # remove erroneous rows
    if "year" in df.columns:
        df = df[df["year"] != 2026]
    if "period" in df.columns:
        df = df[df["period"] != "SCHOOL HOLIDAYS"]

    df = df.dropna(subset=["traffic_count"])
    df = df.reset_index(drop=True)
    return df


@st.cache_resource(show_spinner="🤖 Loading model artefacts …")
def load_model_artefacts():
    """Load saved model, features list, encodings, station lookup."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    features = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))
    encodings = joblib.load(os.path.join(MODELS_DIR, "encodings.pkl"))
    station_df = joblib.load(os.path.join(MODELS_DIR, "station_lookup.pkl"))
    return model, features, encodings, station_df


# ─── Load everything once ───────────────────────────────────────
try:
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df.copy())
    model, FEATURES, encodings, station_df = load_model_artefacts()

    period_map = encodings["period_map"]
    class_map = encodings["class_map"]
    direction_map = encodings["direction_map"]
    traffic_dir_map = encodings["traffic_dir_map"]

    station_map = dict(zip(station_df["station_id"], station_df["station_id_enc"]))
    station_avg_map = dict(zip(station_df["station_id_enc"], station_df["traffic_count"]))
    global_mean = float(np.mean(list(station_avg_map.values())))

    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    st.error(f"⚠️ Failed to load data or model artefacts: {e}")


# ─── Sidebar ────────────────────────────────────────────────────
st.sidebar.markdown("## 🚦 Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "🏠 Overview",
        "📊 Data Explorer",
        "📈 EDA & Visualizations",
        "🤖 Model Performance",
        "🔮 Predict Traffic",
    ],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **About**  
    This app predicts daily traffic volumes at government sensor stations
    across a 20‑year period (2006–2025).

    *Model*: XGBoost (R² ≈ 0.923)  
    *Dataset*: 2,73,913 records · 1,967 stations
    """
)


# ═══════════════════════════════════════════════════════════════
#                          PAGES
# ═══════════════════════════════════════════════════════════════

# ─── 🏠 Overview ────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("# 🚦 Urban Traffic Volume Prediction")
    st.markdown(
        """
        Welcome! This application uses **machine‑learning** to predict traffic 
        volumes for government sensor stations. It was built from a
        comprehensive analysis notebook covering:

        - **Data Loading & Cleaning** — handling missing values, removing 
          erroneous records  
        - **Exploratory Data Analysis** — distribution of traffic counts, 
          direction & period analysis  
        - **Feature Engineering** — label encoding, binary flags, station 
          averages  
        - **Model Training & Comparison** — 7 models compared; **XGBoost wins**  
        - **Interactive Prediction** — enter station, year, period, vehicle 
          type & direction to get a forecast
        
        > Use the **sidebar** to navigate between sections.
        """
    )

    if DATA_LOADED:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Total Records", f"{len(clean_df):,}")
        col2.metric("🏢 Stations", f"{clean_df['station_id'].nunique():,}")
        col3.metric("📅 Year Range",
                     f"{int(clean_df['year'].min())}–{int(clean_df['year'].max())}")
        col4.metric("🤖 Best Model R²", "0.9229")


# ─── 📊 Data Explorer ───────────────────────────────────────────
elif page == "📊 Data Explorer" and DATA_LOADED:
    st.markdown("# 📊 Data Explorer")

    tab_raw, tab_clean, tab_missing = st.tabs(
        ["Raw Data", "Cleaned Data", "Missing Values"]
    )

    with tab_raw:
        st.markdown("### Raw Dataset (first 200 rows)")
        st.dataframe(raw_df.head(200), use_container_width=True, height=400)
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{raw_df.shape[0]:,}")
        c2.metric("Columns", f"{raw_df.shape[1]}")

    with tab_clean:
        st.markdown("### Cleaned Dataset (first 200 rows)")
        st.dataframe(clean_df.head(200), use_container_width=True, height=400)
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{clean_df.shape[0]:,}")
        c2.metric("Columns", f"{clean_df.shape[1]}")

    with tab_missing:
        st.markdown("### Missing Values in Raw Data")
        missing = raw_df.isnull().sum()
        missing_pct = (missing / len(raw_df) * 100).round(2)
        missing_info = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": missing_pct.values
        })
        only_missing = missing_info[missing_info["Missing Count"] > 0].sort_values(
            "Missing %", ascending=False
        )

        if only_missing.empty:
            st.success("No missing values found in the raw data!")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(
                only_missing["Column"],
                only_missing["Missing %"],
                color="#818cf8",
                edgecolor="white",
            )
            ax.set_xlabel("Missing %", fontsize=12, color="white")
            ax.set_title("Columns with Missing Values", fontsize=14,
                         fontweight="bold", color="white")
            ax.tick_params(colors="white")
            fig.patch.set_facecolor("#1a1a3e")
            ax.set_facecolor("#1a1a3e")
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(only_missing, use_container_width=True)


# ─── 📈 EDA & Visualizations ────────────────────────────────────
elif page == "📈 EDA & Visualizations" and DATA_LOADED:
    st.markdown("# 📈 EDA & Visualizations")

    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "Traffic Distribution",
        "Yearly Trends",
        "Period Analysis",
        "Direction Analysis",
    ])

    # --- Traffic Distribution ---
    with eda_tab1:
        st.markdown("### Traffic Count Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#1a1a3e")

        # histogram
        axes[0].hist(
            clean_df["traffic_count"].clip(upper=clean_df["traffic_count"].quantile(0.99)),
            bins=60, color="#6366f1", edgecolor="white", alpha=0.85,
        )
        axes[0].set_title("Distribution (clipped at 99th pctl)", fontsize=12,
                          fontweight="bold", color="white")
        axes[0].set_xlabel("Traffic Count", color="white")
        axes[0].set_ylabel("Frequency", color="white")
        axes[0].set_facecolor("#1a1a3e")
        axes[0].tick_params(colors="white")
        for s in axes[0].spines.values():
            s.set_visible(False)

        # log‑scale
        log_counts = np.log1p(clean_df["traffic_count"])
        axes[1].hist(log_counts, bins=60, color="#818cf8", edgecolor="white",
                     alpha=0.85)
        axes[1].set_title("Log‑Transformed Distribution", fontsize=12,
                          fontweight="bold", color="white")
        axes[1].set_xlabel("log(1 + traffic_count)", color="white")
        axes[1].set_ylabel("Frequency", color="white")
        axes[1].set_facecolor("#1a1a3e")
        axes[1].tick_params(colors="white")
        for s in axes[1].spines.values():
            s.set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        # quick stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{clean_df['traffic_count'].mean():,.0f}")
        col2.metric("Median", f"{clean_df['traffic_count'].median():,.0f}")
        col3.metric("Std Dev", f"{clean_df['traffic_count'].std():,.0f}")
        col4.metric("Max", f"{clean_df['traffic_count'].max():,}")

    # --- Yearly Trends ---
    with eda_tab2:
        st.markdown("### Yearly Traffic Trends")
        yearly = clean_df.groupby("year")["traffic_count"].agg(
            ["mean", "median", "count"]
        ).reset_index()

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#1a1a3e")
        ax.set_facecolor("#1a1a3e")

        ax.plot(yearly["year"], yearly["median"], marker="o",
                color="#818cf8", linewidth=2, label="Median")
        ax.fill_between(yearly["year"], 0, yearly["median"],
                        alpha=0.15, color="#818cf8")
        ax.set_title("Median Traffic Count per Year", fontsize=14,
                     fontweight="bold", color="white")
        ax.set_xlabel("Year", color="white")
        ax.set_ylabel("Median Traffic Count", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1a1a3e", edgecolor="white", labelcolor="white")
        for s in ax.spines.values():
            s.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # --- Period Analysis ---
    with eda_tab3:
        st.markdown("### Traffic by Period")
        if "period" in clean_df.columns:
            period_agg = clean_df.groupby("period")["traffic_count"].median().sort_values()
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#1a1a3e")
            ax.set_facecolor("#1a1a3e")
            ax.barh(period_agg.index, period_agg.values,
                    color="#a78bfa", edgecolor="white")
            ax.set_title("Median Traffic by Period", fontsize=14,
                         fontweight="bold", color="white")
            ax.set_xlabel("Median Traffic Count", color="white")
            ax.tick_params(colors="white")
            for s in ax.spines.values():
                s.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Period column not available after cleaning.")

    # --- Direction Analysis ---
    with eda_tab4:
        st.markdown("### Traffic by Direction")
        dir_cols = [c for c in ("cardinal_direction_name", "traffic_direction_name")
                    if c in clean_df.columns]
        if dir_cols:
            fig, axes = plt.subplots(1, len(dir_cols), figsize=(16, 5))
            fig.patch.set_facecolor("#1a1a3e")
            if len(dir_cols) == 1:
                axes = [axes]
            colors = ["#a29bfe", "#fd79a8"]
            for idx, col in enumerate(dir_cols):
                agg = clean_df.groupby(col)["traffic_count"].median().sort_values()
                axes[idx].barh(agg.index, agg.values,
                               color=colors[idx % 2], edgecolor="white")
                axes[idx].set_title(f"Median Traffic by {col.replace('_', ' ').title()}",
                                    fontsize=12, fontweight="bold", color="white")
                axes[idx].set_xlabel("Median Count", color="white")
                axes[idx].set_facecolor("#1a1a3e")
                axes[idx].tick_params(colors="white")
                for s in axes[idx].spines.values():
                    s.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Direction columns not available after cleaning.")


# ─── 🤖 Model Performance ───────────────────────────────────────
elif page == "🤖 Model Performance" and DATA_LOADED:
    st.markdown("# 🤖 Model Performance")

    # Hard‑coded results from the notebook so we don't retrain here
    results_data = {
        "Model": [
            "XGBoost", "Random Forest", "Decision Tree",
            "Extra Trees", "Ridge", "Linear Regression", "Lasso",
        ],
        "R²": [0.9229, 0.8905, 0.8868, 0.8214, 0.6180, 0.6180, 0.1697],
        "MAE": [1173, 1562, 1161, 2346, 7963, 7963, 5027],
        "RMSE": [2691, 3730, 2717, 5017, 152340, 152345, 10720],
    }
    results_df = pd.DataFrame(results_data).sort_values("R²", ascending=False)

    # R² bar chart
    st.markdown("### Model Comparison — R² Score")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a3e")
    ax.set_facecolor("#1a1a3e")

    bars = ax.barh(
        results_df["Model"], results_df["R²"],
        color=["#22c55e" if m == "XGBoost" else "#6366f1" for m in results_df["Model"]],
        edgecolor="white",
    )
    for bar, val in zip(bars, results_df["R²"]):
        ax.text(val - 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="right",
                fontsize=9, color="white", fontweight="bold")
    ax.set_title("R² Score  (green = best)", fontsize=13,
                 fontweight="bold", color="white")
    ax.set_xlabel("R² Score", color="white")
    ax.set_xlim(0, 1.05)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Table
    st.markdown("### Detailed Metrics")
    display_df = results_df.copy()
    display_df["R²"] = display_df["R²"].apply(lambda x: f"{x:.4f}")
    display_df["MAE"] = display_df["MAE"].apply(lambda x: f"{x:,.0f}")
    display_df["RMSE"] = display_df["RMSE"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    # Feature importance (XGBoost)
    st.markdown("### Feature Importance (XGBoost)")
    try:
        best_model = model.named_steps["model"]
        if hasattr(best_model, "feature_importances_"):
            importances = pd.Series(
                best_model.feature_importances_, index=FEATURES
            ).sort_values()
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            fig2.patch.set_facecolor("#1a1a3e")
            ax2.set_facecolor("#1a1a3e")
            imp_colors = [
                "#e17055" if f == importances.idxmax() else "#74b9ff"
                for f in importances.index
            ]
            importances.plot(kind="barh", ax=ax2, color=imp_colors, edgecolor="white")
            ax2.set_title("Feature Importance", fontsize=13,
                          fontweight="bold", color="white")
            ax2.set_xlabel("Importance", color="white")
            ax2.tick_params(colors="white")
            for s in ax2.spines.values():
                s.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)

            st.markdown("**Top 3 most important features:**")
            for feat, val in importances.sort_values(ascending=False).head(3).items():
                st.markdown(f"- `{feat}` → **{val:.4f}**")
        else:
            st.info("Feature importances not available for this model type.")
    except Exception:
        st.info("Could not extract feature importances from the saved model.")

    # Best model summary
    st.markdown("---")
    st.markdown("### 🏆 Best Model Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", "XGBoost")
    c2.metric("R²", "0.9229")
    c3.metric("MAE", "1,173")


# ─── 🔮 Predict Traffic ─────────────────────────────────────────
elif page == "🔮 Predict Traffic" and DATA_LOADED:
    st.markdown("# 🔮 Traffic Count Predictor")
    st.markdown(
        "Enter the conditions below and hit **Predict** to get the "
        "estimated traffic volume."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        station_id = st.selectbox("🏢 Station", sorted(station_map.keys()))
        year = st.slider("📅 Year", 2006, 2025, 2024)
        period = st.selectbox("⏰ Period", list(period_map.keys()))

    with col_right:
        classification = st.selectbox("🚗 Vehicle Type", list(class_map.keys()))
        direction = st.selectbox("🧭 Cardinal Direction", list(direction_map.keys()))
        traffic_type = st.selectbox("↔️ Traffic Direction", list(traffic_dir_map.keys()))

    st.markdown("---")

    def predict_traffic():
        """Build the feature vector exactly as in the notebook, then predict."""
        sid_enc = station_map.get(station_id, 0)

        is_peak = 1 if "PEAK" in period.upper() else 0
        is_weekend = 1 if "WEEKEND" in period.upper() else 0
        is_holiday = 1 if "HOLIDAY" in period.upper() else 0
        is_both = 1 if "AND" in traffic_type.upper() else 0
        is_heavy = 1 if classification.upper() == "HEAVY VEHICLES" else 0
        decade = (year // 10) * 10
        station_avg = station_avg_map.get(sid_enc, global_mean)

        row = pd.DataFrame([{
            "station_id_enc": sid_enc,
            "traffic_direction_name_enc": traffic_dir_map.get(traffic_type.upper(), 1),
            "cardinal_direction_name_enc": direction_map.get(direction.upper(), 3),
            "classification_type_enc": class_map.get(classification.upper(), 0),
            "period_enc": period_map.get(period.upper(), 5),
            "station_avg": station_avg,
            "year": year,
            "is_peak": is_peak,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "is_both_directions": is_both,
            "is_heavy": is_heavy,
            "decade": decade,
        }])[FEATURES]

        pred = int(np.expm1(model.predict(row)[0]))
        return pred

    if st.button("🚀 Predict Traffic", use_container_width=True):
        with st.spinner("Crunching numbers …"):
            result = predict_traffic()

        st.success(f"### 🚗 Predicted Traffic Count: **{result:,}**")

        # show a mini comparison across periods
        st.markdown("---")
        st.markdown("#### Quick Comparison — All Periods for This Station & Year")
        comparison_rows = []
        for p in period_map.keys():
            sid_enc = station_map.get(station_id, 0)
            is_peak = 1 if "PEAK" in p else 0
            is_weekend = 1 if "WEEKEND" in p else 0
            is_holiday = 1 if "HOLIDAY" in p else 0
            is_both = 1 if "AND" in traffic_type.upper() else 0
            is_heavy = 1 if classification.upper() == "HEAVY VEHICLES" else 0
            decade = (year // 10) * 10
            sa = station_avg_map.get(sid_enc, global_mean)
            row = pd.DataFrame([{
                "station_id_enc": sid_enc,
                "traffic_direction_name_enc": traffic_dir_map.get(traffic_type.upper(), 1),
                "cardinal_direction_name_enc": direction_map.get(direction.upper(), 3),
                "classification_type_enc": class_map.get(classification.upper(), 0),
                "period_enc": period_map.get(p, 5),
                "station_avg": sa,
                "year": year,
                "is_peak": is_peak,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "is_both_directions": is_both,
                "is_heavy": is_heavy,
                "decade": decade,
            }])[FEATURES]
            pred = int(np.expm1(model.predict(row)[0]))
            comparison_rows.append({"Period": p, "Predicted Count": f"{pred:,}"})

        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )


# ─── Footer ─────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center; color:#888; font-size:0.75rem;'>"
    "Built with ❤️ using Streamlit & XGBoost"
    "</div>",
    unsafe_allow_html=True,
)
