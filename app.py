import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NSW Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary   : #0a0e1a;
    --bg-secondary : #111827;
    --bg-card      : #1a2235;
    --accent-cyan  : #00d4ff;
    --accent-orange: #ff6b35;
    --accent-green : #00ff88;
    --accent-purple: #a855f7;
    --text-primary : #e2e8f0;
    --text-muted   : #64748b;
    --border       : #1e293b;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}
.main { background-color: var(--bg-primary); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.hero-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1040 50%, #0a1628 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #a855f7, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0; line-height: 1.2;
}
.hero-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 8px;
    font-weight: 400;
    letter-spacing: 0.05em;
}
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00d4ff;
    border-left: 3px solid #00d4ff;
    padding-left: 12px;
    margin: 28px 0 16px 0;
}
.pred-result {
    background: linear-gradient(135deg, #0d2137, #0a1628);
    border: 1px solid #00d4ff44;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.pred-number {
    font-family: 'Space Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    color: #00d4ff;
    line-height: 1;
}
.pred-label {
    color: #64748b;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 8px;
}
.info-box {
    background: #0d2137;
    border: 1px solid #00d4ff33;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.6;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #64748b;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #00d4ff !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #a855f722);
    border: 1px solid #00d4ff55;
    color: #00d4ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    border-radius: 8px;
    padding: 10px 24px;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #a855f744);
    border-color: #00d4ff;
}
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="stMetric"] label { color: #64748b !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-family: 'Space Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#111827',
    'axes.facecolor'   : '#1a2235',
    'axes.edgecolor'   : '#1e293b',
    'axes.labelcolor'  : '#94a3b8',
    'axes.titlecolor'  : '#e2e8f0',
    'xtick.color'      : '#64748b',
    'ytick.color'      : '#64748b',
    'grid.color'       : '#1e293b',
    'grid.linewidth'   : 0.8,
    'text.color'       : '#94a3b8',
    'font.family'      : 'monospace',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

CYAN   = '#00d4ff'
ORANGE = '#ff6b35'
GREEN  = '#00ff88'
PURPLE = '#a855f7'
RED    = '#ff4466'
YELLOW = '#f5c542'

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('Traffic_Data_Gov.csv')
    drop1 = ['the_geom','the_geom_webmercator','record_id',
             'latest_date','data_start_date','data_end_date',
             'data_duration','updated_on','md5','cartodb_id',
             'count_type','publish','data_quality_indicator']
    df.drop(columns=drop1, inplace=True)
    df['data_availability'].replace(-1, np.nan, inplace=True)
    df['data_reliability'].replace(-1, np.nan, inplace=True)
    df['data_availability'].fillna(df['data_availability'].median(), inplace=True)
    df['data_reliability'].fillna(df['data_reliability'].median(), inplace=True)
    df.drop(df[df['year'] == 2026].index, inplace=True)
    df.drop(df[df['period'] == 'SCHOOL HOLIDAYS'].index, inplace=True)
    drop2 = ['station_key','traffic_direction_seq',
             'cardinal_direction_seq','classification_seq','partial_year']
    df.drop(columns=drop2, inplace=True)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ['station_id','traffic_direction_name',
                'cardinal_direction_name','classification_type','period']:
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    df['is_peak']            = df['period'].str.contains('PEAK').astype(int)
    df['is_weekend']         = df['period'].str.contains('WEEKEND').astype(int)
    df['is_holiday']         = df['period'].str.contains('HOLIDAY').astype(int)
    df['is_both_directions'] = df['traffic_direction_name'].str.contains('AND').astype(int)
    df['is_heavy']           = (df['classification_type'] == 'HEAVY VEHICLES').astype(int)
    df['decade']             = (df['year'] // 10) * 10
    return df

@st.cache_resource
def load_model():
    try:
        model          = joblib.load('models/best_model.pkl')
        features       = joblib.load('models/features.pkl')
        station_lookup = joblib.load('models/station_lookup.pkl')
        return model, features, station_lookup
    except:
        return None, None, None

df                          = load_data()
model, FEATURES, station_lk = load_model()

# ─────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-title">🚦 NSW Traffic Intelligence</p>
    <p class="hero-subtitle">
        ROAD TRAFFIC VOLUME FORECASTING &nbsp;·&nbsp;
        NSW TRANSPORT DATA &nbsp;·&nbsp; 2006–2025
    </p>
</div>
""", unsafe_allow_html=True)

# TOP METRICS
c1, c2, c3, c4 = st.columns(4)
with c1:  st.metric("Total Records",       f"{len(df):,}",              "265K+ data points")
with c2:  st.metric("Monitoring Stations", f"{df['station_id'].nunique():,}", "Across NSW")
with c3:  st.metric("Years of Data",       "2006 – 2025",               "19 years")
with c4:  st.metric("Best Model",          "Random Forest",             "R² = 0.9853")

st.divider()

# ─────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='font-family:Space Mono,monospace;font-size:1rem;
    color:#00d4ff;letter-spacing:0.1em;padding:16px 0 8px'>⚙ CONTROLS</div>""",
    unsafe_allow_html=True)

    year_range = st.slider("📅 Year Range", 2006, 2025, (2015, 2025))

    veh_types = st.multiselect("🚗 Vehicle Type",
        options=sorted(df['classification_type'].unique()),
        default=sorted(df['classification_type'].unique()))

    directions = st.multiselect("📍 Direction",
        options=sorted(df['cardinal_direction_name'].unique()),
        default=sorted(df['cardinal_direction_name'].unique()))

    periods = st.multiselect("⏰ Period",
        options=sorted(df['period'].unique()),
        default=sorted(df['period'].unique()))

    st.divider()
    chart_palette = st.selectbox("🎨 Color Palette",
        ["Cyan/Orange", "Green/Purple", "Red/Blue"])
    chart_height = st.slider("📏 Chart Height", 300, 700, 420, 20)

    palette_map = {
        "Cyan/Orange" : [CYAN, ORANGE, GREEN, PURPLE],
        "Green/Purple": [GREEN, PURPLE, CYAN, ORANGE],
        "Red/Blue"    : [RED, CYAN, ORANGE, GREEN],
    }
    COLORS = palette_map[chart_palette]

# FILTER
mask = (
    df['year'].between(*year_range) &
    df['classification_type'].isin(veh_types) &
    df['cardinal_direction_name'].isin(directions) &
    df['period'].isin(periods)
)
dff = df[mask]

if len(dff) == 0:
    st.warning("⚠️ No data matches filters. Adjust sidebar.")
    st.stop()

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "🔍  Deep Analysis",
    "🔮  Predict Traffic",
    "🏆  Model Results"
])

# ══════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">TRAFFIC DISTRIBUTION & YEAR TREND</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        vals = np.log1p(dff['traffic_count'])
        ax.hist(vals, bins=60, color=COLORS[0], alpha=0.85,
                edgecolor='#0a0e1a', linewidth=0.4)
        ax.axvline(vals.mean(),   color=COLORS[1], linewidth=2,
                   linestyle='--', label=f'Mean: {dff["traffic_count"].mean():,.0f}')
        ax.axvline(vals.median(), color=COLORS[2], linewidth=2,
                   linestyle=':', label=f'Median: {dff["traffic_count"].median():,.0f}')
        ax.set_title('Traffic Count Distribution (Log Scale)', fontsize=11, pad=12)
        ax.set_xlabel('log(Traffic Count)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        yr_data = dff.groupby('year')['traffic_count'].median()
        ax.plot(yr_data.index, yr_data.values,
                color=COLORS[0], linewidth=2.5, marker='o', markersize=5, zorder=3)
        ax.fill_between(yr_data.index, yr_data.values, alpha=0.12, color=COLORS[0])
        if 2020 in yr_data.index:
            ax.annotate('COVID\ndip', xy=(2020, yr_data[2020]),
                        xytext=(2017, yr_data[2020]*0.82),
                        arrowprops=dict(arrowstyle='->', color=RED),
                        fontsize=8, color=RED)
        ax.set_title('Median Traffic by Year', fontsize=11, pad=12)
        ax.set_xlabel('Year')
        ax.set_ylabel('Median Traffic Count')
        ax.set_xticks(yr_data.index)
        ax.set_xticklabels(yr_data.index, rotation=45, ha='right', fontsize=7)
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">PERIOD & VEHICLE ANALYSIS</div>',
                unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        p_data = dff.groupby('period')['traffic_count'].median().sort_values()
        bar_c  = [COLORS[0] if 'PEAK' in p
                  else COLORS[1] if 'WEEKEND' in p
                  else COLORS[2] for p in p_data.index]
        bars = ax.barh(p_data.index, p_data.values,
                       color=bar_c, edgecolor='#0a0e1a', linewidth=0.5, height=0.65)
        for bar, val in zip(bars, p_data.values):
            ax.text(val + p_data.max()*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:,.0f}', va='center', fontsize=8)
        ax.set_title('Median Traffic by Period', fontsize=11, pad=12)
        ax.set_xlabel('Median Traffic Count')
        ax.xaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        v_data = dff.groupby('classification_type')['traffic_count']\
                    .median().sort_values(ascending=False)
        bars = ax.bar(v_data.index, v_data.values,
                      color=COLORS[:len(v_data)],
                      edgecolor='#0a0e1a', linewidth=0.5, width=0.55)
        for bar, val in zip(bars, v_data.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + v_data.max()*0.01,
                    f'{val:,.0f}', ha='center', fontsize=8, fontweight='bold')
        ax.set_title('Median Traffic by Vehicle Type', fontsize=11, pad=12)
        ax.set_ylabel('Median Traffic Count')
        ax.set_xticklabels(v_data.index, rotation=15, ha='right')
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">TOP 15 BUSIEST STATIONS</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, chart_height/80))
    top15 = dff.groupby('station_id')['traffic_count']\
               .median().nlargest(15).sort_values()
    grad  = plt.cm.YlOrRd(np.linspace(0.35, 0.9, 15))
    bars  = ax.barh(top15.index.astype(str), top15.values,
                    color=grad, edgecolor='#0a0e1a', linewidth=0.4, height=0.65)
    for bar, val in zip(bars, top15.values):
        ax.text(val + top15.max()*0.005,
                bar.get_y() + bar.get_height()/2,
                f'{val:,.0f}', va='center', fontsize=9, fontweight='bold')
    ax.set_title('Top 15 Busiest Stations (Median Daily Traffic)', fontsize=11, pad=12)
    ax.set_xlabel('Median Traffic Count')
    ax.xaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════
# TAB 2 — DEEP ANALYSIS
# ══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">DIRECTION & CORRELATION</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        d_data  = dff.groupby('cardinal_direction_name')['traffic_count']\
                     .median().sort_values()
        bar_c   = [COLORS[1] if 'AND' in d else COLORS[0] for d in d_data.index]
        bars    = ax.barh(d_data.index, d_data.values,
                          color=bar_c, edgecolor='#0a0e1a', linewidth=0.5, height=0.6)
        for bar, val in zip(bars, d_data.values):
            ax.text(val + d_data.max()*0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:,.0f}', va='center', fontsize=8)
        ax.set_title('Median Traffic by Direction', fontsize=11, pad=12)
        ax.set_xlabel('Median Traffic Count')
        ax.xaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        temp    = dff[['year','traffic_count','data_availability',
                       'data_reliability','station_id_enc','period_enc',
                       'classification_type_enc','is_peak',
                       'is_heavy','is_weekend']].copy()
        tc = temp.corr()['traffic_count'].drop('traffic_count').sort_values()
        colors_c = [COLORS[0] if v > 0 else RED for v in tc.values]
        ax.barh(tc.index, tc.values,
                color=colors_c, edgecolor='#0a0e1a', linewidth=0.5, height=0.65)
        ax.axvline(0, color='#64748b', linewidth=1)
        ax.set_title('Correlation with Traffic Count', fontsize=11, pad=12)
        ax.set_xlabel('Correlation Coefficient')
        ax.xaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">FULL CORRELATION HEATMAP</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, chart_height/75))
    num_cols = ['year','traffic_count','data_availability','data_reliability',
                'station_id_enc','period_enc','classification_type_enc',
                'traffic_direction_name_enc','cardinal_direction_name_enc',
                'is_peak','is_heavy','is_weekend','is_holiday','decade']
    corr_full  = dff[num_cols].corr()
    mask_upper = np.triu(np.ones_like(corr_full, dtype=bool), k=1)
    sns.heatmap(corr_full, ax=ax, mask=mask_upper,
                annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, linecolor='#0a0e1a',
                annot_kws={'size': 7},
                cbar_kws={'shrink': 0.8})
    ax.set_title('Full Feature Correlation Matrix', fontsize=11, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">OUTLIERS & DECADE TRENDS</div>',
                unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        types   = sorted(dff['classification_type'].unique())
        bp_data = [dff[dff['classification_type']==t]['traffic_count'].values
                   for t in types]
        bp = ax.boxplot(bp_data, patch_artist=True,
                        medianprops=dict(color=YELLOW, linewidth=2.5),
                        whiskerprops=dict(color='#64748b'),
                        capprops=dict(color='#64748b'),
                        flierprops=dict(marker='.', color=COLORS[1],
                                        alpha=0.3, markersize=3))
        for patch, c in zip(bp['boxes'], COLORS[:len(types)]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_xticklabels(types, rotation=15, ha='right', fontsize=8)
        ax.set_title('Traffic Distribution by Vehicle Type', fontsize=11, pad=12)
        ax.set_ylabel('Traffic Count')
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        dec_data = dff.groupby('decade')['traffic_count'].agg(['median','mean'])
        x = np.arange(len(dec_data)); w = 0.35
        b1 = ax.bar(x-w/2, dec_data['median'], w, label='Median',
                    color=COLORS[0], alpha=0.85, edgecolor='#0a0e1a')
        b2 = ax.bar(x+w/2, dec_data['mean'],   w, label='Mean',
                    color=COLORS[1], alpha=0.85, edgecolor='#0a0e1a')
        for bar, val in zip(b1, dec_data['median']):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+dec_data['mean'].max()*0.01,
                    f'{val:,.0f}', ha='center', fontsize=7)
        for bar, val in zip(b2, dec_data['mean']):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+dec_data['mean'].max()*0.01,
                    f'{val:,.0f}', ha='center', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}s" for d in dec_data.index])
        ax.set_title('Traffic by Decade: Mean vs Median', fontsize=11, pad=12)
        ax.set_ylabel('Traffic Count')
        ax.legend(); ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">STATION DEEP DIVE</div>',
                unsafe_allow_html=True)
    top_stations     = df.groupby('station_id')['traffic_count']\
                         .median().nlargest(50).index.tolist()
    selected_station = st.selectbox("Select Station", options=top_stations)
    s_data           = df[df['station_id'] == selected_station]

    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        yr_s = s_data.groupby('year')['traffic_count'].mean()
        ax.plot(yr_s.index, yr_s.values,
                color=COLORS[0], linewidth=2.5, marker='o', markersize=6)
        ax.fill_between(yr_s.index, yr_s.values, alpha=0.12, color=COLORS[0])
        ax.set_title(f'Station {selected_station} — Year Trend', fontsize=11, pad=12)
        ax.set_xlabel('Year'); ax.set_ylabel('Avg Traffic Count')
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col6:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        per_s = s_data.groupby('period')['traffic_count'].median().sort_values()
        bar_c = [COLORS[0] if 'PEAK' in p
                 else COLORS[1] if 'WEEKEND' in p
                 else COLORS[2] for p in per_s.index]
        bars  = ax.barh(per_s.index, per_s.values,
                        color=bar_c, edgecolor='#0a0e1a', linewidth=0.5, height=0.6)
        for bar, val in zip(bars, per_s.values):
            ax.text(val + per_s.max()*0.01,
                    bar.get_y()+bar.get_height()/2,
                    f'{val:,.0f}', va='center', fontsize=8)
        ax.set_title(f'Station {selected_station} — By Period', fontsize=11, pad=12)
        ax.set_xlabel('Median Traffic Count')
        ax.xaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════
# TAB 3 — PREDICT
# ══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">TRAFFIC VOLUME PREDICTOR</div>',
                unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model not found! Make sure models/ folder is in same directory.")
    else:
        col_form, col_result = st.columns([1.2, 1])

        with col_form:
            if station_lk is not None:
                station_lk['label'] = (
                    station_lk['station_id'].astype(str) +
                    "  |  enc=" + station_lk['station_id_enc'].astype(str) +
                    "  |  median=" +
                    station_lk['traffic_count'].apply(lambda x: f"{x:,.0f}")
                )
                sel = st.selectbox("📍 Station", station_lk['label'].tolist())
                station_enc = int(sel.split("enc=")[1].split(" ")[0])
            else:
                station_enc = st.number_input("Station ID Encoded", 0, 2000, 1)

            c1, c2 = st.columns(2)
            with c1: pred_year   = st.slider("📅 Year", 2006, 2030, 2024)
            with c2: pred_period = st.selectbox("⏰ Period", [
                'WEEKDAYS','ALL DAYS','AM PEAK','PM PEAK',
                'OFF PEAK','WEEKENDS','PUBLIC HOLIDAYS'])

            c3, c4 = st.columns(2)
            with c3: pred_class = st.selectbox("🚗 Vehicle Type", [
                'UNCLASSIFIED','ALL VEHICLES','LIGHT VEHICLES','HEAVY VEHICLES'])
            with c4: pred_dir   = st.selectbox("🧭 Direction", [
                'NORTH','SOUTH','EAST','WEST','BOTH',
                'NORTHBOUND AND SOUTHBOUND','EASTBOUND AND WESTBOUND'])

            pred_tdir = st.selectbox("↔️ Traffic Direction", [
                'PRESCRIBED','COUNTER','PRESCRIBED AND COUNTER'])

            c5, c6 = st.columns(2)
            with c5: pred_avail = st.slider("📶 Data Availability %", 0, 100, 93)
            with c6: pred_rel   = st.slider("✅ Data Reliability %",   0, 100, 93)

            predict_btn = st.button("🔮  PREDICT TRAFFIC VOLUME")

        with col_result:
            if predict_btn:
                period_map = {
                    'ALL DAYS':0,'AM PEAK':1,'OFF PEAK':2,
                    'PM PEAK':3,'PUBLIC HOLIDAYS':4,'WEEKDAYS':5,'WEEKENDS':6
                }
                class_map  = {
                    'ALL VEHICLES':0,'HEAVY VEHICLES':1,
                    'LIGHT VEHICLES':2,'UNCLASSIFIED':3
                }
                dir_map    = {
                    'BOTH':0,'EAST':1,'EASTBOUND AND WESTBOUND':2,
                    'NORTH':3,'NORTHBOUND AND SOUTHBOUND':4,'SOUTH':5,'WEST':6
                }
                tdir_map   = {
                    'COUNTER':0,'PRESCRIBED':1,'PRESCRIBED AND COUNTER':2
                }

                is_peak  = 1 if 'PEAK'    in pred_period else 0
                is_wkend = 1 if 'WEEKEND' in pred_period else 0
                is_hol   = 1 if 'HOLIDAY' in pred_period else 0
                is_both  = 1 if 'AND'     in pred_tdir   else 0
                is_heavy = 1 if pred_class == 'HEAVY VEHICLES' else 0
                decade   = (pred_year // 10) * 10

                inp = pd.DataFrame([{
                    'station_id_enc'              : station_enc,
                    'traffic_direction_name_enc'  : tdir_map.get(pred_tdir, 1),
                    'cardinal_direction_name_enc' : dir_map.get(pred_dir, 3),
                    'classification_type_enc'     : class_map.get(pred_class, 0),
                    'period_enc'                  : period_map.get(pred_period, 5),
                    'year'                        : pred_year,
                    'data_availability'           : pred_avail,
                    'data_reliability'            : pred_rel,
                    'is_peak'                     : is_peak,
                    'is_weekend'                  : is_wkend,
                    'is_holiday'                  : is_hol,
                    'is_both_directions'          : is_both,
                    'is_heavy'                    : is_heavy,
                    'decade'                      : decade
                }])[FEATURES]

                pred_val = int(np.expm1(model.predict(inp)[0]))

                if pred_val > 50000:
                    label, color = "EXTREME VOLUME", RED
                elif pred_val > 20000:
                    label, color = "HIGH VOLUME",    ORANGE
                elif pred_val > 5000:
                    label, color = "MEDIUM VOLUME",  YELLOW
                else:
                    label, color = "LOW VOLUME",     GREEN

                st.markdown(f"""
                <div class="pred-result">
                    <div class="pred-label">PREDICTED TRAFFIC VOLUME</div>
                    <div class="pred-number" style="color:{color}">{pred_val:,}</div>
                    <div class="pred-label">vehicles</div>
                    <div style="margin-top:16px;padding:8px 20px;
                    background:{color}22;border-radius:8px;color:{color};
                    font-family:Space Mono,monospace;font-size:0.8rem;
                    letter-spacing:0.1em">{label}</div>
                    <div style="margin-top:24px;text-align:left;
                    font-size:0.8rem;color:#475569;line-height:1.8">
                        <b style="color:#94a3b8">Station enc :</b> {station_enc}<br>
                        <b style="color:#94a3b8">Year        :</b> {pred_year}<br>
                        <b style="color:#94a3b8">Period      :</b> {pred_period}<br>
                        <b style="color:#94a3b8">Vehicle     :</b> {pred_class}<br>
                        <b style="color:#94a3b8">Direction   :</b> {pred_dir}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box" style="text-align:center;padding:60px 24px">
                    <div style="font-size:3rem;margin-bottom:16px">🔮</div>
                    <div style="color:#64748b;font-size:0.9rem">
                        Configure parameters on the left<br>
                        then click <b style="color:#00d4ff">PREDICT</b>
                    </div>
                </div>""", unsafe_allow_html=True)

        # Year forecast chart
        st.markdown('<div class="section-header">YEAR FORECAST CHART</div>',
                    unsafe_allow_html=True)
        if predict_btn:
            years_c = list(range(2010, 2031))
            preds_c = []
            for yr in years_c:
                inp_c = pd.DataFrame([{
                    'station_id_enc'              : station_enc,
                    'traffic_direction_name_enc'  : tdir_map.get(pred_tdir, 1),
                    'cardinal_direction_name_enc' : dir_map.get(pred_dir, 3),
                    'classification_type_enc'     : class_map.get(pred_class, 0),
                    'period_enc'                  : period_map.get(pred_period, 5),
                    'year'                        : yr,
                    'data_availability'           : pred_avail,
                    'data_reliability'            : pred_rel,
                    'is_peak'                     : is_peak,
                    'is_weekend'                  : is_wkend,
                    'is_holiday'                  : is_hol,
                    'is_both_directions'          : is_both,
                    'is_heavy'                    : is_heavy,
                    'decade'                      : (yr // 10) * 10
                }])[FEATURES]
                preds_c.append(int(np.expm1(model.predict(inp_c)[0])))

            fig, ax = plt.subplots(figsize=(14, chart_height/100))
            ax.fill_between(years_c, preds_c, alpha=0.15, color=COLORS[0])
            ax.plot(years_c, preds_c, color=COLORS[0], linewidth=2.5,
                    marker='o', markersize=6, zorder=3)
            ax.axvline(2025, color='#64748b', linestyle='--',
                       linewidth=1, label='Forecast →')
            ax.scatter([pred_year], [pred_val], color=COLORS[1],
                       s=150, zorder=5,
                       label=f'Selected: {pred_year} = {pred_val:,}')
            for yr, pv in zip(years_c, preds_c):
                ax.text(yr, pv + max(preds_c)*0.012, f'{pv:,}',
                        ha='center', fontsize=7, color='#94a3b8')
            ax.set_title(
                f'Traffic Forecast 2010–2030  ·  {pred_period} | {pred_class} | Station enc={station_enc}',
                fontsize=11, pad=12)
            ax.set_xlabel('Year'); ax.set_ylabel('Predicted Traffic')
            ax.set_xticks(years_c)
            ax.set_xticklabels(years_c, rotation=45, ha='right', fontsize=8)
            ax.yaxis.grid(True, alpha=0.3); ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            st.markdown("""<div class="info-box" style="text-align:center;padding:32px">
            Run a prediction above to see the year forecast chart 📈
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 4 — MODEL RESULTS
# ══════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">MODEL LEADERBOARD</div>',
                unsafe_allow_html=True)

    res = {
        'Model': ['Random Forest','Decision Tree','Extra Trees','XGBoost',
                  'KNN','Gradient Boosting','Linear Regression',
                  'Ridge Regression','Lasso Regression'],
        'R²'   : [0.9853,0.9790,0.9256,0.7749,0.2913,
                  -0.0971,-4.766,-4.766,-506272.42],
        'MAE'  : [537,566,1137,2256,3772,3682,5832,5832,7097],
        'RMSE' : [1527,1850,3222,5312,8252,8301,11780,11780,14000],
    }
    res_df = pd.DataFrame(res)

    col1, col2 = st.columns(2)
    bar_colors = [YELLOW if m=='Random Forest'
                  else COLORS[0] if r>0.8
                  else COLORS[2] if r>0
                  else RED
                  for m, r in zip(res['Model'], res['R²'])]

    with col1:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        r2_disp = [max(r, -5) for r in res['R²']]
        ax.bar(res['Model'], r2_disp, color=bar_colors,
               edgecolor='#0a0e1a', linewidth=0.5, width=0.6)
        ax.axhline(0.8, color=GREEN, linestyle='--',
                   linewidth=1.5, label='0.8 threshold')
        ax.axhline(0, color='#64748b', linewidth=1)
        ax.set_title('R² Score Comparison', fontsize=11, pad=12)
        ax.set_ylabel('R² Score')
        ax.set_xticklabels(res['Model'], rotation=30, ha='right', fontsize=7)
        ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, chart_height/100))
        bars = ax.bar(res['Model'], res['MAE'], color=bar_colors,
                      edgecolor='#0a0e1a', linewidth=0.5, width=0.6)
        for bar, val in zip(bars, res['MAE']):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+50,
                    f'{val:,}', ha='center', fontsize=7)
        ax.set_title('MAE Comparison (Lower = Better)', fontsize=11, pad=12)
        ax.set_ylabel('Mean Absolute Error')
        ax.set_xticklabels(res['Model'], rotation=30, ha='right', fontsize=7)
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">FEATURE IMPORTANCE — RANDOM FOREST</div>',
                unsafe_allow_html=True)
    feat_imp = pd.Series({
        'station_id_enc'              : 0.498,
        'is_heavy'                    : 0.298,
        'is_peak'                     : 0.082,
        'year'                        : 0.051,
        'classification_type_enc'     : 0.028,
        'data_availability'           : 0.018,
        'period_enc'                  : 0.011,
        'traffic_direction_name_enc'  : 0.006,
        'is_both_directions'          : 0.003,
        'cardinal_direction_name_enc' : 0.002,
        'data_reliability'            : 0.001,
        'is_holiday'                  : 0.001,
        'decade'                      : 0.001,
        'is_weekend'                  : 0.000,
    }).sort_values()

    fig, ax = plt.subplots(figsize=(12, chart_height/80))
    fi_c = [YELLOW if f=='station_id_enc'
            else COLORS[1] if f=='is_heavy'
            else COLORS[0] for f in feat_imp.index]
    bars = ax.barh(feat_imp.index, feat_imp.values,
                   color=fi_c, edgecolor='#0a0e1a', linewidth=0.5, height=0.65)
    for bar, val in zip(bars, feat_imp.values):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.set_title('Feature Importance — Random Forest', fontsize=11, pad=12)
    ax.set_xlabel('Importance Score')
    ax.xaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">FULL RESULTS TABLE</div>',
                unsafe_allow_html=True)
    disp = res_df.copy()
    disp['R²']   = disp['R²'].apply(lambda x: f"{x:.4f}" if x>-1000 else f"{x:,.0f}")
    disp['MAE']  = disp['MAE'].apply(lambda x: f"{x:,}")
    disp['RMSE'] = disp['RMSE'].apply(lambda x: f"{x:,}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#334155;font-family:Space Mono,monospace;
font-size:0.7rem;letter-spacing:0.1em;padding:12px'>
NSW TRAFFIC INTELLIGENCE · RANDOM FOREST R²=0.9853 ·
265,291 RECORDS · 1,967 STATIONS · 2006–2025
</div>
""", unsafe_allow_html=True)