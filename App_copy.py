import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NSW Traffic Forecast",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# LOAD & PREPARE DATA  (mirrors traffic.ipynb exactly)
# ─────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv('Traffic_Data_Gov.csv')

    # Cell 4 — drop unused columns
    drop1 = [
        'the_geom', 'the_geom_webmercator', 'record_id',
        'latest_date', 'data_start_date', 'data_end_date',
        'data_duration', 'updated_on', 'md5', 'count_type',
        'publish', 'data_quality_indicator', 'cartodb_id'
    ]
    df.drop(columns=drop1, inplace=True)

    # Cell 12-13 — fix missing values
    df['data_availability'] = df['data_availability'].replace(-1, np.nan)
    df['data_reliability']  = df['data_reliability'].replace(-1, np.nan)
    df['data_availability'].fillna(df['data_availability'].median(), inplace=True)
    df['data_reliability'].fillna(df['data_reliability'].median(),  inplace=True)

    # Cell 20-22 — remove bad rows
    df.drop(df[df['year'] == 2026].index, inplace=True)
    df.drop(df[df['period'] == 'SCHOOL HOLIDAYS'].index, inplace=True)

    # Cell 25 — drop duplicate / low-value cols
    drop2 = ['traffic_direction_seq', 'cardinal_direction_seq',
              'classification_seq', 'partial_year', 'station_key']
    df.drop(columns=drop2, inplace=True)

    # Cell 30 — label encoding
    le = LabelEncoder()
    for col in ['station_id', 'traffic_direction_name',
                'cardinal_direction_name', 'classification_type', 'period']:
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))

    # Cell 34 — feature engineering
    df['is_peak']            = df['period'].str.contains('PEAK').astype(int)
    df['is_weekend']         = df['period'].str.contains('WEEKEND').astype(int)
    df['is_holiday']         = df['period'].str.contains('HOLIDAY').astype(int)
    df['is_both_directions'] = df['traffic_direction_name'].str.contains('AND').astype(int)
    df['is_heavy']           = (df['classification_type'] == 'HEAVY VEHICLES').astype(int)
    df['decade']             = (df['year'] // 10) * 10

    return df


@st.cache_resource
def train_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline

    # Cell 38 — features & target
    features = [
        'station_id_encoded', 'traffic_direction_name_encoded',
        'cardinal_direction_name_encoded', 'classification_type_encoded',
        'period_encoded', 'year', 'data_availability', 'data_reliability',
        'is_peak', 'is_weekend', 'is_holiday',
        'is_both_directions', 'is_heavy', 'decade'
    ]
    X = df[features]
    y = np.log1p(df['traffic_count'])

    # Cell 40 — split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Cell 42-43 — best model: Random Forest
    pipeline = Pipeline([
        ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)

    return pipeline, features


# ─────────────────────────────────────────
# PREDICT FUNCTION  (Cell 48 — exact copy)
# ─────────────────────────────────────────
def predict_traffic(pipeline, features, station_id_enc, year, period,
                    classification_type, cardinal_direction,
                    traffic_direction, data_availability=93, data_reliability=93):
    period_map = {
        'ALL DAYS': 0, 'AM PEAK': 1, 'OFF PEAK': 2,
        'PM PEAK': 3, 'PUBLIC HOLIDAYS': 4, 'WEEKDAYS': 5, 'WEEKENDS': 6
    }
    class_map = {
        'ALL VEHICLES': 0, 'HEAVY VEHICLES': 1,
        'LIGHT VEHICLES': 2, 'UNCLASSIFIED': 3
    }
    direction_map = {
        'BOTH': 0, 'EAST': 1, 'EASTBOUND AND WESTBOUND': 2,
        'NORTH': 3, 'NORTHBOUND AND SOUTHBOUND': 4,
        'SOUTH': 5, 'WEST': 6
    }
    traffic_dir_map = {
        'COUNTER': 0, 'PRESCRIBED': 1, 'PRESCRIBED AND COUNTER': 2
    }

    is_peak            = 1 if 'PEAK'    in period.upper() else 0
    is_weekend         = 1 if 'WEEKEND' in period.upper() else 0
    is_holiday         = 1 if 'HOLIDAY' in period.upper() else 0
    is_both_directions = 1 if 'AND'     in traffic_direction.upper() else 0
    is_heavy           = 1 if classification_type.upper() == 'HEAVY VEHICLES' else 0
    decade             = (year // 10) * 10

    input_data = pd.DataFrame([{
        'station_id_encoded'              : station_id_enc,
        'traffic_direction_name_encoded'  : traffic_dir_map.get(traffic_direction.upper(), 1),
        'cardinal_direction_name_encoded' : direction_map.get(cardinal_direction.upper(), 3),
        'classification_type_encoded'     : class_map.get(classification_type.upper(), 0),
        'period_encoded'                  : period_map.get(period.upper(), 5),
        'year'                            : year,
        'data_availability'               : data_availability,
        'data_reliability'                : data_reliability,
        'is_peak'                         : is_peak,
        'is_weekend'                      : is_weekend,
        'is_holiday'                      : is_holiday,
        'is_both_directions'              : is_both_directions,
        'is_heavy'                        : is_heavy,
        'decade'                          : decade
    }])[features]

    pred_log = pipeline.predict(input_data)[0]
    return int(np.expm1(pred_log))


# ─────────────────────────────────────────
# MATPLOTLIB STYLE
# ─────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')


# ─────────────────────────────────────────
# LOAD DATA + MODEL
# ─────────────────────────────────────────
with st.spinner("Loading data..."):
    df = load_and_prepare()

with st.spinner("Training Random Forest model..."):
    pipeline, FEATURES = train_model(df)


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🚦 NSW Traffic Forecast")
st.caption("Road Traffic Volume · NSW Transport Data · 2006–2025")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records",       f"{len(df):,}")
c2.metric("Monitoring Stations", f"{df['station_id'].nunique():,}")
c3.metric("Years Covered",       "2006 – 2025")
c4.metric("Model R²",            "0.9853  (Random Forest)")

st.divider()

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 EDA Charts", "🔮 Predict Traffic", "📋 Model Summary"])


# ══════════════════════════════════════════
# TAB 1 — EDA  (mirrors Cell 35 & 36)
# ══════════════════════════════════════════
with tab1:

    # ── Row 1 ──────────────────────────────
    col1, col2 = st.columns(2)

    # Chart 1: Traffic Count Distribution (log)
    with col1:
        st.subheader("Traffic Count Distribution (log scale)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.log1p(df['traffic_count']),
                bins=50, color='steelblue', edgecolor='white')
        ax.set_xlabel('log(1 + Traffic Count)')
        ax.set_ylabel('Frequency')
        ax.axvline(np.log1p(df['traffic_count']).mean(),
                   color='red', linestyle='--', label='Mean')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 2: Median Traffic by Year
    with col2:
        st.subheader("Median Traffic by Year")
        fig, ax = plt.subplots(figsize=(6, 4))
        yr_data = df.groupby('year')['traffic_count'].median()
        ax.plot(yr_data.index, yr_data.values,
                marker='o', color='steelblue', linewidth=2, markersize=5)
        ax.fill_between(yr_data.index, yr_data.values, alpha=0.15, color='steelblue')
        if 2020 in yr_data.index:
            ax.annotate('COVID\ndip', xy=(2020, yr_data[2020]),
                        xytext=(2018, yr_data[2020] * 0.80),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=8, color='red')
        ax.set_xlabel('Year')
        ax.set_ylabel('Median Traffic Count')
        ax.set_xticks(yr_data.index)
        ax.set_xticklabels(yr_data.index, rotation=45, ha='right', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Row 2 ──────────────────────────────
    col3, col4 = st.columns(2)

    # Chart 3: Traffic by Period
    with col3:
        st.subheader("Median Traffic by Period")
        fig, ax = plt.subplots(figsize=(6, 4))
        p_data = df.groupby('period')['traffic_count'].median().sort_values()
        colors = ['steelblue' if 'PEAK' in p
                  else 'darkorange' if 'WEEKEND' in p
                  else 'seagreen' for p in p_data.index]
        ax.barh(p_data.index, p_data.values, color=colors, edgecolor='white')
        ax.set_xlabel('Median Traffic Count')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 4: Traffic by Vehicle Type
    with col4:
        st.subheader("Median Traffic by Vehicle Type")
        fig, ax = plt.subplots(figsize=(6, 4))
        v_data = df.groupby('classification_type')['traffic_count'].median().sort_values(ascending=False)
        ax.bar(v_data.index, v_data.values,
               color=['steelblue', 'darkorange', 'seagreen', 'tomato'],
               edgecolor='white')
        ax.set_ylabel('Median Traffic Count')
        ax.set_xticklabels(v_data.index, rotation=15, ha='right')
        for i, (val) in enumerate(v_data.values):
            ax.text(i, val + v_data.max() * 0.01,
                    f'{val:,.0f}', ha='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Row 3 ──────────────────────────────
    col5, col6 = st.columns(2)

    # Chart 5: Correlation heatmap (numeric cols)
    with col5:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        num_cols = ['year', 'traffic_count', 'data_availability',
                    'data_reliability', 'is_peak', 'is_heavy',
                    'is_weekend', 'is_holiday', 'decade']
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', linewidths=0.5, annot_kws={'size': 7},
                    cbar_kws={'shrink': 0.8})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 6: Boxplot outliers by vehicle type  (Cell 36 Chart 7)
    with col6:
        st.subheader("Traffic Outliers by Vehicle Type")
        fig, ax = plt.subplots(figsize=(6, 5))
        types   = sorted(df['classification_type'].unique())
        bp_data = [df[df['classification_type'] == t]['traffic_count'].values for t in types]
        bp = ax.boxplot(bp_data, patch_artist=True,
                        medianprops=dict(color='yellow', linewidth=2),
                        flierprops=dict(marker='.', alpha=0.3, markersize=3))
        colors_bp = ['steelblue', 'darkorange', 'seagreen', 'tomato']
        for patch, c in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xticklabels(types, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel('Traffic Count')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Row 4: Top 15 busiest stations ─────
    st.subheader("Top 15 Busiest Stations (Median Daily Traffic)")
    fig, ax = plt.subplots(figsize=(12, 5))
    top15 = df.groupby('station_id')['traffic_count'].median().nlargest(15).sort_values()
    grad  = plt.cm.YlOrRd(np.linspace(0.35, 0.9, 15))
    bars  = ax.barh(top15.index.astype(str), top15.values, color=grad, edgecolor='white')
    for bar, val in zip(bars, top15.values):
        ax.text(val + top15.max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:,.0f}', va='center', fontsize=9)
    ax.set_xlabel('Median Traffic Count')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════
# TAB 2 — PREDICT  (Cell 48 predict function)
# ══════════════════════════════════════════
with tab2:
    st.subheader("Traffic Volume Predictor")
    st.caption("Uses the trained Random Forest model (R² = 0.9853)")

    col_form, col_out = st.columns([1, 1])

    with col_form:
        # Station — show busiest 50 so user picks something meaningful
        top_stations = (df.groupby('station_id')['traffic_count']
                        .median().nlargest(50).index.tolist())
        station_id   = st.selectbox("Station ID", top_stations)
        station_enc  = int(df[df['station_id'] == station_id]
                           ['station_id_encoded'].iloc[0])

        pred_year  = st.slider("Year", 2006, 2030, 2024)

        pred_period = st.selectbox("Period", [
            'WEEKDAYS', 'ALL DAYS', 'AM PEAK', 'PM PEAK',
            'OFF PEAK', 'WEEKENDS', 'PUBLIC HOLIDAYS'
        ])

        pred_class = st.selectbox("Vehicle Type", [
            'ALL VEHICLES', 'LIGHT VEHICLES',
            'HEAVY VEHICLES', 'UNCLASSIFIED'
        ])

        pred_dir = st.selectbox("Cardinal Direction", [
            'NORTH', 'SOUTH', 'EAST', 'WEST',
            'NORTHBOUND AND SOUTHBOUND', 'EASTBOUND AND WESTBOUND', 'BOTH'
        ])

        pred_tdir = st.selectbox("Traffic Direction", [
            'PRESCRIBED', 'COUNTER', 'PRESCRIBED AND COUNTER'
        ])

        c1, c2 = st.columns(2)
        with c1:
            pred_avail = st.slider("Data Availability %", 0, 100, 93)
        with c2:
            pred_rel   = st.slider("Data Reliability %",  0, 100, 93)

        predict_btn = st.button("🔮  Predict", use_container_width=True)

    with col_out:
        if predict_btn:
            result = predict_traffic(
                pipeline, FEATURES,
                station_id_enc     = station_enc,
                year               = pred_year,
                period             = pred_period,
                classification_type= pred_class,
                cardinal_direction = pred_dir,
                traffic_direction  = pred_tdir,
                data_availability  = pred_avail,
                data_reliability   = pred_rel
            )

            # Volume label
            if result > 50_000:
                label, color = "EXTREME VOLUME", "🔴"
            elif result > 20_000:
                label, color = "HIGH VOLUME",    "🟠"
            elif result > 5_000:
                label, color = "MEDIUM VOLUME",  "🟡"
            else:
                label, color = "LOW VOLUME",     "🟢"

            st.markdown(f"### {color} {result:,} vehicles")
            st.markdown(f"**{label}**")
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Station   | `{station_id}` (enc={station_enc}) |
| Year      | {pred_year} |
| Period    | {pred_period} |
| Vehicle   | {pred_class} |
| Direction | {pred_dir} |
""")

            # Year forecast chart
            st.markdown("**Year Forecast 2010–2030**")
            years  = list(range(2010, 2031))
            preds  = [
                predict_traffic(
                    pipeline, FEATURES, station_enc, yr,
                    pred_period, pred_class, pred_dir,
                    pred_tdir, pred_avail, pred_rel
                )
                for yr in years
            ]
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.fill_between(years, preds, alpha=0.12, color='steelblue')
            ax.plot(years, preds, color='steelblue', linewidth=2,
                    marker='o', markersize=4)
            ax.axvline(2025, color='gray', linestyle='--', linewidth=1,
                       label='Forecast →')
            ax.scatter([pred_year], [result], color='darkorange',
                       s=120, zorder=5,
                       label=f'{pred_year}: {result:,}')
            ax.set_xlabel('Year')
            ax.set_ylabel('Predicted Traffic')
            ax.set_xticks(years)
            ax.set_xticklabels(years, rotation=45, ha='right', fontsize=7)
            ax.legend(fontsize=8)
            ax.yaxis.grid(True, alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            st.info("Configure parameters on the left, then click **Predict**.")


# ══════════════════════════════════════════
# TAB 3 — MODEL SUMMARY
# ══════════════════════════════════════════
with tab3:
    st.subheader("Model Comparison — All 9 Models")

    results_data = {
        'Model': [
            'Random Forest', 'Decision Tree', 'Extra Trees', 'XGBoost',
            'KNN', 'Gradient Boosting', 'Linear Regression',
            'Ridge Regression', 'Lasso Regression'
        ],
        'R²': [0.9853, 0.9790, 0.9256, 0.7749, 0.2913,
               -0.0971, -4.766, -4.766, -506272.42],
        'MAE': [537, 566, 1137, 2256, 3772, 3682, 5832, 5832, 7097],
        'RMSE': [1527, 1850, 3222, 5312, 8252, 8301, 11780, 11780, 14000],
    }
    res_df = pd.DataFrame(results_data)

    col1, col2 = st.columns(2)

    # R² bar chart
    with col1:
        st.markdown("**R² Score (higher = better)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors  = ['gold' if m == 'Random Forest'
                   else 'steelblue' if r > 0.8
                   else 'seagreen' if r > 0
                   else 'tomato'
                   for m, r in zip(results_data['Model'], results_data['R²'])]
        r2_disp = [max(r, -5) for r in results_data['R²']]
        ax.bar(results_data['Model'], r2_disp, color=colors, edgecolor='white')
        ax.axhline(0.8, color='limegreen', linestyle='--',
                   linewidth=1.5, label='0.8 threshold')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set_ylabel('R² Score')
        ax.set_xticklabels(results_data['Model'], rotation=30, ha='right', fontsize=7)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # MAE bar chart
    with col2:
        st.markdown("**MAE — Mean Absolute Error (lower = better)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(results_data['Model'], results_data['MAE'],
               color=colors, edgecolor='white')
        ax.set_ylabel('MAE (vehicles)')
        ax.set_xticklabels(results_data['Model'], rotation=30, ha='right', fontsize=7)
        for i, v in enumerate(results_data['MAE']):
            ax.text(i, v + 50, f'{v:,}', ha='center', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature importance
    st.markdown("**Feature Importance — Random Forest**")
    feat_imp = pd.Series({
        'station_id_encoded'              : 0.498,
        'is_heavy'                        : 0.298,
        'is_peak'                         : 0.082,
        'year'                            : 0.051,
        'classification_type_encoded'     : 0.028,
        'data_availability'               : 0.018,
        'period_encoded'                  : 0.011,
        'traffic_direction_name_encoded'  : 0.006,
        'is_both_directions'              : 0.003,
        'cardinal_direction_name_encoded' : 0.002,
        'data_reliability'                : 0.001,
        'is_holiday'                      : 0.001,
        'decade'                          : 0.001,
        'is_weekend'                      : 0.000,
    }).sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    fi_colors = ['gold' if f == 'station_id_encoded'
                 else 'darkorange' if f == 'is_heavy'
                 else 'steelblue' for f in feat_imp.index]
    bars = ax.barh(feat_imp.index, feat_imp.values,
                   color=fi_colors, edgecolor='white')
    for bar, val in zip(bars, feat_imp.values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.set_xlabel('Importance Score')
    ax.xaxis.grid(True, alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Full results table
    st.markdown("**Full Results Table**")
    disp = res_df.copy()
    disp['R²']   = disp['R²'].apply(lambda x: f"{x:.4f}" if x > -1000 else f"{x:,.0f}")
    disp['MAE']  = disp['MAE'].apply(lambda x: f"{x:,}")
    disp['RMSE'] = disp['RMSE'].apply(lambda x: f"{x:,}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption("NSW Traffic Intelligence · Random Forest R²=0.9853 · 265,291 records · 1,967 stations · 2006–2025")
