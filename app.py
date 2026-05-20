import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
 
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart City Traffic Forecaster",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
 
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
 
.main { background: #0f1117; }
 
.metric-card {
    background: linear-gradient(135deg, #1a1f2e, #252b3b);
    border: 1px solid #2d3348;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #4ade80; }
.metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
 
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #e2e8f0;
    border-left: 3px solid #4ade80;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}
 
.info-box {
    background: #1a2332;
    border: 1px solid #2d4a6e;
    border-radius: 8px;
    padding: 14px 18px;
    color: #93c5fd;
    font-size: 0.9rem;
    margin-bottom: 16px;
}
 
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1f2937;
}
 
div[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2d3348;
    border-radius: 10px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Smart City Traffic")
    st.markdown("**Project 9 — UCT ML Internship**")
    st.markdown("---")
 
    st.markdown("### 📂 Upload Data")
    train_file = st.file_uploader("Training CSV", type="csv", key="train")
    test_file  = st.file_uploader("Test CSV",     type="csv", key="test")
 
    st.markdown("---")
    st.markdown("### ⚙️ Forecast Settings")
    forecast_hours = st.slider("Forecast horizon (hours)", 1, 168, 24)
    junction_filter = st.multiselect("Junctions to show", [1, 2, 3, 4], default=[1, 2, 3, 4])
    show_decomp = st.checkbox("Show seasonal decomposition", value=False)
 
    st.markdown("---")
    st.markdown("### 📊 Model")
    model_choice = st.selectbox("Forecasting model", ["Moving Average", "Exponential Smoothing", "SARIMA (simple)"])
 
# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    # Flexible datetime parsing — infer_datetime_format removed in pandas 2.x
    dt_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if dt_cols:
        try:
            df[dt_cols[0]] = pd.to_datetime(df[dt_cols[0]])
        except Exception:
            df[dt_cols[0]] = pd.to_datetime(df[dt_cols[0]], format='mixed', dayfirst=False)
        df = df.rename(columns={dt_cols[0]: 'DateTime'})
    return df
 
def moving_average_forecast(series, window=12, steps=24):
    ma = series.rolling(window, min_periods=1).mean()
    last_val = ma.iloc[-1]
    forecast = [last_val] * steps
    return forecast
 
def exp_smoothing_forecast(series, alpha=0.3, steps=24):
    smoothed = series.ewm(alpha=alpha).mean()
    last = smoothed.iloc[-1]
    forecast = []
    val = last
    for _ in range(steps):
        val = alpha * series.iloc[-1] + (1 - alpha) * val
        forecast.append(val)
    return forecast
 
def sarima_forecast(series, steps=24):
    """Simple seasonal naive forecast (period=24h)"""
    period = 24
    tail = series.tail(period).values
    forecast = list(np.tile(tail, steps // period + 1))[:steps]
    return forecast
 
def get_forecast(series, model, steps):
    if model == "Moving Average":
        return moving_average_forecast(series, steps=steps)
    elif model == "Exponential Smoothing":
        return exp_smoothing_forecast(series, steps=steps)
    else:
        return sarima_forecast(series, steps=steps)
 
plt.style.use('dark_background')
COLORS = ['#4ade80', '#60a5fa', '#f472b6', '#fb923c']
 
# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("# 🚦 Smart City Traffic Pattern Forecaster")
st.markdown("Forecast traffic volumes across city junctions using historical sensor data.")
 
# ── Load data ─────────────────────────────────────────────────────────────────
if train_file is None:
    st.markdown('<div class="info-box">👆 Upload your <strong>train_aWnotuB.csv</strong> (and optionally the test CSV) from the sidebar to get started.</div>', unsafe_allow_html=True)
 
    # Demo mode with synthetic data
    st.markdown("### 🔬 Demo Mode — Synthetic Data")
    np.random.seed(42)
    n = 24 * 90  # 90 days hourly
    dates = pd.date_range("2021-01-01", periods=n, freq="H")
    demo_rows = []
    for j in range(1, 5):
        base = 20 + j * 5
        traffic = (base
                   + 10 * np.sin(2 * np.pi * np.arange(n) / 24)   # daily cycle
                   + 5  * np.sin(2 * np.pi * np.arange(n) / (24*7)) # weekly
                   + np.random.normal(0, 3, n)).clip(0)
        for i, d in enumerate(dates):
            demo_rows.append({'DateTime': d, 'Junction': j, 'Vehicles': int(traffic[i]), 'ID': i})
    df_train = pd.DataFrame(demo_rows)
    st.info("No file uploaded — showing synthetic demo data. Upload your CSV for real analysis.")
else:
    df_train = load_data(train_file)
 
# ── Detect columns ────────────────────────────────────────────────────────────
junc_col = next((c for c in df_train.columns if 'junction' in c.lower()), None)
veh_col  = next((c for c in df_train.columns if 'vehicle' in c.lower() or 'count' in c.lower() or 'traffic' in c.lower()), None)
dt_col   = 'DateTime' if 'DateTime' in df_train.columns else df_train.columns[0]
 
if junc_col and veh_col:
    df_train[junc_col] = df_train[junc_col].astype(int)
    df_train[veh_col]  = pd.to_numeric(df_train[veh_col], errors='coerce')
    df_train = df_train.dropna(subset=[veh_col])
    all_junctions = sorted(df_train[junc_col].unique())
    junctions = [j for j in junction_filter if j in all_junctions]
else:
    st.error("Could not detect Junction / Vehicles columns. Please check your CSV format.")
    st.stop()
 
# ── KPI metrics ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Overview Metrics</div>', unsafe_allow_html=True)
 
col1, col2, col3, col4 = st.columns(4)
total_records = len(df_train)
avg_traffic   = df_train[veh_col].mean()
peak_traffic  = df_train[veh_col].max()
date_range    = (df_train[dt_col].max() - df_train[dt_col].min()).days
 
with col1: st.metric("Total Records",    f"{total_records:,}")
with col2: st.metric("Avg Vehicles/hr",  f"{avg_traffic:.1f}")
with col3: st.metric("Peak Vehicles/hr", f"{peak_traffic:.0f}")
with col4: st.metric("Days of Data",     f"{date_range}")
 
# ── Traffic over time ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🕐 Traffic Volume Over Time</div>', unsafe_allow_html=True)
 
fig, ax = plt.subplots(figsize=(14, 4), facecolor='#0f1117')
ax.set_facecolor('#0f1117')
 
for i, j in enumerate(junctions):
    sub = df_train[df_train[junc_col] == j].set_index(dt_col)[veh_col].resample('H').mean()
    ax.plot(sub.index, sub.values, color=COLORS[i % 4], lw=1.2, alpha=0.85, label=f'Junction {j}')
 
ax.set_xlabel('Date', color='#94a3b8')
ax.set_ylabel('Vehicles / hr', color='#94a3b8')
ax.tick_params(colors='#94a3b8')
ax.spines[:].set_color('#2d3348')
ax.legend(facecolor='#1a1f2e', edgecolor='#2d3348', labelcolor='#e2e8f0')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
st.pyplot(fig)
plt.close()
 
# ── Heatmap: avg traffic by hour & day ───────────────────────────────────────
st.markdown('<div class="section-header">🗓️ Traffic Heatmap — Hour × Day of Week</div>', unsafe_allow_html=True)
 
tab_labels = [f"Junction {j}" for j in junctions]
tabs = st.tabs(tab_labels)
 
for tab, j in zip(tabs, junctions):
    with tab:
        sub = df_train[df_train[junc_col] == j].copy()
        sub['hour'] = sub[dt_col].dt.hour
        sub['dow']  = sub[dt_col].dt.day_name()
        pivot = sub.groupby(['hour', 'dow'])[veh_col].mean().unstack()
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        pivot = pivot[[d for d in day_order if d in pivot.columns]]
 
        fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0f1117')
        ax2.set_facecolor('#0f1117')
        sns.heatmap(pivot, ax=ax2, cmap='YlOrRd', linewidths=0.3,
                    cbar_kws={'label': 'Avg Vehicles'})
        ax2.set_title(f'Junction {j} — Avg Traffic by Hour & Day', color='#e2e8f0', pad=12)
        ax2.set_xlabel('Day of Week', color='#94a3b8')
        ax2.set_ylabel('Hour of Day', color='#94a3b8')
        ax2.tick_params(colors='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
 
# ── Peak hour analysis ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⏰ Peak Hour Distribution</div>', unsafe_allow_html=True)
 
df_train['hour'] = df_train[dt_col].dt.hour
fig3, axes = plt.subplots(1, len(junctions), figsize=(4 * len(junctions), 4), facecolor='#0f1117')
if len(junctions) == 1: axes = [axes]
 
for ax, j, color in zip(axes, junctions, COLORS):
    sub = df_train[df_train[junc_col] == j]
    hourly = sub.groupby('hour')[veh_col].mean()
    ax.set_facecolor('#0f1117')
    ax.bar(hourly.index, hourly.values, color=color, alpha=0.8, width=0.8)
    ax.set_title(f'Junction {j}', color='#e2e8f0')
    ax.set_xlabel('Hour', color='#94a3b8')
    ax.set_ylabel('Avg Vehicles', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.spines[:].set_color('#2d3348')
    peak = hourly.idxmax()
    ax.axvline(peak, color='white', lw=1.2, ls='--', alpha=0.6)
    ax.text(peak + 0.3, hourly.max() * 0.95, f'Peak: {peak}:00', color='white', fontsize=8)
 
plt.tight_layout()
st.pyplot(fig3)
plt.close()
 
# ── Forecast section ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔮 Traffic Forecast</div>', unsafe_allow_html=True)
 
st.markdown(f"Forecasting **{forecast_hours} hours** ahead using **{model_choice}**.")
 
fig4, ax4 = plt.subplots(figsize=(14, 5), facecolor='#0f1117')
ax4.set_facecolor('#0f1117')
 
for i, j in enumerate(junctions):
    sub = df_train[df_train[junc_col] == j].set_index(dt_col)[veh_col].resample('H').mean().dropna()
    history_tail = sub.tail(72)  # last 3 days
 
    forecast_vals = get_forecast(sub, model_choice, forecast_hours)
    last_dt = sub.index[-1]
    forecast_dates = pd.date_range(last_dt + timedelta(hours=1), periods=forecast_hours, freq='H')
 
    color = COLORS[i % 4]
    ax4.plot(history_tail.index, history_tail.values, color=color, lw=1.5, alpha=0.9, label=f'J{j} History')
    ax4.plot(forecast_dates, forecast_vals, color=color, lw=2, ls='--', alpha=0.7, label=f'J{j} Forecast')
    ax4.fill_between(forecast_dates,
                     [max(0, v * 0.85) for v in forecast_vals],
                     [v * 1.15 for v in forecast_vals],
                     color=color, alpha=0.08)
 
ax4.axvline(sub.index[-1], color='#f1f5f9', lw=1, ls=':', alpha=0.5)
ax4.text(sub.index[-1], ax4.get_ylim()[1] * 0.95, '  Now', color='#f1f5f9', fontsize=9)
ax4.set_xlabel('Date / Time', color='#94a3b8')
ax4.set_ylabel('Vehicles / hr', color='#94a3b8')
ax4.set_title('Traffic Forecast — All Junctions', color='#e2e8f0', pad=12)
ax4.tick_params(colors='#94a3b8')
ax4.spines[:].set_color('#2d3348')
ax4.legend(facecolor='#1a1f2e', edgecolor='#2d3348', labelcolor='#e2e8f0', ncol=2)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
plt.tight_layout()
st.pyplot(fig4)
plt.close()
 
# ── Forecast table ────────────────────────────────────────────────────────────
with st.expander("📋 Forecast Values Table"):
    forecast_df_rows = []
    last_dt = df_train[dt_col].max()
    for j in junctions:
        sub = df_train[df_train[junc_col] == j].set_index(dt_col)[veh_col].resample('H').mean().dropna()
        fvals = get_forecast(sub, model_choice, forecast_hours)
        for h, v in enumerate(fvals):
            forecast_df_rows.append({
                'DateTime': last_dt + timedelta(hours=h+1),
                'Junction': j,
                'Forecasted Vehicles': round(v, 1)
            })
    forecast_out = pd.DataFrame(forecast_df_rows)
    st.dataframe(forecast_out, use_container_width=True)
    csv = forecast_out.to_csv(index=False).encode()
    st.download_button("⬇️ Download Forecast CSV", csv, "traffic_forecast.csv", "text/csv")
 
# ── Weekly pattern ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📅 Weekly Traffic Pattern</div>', unsafe_allow_html=True)
 
df_train['dow_num'] = df_train[dt_col].dt.dayofweek
df_train['dow_name'] = df_train[dt_col].dt.day_name()
day_order_num = list(range(7))
day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
 
fig5, ax5 = plt.subplots(figsize=(10, 4), facecolor='#0f1117')
ax5.set_facecolor('#0f1117')
 
for i, j in enumerate(junctions):
    sub = df_train[df_train[junc_col] == j]
    weekly = sub.groupby('dow_num')[veh_col].mean()
    ax5.plot(weekly.index, weekly.values, marker='o', color=COLORS[i % 4], lw=2, label=f'Junction {j}')
 
ax5.set_xticks(range(7))
ax5.set_xticklabels(day_names, color='#94a3b8')
ax5.set_xlabel('Day of Week', color='#94a3b8')
ax5.set_ylabel('Avg Vehicles', color='#94a3b8')
ax5.set_title('Average Traffic by Day of Week', color='#e2e8f0', pad=10)
ax5.tick_params(colors='#94a3b8')
ax5.spines[:].set_color('#2d3348')
ax5.legend(facecolor='#1a1f2e', edgecolor='#2d3348', labelcolor='#e2e8f0')
ax5.axvspan(4.5, 6.5, color='#4ade80', alpha=0.05)
ax5.text(5, ax5.get_ylim()[1] * 0.98, 'Weekend', color='#4ade80', fontsize=8, ha='center')
plt.tight_layout()
st.pyplot(fig5)
plt.close()
 
# ── Test file comparison ──────────────────────────────────────────────────────
if test_file:
    st.markdown('<div class="section-header">🧪 Test Set Analysis</div>', unsafe_allow_html=True)
    df_test = load_data(test_file)
    st.write(f"Test set shape: **{df_test.shape[0]:,} rows × {df_test.shape[1]} cols**")
    st.dataframe(df_test.head(20), use_container_width=True)
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="color:#475569;font-size:0.82rem;text-align:center">'
    'Smart City Traffic Forecaster · Project 9 · UCT ML Internship · Built with Streamlit'
    '</p>',
    unsafe_allow_html=True
)
