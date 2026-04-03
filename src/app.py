"""Streamlit demo app with kid-friendly dashboard and alerts."""

import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from src.config import DEFAULT_OUTPUT
    from src.data_prep import preprocess
    from src.features import add_features
except ModuleNotFoundError:
    # Fallback for platforms that run this file as a script from within src/
    from config import DEFAULT_OUTPUT
    from data_prep import preprocess
    from features import add_features


st.set_page_config(page_title="Smart Energy Tracker", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    :root {
        --primary-color: #00D9FF;
        --secondary-color: #FFB703;
        --success-color: #06D6A0;
        --danger-color: #EF476F;
        --dark-bg: #0F1419;
        --card-bg: #1A1F2E;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2C5F8D 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #00D9FF;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #00D9FF;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        color: #B0BEC5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .section-header {
        font-size: 1.8em;
        font-weight: bold;
        color: #00D9FF;
        margin: 30px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #FFB703;
    }
    
    .tip-card {
        background: linear-gradient(135deg, #1F4620 0%, #2D5A2D 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #06D6A0;
        margin: 10px 0;
        color: #E0F7E0;
    }
    
    .alert-card {
        background: linear-gradient(135deg, #4A1F1F 0%, #6B2C2C 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #EF476F;
        margin: 10px 0;
        color: #FFD6D6;
    }
    
    .progress-bar {
        height: 12px;
        border-radius: 10px;
        background: linear-gradient(90deg, #00D9FF, #FFB703);
    }
    </style>
    """, unsafe_allow_html=True)

# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# ⚡")
with col2:
    st.markdown("<h1 style='color: #00D9FF; margin-bottom: 0;'>Smart Energy Tracker</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #B0BEC5; margin-top: 5px; font-size: 1.1em;'>Real-time energy monitoring & analytics for your home</p>", unsafe_allow_html=True)

def window_usage(df: pd.DataFrame, end_date: date, days: int) -> tuple[float, float]:
    start = end_date - timedelta(days=days - 1)
    window = df[(df["date"] >= start) & (df["date"] <= end_date)]
    usage = float(window["usage_kwh"].sum()) if not window.empty else 0.0
    cost = float(window["cost"].sum()) if not window.empty else 0.0
    return usage, cost

def appliance_window_summary(df: pd.DataFrame, pick_date: date) -> pd.DataFrame:
    day_mask = df["date"] == pick_date
    week_mask = (df["date"] >= pick_date - timedelta(days=6)) & (df["date"] <= pick_date)
    month_mask = (df["date"] >= pick_date - timedelta(days=29)) & (df["date"] <= pick_date)

    daily = df[day_mask].groupby("appliance_name")["usage_kwh"].sum()
    weekly = df[week_mask].groupby("appliance_name")["usage_kwh"].sum()
    monthly = df[month_mask].groupby("appliance_name")["usage_kwh"].sum()
    weekly_cost = df[week_mask].groupby("appliance_name")["cost"].sum()

    summary = pd.DataFrame({
        "daily_kwh": daily,
        "weekly_kwh": weekly,
        "monthly_kwh": monthly,
        "weekly_cost": weekly_cost,
    }).fillna(0)

    return summary.sort_values("weekly_kwh", ascending=False)

def generate_recommendations(
    df: pd.DataFrame,
    pick_date: date,
    usage_today: float,
    cost_today: float,
    usage_limit: float,
    cost_limit: float,
) -> list[str]:
    recs: list[str] = []
    week_df = df[(df["date"] >= pick_date - timedelta(days=6)) & (df["date"] <= pick_date)]

    if usage_today > usage_limit:
        recs.append(f"Reduce usage later today to stay under {usage_limit:.2f} kWh.")
    if cost_today > cost_limit:
        recs.append(f"Cost already passed ₹{cost_limit:.2f}; shift any non-urgent loads to off-peak hours.")

    if not week_df.empty:
        peak_usage = week_df[week_df.get("is_peak", 0) == 1]["usage_kwh"].sum()
        total_week_usage = max(week_df["usage_kwh"].sum(), 1e-6)
        if peak_usage / total_week_usage > 0.55:
            recs.append("Over half of usage is in peak (5-10pm). Run washers/chargers earlier in the day.")

        appliance_week = week_df.groupby("appliance_name")["usage_kwh"].sum().sort_values(ascending=False)
        if not appliance_week.empty:
            top_app = appliance_week.index[0]
            top_val = appliance_week.iloc[0]
            recs.append(f"{top_app} used {top_val:.2f} kWh this week. Try shorter cycles or smart plugs to cap standby draw.")

        daily_totals = week_df.groupby("date")["usage_kwh"].sum()
        if not daily_totals.empty:
            weekday_avg = daily_totals.mean()
            if usage_today > weekday_avg + 0.5:
                recs.append("Today's use is above your 7-day average; unplug idle devices overnight.")

    if not recs:
        recs.append("Nice work staying within limits. Keep stacking off-peak runs and brief appliance cycles.")

    return recs[:4]

uploaded = st.file_uploader("Upload processed CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info(f"Using default dataset: {DEFAULT_OUTPUT}")
    try:
        df = pd.read_csv(DEFAULT_OUTPUT)
    except FileNotFoundError:
        st.warning("No data available. Upload a processed CSV to continue.")
        st.stop()

with st.spinner("Cleaning data"):
    df = preprocess(df)
    df = add_features(df)

st.subheader("Dashboard")

# Sidebar: Limits & Alerts
with st.sidebar:
    st.markdown("<h2 style='color: #00D9FF; text-align: center;'>⚙️ Settings & Limits</h2>", unsafe_allow_html=True)
    st.divider()
    pick_date = st.date_input("📅 Pick a day", value=pd.to_datetime(df["timestamp"]).dt.date.max())
    st.divider()
    st.markdown("<p style='color: #FFB703; font-weight: bold;'>💡 Daily Limits</p>", unsafe_allow_html=True)
    usage_limit = st.number_input("⚡ Energy limit (kWh)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
    cost_limit = st.number_input("💰 Cost limit (₹)", min_value=1.0, max_value=1000.0, value=20.0, step=1.0)

# Compute today's usage/cost
df["date"] = pd.to_datetime(df["timestamp"]).dt.date
today_df = df[df["date"] == pick_date]
usage_today = float(today_df["usage_kwh"].sum()) if not today_df.empty else 0.0
cost_today = float(today_df["cost"].sum()) if not today_df.empty else 0.0
week_usage, week_cost = window_usage(df, pick_date, 7)
month_usage, month_cost = window_usage(df, pick_date, 30)

# Status determination
status_ok = usage_today <= usage_limit and cost_today <= cost_limit
status_text = "✅ All Good" if status_ok else "⚠️ Limit Exceeded"
status_color = "✅" if status_ok else "🚨"

# Top metrics with beautiful styling
st.markdown("<div class='section-header'>📊 Today's Overview</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>⚡ Today Energy</div>
        <div class='metric-value'>{usage_today:.2f}</div>
        <div style='color: #B0BEC5; font-size: 0.9em;'>kWh / {usage_limit:.2f} kWh</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>💰 Today Cost</div>
        <div class='metric-value'>₹{cost_today:.2f}</div>
        <div style='color: #B0BEC5; font-size: 0.9em;'>/ ₹{cost_limit:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Status</div>
        <div class='metric-value'>{status_color}</div>
        <div style='color: #B0BEC5; font-size: 0.9em;'>{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

# Progress bars
st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

progress_usage = min(usage_today / max(usage_limit, 1e-6), 1.0)
progress_cost = min(cost_today / max(cost_limit, 1e-6), 1.0)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<p style='color: #00D9FF; font-weight: bold; margin-bottom: 8px;'>⚡ Energy Usage Progress</p>", unsafe_allow_html=True)
    st.progress(progress_usage, text=f"{usage_today:.2f} / {usage_limit:.2f} kWh ({progress_usage*100:.0f}%)")

with col2:
    st.markdown(f"<p style='color: #FFB703; font-weight: bold; margin-bottom: 8px;'>💰 Cost Progress</p>", unsafe_allow_html=True)
    st.progress(progress_cost, text=f"₹{cost_today:.2f} / ₹{cost_limit:.2f} ({progress_cost*100:.0f}%)")

st.markdown("---")

# Period Estimates
st.markdown("<div class='section-header'>📈 Usage & Cost Trends</div>", unsafe_allow_html=True)

col_est1, col_est2, col_est3 = st.columns(3, gap="medium")
with col_est1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>📅 Daily</div>
        <div class='metric-value'>{usage_today:.2f}</div>
        <div style='color: #B0BEC5; font-size: 0.9em;'>kWh</div>
    </div>
    """, unsafe_allow_html=True)

with col_est2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>📊 7-Day Total</div>
        <div class='metric-value'>{week_usage:.2f}</div>
        <div style='color: #B0BEC5; font-size: 0.9em;'>kWh</div>
    </div>
    """, unsafe_allow_html=True)

with col_est3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>📆 30-Day Total</div>
        <div class='metric-value'>{month_usage:.2f}</div>
        <div style='color: #B0BEC5; font-size: 0.9em;'>kWh</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"<p style='color: #B0BEC5; text-align: center; margin-top: 15px;'>7-day cost: <span style='color: #FFB703; font-weight: bold;'>₹{week_cost:.2f}</span> | 30-day cost: <span style='color: #FFB703; font-weight: bold;'>₹{month_cost:.2f}</span></p>", unsafe_allow_html=True)

# Daily goal badge
def goal_badge(value: float, limit: float, label: str) -> None:
    ratio = value / max(limit, 1e-6)
    if ratio <= 0.5:
        st.markdown(f"<div class='tip-card'>🏅 <b>Excellent!</b> {label} usage is well under half the limit. Keep it up!</div>", unsafe_allow_html=True)
    elif ratio <= 1.0:
        st.markdown(f"<div class='alert-card'>⚠️ <b>Careful!</b> {label} usage is over half but within limit. Monitor closely.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='alert-card'>🚨 <b>Alert!</b> {label} limit exceeded by {(ratio-1)*100:.0f}%!</div>", unsafe_allow_html=True)

st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📋 Performance Status</div>", unsafe_allow_html=True)
goal_badge(usage_today, usage_limit, "Energy")
goal_badge(cost_today, cost_limit, "Cost")

def trigger_alert(msg: str) -> None:
    st.warning(msg)
    st.toast(msg)

# Alert triggers
if usage_today > usage_limit:
    st.markdown(f"""
    <div class='alert-card'>
        <b>🚨 Energy Limit Exceeded!</b><br>
        Current: {usage_today:.2f} kWh | Limit: {usage_limit:.2f} kWh | Over by: {usage_today - usage_limit:.2f} kWh
    </div>
    """, unsafe_allow_html=True)

if cost_today > cost_limit:
    st.markdown(f"""
    <div class='alert-card'>
        <b>🚨 Cost Limit Exceeded!</b><br>
        Current: ₹{cost_today:.2f} | Limit: ₹{cost_limit:.2f} | Over by: ₹{cost_today - cost_limit:.2f}
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🔔 Test Alert", use_container_width=True):
        trigger_alert("✅ Test alert: System is working perfectly!")

st.markdown("---")

# Energy Trends Chart
st.markdown("<div class='section-header'>📊 Energy & Cost Trends</div>", unsafe_allow_html=True)
chart_df = df.sort_values("timestamp")[ ["timestamp", "usage_kwh", "cost"] ].set_index("timestamp")

# Create custom chart with colors
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor='#0F1419')
ax1.plot(chart_df.index, chart_df['usage_kwh'], color='#00D9FF', linewidth=2.5, label='Usage (kWh)')
ax1.fill_between(chart_df.index, chart_df['usage_kwh'], alpha=0.2, color='#00D9FF')
ax1.set_ylabel('Usage (kWh)', color='#B0BEC5')
ax1.set_facecolor('#1A1F2E')
ax1.grid(alpha=0.2, color='#00D9FF')
ax1.tick_params(colors='#B0BEC5')
ax1.set_title('Energy Usage Over Time', color='#00D9FF', fontsize=12, fontweight='bold')

ax2.plot(chart_df.index, chart_df['cost'], color='#FFB703', linewidth=2.5, label='Cost (₹)')
ax2.fill_between(chart_df.index, chart_df['cost'], alpha=0.2, color='#FFB703')
ax2.set_ylabel('Cost (₹)', color='#B0BEC5')
ax2.set_xlabel('Time', color='#B0BEC5')
ax2.set_facecolor('#1A1F2E')
ax2.grid(alpha=0.2, color='#FFB703')
ax2.tick_params(colors='#B0BEC5')
ax2.set_title('Cost Over Time', color='#FFB703', fontsize=12, fontweight='bold')

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# Weekly summary (last 7 days relative to selected date)
st.markdown("<div class='section-header'>📅 Weekly Summary (Last 7 Days)</div>", unsafe_allow_html=True)
start_day = pick_date - timedelta(days=6)
week_df = df[(df["date"] >= start_day) & (df["date"] <= pick_date)].copy()
if not week_df.empty:
    daily = week_df.groupby("date").agg({"usage_kwh": "sum", "cost": "sum"}).sort_index()
    
    # Beautiful bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor='#0F1419')
    
    ax1.bar(range(len(daily)), daily['usage_kwh'], color='#00D9FF', alpha=0.8, edgecolor='#00D9FF')
    ax1.set_xlabel('Day', color='#B0BEC5')
    ax1.set_ylabel('Usage (kWh)', color='#B0BEC5')
    ax1.set_facecolor('#1A1F2E')
    ax1.set_title('Daily Energy Usage', color='#00D9FF', fontweight='bold')
    ax1.tick_params(colors='#B0BEC5')
    ax1.grid(axis='y', alpha=0.2, color='#00D9FF')
    
    ax2.bar(range(len(daily)), daily['cost'], color='#FFB703', alpha=0.8, edgecolor='#FFB703')
    ax2.set_xlabel('Day', color='#B0BEC5')
    ax2.set_ylabel('Cost (₹)', color='#B0BEC5')
    ax2.set_facecolor('#1A1F2E')
    ax2.set_title('Daily Cost', color='#FFB703', fontweight='bold')
    ax2.tick_params(colors='#B0BEC5')
    ax2.grid(axis='y', alpha=0.2, color='#FFB703')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    col_w1, col_w2 = st.columns(2, gap="medium")
    with col_w1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>⚡ 7-Day kWh</div>
            <div class='metric-value'>{daily['usage_kwh'].sum():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_w2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>💰 7-Day Cost</div>
            <div class='metric-value'>₹{daily['cost'].sum():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("📊 No data in the selected 7-day window.")

st.markdown("<div class='section-header'>🔌 Appliance Insights</div>", unsafe_allow_html=True)
st.markdown("<p style='color: #B0BEC5; font-size: 0.95em;'>Select appliances to analyze their usage patterns and costs</p>", unsafe_allow_html=True)

appliance_options = sorted(df.get("appliance_name", pd.Series()).dropna().unique())
default_appliances = appliance_options[: min(4, len(appliance_options))]
selected_appliances = st.multiselect(
    "🏠 Focus on appliances",
    options=appliance_options,
    default=list(default_appliances),
    help="Filter the charts and table to the appliances you care about",
)
appliance_df = df[df["appliance_name"].isin(selected_appliances)] if selected_appliances else df
app_summary = appliance_window_summary(appliance_df, pick_date)

if not app_summary.empty:
    # Beautiful dataframe styling
    st.markdown("<p style='color: #00D9FF; font-weight: bold; margin-top: 15px;'>📊 Appliance Usage Summary</p>", unsafe_allow_html=True)
    st.dataframe(
        app_summary[["daily_kwh", "weekly_kwh", "monthly_kwh", "weekly_cost"]].round(2),
        use_container_width=True,
        height=300
    )
    
    # Appliance charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0F1419')
        app_summary[["weekly_kwh"]].sort_values("weekly_kwh").plot(kind='barh', ax=ax, color='#00D9FF', edgecolor='#00D9FF')
        ax.set_xlabel('Weekly Usage (kWh)', color='#B0BEC5')
        ax.set_title('Weekly Energy by Appliance', color='#00D9FF', fontweight='bold')
        ax.set_facecolor('#1A1F2E')
        ax.tick_params(colors='#B0BEC5')
        ax.grid(axis='x', alpha=0.2, color='#00D9FF')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0F1419')
        app_summary[["weekly_cost"]].sort_values("weekly_cost").plot(kind='barh', ax=ax, color='#FFB703', edgecolor='#FFB703')
        ax.set_xlabel('Weekly Cost (₹)', color='#B0BEC5')
        ax.set_title('Weekly Cost by Appliance', color='#FFB703', fontweight='bold')
        ax.set_facecolor('#1A1F2E')
        ax.tick_params(colors='#B0BEC5')
        ax.grid(axis='x', alpha=0.2, color='#FFB703')
        st.pyplot(fig, use_container_width=True)

    # Area chart for weekly breakdown
    last_week = appliance_df[(appliance_df["date"] >= pick_date - timedelta(days=6)) & (appliance_df["date"] <= pick_date)]
    if not last_week.empty:
        pivot_week = last_week.pivot_table(
            index="timestamp",
            columns="appliance_name",
            values="usage_kwh",
            aggfunc="sum",
        ).fillna(0)
        
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0F1419')
        ax.stackplot(pivot_week.index, *[pivot_week[col] for col in pivot_week.columns], 
                     labels=pivot_week.columns, alpha=0.7)
        ax.set_xlabel('Time', color='#B0BEC5')
        ax.set_ylabel('Usage (kWh)', color='#B0BEC5')
        ax.set_title('Appliance Usage Breakdown Over Time', color='#00D9FF', fontweight='bold')
        ax.set_facecolor('#1A1F2E')
        ax.tick_params(colors='#B0BEC5')
        ax.legend(loc='upper left', facecolor='#1A1F2E', edgecolor='#B0BEC5')
        ax.grid(alpha=0.2, color='#00D9FF')
        st.pyplot(fig, use_container_width=True)
else:
    st.info("📊 No appliance data available for the selected window.")

st.markdown("<div class='section-header'>💡 Smart Recommendations</div>", unsafe_allow_html=True)
recommendations = generate_recommendations(df, pick_date, usage_today, cost_today, usage_limit, cost_limit)

if recommendations:
    for i, tip in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class='tip-card'>
            <b>💡 Tip {i}:</b> {tip}
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='tip-card'>
        <b>🎉 Great Job!</b> No recommendations at this time. Keep maintaining your energy habits!
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='section-header'>🧩 Appliance Clustering Analysis</div>", unsafe_allow_html=True)
st.markdown("<p style='color: #B0BEC5; font-size: 0.95em;'>Discover usage patterns and group similar appliances together</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    n_clusters = st.slider("🎚️ Number of clusters", min_value=2, max_value=7, value=3)
with col2:
    run_clustering = st.button("▶️ Run Clustering", use_container_width=True)

if run_clustering:
    with st.spinner("🔄 Analyzing appliance patterns..."):
        feature_cols = [c for c in df.columns if c not in {"timestamp", "appliance_name", "appliance_id"}]
        X = df[feature_cols].select_dtypes(include=["number", "bool"]).fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        df_clustered = df.copy()
        df_clustered["cluster"] = labels

        # Beautiful clustering visualization
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0F1419')
        cluster_means = df_clustered.groupby("cluster")["usage_kwh"].mean()
        colors = ['#00D9FF', '#FFB703', '#06D6A0', '#EF476F', '#FF006E', '#8338EC', '#3A86FF']
        bars = ax.bar(cluster_means.index, cluster_means.values, 
                     color=[colors[i % len(colors)] for i in range(len(cluster_means))],
                     edgecolor='white', linewidth=2, alpha=0.85)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', color='#B0BEC5', fontweight='bold')
        
        ax.set_xlabel('Cluster', color='#B0BEC5', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Usage (kWh)', color='#B0BEC5', fontsize=11, fontweight='bold')
        ax.set_title('Mean Energy Usage by Cluster', color='#00D9FF', fontsize=13, fontweight='bold')
        ax.set_facecolor('#1A1F2E')
        ax.tick_params(colors='#B0BEC5')
        ax.grid(axis='y', alpha=0.2, color='#00D9FF')
        
        st.pyplot(fig, use_container_width=True)

        # Clustering metrics
        col1, col2, col3 = st.columns(3, gap="medium")
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>📊 Inertia Score</div>
                <div class='metric-value'>{kmeans.inertia_:.2f}</div>
                <div style='color: #B0BEC5; font-size: 0.9em;'>Lower is better</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>🎯 Number of Clusters</div>
                <div class='metric-value'>{n_clusters}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>📈 Total Points</div>
                <div class='metric-value'>{len(df_clustered)}</div>
            </div>
            """, unsafe_allow_html=True)

        # Cluster statistics table
        st.markdown("<p style='color: #00D9FF; font-weight: bold; margin-top: 20px;'>📋 Cluster Statistics</p>", unsafe_allow_html=True)
        cluster_stats = df_clustered.groupby("cluster").agg({
            "usage_kwh": "mean",
            "cost": "mean",
            "appliance_name": "count"
        }).rename(columns={"appliance_name": "count"}).round(4)
        
        st.dataframe(
            cluster_stats,
            use_container_width=True,
            height=250
        )
        
        st.success("✅ Clustering analysis complete!")

