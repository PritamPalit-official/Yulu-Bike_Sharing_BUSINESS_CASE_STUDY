import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from scipy import stats

# Page Configuration
st.set_page_config(
    page_title="🚲 Yulu Rental Demand Analyzer",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Yulu themed: Teal #00A896 & dark gray)
st.markdown("""
<style>
    .reportview-container {
        background: #111111;
    }
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #00A896;
        font-size: 38px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .accent-header {
        color: #f5f5f1;
        font-size: 16px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #1f1f1f;
        border-left: 5px solid #00A896;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-card-grey {
        background-color: #1f1f1f;
        border-left: 5px solid #564d4d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-val {
        color: #f5f5f1;
        font-size: 28px;
        font-weight: 800;
    }
    .metric-lbl {
        color: #b3b3b3;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
</style>
""", unsafe_allowed_html=True)

# Data Loading (Cached for performance)
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'yulu_data.csv')
    df = pd.read_csv(csv_path)
    
    # Datetime conversions
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    
    # Categorical labeling for UI
    df["Season_Name"] = df["season"].map({1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"})
    df["Weather_Name"] = df["weather"].map({
        1: "Clear / Partly Cloudy",
        2: "Mist / Cloudy",
        3: "Light Rain / Snow",
        4: "Heavy Rain / Ice Pallets"
    })
    df["Working_Day_Label"] = df["workingday"].map({0: "Weekend/Holiday", 1: "Working Day"})
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading yulu_data.csv: {e}")
    st.stop()

# ── Sidebar Filter Setup ──────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/ea/Bicycle_icon_completed.svg", width=60)
st.sidebar.title("🚲 Filter Rental Data")

# 1. Season Filter
seasons = ["Spring", "Summer", "Fall", "Winter"]
selected_seasons = st.sidebar.multiselect("Seasons", seasons, default=seasons)

# 2. Weather Filter
weathers = list(df["Weather_Name"].dropna().unique())
selected_weathers = st.sidebar.multiselect("Weather Conditions", weathers, default=weathers)

# 3. Working Day Filter
working_options = ["Working Day", "Weekend/Holiday"]
selected_working = st.sidebar.multiselect("Day Type", working_options, default=working_options)

# 4. Temperature Slider
min_temp = float(df["temp"].min())
max_temp = float(df["temp"].max())
selected_temp_range = st.sidebar.slider(
    "Temperature (°C)", 
    min_temp, 
    max_temp, 
    (min_temp, max_temp)
)

# Apply filters
df_filtered = df.copy()

if selected_seasons:
    df_filtered = df_filtered[df_filtered["Season_Name"].isin(selected_seasons)]
if selected_weathers:
    df_filtered = df_filtered[df_filtered["Weather_Name"].isin(selected_weathers)]
if selected_working:
    df_filtered = df_filtered[df_filtered["Working_Day_Label"].isin(selected_working)]

df_filtered = df_filtered[
    (df_filtered["temp"] >= selected_temp_range[0]) & 
    (df_filtered["temp"] <= selected_temp_range[1])
]

# Page Header
st.markdown("<div class='main-header'>🚲 Yulu Rental Demand Analyzer</div>", unsafe_allowed_html=True)
st.markdown("<div class='accent-header'>Operational metrics, hourly demand profiles, and live hypothesis testing sandbox</div>", unsafe_allowed_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_rentals = df_filtered["count"].sum() if not df_filtered.empty else 0
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val'>{total_rentals:,}</div>
        <div class='metric-lbl'>Total Bike Rentals</div>
    </div>
    """, unsafe_allowed_html=True)

with col2:
    avg_rentals = df_filtered["count"].mean() if not df_filtered.empty else 0
    st.markdown(f"""
    <div class='metric-card-grey'>
        <div class='metric-val'>{avg_rentals:.1f} / hr</div>
        <div class='metric-lbl'>Average Hourly Rentals</div>
    </div>
    """, unsafe_allowed_html=True)

with col3:
    total_registered = df_filtered["registered"].sum() if not df_filtered.empty else 0
    reg_pct = (total_registered / total_rentals * 100) if total_rentals > 0 else 0
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val'>{reg_pct:.1f}%</div>
        <div class='metric-lbl'>Registered User Share</div>
    </div>
    """, unsafe_allowed_html=True)

with col4:
    total_casual = df_filtered["casual"].sum() if not df_filtered.empty else 0
    cas_pct = (total_casual / total_rentals * 100) if total_rentals > 0 else 0
    st.markdown(f"""
    <div class='metric-card-grey'>
        <div class='metric-val'>{cas_pct:.1f}%</div>
        <div class='metric-lbl'>Casual User Share</div>
    </div>
    """, unsafe_allowed_html=True)

st.markdown("<br>", unsafe_allowed_html=True)

# Tabs setup
tab1, tab2, tab3 = st.tabs(["📊 Demand Trends", "🧪 Hypothesis Testing Sandbox", "📂 Data Explorer & Recommendations"])

# ── TAB 1: Demand Trends ──────────────────────────────────────────────────
with tab1:
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("⏰ Hourly Demand Profile (Working Day vs. Weekend)")
        if not df_filtered.empty:
            hourly_summary = df_filtered.groupby(["hour", "Working_Day_Label"])["count"].mean().reset_index()
            fig_hourly = px.line(
                hourly_summary,
                x="hour",
                y="count",
                color="Working_Day_Label",
                markers=True,
                color_discrete_sequence=["#00A896", "#FF4B2B"],
                template="plotly_dark",
                labels={"count": "Average Rentals", "hour": "Hour of the Day (24h)"}
            )
            fig_hourly.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340)
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            st.info("No data matches current filters.")
            
    with row1_col2:
        st.subheader("🌡️ Temperature vs. Rental Volume")
        if not df_filtered.empty:
            # Sample data points if too large to prevent slow render
            sample_df = df_filtered.sample(n=min(len(df_filtered), 2000), random_state=42)
            fig_temp = px.scatter(
                sample_df,
                x="temp",
                y="count",
                color="Season_Name",
                trendline="ols",
                color_discrete_sequence=px.colors.qualitative.Safe,
                template="plotly_dark",
                labels={"count": "Hourly Rentals", "temp": "Temperature (°C)"}
            )
            fig_temp.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340)
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("No data matches current filters.")

    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.subheader("🌦️ Rental Counts across Weather Conditions")
        if not df_filtered.empty:
            fig_weather_box = px.box(
                df_filtered,
                x="Weather_Name",
                y="count",
                color="Weather_Name",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_dark",
                labels={"count": "Hourly Rentals", "Weather_Name": "Weather Type"}
            )
            fig_weather_box.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340, showlegend=False)
            st.plotly_chart(fig_weather_box, use_container_width=True)
        else:
            st.info("No data available.")
            
    with row2_col2:
        st.subheader("🍂 Seasonal Rental Distributions")
        if not df_filtered.empty:
            fig_season_box = px.box(
                df_filtered,
                x="Season_Name",
                y="count",
                color="Season_Name",
                color_discrete_sequence=["#FFD166", "#06D6A0", "#118AB2", "#073B4C"],
                template="plotly_dark",
                labels={"count": "Hourly Rentals", "Season_Name": "Season"}
            )
            fig_season_box.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340, showlegend=False)
            st.plotly_chart(fig_season_box, use_container_width=True)
        else:
            st.info("No data available.")

# ── TAB 2: Hypothesis Testing Sandbox ─────────────────────────────────────
with tab2:
    st.subheader("🧪 Live Hypothesis Testing Sandbox")
    st.markdown("""
    Perform standard statistical hypothesis tests on the Yulu rental database dynamically. 
    Select a test below to calculate test statistics, p-values, and view statistical conclusions.
    """)
    
    test_type = st.selectbox("Select Statistical Test", [
        "2-Sample t-test (Working Day vs. Weekend/Holiday)",
        "One-way ANOVA (Weather impact on Rentals)",
        "One-way ANOVA (Season impact on Rentals)",
        "Chi-Square Test (Season vs. Weather Independence)"
    ], key="yulu_test_select")
    
    run_test = st.button("Run Hypothesis Test 🔬", use_container_width=True, key="yulu_run_test")
    
    if run_test:
        st.markdown("### 📊 Test Results")
        
        if test_type == "2-Sample t-test (Working Day vs. Weekend/Holiday)":
            group1 = df[df["workingday"] == 1]["count"]
            group2 = df[df["workingday"] == 0]["count"]
            
            stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
            st.write(f"**Test Statistic (t-value)**: `{stat:.4f}`")
            st.write(f"**P-Value**: `{p_val:.6e}`")
            
            if p_val < 0.05:
                st.success("✅ **Reject Null Hypothesis (Significant)**: There is a statistically significant difference in bike rentals between working days and weekends/holidays.")
            else:
                st.warning("⚠️ **Fail to Reject Null Hypothesis (Not Significant)**: No statistically significant difference in bike rentals was found between working days and weekends/holidays.")
                
        elif test_type == "One-way ANOVA (Weather impact on Rentals)":
            w1 = df[df["weather"] == 1]["count"]
            w2 = df[df["weather"] == 2]["count"]
            w3 = df[df["weather"] == 3]["count"]
            w4 = df[df["weather"] == 4]["count"]
            
            groups = [w for w in [w1, w2, w3, w4] if len(w) > 0]
            stat, p_val = stats.f_oneway(*groups)
            st.write(f"**Test Statistic (F-value)**: `{stat:.4f}`")
            st.write(f"**P-Value**: `{p_val:.6e}`")
            
            if p_val < 0.05:
                st.success("✅ **Reject Null Hypothesis (Significant)**: Different weather conditions have a statistically significant impact on the hourly rental volume.")
            else:
                st.warning("⚠️ **Fail to Reject Null Hypothesis (Not Significant)**: Weather conditions do not show a statistically significant impact on rental volume.")
                
        elif test_type == "One-way ANOVA (Season impact on Rentals)":
            s1 = df[df["season"] == 1]["count"]
            s2 = df[df["season"] == 2]["count"]
            s3 = df[df["season"] == 3]["count"]
            s4 = df[df["season"] == 4]["count"]
            
            stat, p_val = stats.f_oneway(s1, s2, s3, s4)
            st.write(f"**Test Statistic (F-value)**: `{stat:.4f}`")
            st.write(f"**P-Value**: `{p_val:.6e}`")
            
            if p_val < 0.05:
                st.success("✅ **Reject Null Hypothesis (Significant)**: Seasons show a statistically significant difference in customer bike rental behavior.")
            else:
                st.warning("⚠️ **Fail to Reject Null Hypothesis (Not Significant)**: Seasons do not show a statistically significant difference in hourly rentals.")
                
        elif test_type == "Chi-Square Test (Season vs. Weather Independence)":
            contingency_table = pd.crosstab(df["season"], df["weather"])
            st.write("**Contingency Table (Season vs Weather Counts)**:")
            st.dataframe(contingency_table)
            
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
            st.write(f"**Chi-Square Statistic**: `{chi2:.4f}`")
            st.write(f"**P-Value**: `{p_val:.6e}`")
            st.write(f"**Degrees of Freedom**: `{dof}`")
            
            if p_val < 0.05:
                st.success("✅ **Reject Null Hypothesis (Dependent)**: Season and weather conditions are statistically dependent on each other.")
            else:
                st.warning("⚠️ **Fail to Reject Null Hypothesis (Independent)**: Weather distributions and seasons are statistically independent.")

# ── TAB 3: Data Explorer & Recommendations ────────────────────────────────
with tab3:
    # Strategic Insights Expander
    st.markdown("### 💡 Operational Insights & Business Actions")
    
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.markdown("""
        #### Key Findings
        1. **Weekday vs Weekend Demand**: Peak demand hours on working days align with office rush hours (**8:00 AM** and **5:00 PM–6:00 PM**). On weekends, demand is spread out, peaking between **12:00 PM and 4:00 PM**.
        2. **Weather Impact**: Clear weather (Weather 1) and mist (Weather 2) show high demand. Light rain/snow (Weather 3) severely drops demand, while heavy precipitation (Weather 4) halts rental activity.
        3. **Seasonal Variation**: Fall (Season 3) and Summer (Season 2) show the highest demand, while Spring (Season 1) shows the lowest demand.
        """)
        
    with rec_col2:
        st.markdown("""
        #### Actionable Recommendations
        - **Dynamic Fleets & Rebalancing**: Ensure maximum bike availability near major business parks and transit stations during peak commute hours (8 AM & 5-6 PM) on working days.
        - **Rainy Season Promos**: Offer promotions or safety-themed vouchers during light mist/rain to encourage ridership.
        - **Maintenance Schedules**: Perform major bike fleet maintenance during the winter/spring season when demand is naturally low.
        - **Dynamic Pricing**: Implement lower rates on weekend afternoons (12 PM-4 PM) to capture casual leisure riders.
        """)
        
    st.markdown("---")
    st.markdown("#### 📂 Rental Records Explorer")
    st.write(f"Displaying **{len(df_filtered):,}** records matching active filters.")
    
    display_cols = ["datetime", "Season_Name", "Working_Day_Label", "Weather_Name", "temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"]
    st.dataframe(df_filtered[display_cols], use_container_width=True)
    
    # Download Button
    csv = df_filtered[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="yulu_filtered_rentals.csv",
        mime="text/csv",
    )
