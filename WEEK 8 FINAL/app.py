import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

st.set_page_config(
    page_title="AI Traffic Intelligence",
    page_icon="🚦",
    layout="wide"
)

sns.set_style("whitegrid")

# -----------------------------
# LOAD MODELS
# -----------------------------
time_model = pickle.load(open("travel_time_model.pkl","rb"))
cong_model = pickle.load(open("congestion_model.pkl","rb"))

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1AMBzflTbushZN3zzZL-pl390QCm3e5Qx"
    df = pd.read_csv(url)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["day"] = df["Date"].dt.day_name()

    return df

df = load_data()

# -----------------------------
# CURRENT TIME
# -----------------------------
now = datetime.now()
current_hour = now.hour
current_day = now.strftime("%A")

# -----------------------------
# HEADER
# -----------------------------
st.title("🚦 AI Traffic Intelligence Dashboard")
st.caption(f"📍 {current_day} | ⏰ {current_hour}:00 (Live Estimate)")

# -----------------------------
# AREA SELECTION
# -----------------------------
areas = sorted(df["Area Name"].unique())
area = st.sidebar.selectbox("Select Area", areas)

area_data = df[df["Area Name"] == area]

# -----------------------------
# SAME DAY FILTER
# -----------------------------
day_data = area_data[area_data["day"] == current_day]
if day_data.empty:
    day_data = area_data

# -----------------------------
# BASIC STATS
# -----------------------------
avg_congestion = day_data["Congestion Level"].mean()
avg_speed = day_data["Average Speed"].mean()
traffic_volume = day_data["Traffic Volume"].mean()
incident_factor = day_data["Incident Reports"].mean()

# -----------------------------
# ENCODING
# -----------------------------
weather_map = {"Clear":0,"Overcast":1,"Rain":2,"Fog":3}
roadwork_map = {"No":0,"Yes":1}

weather_code = weather_map.get(day_data["Weather Conditions"].mode()[0],0)
roadwork_code = roadwork_map.get(day_data["Roadwork and Construction Activity"].mode()[0],0)

# -----------------------------
# MODEL INPUT
# -----------------------------
features = pd.DataFrame([{
"Traffic Volume": traffic_volume,
"Average Speed": avg_speed,
"Road Capacity Utilization": day_data["Road Capacity Utilization"].mean(),
"Incident Reports": day_data["Incident Reports"].mean(),
"Environmental Impact": day_data["Environmental Impact"].mean(),
"Public Transport Usage": day_data["Public Transport Usage"].mean(),
"Traffic Signal Compliance": day_data["Traffic Signal Compliance"].mean(),
"Parking Usage": day_data["Parking Usage"].mean(),
"Pedestrian and Cyclist Count": day_data["Pedestrian and Cyclist Count"].mean(),
"Weather Conditions": weather_code,
"Roadwork and Construction Activity": roadwork_code,
"day": now.weekday()
}])

# -----------------------------
# PREDICTIONS
# -----------------------------
travel_time_index = time_model.predict(features)[0]
traffic_class = cong_model.predict(features)[0]

traffic_map = {0:"Low",1:"Medium",2:"High"}
traffic_label = traffic_map.get(traffic_class,"Unknown")
import numpy as np

base = avg_congestion

volume_factor = traffic_volume / 60000
speed_factor = 50 / max(avg_speed,1)
incident_factor = 1 + (incident_factor / 5)

# 🔥 generate area-specific peak shift
peak_shift = int((volume_factor * 6 + incident_factor * 4) % 24)

hourly_profile = []

for h in range(24):

    # dynamic peak centered at different hour per area
    peak_intensity = np.exp(-((h - peak_shift) ** 2) / 20)

    val = base * (0.6 + peak_intensity + speed_factor*0.1)

    hourly_profile.append(val)

hourly_df = pd.DataFrame({
    "hour": list(range(24)),
    "congestion": hourly_profile
})
# -----------------------------
# PEAK / BEST
# -----------------------------
peak_hour = int(hourly_df.loc[hourly_df["congestion"].idxmax(), "hour"])
best_hour = int(hourly_df.loc[hourly_df["congestion"].idxmin(), "hour"])

current_congestion = hourly_df.loc[
    hourly_df["hour"]==current_hour,"congestion"
].values[0]

# -----------------------------
# TRAFFIC SCORE (0–100)
# -----------------------------
traffic_score = min(100, (current_congestion + incident_factor*10))

# -----------------------------
# KPI CARDS
# -----------------------------
st.subheader("📊 Live Traffic Overview")

c1,c2,c3,c4 = st.columns(4)

c1.metric("Traffic Level", traffic_label)
c2.metric("Congestion Score", f"{traffic_score:.0f}/100")
c3.metric("Avg Speed", f"{avg_speed:.1f} km/h")
c4.metric("Travel Time Index", f"{travel_time_index:.2f}")

# -----------------------------
# INSIGHTS
# -----------------------------
st.subheader("🧠 Smart Insights")

if traffic_score > 75:
    st.error("🚨 Severe congestion — avoid travel if possible.")
elif traffic_score > 50:
    st.warning("⚠ Moderate congestion — expect delays.")
else:
    st.success("✅ Traffic conditions are smooth.")

if incident_factor > 2:
    st.error("⚠ High accident probability zone.")

if avg_speed < 30:
    st.warning("🚗 Low speeds indicate heavy traffic.")

st.write(f"📍 Peak Hour: **{peak_hour}:00**")
st.write(f"📍 Best Hour: **{best_hour}:00**")


# -----------------------------
# HOURLY CHART
# -----------------------------
st.subheader("📈 Hourly Congestion Trend")


fig, ax = plt.subplots(figsize=(8,4))

ax.plot(
    hourly_df["hour"],
    hourly_df["congestion"],
    marker="o",
    linewidth=2
)

# highlight current hour
ax.axvline(current_hour, linestyle="--")

# highlight peak
ax.scatter(
    peak_hour,
    hourly_df.loc[hourly_df["hour"]==peak_hour,"congestion"],
    s=120
)

ax.set_title(f"{area} - Hourly Congestion Pattern")

st.pyplot(fig)

# -----------------------------
# ACCIDENT PRONE
# -----------------------------
st.subheader("⚠ Accident Prone Intersections")

accidents = (
    day_data.groupby("Road/Intersection Name")["Incident Reports"]
    .sum()
    .sort_values(ascending=False)
    .head(8)
)

st.bar_chart(accidents)

# -----------------------------
# HIGH TRAFFIC ROADS
# -----------------------------
st.subheader("🚗 High Traffic Roads")

traffic = (
    day_data.groupby("Road/Intersection Name")["Traffic Volume"]
    .mean()
    .sort_values(ascending=False)
    .head(8)
)

st.bar_chart(traffic)

# -----------------------------
# CONGESTION DRIVERS
# -----------------------------
st.subheader("📉 Congestion Drivers")

corr = day_data[[
"Traffic Volume",
"Average Speed",
"Incident Reports",
"Pedestrian and Cyclist Count",
"Congestion Level"
]].corr()

fig2, ax2 = plt.subplots()

sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)

st.pyplot(fig2)