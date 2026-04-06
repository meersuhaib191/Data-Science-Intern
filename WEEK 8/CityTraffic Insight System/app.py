from datetime import datetime
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="CityTraffic Insight Pro", page_icon="🚦", layout="wide")
sns.set_style("whitegrid")


st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .hero-card {
          background: linear-gradient(120deg, #101828 0%, #1d4ed8 100%);
          padding: 1rem 1.2rem;
          border-radius: 14px;
          color: white;
          margin-bottom: 1rem;
      }
      .insight-card {
          border: 1px solid rgba(49, 130, 206, 0.35);
          background: rgba(59, 130, 246, 0.08);
          border-radius: 12px;
          padding: 0.7rem 0.9rem;
          margin-bottom: 0.5rem;
      }
      .kpi-card {
          border: 1px solid rgba(15, 23, 42, 0.14);
          border-radius: 14px;
          padding: 0.9rem 1rem;
          background: linear-gradient(160deg, rgba(248, 250, 252, 0.95), rgba(241, 245, 249, 0.95));
          min-height: 92px;
      }
      .kpi-card--blue {
          border-color: rgba(37, 99, 235, 0.28);
          background: linear-gradient(160deg, rgba(219, 234, 254, 0.8), rgba(239, 246, 255, 0.9));
      }
      .kpi-card--amber {
          border-color: rgba(245, 158, 11, 0.30);
          background: linear-gradient(160deg, rgba(254, 243, 199, 0.9), rgba(255, 251, 235, 0.95));
      }
      .kpi-card--green {
          border-color: rgba(22, 163, 74, 0.28);
          background: linear-gradient(160deg, rgba(220, 252, 231, 0.86), rgba(240, 253, 244, 0.95));
      }
      .kpi-card--violet {
          border-color: rgba(124, 58, 237, 0.26);
          background: linear-gradient(160deg, rgba(237, 233, 254, 0.88), rgba(245, 243, 255, 0.95));
      }
      .kpi-card--rose {
          border-color: rgba(225, 29, 72, 0.24);
          background: linear-gradient(160deg, rgba(255, 228, 230, 0.86), rgba(255, 241, 242, 0.95));
      }
      .kpi-title {
          font-size: 0.8rem;
          color: #334155;
          margin-bottom: 0.2rem;
          font-weight: 600;
      }
      .kpi-value {
          font-size: 1.4rem;
          font-weight: 800;
          color: #0f172a;
          margin: 0;
      }
      .chip-row {
          display: flex;
          gap: 0.5rem;
          flex-wrap: wrap;
          margin-bottom: 0.7rem;
      }
      .chip {
          border: 1px solid rgba(15, 23, 42, 0.12);
          border-radius: 999px;
          padding: 0.25rem 0.65rem;
          font-size: 0.75rem;
          font-weight: 600;
          background: #f8fafc;
          color: #334155;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_FILES = {
    "time": Path("travel_time_model.pkl"),
    "congestion": Path("congestion_model.pkl"),
}
DATA_URL = "https://drive.google.com/uc?id=1AMBzflTbushZN3zzZL-pl390QCm3e5Qx"
TRAFFIC_MAP = {0: "Low", 1: "Medium", 2: "High"}
WEATHER_MAP = {"Clear": 0, "Overcast": 1, "Rain": 2, "Fog": 3}
ROADWORK_MAP = {"No": 0, "Yes": 1}
ROUTE_PROFILES = {
    "Driving": "driving",
    "Cycling": "cycling",
    "Walking": "foot",
}


@st.cache_resource
def load_models():
    time_model = None
    cong_model = None
    warnings = []

    if MODEL_FILES["time"].exists():
        with MODEL_FILES["time"].open("rb") as f:
            time_model = pickle.load(f)
    else:
        warnings.append("`travel_time_model.pkl` not found. Using heuristic estimate.")

    if MODEL_FILES["congestion"].exists():
        with MODEL_FILES["congestion"].open("rb") as f:
            cong_model = pickle.load(f)
    else:
        warnings.append("`congestion_model.pkl` not found. Using rule-based classification.")

    return time_model, cong_model, warnings


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["day"] = df["Date"].dt.day_name()
    return df


def build_feature_row(filtered_df: pd.DataFrame, day_index: int, weather: str, roadwork: str):
    return pd.DataFrame(
        [
            {
                "Traffic Volume": filtered_df["Traffic Volume"].mean(),
                "Average Speed": filtered_df["Average Speed"].mean(),
                "Road Capacity Utilization": filtered_df["Road Capacity Utilization"].mean(),
                "Incident Reports": filtered_df["Incident Reports"].mean(),
                "Environmental Impact": filtered_df["Environmental Impact"].mean(),
                "Public Transport Usage": filtered_df["Public Transport Usage"].mean(),
                "Traffic Signal Compliance": filtered_df["Traffic Signal Compliance"].mean(),
                "Parking Usage": filtered_df["Parking Usage"].mean(),
                "Pedestrian and Cyclist Count": filtered_df["Pedestrian and Cyclist Count"].mean(),
                "Weather Conditions": WEATHER_MAP.get(weather, 0),
                "Roadwork and Construction Activity": ROADWORK_MAP.get(roadwork, 0),
                "day": day_index,
            }
        ]
    )


def fallback_predictions(features: pd.DataFrame):
    speed = float(features["Average Speed"].iloc[0])
    volume = float(features["Traffic Volume"].iloc[0])
    incidents = float(features["Incident Reports"].iloc[0])

    time_index = max(0.6, min(1.8, 1 + (volume / 90000) + (incidents * 0.08) - (speed / 130)))
    raw_score = (volume / 1500) + (incidents * 7) + max(0, (35 - speed)) * 1.8

    if raw_score < 45:
        cls = 0
    elif raw_score < 85:
        cls = 1
    else:
        cls = 2

    return time_index, cls


def build_hourly_profile(avg_congestion, avg_speed, traffic_volume, incident_reports):
    base = float(avg_congestion)
    speed_factor = 50 / max(float(avg_speed), 1)
    incident_factor = 1 + (float(incident_reports) / 5)
    volume_factor = float(traffic_volume) / 60000
    peak_shift = int((volume_factor * 6 + incident_factor * 4) % 24)

    hourly_profile = []
    for hour in range(24):
        peak_intensity = np.exp(-((hour - peak_shift) ** 2) / 20)
        value = base * (0.6 + peak_intensity + speed_factor * 0.1)
        hourly_profile.append(value)

    return pd.DataFrame({"hour": list(range(24)), "congestion": hourly_profile})


def traffic_health_message(score):
    if score > 75:
        return "Severe", "🚨 Severe congestion expected. Avoid non-essential travel."
    if score > 50:
        return "Moderate", "⚠ Moderate congestion expected. Keep 10-15 minutes buffer."
    return "Smooth", "✅ Smooth traffic expected for most routes."


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_city(city_name: str):
    if not city_name.strip():
        return None

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city_name, "format": "jsonv2", "limit": 1}
    headers = {"User-Agent": "citytraffic-insight-pro/1.0 (educational project)"}
    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()

    if not data:
        return None

    row = data[0]
    return {
        "display_name": row["display_name"],
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_route_options(origin_lat, origin_lon, dest_lat, dest_lon, profile, max_routes=3):
    url = (
        f"https://router.project-osrm.org/route/v1/{profile}/"
        f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    )
    params = {"overview": "full", "geometries": "geojson", "alternatives": "true", "steps": "true"}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    routes = data.get("routes", [])
    if not routes:
        return []

    def infer_route_name(route_obj, fallback_id):
        road_counts = {}
        for leg in route_obj.get("legs", []):
            for step in leg.get("steps", []):
                road_name = str(step.get("name", "")).strip()
                if not road_name:
                    continue
                road_counts[road_name] = road_counts.get(road_name, 0) + 1

        if road_counts:
            primary_road = max(road_counts.items(), key=lambda item: item[1])[0]
            return f"Via {primary_road}"
        return f"Route {fallback_id}"

    route_options = []
    seen_names = {}
    for idx, route in enumerate(routes[:max_routes], start=1):
        base_name = infer_route_name(route, idx)
        seen_names[base_name] = seen_names.get(base_name, 0) + 1
        route_name = base_name if seen_names[base_name] == 1 else f"{base_name} ({seen_names[base_name]})"
        route_options.append(
            {
                "route_id": idx,
                "route_name": route_name,
                "distance_km": route["distance"] / 1000.0,
                "duration_min": route["duration"] / 60.0,
                "geometry": route["geometry"]["coordinates"],
            }
        )
    return route_options


def traffic_multiplier_for_hour(traffic_score, weather, roadwork, departure_hour):
    weather_multiplier = {"Clear": 1.0, "Overcast": 1.05, "Rain": 1.2, "Fog": 1.25}.get(weather, 1.0)
    roadwork_multiplier = 1.18 if roadwork == "Yes" else 1.0
    peak_multiplier = 1.18 if departure_hour in {8, 9, 10, 17, 18, 19, 20} else 1.0
    traffic_multiplier = 1.0 + (traffic_score / 300.0)
    return weather_multiplier * roadwork_multiplier * peak_multiplier * traffic_multiplier


def estimate_trip_metrics(
    base_duration_min,
    distance_km,
    traffic_score,
    weather,
    roadwork,
    departure_hour,
    mode,
):
    adjusted_duration = base_duration_min * traffic_multiplier_for_hour(
        traffic_score=traffic_score,
        weather=weather,
        roadwork=roadwork,
        departure_hour=departure_hour,
    )

    mode_efficiency = {"Driving": 0.18, "Cycling": 0.0, "Walking": 0.0}.get(mode, 0.0)
    carbon_kg = distance_km * mode_efficiency
    fuel_liters = distance_km / 14.0 if mode == "Driving" else 0.0

    risk_score = min(
        100,
        (traffic_score * 0.55)
        + (18 if weather in {"Rain", "Fog"} else 0)
        + (12 if roadwork == "Yes" else 0)
        + (8 if departure_hour < 6 or departure_hour > 21 else 0),
    )

    return adjusted_duration, carbon_kg, fuel_liters, risk_score


def get_best_departure_window(base_duration_min, traffic_score, weather, roadwork, start_hour):
    candidate_hours = [(start_hour + offset) % 24 for offset in range(0, 18)]
    projections = []
    for hour in candidate_hours:
        multiplier = traffic_multiplier_for_hour(
            traffic_score=traffic_score,
            weather=weather,
            roadwork=roadwork,
            departure_hour=hour,
        )
        eta = base_duration_min * multiplier
        projections.append({"hour": hour, "eta_min": eta})

    ranked = sorted(projections, key=lambda row: row["eta_min"])
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else ranked[0]
    return best, second, pd.DataFrame(projections).sort_values("hour")


def build_route_congestion_segments(geometry, traffic_score, departure_hour, weather, roadwork):
    if len(geometry) < 2:
        return []

    weather_penalty = 10 if weather in {"Rain", "Fog"} else 0
    roadwork_penalty = 8 if roadwork == "Yes" else 0
    peak_penalty = 10 if departure_hour in {8, 9, 10, 17, 18, 19, 20} else 0

    chunk = max(2, len(geometry) // 12)
    segments = []
    for i in range(0, len(geometry) - 1, chunk):
        path = geometry[i : min(i + chunk + 1, len(geometry))]
        progress = i / max(1, len(geometry) - 1)
        wave = np.sin(progress * np.pi * 2.4)
        score = min(100, max(10, (traffic_score * 0.7) + weather_penalty + roadwork_penalty + peak_penalty + wave * 12))

        if score >= 75:
            color = [220, 38, 38]
            level = "High congestion"
        elif score >= 55:
            color = [245, 158, 11]
            level = "Moderate congestion"
        else:
            color = [22, 163, 74]
            level = "Low congestion"

        segments.append({"path": path, "color": color, "score": round(float(score), 1), "level": level})
    return segments


def render_kpi_card(title, value, tone="blue"):
    st.markdown(
        f"""
        <div class="kpi-card kpi-card--{tone}">
            <div class="kpi-title">{title}</div>
            <p class="kpi-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    now = datetime.now()
    current_hour = now.hour
    current_day = now.strftime("%A")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    st.markdown(
        f"""
        <div class="hero-card">
            <h2 style="margin:0;">CityTraffic Insight Pro</h2>
            <p style="margin:0.3rem 0 0 0;">AI traffic intelligence for Bengaluru · {current_day} · {current_hour:02d}:00</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    time_model, cong_model, model_warnings = load_models()

    for warning in model_warnings:
        st.warning(warning)

    with st.sidebar:
        st.header("Control Panel")
        analysis_mode = st.segmented_control(
            "Analysis View",
            ["City Traffic Analysis", "Intercity Analysis"],
            default="City Traffic Analysis",
            selection_mode="single",
        )
        areas = sorted(df["Area Name"].dropna().unique())
        area = st.selectbox("Area", areas)
        selected_day = st.selectbox("Day", day_names, index=day_names.index(current_day))
        weather = st.selectbox("Weather", list(WEATHER_MAP.keys()), index=0)
        roadwork = st.selectbox("Roadwork", list(ROADWORK_MAP.keys()), index=0)
        departure_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=current_hour)
        top_n = st.slider("Top roads to show", min_value=5, max_value=12, value=8)
        use_same_day = st.toggle("Use selected weekday slice", value=True)

    area_df = df[df["Area Name"] == area].copy()
    if use_same_day:
        filtered_df = area_df[area_df["day"] == selected_day]
        if filtered_df.empty:
            filtered_df = area_df
            st.info("No rows for selected weekday in this area. Using full area data.")
    else:
        filtered_df = area_df

    if filtered_df.empty:
        st.error("No data available for this selection.")
        st.stop()

    day_index = day_names.index(selected_day)
    features = build_feature_row(filtered_df, day_index, weather, roadwork)

    if time_model is not None:
        travel_time_index = float(time_model.predict(features)[0])
    else:
        travel_time_index, _ = fallback_predictions(features)

    if cong_model is not None:
        traffic_class = int(cong_model.predict(features)[0])
    else:
        _, traffic_class = fallback_predictions(features)

    traffic_label = TRAFFIC_MAP.get(traffic_class, "Unknown")

    avg_speed = float(filtered_df["Average Speed"].mean())
    traffic_volume = float(filtered_df["Traffic Volume"].mean())
    incident_reports = float(filtered_df["Incident Reports"].mean())
    avg_congestion = float(filtered_df["Congestion Level"].mean())
    hourly_df = build_hourly_profile(avg_congestion, avg_speed, traffic_volume, incident_reports)

    peak_hour = int(hourly_df.loc[hourly_df["congestion"].idxmax(), "hour"])
    best_hour = int(hourly_df.loc[hourly_df["congestion"].idxmin(), "hour"])
    current_congestion = float(hourly_df.loc[hourly_df["hour"] == current_hour, "congestion"].iloc[0])
    traffic_score = min(100.0, current_congestion + (1 + incident_reports / 5) * 10)
    health_level, health_msg = traffic_health_message(traffic_score)

    st.markdown(
        f"""
        <div class="chip-row">
            <div class="chip">Area: {area}</div>
            <div class="chip">Day: {selected_day}</div>
            <div class="chip">Weather: {weather}</div>
            <div class="chip">Roadwork: {roadwork}</div>
            <div class="chip">Departure: {departure_hour:02d}:00</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Traffic Level", traffic_label, "blue")
    with col2:
        render_kpi_card("Congestion Score", f"{traffic_score:.0f}/100", "amber")
    with col3:
        render_kpi_card("Average Speed", f"{avg_speed:.1f} km/h", "green")
    with col4:
        render_kpi_card("Travel Time Index", f"{travel_time_index:.2f}", "violet")
    with col5:
        render_kpi_card("Predicted Peak", f"{peak_hour:02d}:00", "rose")

    if analysis_mode == "City Traffic Analysis":
        city_tabs = st.tabs(["Live Intelligence", "Road Risk Map", "Data Explorer"])

    if analysis_mode == "City Traffic Analysis":
        with city_tabs[0]:
            st.markdown(
                f"<div class='insight-card'><strong>Status:</strong> {health_level} traffic · {health_msg}</div>",
                unsafe_allow_html=True,
            )
            st.write(f"Best time to travel today in **{area}** is around **{best_hour:02d}:00**.")
            st.progress(int(max(0, min(100, traffic_score))), text=f"Current congestion severity: {traffic_score:.0f}/100")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hourly_df["hour"], hourly_df["congestion"], marker="o", linewidth=2, color="#2563eb")
            ax.axvline(current_hour, linestyle="--", color="#f97316", linewidth=1.5, label="Current Hour")
            ax.scatter(
                peak_hour,
                hourly_df.loc[hourly_df["hour"] == peak_hour, "congestion"],
                s=120,
                color="#dc2626",
                label="Peak",
            )
            ax.set_title(f"{area} - Hourly Congestion Pattern")
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Congestion Index")
            ax.legend()
            st.pyplot(fig)

            c_left, c_right = st.columns([1.3, 1])
            with c_left:
                st.subheader("Top Congestion Hotspots")
                hotspot_df = (
                    filtered_df.groupby("Road/Intersection Name")["Congestion Level"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(6)
                    .reset_index()
                )
                st.dataframe(
                    hotspot_df.rename(
                        columns={
                            "Road/Intersection Name": "Road / Intersection",
                            "Congestion Level": "Avg Congestion",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
            with c_right:
                st.subheader("Travel Readiness")
                readiness = max(0, 100 - int(traffic_score))
                st.progress(readiness, text=f"Readiness score: {readiness}/100")
                st.caption("Higher readiness means better conditions for immediate travel.")

    if analysis_mode == "City Traffic Analysis":
        with city_tabs[1]:
            left, right = st.columns(2)
            with left:
                st.subheader("Accident-Prone Intersections")
                accidents = (
                    filtered_df.groupby("Road/Intersection Name")["Incident Reports"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(top_n)
                )
                st.bar_chart(accidents)
            with right:
                st.subheader("Highest Traffic Roads")
                traffic = (
                    filtered_df.groupby("Road/Intersection Name")["Traffic Volume"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(top_n)
                )
                st.bar_chart(traffic)

            st.subheader("Congestion Drivers Correlation")
            corr = filtered_df[
                [
                    "Traffic Volume",
                    "Average Speed",
                    "Incident Reports",
                    "Pedestrian and Cyclist Count",
                    "Congestion Level",
                ]
            ].corr()
            fig2, ax2 = plt.subplots(figsize=(9, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2, fmt=".2f")
            st.pyplot(fig2)

    if analysis_mode == "Intercity Analysis":
        st.subheader("Bengaluru Intra-City Navigation")
        st.caption("Fastest route analytics for Bengaluru commuters")
        
        bangalore_areas = sorted(df["Area Name"].dropna().unique())
        default_origin_index = bangalore_areas.index(area) if area in bangalore_areas else 0
        default_destination_index = 0 if default_origin_index != 0 else min(1, len(bangalore_areas) - 1)

        trip_left, trip_mid, trip_right = st.columns([1.2, 1.2, 1])
        with trip_left:
            origin_area = st.selectbox("Origin Area (Bengaluru)", bangalore_areas, index=default_origin_index)
        with trip_mid:
            destination_area = st.selectbox(
                "Destination Area (Bengaluru)", bangalore_areas, index=default_destination_index
            )
        with trip_right:
            travel_mode = st.selectbox("Mode", list(ROUTE_PROFILES.keys()), index=0)

        if origin_area == destination_area:
            st.warning("Choose different origin and destination areas.")
        else:
            try:
                origin_geo = geocode_city(f"{origin_area}, Bengaluru, Karnataka, India")
                destination_geo = geocode_city(f"{destination_area}, Bengaluru, Karnataka, India")

                if origin_geo is None or destination_geo is None:
                    st.error("Could not geocode one of the selected areas. Try another Bengaluru area.")
                else:
                    route_options = get_route_options(
                        origin_geo["lat"],
                        origin_geo["lon"],
                        destination_geo["lat"],
                        destination_geo["lon"],
                        ROUTE_PROFILES[travel_mode],
                    )
                    if not route_options:
                        st.error("No route found for this area pair and transport mode.")
                    else:
                        evaluated_routes = []
                        for route in route_options:
                            adjusted_eta, carbon_kg, fuel_liters, risk_score = estimate_trip_metrics(
                                base_duration_min=route["duration_min"],
                                distance_km=route["distance_km"],
                                traffic_score=traffic_score,
                                weather=weather,
                                roadwork=roadwork,
                                departure_hour=departure_hour,
                                mode=travel_mode,
                            )
                            evaluated_routes.append(
                                {
                                    **route,
                                    "adjusted_eta_min": adjusted_eta,
                                    "carbon_kg": carbon_kg,
                                    "fuel_liters": fuel_liters,
                                    "risk_score": risk_score,
                                }
                            )

                        fastest_route = min(evaluated_routes, key=lambda row: row["adjusted_eta_min"])
                        best_window, alt_window, hourly_eta_df = get_best_departure_window(
                            base_duration_min=fastest_route["duration_min"],
                            traffic_score=traffic_score,
                            weather=weather,
                            roadwork=roadwork,
                            start_hour=departure_hour,
                        )

                        metric_a, metric_b, metric_c, metric_d = st.columns(4)
                        metric_a.metric("Fastest Distance", f"{fastest_route['distance_km']:.1f} km")
                        metric_b.metric("Base ETA", f"{fastest_route['duration_min']:.0f} min")
                        metric_c.metric("Fastest ETA Now", f"{fastest_route['adjusted_eta_min']:.0f} min")
                        metric_d.metric("Congestion Risk", f"{fastest_route['risk_score']:.0f}/100")

                        ops_a, ops_b = st.columns(2)
                        ops_a.metric("Estimated CO2", f"{fastest_route['carbon_kg']:.2f} kg")
                        ops_b.metric("Estimated Fuel", f"{fastest_route['fuel_liters']:.1f} L")

                        route_table = pd.DataFrame(
                            [
                                {
                                    "Route": row["route_name"],
                                    "Distance (km)": round(row["distance_km"], 1),
                                    "Base ETA (min)": round(row["duration_min"]),
                                    "Adjusted ETA (min)": round(row["adjusted_eta_min"]),
                                    "Risk (/100)": round(row["risk_score"]),
                                }
                                for row in sorted(evaluated_routes, key=lambda r: r["adjusted_eta_min"])
                            ]
                        )
                        st.dataframe(route_table, width="stretch")
                        st.bar_chart(route_table.set_index("Route")["Adjusted ETA (min)"], width="stretch")

                        st.info(
                            f"Fastest route right now: **{fastest_route['route_name']}**. "
                            f"Best departure window: **{best_window['hour']:02d}:00** "
                            f"(next best: **{alt_window['hour']:02d}:00**)."
                        )

                        fastest_segments = build_route_congestion_segments(
                            geometry=fastest_route["geometry"],
                            traffic_score=traffic_score,
                            departure_hour=departure_hour,
                            weather=weather,
                            roadwork=roadwork,
                        )
                        segment_summary = pd.Series([seg["level"] for seg in fastest_segments]).value_counts()
                        high_count = int(segment_summary.get("High congestion", 0))
                        moderate_count = int(segment_summary.get("Moderate congestion", 0))
                        low_count = int(segment_summary.get("Low congestion", 0))

                        con_a, con_b, con_c = st.columns(3)
                        con_a.metric("High Congestion Segments", high_count)
                        con_b.metric("Moderate Segments", moderate_count)
                        con_c.metric("Low Congestion Segments", low_count)

                        congestion_df = pd.DataFrame(
                            [
                                {"Segment": idx + 1, "Congestion Level": seg["level"], "Score (/100)": seg["score"]}
                                for idx, seg in enumerate(fastest_segments)
                            ]
                        )
                        st.dataframe(congestion_df, width="stretch")

                        st.markdown(
                            f"""
                            <div class='insight-card'>
                            <strong>Commute recommendation:</strong>
                            For <strong>{origin_area} → {destination_area}</strong>, leave around
                            <strong>{best_window['hour']:02d}:00</strong> to minimize ETA.
                            Route congestion analytics are shown segment-wise below.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.line_chart(hourly_eta_df.set_index("hour")["eta_min"])
            except requests.RequestException as err:
                st.error(f"Route service unavailable: {err}")

    if analysis_mode == "City Traffic Analysis":
        with city_tabs[2]:
            st.subheader("Filtered Sample")
            sample_cols = [
                "Date",
                "Area Name",
                "Road/Intersection Name",
                "Traffic Volume",
                "Average Speed",
                "Congestion Level",
                "Incident Reports",
                "Weather Conditions",
            ]
            preview_df = filtered_df[sample_cols].sort_values("Date", ascending=False).head(200)
            st.dataframe(preview_df, width="stretch")

            report_row = pd.DataFrame(
                [
                    {
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "area": area,
                        "selected_day": selected_day,
                        "weather": weather,
                        "roadwork": roadwork,
                        "traffic_level": traffic_label,
                        "congestion_score": round(traffic_score, 2),
                        "avg_speed_kmh": round(avg_speed, 2),
                        "travel_time_index": round(travel_time_index, 3),
                        "peak_hour": f"{peak_hour:02d}:00",
                        "best_hour": f"{best_hour:02d}:00",
                    }
                ]
            )
            st.download_button(
                "Download Snapshot (CSV)",
                data=report_row.to_csv(index=False).encode("utf-8"),
                file_name=f"traffic_snapshot_{area.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )

    st.caption(
        "Built for analytics exploration. Predictions are data-driven estimates and not a substitute for live GPS traffic APIs."
    )


if __name__ == "__main__":
    main()