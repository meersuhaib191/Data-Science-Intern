import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, AntPath
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Traffic Intelligence System",
    page_icon="🚦",
    layout="wide"
)

ox.settings.use_cache = True


# ------------------------------------------------
# LOAD ROAD NETWORK (ONLY ONCE)
# ------------------------------------------------

@st.cache_resource
def load_graph():
    G = ox.graph_from_place(
        "Bengaluru, Karnataka, India",
        network_type="drive"
    )
    return G

G = load_graph()


# ------------------------------------------------
# AREA COORDINATES
# ------------------------------------------------

coords = {
"Indiranagar":[12.9719,77.6412],
"Whitefield":[12.9698,77.7499],
"Koramangala":[12.9352,77.6245],
"BTM":[12.9166,77.6101],
"Electronic City":[12.8399,77.6770],
"Yelahanka":[13.1007,77.5963],
"Hebbal":[13.0352,77.5970],
"Marathahalli":[12.9591,77.6974],
"MG Road":[12.9755,77.6065],
"Jayanagar":[12.9250,77.5938]
}


# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("🚦 AI Traffic Intelligence System")
st.write("Predict congestion and optimize travel routes in Bengaluru.")


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header("Travel Controls")

areas = list(coords.keys())

source = st.sidebar.selectbox(
    "Select Source",
    ["Select Source"] + areas
)

destination = st.sidebar.selectbox(
    "Select Destination",
    ["Select Destination"] + areas
)

weather = st.sidebar.selectbox(
    "Weather",
    ["Clear","Cloudy","Rainy"]
)

hour = st.sidebar.slider("Travel Hour",0,23,8)

if source == "Select Source" or destination == "Select Destination":
    st.info("Please select source and destination.")
    st.stop()

if source == destination:
    st.warning("Source and destination must differ.")
    st.stop()


# ------------------------------------------------
# ROUTING
# ------------------------------------------------

source_coord = coords[source]
dest_coord = coords[destination]

orig = ox.distance.nearest_nodes(G, source_coord[1], source_coord[0])
dest = ox.distance.nearest_nodes(G, dest_coord[1], dest_coord[0])

route = nx.shortest_path(G, orig, dest, weight="length")

route_coords = [(float(G.nodes[n]["y"]), float(G.nodes[n]["x"])) for n in route]

distance = nx.shortest_path_length(G, orig, dest, weight="length")
distance_km = distance / 1000


# ------------------------------------------------
# REALISTIC TRAFFIC MODEL
# ------------------------------------------------

traffic_score = 0

# peak hour
if 7 <= hour <= 10:
    traffic_score += 3

if 17 <= hour <= 20:
    traffic_score += 3

# weather
if weather == "Rainy":
    traffic_score += 2
elif weather == "Cloudy":
    traffic_score += 1

# distance
if distance_km > 12:
    traffic_score += 2
elif distance_km > 6:
    traffic_score += 1


# classify traffic
if traffic_score <= 2:
    traffic_label = "Low Traffic"
    heat_weight = 0.3
    speed = 40

elif traffic_score <= 4:
    traffic_label = "Medium Traffic"
    heat_weight = 0.6
    speed = 28

else:
    traffic_label = "High Traffic"
    heat_weight = 1.0
    speed = 18


travel_time = distance_km / (speed/60)


# ------------------------------------------------
# PREDICTION PANEL
# ------------------------------------------------

st.subheader("🚦 Traffic Prediction")

col1,col2,col3 = st.columns(3)

if traffic_label == "Low Traffic":
    col1.success(traffic_label)

elif traffic_label == "Medium Traffic":
    col1.warning(traffic_label)

else:
    col1.error(traffic_label)

col2.metric("Route Distance",f"{round(distance_km,2)} km")
col3.metric("Estimated Time",f"{round(travel_time,1)} minutes")


# ------------------------------------------------
#   MAP
# ------------------------------------------------

st.subheader("🗺 Route Map")

route_key = f"{source}-{destination}-{weather}-{hour}"

if st.session_state.get("route_key") != route_key:

    m = folium.Map(location=source_coord, zoom_start=12)

    AntPath(
        route_coords,
        color="blue",
        delay=700
    ).add_to(m)

    folium.Marker(
        source_coord,
        popup="Source"
    ).add_to(m)

    folium.Marker(
        dest_coord,
        popup="Destination"
    ).add_to(m)

    heat_points = []

    for lat, lon in route_coords:
        heat_points.append([
            float(lat),
            float(lon),
            heat_weight
        ])

    HeatMap(
        heat_points,
        radius=15,
        blur=20
    ).add_to(m)

    st.session_state.map = m
    st.session_state.route_key = route_key

st_folium(
    st.session_state.map,
    width=900,
    height=500,
    returned_objects=[]
)


# ------------------------------------------------
#   AI TRAVEL INSIGHTS
# ------------------------------------------------

st.subheader("🤖 AI Travel Insights")

best_day = "Wednesday"
best_hour = "14:00"
peak_hour = "18:00"

col1,col2,col3 = st.columns(3)

col1.metric("Best Day to Travel",best_day)
col2.metric("Best Hour",best_hour)
col3.metric("Peak Traffic Hour",peak_hour)


# ------------------------------------------------
#  TRAFFIC FORECAST
# ------------------------------------------------

st.subheader("📈 Next 6 Hours Traffic Forecast")

hours = [(hour+i)%24 for i in range(6)]

forecast = []

for h in hours:

    score = 0

    if 7 <= h <= 10:
        score += 3

    if 17 <= h <= 20:
        score += 3

    if weather == "Rainy":
        score += 2

    if distance_km > 10:
        score += 2

    forecast.append(score*100)

fig, ax = plt.subplots()

ax.plot(hours, forecast, marker="o")

ax.set_xlabel("Hour")
ax.set_ylabel("Traffic Index")

st.pyplot(fig)


# ------------------------------------------------
# GOOGLE MAPS NAVIGATION
# ------------------------------------------------

st.subheader("🧭 Navigation")

google_maps_url = f"https://www.google.com/maps/dir/{source}/{destination}"

st.link_button(
    "Open in Google Maps",
    google_maps_url
)