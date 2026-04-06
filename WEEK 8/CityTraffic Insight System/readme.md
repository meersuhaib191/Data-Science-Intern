# CityTraffic Insight Pro

Premium AI-powered traffic analytics dashboard for Bengaluru built with Machine Learning + Streamlit.

## Why this project stands out

CityTraffic Insight Pro combines model-based predictions with a polished analytics UX. It helps users understand traffic behavior, estimate travel friction, and identify high-risk intersections through a clean, decision-friendly interface.

## Core capabilities

- Travel Time Index prediction (regression model)
- Traffic level prediction (classification model)
- Hourly congestion curve with peak-hour and best-hour detection
- Risk view for accident-prone intersections and busiest roads
- Correlation heatmap for congestion drivers
- Scenario simulation controls (day, weather, roadwork)
- Bengaluru intra-city fastest-route navigation (area-to-area commute planning)
- Operational trip intelligence (adjusted ETA, risk, fuel and CO2 estimate)
- Route alternatives comparison with fastest option auto-selected
- Congestion analytics by route segments and best departure hour recommendation
- Downloadable traffic snapshot report (CSV)
- Robust fallback logic when `.pkl` models are unavailable

## Project architecture

- `train_models.ipynb` — training notebook for regression and classification models
- `app.py` — Streamlit dashboard application
- `travel_time_model.pkl` — optional serialized regressor
- `congestion_model.pkl` — optional serialized classifier
- `requirements.txt` — Python dependencies

## Dataset

The app uses Bengaluru traffic observations with signals such as:

- Traffic Volume
- Average Speed
- Road Capacity Utilization
- Incident Reports
- Environmental Impact
- Public Transport Usage
- Traffic Signal Compliance
- Parking Usage
- Pedestrian and Cyclist Count
- Weather Conditions

Data source in app: Google Drive CSV endpoint.

## Tech stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib, Seaborn
- Requests

## Local setup

1) Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Run the dashboard:

```bash
streamlit run app.py
```

## Notes

- If model files are present, the app uses trained models for predictions.
- If model files are missing, the app falls back to heuristic predictions so the dashboard still works.
- This is a historical-data intelligence tool, not a real-time GPS navigation replacement.

