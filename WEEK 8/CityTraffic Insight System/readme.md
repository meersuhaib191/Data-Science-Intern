# 🚦 CityTraffic Insight System

An AI-powered traffic intelligence system that analyzes urban traffic patterns across Bengaluru and provides smart insights for congestion, travel time, and route efficiency.

---



##  📌 Overview

**CityTraffic Insight System** is a data-driven traffic analytics platform built using Machine Learning and Streamlit. It leverages historical traffic data from multiple areas in Bengaluru to generate intelligent insights such as:

- Traffic congestion levels  
- Travel time estimation  
- Peak traffic hours  
- Accident-prone intersections  
- Smart travel recommendations  

---



## 🎯 Objectives

- Predict **Travel Time Index** using regression models  
- Classify **Traffic Congestion Levels** (Low / Medium / High)  
- Provide **area-based traffic insights**  
- Enable **data-driven decision making for travel planning**  

---

## 🧠 Machine Learning Models

### 🔹 Random Forest Regressor
- Predicts **Travel Time Index**
- Helps estimate delays and travel efficiency  

### 🔹 Random Forest Classifier
- Predicts **Traffic Congestion Level**
- Outputs:
  - Low Traffic  
  - Medium Traffic  
  - High Traffic  

---

## 📊 Dataset

The dataset includes traffic data collected from various **areas and intersections across Bengaluru**, with features such as:

- Traffic Volume  
- Average Speed  
- Road Capacity Utilization  
- Incident Reports  
- Environmental Impact  
- Public Transport Usage  
- Traffic Signal Compliance  
- Parking Usage  
- Pedestrian & Cyclist Count  
- Weather Conditions  

---

## 🖥️ Dashboard Features

The Streamlit dashboard provides:

- 📍 Area-based traffic analysis  
- 📈 Congestion trend visualization  
- 🚗 Travel time estimation  
- ⚠ Accident-prone area detection  
- 🧠 Smart traffic insights  
- 📊 Congestion drivers analysis  

---

## ⚙️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Matplotlib, Seaborn**
- **Streamlit**

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/citytraffic-insight-system.git
cd citytraffic-insight-system
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the app
```bash
streamlit run app.py
```
```
### 📁 Project Structure
├── app.py
├── train_model.ipynb
├── travel_time_model.pkl
├── congestion_model.pkl
├── requirements.txt
└── README.md
```
### 📌 Key Highlights

AI-powered traffic prediction

Data-driven urban traffic insights

Real-time style dashboard (based on historical data)

Scalable for smart city applications

### ⚠️ Note

This system uses historical traffic data and provides data-informed predictions, not real-time live traffic.

### 📈 Future Improvements

Integration with real-time traffic APIs

Route optimization using graph algorithms

Interactive maps (Folium / Mapbox)

Advanced deep learning models


### 👨‍💻 Author

Mir Suhaib

