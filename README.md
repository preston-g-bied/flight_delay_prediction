# Flight Delay Prediction

## Overview
This project aims to predict whether a flight will be delayed for more than 15 minutes based on various flight features. The target metric is ROC AUC.

## Dataset
The dataset includes the following features:
- Month, DayofMonth, DayOfWeek - Date information
- DepTime - Departure time
- UniqueCarrier - Code of the airline carrier
- Origin - Flight origin airport
- Dest - Flight destination airport
- Distance - Distance between Origin and Dest airports
- dep_delayed_15min - Target variable (whether flight was delayed > 15 min)

## Project Structure
```
flight-delay-prediction/
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   ├── processed/             # Cleaned data
│   └── external/              # External data
│
├── notebooks/                 # Jupyter notebooks
│   └── exploratory/           # Data exploration
│
├── src/                       # Source code
│   ├── data/                  # Data processing scripts
│   ├── features/              # Feature engineering
│   ├── models/                # Model training and prediction
│   └── visualization/         # Visualization code
│
├── models/                    # Saved models
│
└── reports/                   # Analysis reports and figures
```

## Getting Started
1. Clone this repository
2. Install dependencies with `pip install -r requirements.txt`
3. Place the raw data files in the `data/raw/` directory
4. Run the data processing scripts to generate processed datasets
5. Explore the data using the Jupyter notebooks
6. Train models and make predictions