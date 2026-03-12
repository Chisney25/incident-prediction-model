# Incident Prediction Model

## Overview
This project implements a machine learning model to predict system incidents within a future horizon ($H=10m$) based on a sliding window of historical metrics ($W=30m$).

## Technical Choices
- **Dataset:** Synthetic CPU metrics with a diurnal (daily) cycle and random spikes.
- **Problem Formulation:** Sliding window binary classification.
- **Model:** Random Forest Classifier. Chosen for its interpretability and ability to handle non-linear time-series patterns.
- **Evaluation Strategy:** Chronological split (80/20) to prevent data leakage.

## Performance Analysis
The model achieved **100% precision** for incident prediction, meaning it generates zero false positives. The **recall of 43%** suggests the model is conservative, identifying the most obvious "pre-incident" patterns while missing more subtle fluctuations. 

## How to Adapt to a Real System
To move this to production (e.g., in a JetBrains environment):
1. **Data Source:** Replace the synthetic CSV with a live feed from **Prometheus** or **Grafana**.
2. **Deployment:** Wrap the `incident_model.pkl` in a Flask or FastAPI microservice.
3. **Action:** If the model predicts an incident (1), trigger a webhook to PagerDuty or Slack.