import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os

def create_sliding_window(df, window_size=30, horizon=10):
    """
    Transforms time-series into features (X) and labels (y).
    X: The last 'window_size' minutes of CPU usage.
    y: 1 if an incident occurs in the NEXT 'horizon' minutes, else 0.
    """
    X, y = [], []
    data = df['cpu_usage'].values
    labels = df['is_incident'].values

    for i in range(len(data) - window_size - horizon):
        # Features: The window of past data
        X.append(data[i : i + window_size])
        
        # Target: Is there ANY incident in the future horizon?
        future_window = labels[i + window_size : i + window_size + horizon]
        y.append(1 if np.any(future_window == 1) else 0)

    return np.array(X), np.array(y)

def train_model():
    # 1. Load Data
    if not os.path.exists('data/synthetic_metrics.csv'):
        print("Error: Run generate_data.py first!")
        return

    df = pd.read_csv('data/synthetic_metrics.csv')
    
    # 2. Prepare Windows
    W, H = 30, 10
    print(f"Formulating problem: Window={W}m, Horizon={H}m...")
    X, y = create_sliding_window(df, window_size=W, horizon=H)

    # 3. Split Data (Time-series aware split)
    # We take the first 80% for training and last 20% for testing
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 4. Train Model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    print("\n--- MODEL EVALUATION ---")
    print(classification_report(y_test, y_pred))
    
    # 6. Save the model for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/incident_model.pkl')
    print("Model saved to models/incident_model.pkl")

if __name__ == "__main__":
    train_model()