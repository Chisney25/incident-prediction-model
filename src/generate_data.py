import pandas as pd
import numpy as np
import os

def generate_incident_data(n_steps=5000, threshold=90):
    """
    Generates synthetic time-series data representing CPU usage.
    Incidents are defined as spikes that cross the 'threshold'.
    """
    np.random.seed(42)
    
    # 1. Create a base timeline (5000 minutes)
    # Using 'min' instead of 'T' to support pandas 2.0+
    timestamps = pd.date_range(start='2026-01-01', periods=n_steps, freq='min')
    
    # 2. Create a base 'Server Load' (Daily cycle + Noise)
    # This simulates a server that is busier during certain times of day
    time_index = np.arange(n_steps)
    base_load = 40 + 15 * np.sin(2 * np.pi * time_index / 1440) # 1440 mins in a day
    noise = np.random.normal(0, 3, n_steps)
    load = base_load + noise
    
    # 3. Inject Random 'Incidents' (High-load spikes)
    # We create 30 incidents where the load jumps significantly
    num_incidents = 30
    incident_starts = np.random.choice(range(100, n_steps - 50), num_incidents, replace=False)
    
    for start in incident_starts:
        duration = np.random.randint(5, 15)
        magnitude = np.random.uniform(35, 50)
        load[start:start+duration] += magnitude

    # 4. Construct the DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': np.clip(load, 0, 100) # Ensure values stay between 0-100%
    })

    # 5. Define the Label (Ground Truth)
    # 1 if usage > threshold, else 0
    df['is_incident'] = (df['cpu_usage'] > threshold).astype(int)

    # 6. Save the data
    os.makedirs('data', exist_ok=True)
    file_path = 'data/synthetic_metrics.csv'
    df.to_csv(file_path, index=False)
    
    print("-" * 30)
    print(f"SUCCESS: Data generated!")
    print(f"File: {file_path}")
    print(f"Total Steps: {len(df)}")
    print(f"Total Incidents Found: {df['is_incident'].sum()}")
    print("-" * 30)

if __name__ == "__main__":
    generate_incident_data()