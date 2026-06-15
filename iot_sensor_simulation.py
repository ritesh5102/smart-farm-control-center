import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import os

def generate_sensor_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate normal distributions
    temperature = np.random.uniform(20.0, 40.0, n_samples)
    humidity = np.random.uniform(40.0, 90.0, n_samples)
    soil_moisture = np.random.uniform(20.0, 80.0, n_samples)
    ph = np.random.uniform(5.5, 7.5, n_samples)
    rainfall = np.random.uniform(0.0, 300.0, n_samples)
    
    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'soil_moisture': soil_moisture,
        'ph': ph,
        'rainfall': rainfall
    })
    
    # Introduce deliberate anomalies (~5% of the data)
    num_anomalies = int(n_samples * 0.05)
    anomaly_indices = np.random.choice(n_samples, num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['temp_high', 'ph_low', 'moisture_high', 'rainfall_extreme'])
        if anomaly_type == 'temp_high':
            df.loc[idx, 'temperature'] = np.random.uniform(45.0, 55.0)
        elif anomaly_type == 'ph_low':
            df.loc[idx, 'ph'] = np.random.uniform(3.0, 4.5)
        elif anomaly_type == 'moisture_high':
            df.loc[idx, 'soil_moisture'] = np.random.uniform(90.0, 100.0)
        elif anomaly_type == 'rainfall_extreme':
            df.loc[idx, 'rainfall'] = np.random.uniform(400.0, 500.0)
            
    return df

def train_anomaly_detector(df):
    # Features for Isolation Forest
    features = ['temperature', 'humidity', 'soil_moisture', 'ph', 'rainfall']
    X = df[features]
    
    # Isolation Forest
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(X)
    
    # Predict (-1 for anomaly, 1 for normal)
    predictions = clf.predict(X)
    
    # Map to 1 for anomaly, 0 for normal
    df['is_anomaly'] = np.where(predictions == -1, 1, 0)
    
    return clf, df

def main():
    print("Generating simulated IoT sensor data...")
    df = generate_sensor_data(1000)
    
    print("Applying Isolation Forest anomaly detection...")
    model, df_with_anomalies = train_anomaly_detector(df)
    
    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), "sensor_data.csv")
    df_with_anomalies.to_csv(csv_path, index=False)
    print(f"Sensor data saved to {csv_path}")
    print(f"Detected {df_with_anomalies['is_anomaly'].sum()} anomalies out of {len(df_with_anomalies)} rows.")
    
    # Save Isolation Forest Model
    model_path = os.path.join(os.path.dirname(__file__), "sensor_anomaly_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Anomaly detector model saved to {model_path}")

if __name__ == "__main__":
    main()
