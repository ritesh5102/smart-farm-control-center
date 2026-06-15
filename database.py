import sqlite3
import datetime
import os

DB_NAME = os.path.join(os.path.dirname(__file__), "smart_farm.db")

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Sensor readings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        temperature REAL NOT NULL,
        humidity REAL NOT NULL,
        soil_moisture REAL NOT NULL,
        ph REAL NOT NULL,
        rainfall REAL NOT NULL,
        is_anomaly INTEGER NOT NULL
    )
    """)
    
    # Crop predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crop_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        N REAL NOT NULL,
        P REAL NOT NULL,
        K REAL NOT NULL,
        temperature REAL NOT NULL,
        humidity REAL NOT NULL,
        ph REAL NOT NULL,
        rainfall REAL NOT NULL,
        recommended_crop_1 TEXT NOT NULL,
        recommended_crop_2 TEXT NOT NULL,
        recommended_crop_3 TEXT NOT NULL
    )
    """)
    
    # Disease detections table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS disease_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        filename TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        confidence REAL NOT NULL
    )
    """)
    
    # Yield predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS yield_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        crop TEXT NOT NULL,
        area REAL NOT NULL,
        season TEXT NOT NULL,
        rainfall REAL NOT NULL,
        fertilizer REAL NOT NULL,
        predicted_yield REAL NOT NULL
    )
    """)
    
    # Fertilizer predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fertilizer_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        N REAL NOT NULL,
        P REAL NOT NULL,
        K REAL NOT NULL,
        ph REAL NOT NULL,
        moisture REAL NOT NULL,
        temperature REAL NOT NULL,
        crop_type TEXT NOT NULL,
        recommended_fertilizer TEXT NOT NULL
    )
    """)
    
    conn.commit()
    conn.close()

def log_sensor_reading(temperature, humidity, soil_moisture, ph, rainfall, is_anomaly):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO sensor_readings (timestamp, temperature, humidity, soil_moisture, ph, rainfall, is_anomaly)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, temperature, humidity, soil_moisture, ph, rainfall, int(is_anomaly)))
    conn.commit()
    conn.close()

def log_crop_prediction(N, P, K, temperature, humidity, ph, rainfall, crop1, crop2, crop3):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO crop_predictions (timestamp, N, P, K, temperature, humidity, ph, rainfall, recommended_crop_1, recommended_crop_2, recommended_crop_3)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, N, P, K, temperature, humidity, ph, rainfall, crop1, crop2, crop3))
    conn.commit()
    conn.close()

def log_disease_detection(filename, predicted_class, confidence):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO disease_detections (timestamp, filename, predicted_class, confidence)
    VALUES (?, ?, ?, ?)
    """, (timestamp, filename, predicted_class, float(confidence)))
    conn.commit()
    conn.close()

def log_yield_prediction(crop, area, season, rainfall, fertilizer, predicted_yield):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO yield_predictions (timestamp, crop, area, season, rainfall, fertilizer, predicted_yield)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, crop, float(area), season, float(rainfall), float(fertilizer), float(predicted_yield)))
    conn.commit()
    conn.close()

def log_fertilizer_prediction(N, P, K, ph, moisture, temperature, crop_type, recommended_fertilizer):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO fertilizer_predictions (timestamp, N, P, K, ph, moisture, temperature, crop_type, recommended_fertilizer)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, float(N), float(P), float(K), float(ph), float(moisture), float(temperature), crop_type, recommended_fertilizer))
    conn.commit()
    conn.close()

def get_last_10(table_name):
    conn = get_connection()
    cursor = conn.cursor()
    # Safely select from one of the known tables (to prevent SQL injection)
    allowed_tables = ["sensor_readings", "crop_predictions", "disease_detections", "yield_predictions", "fertilizer_predictions"]
    if table_name not in allowed_tables:
        raise ValueError(f"Table name {table_name} is not allowed")
    
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 10")
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        result.append(dict(zip(columns, r)))
    return result

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
