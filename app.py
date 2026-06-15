import os
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pandas as pd
import pickle
import sqlite3
import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from flask_cors import CORS

import database

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
plant_disease_model = None
crop_rec_model = None
yield_model_package = None
anomaly_detector = None
fertilizer_model_package = None

CLASSES = ['Healthy', 'Early_Blight', 'Late_Blight', 'Leaf_Mold']

# Helper function to check if models are loaded
def load_models():
    global plant_disease_model, crop_rec_model, yield_model_package, anomaly_detector, fertilizer_model_package
    
    # Initialize database tables
    database.init_db()
    
    base_dir = os.path.dirname(__file__)
    
    # 1. Load Plant Disease Model
    disease_path = os.path.join(base_dir, "plant_disease_model.h5")
    if os.path.exists(disease_path):
        try:
            plant_disease_model = tf.keras.models.load_model(disease_path)
            print("Loaded plant disease CNN model.")
        except Exception as e:
            print(f"Error loading plant disease model: {e}")
            
    # 2. Load Crop Recommendation Model
    crop_rec_path = os.path.join(base_dir, "crop_recommendation_model.pkl")
    if os.path.exists(crop_rec_path):
        try:
            with open(crop_rec_path, 'rb') as f:
                crop_rec_model = pickle.load(f)
            print("Loaded crop recommendation Random Forest model.")
        except Exception as e:
            print(f"Error loading crop recommendation model: {e}")
            
    # 3. Load Yield Prediction Model Package
    yield_path = os.path.join(base_dir, "yield_prediction_model.pkl")
    if os.path.exists(yield_path):
        try:
            with open(yield_path, 'rb') as f:
                yield_model_package = pickle.load(f)
            print("Loaded yield prediction XGBoost model package.")
        except Exception as e:
            print(f"Error loading yield prediction model: {e}")
            
    # 4. Load Anomaly Detector Model
    anomaly_path = os.path.join(base_dir, "sensor_anomaly_model.pkl")
    if os.path.exists(anomaly_path):
        try:
            with open(anomaly_path, 'rb') as f:
                anomaly_detector = pickle.load(f)
            print("Loaded sensor anomaly Isolation Forest model.")
        except Exception as e:
            print(f"Error loading anomaly detector: {e}")
            
    # 5. Load Fertilizer Recommendation Model Package
    fert_path = os.path.join(base_dir, "fertilizer_recommendation_model.pkl")
    if os.path.exists(fert_path):
        try:
            with open(fert_path, 'rb') as f:
                fertilizer_model_package = pickle.load(f)
            print("Loaded fertilizer recommendation Random Forest model package.")
        except Exception as e:
            print(f"Error loading fertilizer model: {e}")

@app.before_request
def startup():
    if plant_disease_model is None:
        load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    if plant_disease_model is None:
        return jsonify({'error': 'CNN model not trained or loaded yet'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # Save file
    filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Preprocess with OpenCV
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({'error': 'Could not read uploaded image'}), 400
            
        img_resized = cv2.resize(img, (128, 128))
        img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
        img_normalized = img_blurred.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        preds = plant_disease_model.predict(img_batch)
        class_idx = np.argmax(preds[0])
        confidence = float(preds[0][class_idx])
        disease_name = CLASSES[class_idx].replace('_', ' ')
        
        # Log to Database
        database.log_disease_detection(filename, disease_name, confidence)
        
        return jsonify({
            'success': True,
            'disease': disease_name,
            'confidence': confidence,
            'image_url': f'/static/uploads/{filename}'
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

@app.route('/recommend-crop', methods=['POST'])
def recommend_crop():
    if crop_rec_model is None:
        return jsonify({'error': 'Crop Recommendation model not trained or loaded yet'}), 500
        
    try:
        data = request.get_json(force=True)
        N = float(data.get('N'))
        P = float(data.get('P'))
        K = float(data.get('K'))
        temp = float(data.get('temperature'))
        hum = float(data.get('humidity'))
        ph = float(data.get('ph'))
        rain = float(data.get('rainfall'))
        
        # Prepare feature vector
        features = np.array([[N, P, K, temp, hum, ph, rain]])
        
        # Predict probabilities to get top 3
        probs = crop_rec_model.predict_proba(features)[0]
        classes = crop_rec_model.classes_
        
        # Sort indices by probability descending
        top_indices = np.argsort(probs)[::-1][:3]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'crop': str(classes[idx]).title(),
                'probability': float(probs[idx])
            })
            
        # Log to Database
        top1 = recommendations[0]['crop']
        top2 = recommendations[1]['crop']
        top3 = recommendations[2]['crop']
        database.log_crop_prediction(N, P, K, temp, hum, ph, rain, top1, top2, top3)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': f'Crop recommendation failed: {e}'}), 400

@app.route('/predict-yield', methods=['POST'])
def predict_yield():
    if yield_model_package is None:
        return jsonify({'error': 'Yield Prediction model package not trained or loaded yet'}), 500
        
    try:
        data = request.get_json(force=True)
        crop = data.get('crop') # String e.g. "Rice"
        area = float(data.get('area')) # Area in hectares
        season = data.get('season') # Season name
        rainfall = float(data.get('rainfall')) # mm
        fertilizer = float(data.get('fertilizer')) # kg
        
        # Extract package info
        model = yield_model_package['model']
        crop_encoder = yield_model_package['crop_encoder']
        season_encoder = yield_model_package['season_encoder']
        
        # Encode inputs
        try:
            crop_encoded = crop_encoder.transform([crop])[0]
        except ValueError:
            # Fallback to nearest/first label if unknown
            crop_encoded = crop_encoder.transform([crop_encoder.classes_[0]])[0]
            
        try:
            season_encoded = season_encoder.transform([season])[0]
        except ValueError:
            season_encoded = season_encoder.transform([season_encoder.classes_[0]])[0]
            
        # Feature array: ['Crop', 'Season', 'Area', 'Rainfall', 'Fertilizer']
        features = pd.DataFrame([[crop_encoded, season_encoded, area, rainfall, fertilizer]], 
                                columns=['Crop', 'Season', 'Area', 'Rainfall', 'Fertilizer'])
        
        # Run XGBoost prediction
        pred_yield = float(model.predict(features)[0])
        pred_yield = max(0.0, pred_yield)
        
        # Calculate total production
        total_production = pred_yield * area
        
        # Log to database
        database.log_yield_prediction(crop, area, season, rainfall, fertilizer, pred_yield)
        
        return jsonify({
            'success': True,
            'predicted_yield': pred_yield,
            'total_production': total_production
        })
    except Exception as e:
        return jsonify({'error': f'Yield prediction failed: {e}'}), 400

@app.route('/recommend-fertilizer', methods=['POST'])
def recommend_fertilizer():
    if fertilizer_model_package is None:
        return jsonify({'error': 'Fertilizer Recommendation model package not trained or loaded yet'}), 500
        
    try:
        data = request.get_json(force=True)
        N = float(data.get('N'))
        P = float(data.get('P'))
        K = float(data.get('K'))
        ph = float(data.get('ph'))
        moisture = float(data.get('moisture'))
        temp = float(data.get('temperature'))
        crop = data.get('crop')
        
        model = fertilizer_model_package['model']
        crop_encoder = fertilizer_model_package['crop_encoder']
        
        # Encode crop type
        try:
            crop_encoded = crop_encoder.transform([crop])[0]
        except ValueError:
            crop_encoded = crop_encoder.transform([crop_encoder.classes_[0]])[0]
            
        features = np.array([[N, P, K, ph, moisture, temp, crop_encoded]])
        pred = model.predict(features)[0]
        
        # Log to database
        database.log_fertilizer_prediction(N, P, K, ph, moisture, temp, crop, pred)
        
        return jsonify({
            'success': True,
            'fertilizer': str(pred)
        })
    except Exception as e:
        return jsonify({'error': f'Fertilizer recommendation failed: {e}'}), 400

@app.route('/sensor-status', methods=['GET'])
def sensor_status():
    # Simulate real-time sensor
    # Normally ranges: temp: 20-40, humidity: 40-90, soil_moisture: 20-80, ph: 5.5-7.5, rain: 0-300
    # Occasionally generate an anomalous reading
    is_anomaly_triggered = np.random.rand() < 0.1
    
    if is_anomaly_triggered:
        anomaly_type = np.random.choice(['temp', 'ph', 'moisture', 'rainfall'])
        if anomaly_type == 'temp':
            temp = np.random.uniform(45.0, 52.0)
            hum = np.random.uniform(30.0, 45.0)
            moist = np.random.uniform(20.0, 35.0)
            ph = np.random.uniform(6.0, 7.0)
            rain = np.random.uniform(0.0, 50.0)
        elif anomaly_type == 'ph':
            temp = np.random.uniform(25.0, 30.0)
            hum = np.random.uniform(70.0, 80.0)
            moist = np.random.uniform(60.0, 70.0)
            ph = np.random.uniform(3.0, 4.2)
            rain = np.random.uniform(10.0, 100.0)
        elif anomaly_type == 'moisture':
            temp = np.random.uniform(28.0, 35.0)
            hum = np.random.uniform(80.0, 90.0)
            moist = np.random.uniform(92.0, 98.0)
            ph = np.random.uniform(6.2, 6.8)
            rain = np.random.uniform(150.0, 250.0)
        else: # rainfall
            temp = np.random.uniform(20.0, 25.0)
            hum = np.random.uniform(85.0, 95.0)
            moist = np.random.uniform(75.0, 85.0)
            ph = np.random.uniform(5.5, 6.0)
            rain = np.random.uniform(420.0, 480.0)
    else:
        temp = np.random.uniform(22.0, 38.0)
        hum = np.random.uniform(45.0, 85.0)
        moist = np.random.uniform(25.0, 75.0)
        ph = np.random.uniform(5.8, 7.2)
        rain = np.random.uniform(5.0, 280.0)
        
    # Check with Isolation Forest anomaly detector
    is_anomaly = 0
    if anomaly_detector is not None:
        features = np.array([[temp, hum, moist, ph, rain]])
        pred = anomaly_detector.predict(features)[0]
        if pred == -1:
            is_anomaly = 1
            
    # Log to Database
    database.log_sensor_reading(temp, hum, moist, ph, rain, is_anomaly)
    
    return jsonify({
        'success': True,
        'temperature': temp,
        'humidity': hum,
        'soil_moisture': moist,
        'ph': ph,
        'rainfall': rain,
        'is_anomaly': bool(is_anomaly)
    })

@app.route('/logs/<table_name>', methods=['GET'])
def get_logs(table_name):
    try:
        logs = database.get_last_10(table_name)
        return jsonify({
            'success': True,
            'logs': logs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    load_models()
    app.run(debug=True, port=5000)
