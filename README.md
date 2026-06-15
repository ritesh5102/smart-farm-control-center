# Smart Farm AI/ML Control Center 🌾🤖

A comprehensive, production-ready AI/ML intelligence and telemetry dashboard for modern smart agriculture. This platform integrates real-time IoT anomaly detection, crop recommendation, fertilizer optimization, yield regression, and deep learning leaf disease diagnostics.

---

## 🚀 System Architecture & Features

The platform hosts five distinct machine learning and deep learning pipelines:

1. **Crop Disease Classifier (CNN):**
   - Detects crop diseases (Healthy, Early Blight, Late Blight, and Leaf Mold) from leaf images.
   - Preprocessed using Gaussian Blur and normalized via OpenCV.
   - Built on a 3-layer Convolutional Neural Network (CNN) in TensorFlow/Keras.
   - Achieves **`97.50%`** validation accuracy.

2. **Crop Recommender (Random Forest):**
   - Recommends the top 3 optimal crops based on N, P, K soil nutrients and ambient weather readings.
   - Built on a Random Forest Classifier achieving **`99.55%`** test accuracy.

3. **Fertilizer Suggestion Engine (Random Forest):**
   - Recommends optimal soil additives based on soil composition, moisture, temperature, and crop type.
   - Achieves **`96.33%`** classification accuracy.

4. **Yield Predictor (XGBoost Regressor):**
   - Estimates crop yield in tonnes per hectare and computes gross production.
   - Built on an XGBoost Regressor achieving an R² score of **`90.66%`**.

5. **IoT Sensor Telemetry Anomaly Detector (Isolation Forest):**
   - Analyzes real-time sensor streams to detect outliers and potential hardware faults.
   - Uses an Isolation Forest algorithm with a 5% contamination threshold.

---

## 🛠️ Technology Stack
- **Backend:** Flask, Python 3.11, SQLite
- **Machine Learning & AI:** TensorFlow/Keras, XGBoost, Scikit-learn, OpenCV
- **Analysis & Visualizations:** Matplotlib, Seaborn, Pandas, Numpy
- **Frontend Dashboard:** HTML5, CSS3 (Premium Glassmorphic Dark UI), JavaScript, Chart.js
- **Reverse Proxy & Server:** Nginx, Gunicorn
- **Deployment:** Docker, Docker Compose

---

## 📦 Local Installation & Setup

### Prerequisites
Make sure you have Python 3.11 installed.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Initialize Database & Train Models
Run the training scripts to fit models on the datasets and populate the SQLite database logs:
```bash
# Train Crop Disease CNN
python disease_detection.py

# Train Crop Recommendation
python crop_recommendation.py

# Train Fertilizer Suggestion
python fertilizer_recommendation.py

# Train Yield Prediction
python yield_prediction.py

# Generate Diagnostic Plots
python visualizations.py
```

### Step 3: Start the Flask App
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

## 🐳 Docker Containerization (Production)

To deploy the application to a cloud server using the Nginx reverse proxy and Gunicorn WSGI server:

### Step 1: Run Docker Compose
```bash
docker compose build --no-cache
docker compose up -d
```

### Step 2: Verification
- The application will be served publicly on **port 80** (`http://localhost`).
- Static files are served directly by Nginx for optimized loading speeds.
- Check container health with `docker compose ps` and logs with `docker compose logs -f`.

For detailed cloud setup instructions (firewalls, Ubuntu configuration, SSL setup), refer to [deployment_guide.md](deployment_guide.md).
