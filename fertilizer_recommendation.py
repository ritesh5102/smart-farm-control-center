import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = os.path.join(os.path.dirname(__file__), "fertilizer_data.csv")

def generate_fertilizer_data(n_samples=1200):
    crops = ['Rice', 'Maize', 'Wheat', 'Sugarcane', 'Cotton', 'Jute']
    np.random.seed(42)
    
    data = []
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        n = np.random.uniform(5, 120)
        p = np.random.uniform(5, 120)
        k = np.random.uniform(5, 100)
        ph = np.random.uniform(4.5, 8.5)
        moisture = np.random.uniform(10, 90)
        temp = np.random.uniform(15, 40)
        
        # Heuristics for fertilizer selection
        if ph < 5.2:
            fertilizer = 'Lime + Organic Manure'
        elif ph > 7.8:
            fertilizer = 'Gypsum + Compost'
        elif n < 25:
            fertilizer = 'Urea'
        elif p < 25:
            fertilizer = 'DAP (Diammonium Phosphate)'
        elif k < 20:
            fertilizer = 'MOP (Muriate of Potash)'
        elif n < 50 and p < 50 and k < 50:
            fertilizer = 'NPK 19-19-19'
        elif n > 70 and p > 70 and k < 40:
            fertilizer = 'NPK 10-26-26'
        elif n < 60 and p < 60:
            fertilizer = 'NPK 20-20-0'
        else:
            fertilizer = 'Standard Compost'
            
        data.append([n, p, k, ph, moisture, temp, crop, fertilizer])
        
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'ph', 'moisture', 'temperature', 'crop', 'fertilizer'])
    df.to_csv(CSV_PATH, index=False)
    print("Generated and saved synthetic fertilizer dataset.")
    return df

def train_model():
    df = generate_fertilizer_data(1500)
    
    # Preprocess categorical Crop column
    crop_encoder = LabelEncoder()
    df['crop'] = crop_encoder.fit_transform(df['crop'])
    
    X = df[['N', 'P', 'K', 'ph', 'moisture', 'temperature', 'crop']]
    y = df['fertilizer']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier for fertilizer recommendation...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Fertilizer Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model package
    model_package = {
        'model': model,
        'crop_encoder': crop_encoder,
        'features': ['N', 'P', 'K', 'ph', 'moisture', 'temperature', 'crop']
    }
    
    model_path = os.path.join(os.path.dirname(__file__), "fertilizer_recommendation_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    print(f"Fertilizer model package saved to {model_path}")
    
    return model_package

if __name__ == "__main__":
    train_model()
