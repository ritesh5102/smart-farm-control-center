import pandas as pd
import numpy as np
import pickle
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

DATA_URL = "https://raw.githubusercontent.com/Aswins10/Agricultural-Crop-Yield-in-Indian-States-Dataset/main/crop_yield.csv"
CSV_PATH = os.path.join(os.path.dirname(__file__), "crop_yield.csv")

def download_dataset():
    print(f"Attempting to download Crop Yield Dataset from {DATA_URL}...")
    try:
        response = requests.get(DATA_URL, timeout=15)
        if response.status_code == 200:
            with open(CSV_PATH, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("Successfully downloaded crop yield dataset.")
            # Let's inspect the downloaded file to parse it correctly
            df = pd.read_csv(CSV_PATH)
            # The columns in the dataset are: Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, Yield
            # Some datasets have "Annual_Rainfall" or "Rainfall". We can check and rename/clean it.
            return df
        else:
            raise Exception(f"HTTP response status: {response.status_code}")
    except Exception as e:
        print(f"Download failed due to: {e}. Generating synthetic fallback crop yield dataset...")
        return generate_synthetic_data()

def generate_synthetic_data():
    crops = ['Rice', 'Maize', 'Wheat', 'Sugarcane', 'Cotton', 'Jute']
    seasons = ['Kharif', 'Rabi', 'Summer', 'Whole Year']
    
    np.random.seed(42)
    n_samples = 1500
    
    data = []
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        season = np.random.choice(seasons)
        area = np.random.uniform(0.5, 100.0) # hectares
        rainfall = np.random.uniform(50, 2500) # mm
        fertilizer = area * np.random.uniform(50, 150) # kg
        
        # Calculate yield with some logical relation
        # base yield per hectare
        base_yields = {'Rice': 3.5, 'Maize': 4.0, 'Wheat': 3.2, 'Sugarcane': 70.0, 'Cotton': 2.0, 'Jute': 2.5}
        y_base = base_yields[crop]
        
        # impact of rainfall
        rain_factor = 1.0 - abs(rainfall - 1200) / 3000
        # impact of fertilizer per unit area
        fert_per_ha = fertilizer / area
        fert_factor = 0.8 + 0.4 * (fert_per_ha / 150)
        
        yield_per_ha = y_base * rain_factor * fert_factor + np.random.normal(0, y_base * 0.1)
        yield_per_ha = max(0.1, yield_per_ha)
        
        production = yield_per_ha * area
        
        # Columns match: Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, Yield
        data.append([crop, 2020, season, 'StateX', area, production, rainfall, fertilizer, area * 5, yield_per_ha])
        
    df = pd.DataFrame(data, columns=[
        'Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield'
    ])
    df.to_csv(CSV_PATH, index=False)
    print("Generated and saved synthetic Crop Yield Dataset.")
    return df

def train_model():
    if os.path.exists(CSV_PATH):
        print("Loading existing crop yield dataset from local file...")
        df = pd.read_csv(CSV_PATH)
    else:
        df = download_dataset()
        
    # Standardize column names (different versions might have minor differences)
    # The columns should be: Crop, Season, Area, Rainfall, Fertilizer, and Yield.
    # Check if columns exist and map them
    rename_dict = {}
    for col in df.columns:
        if col.lower() in ['crop', 'croptype']:
            rename_dict[col] = 'Crop'
        elif col.lower() in ['season']:
            rename_dict[col] = 'Season'
        elif col.lower() in ['area']:
            rename_dict[col] = 'Area'
        elif col.lower() in ['annual_rainfall', 'rainfall']:
            rename_dict[col] = 'Rainfall'
        elif col.lower() in ['fertilizer', 'fertiliser']:
            rename_dict[col] = 'Fertilizer'
        elif col.lower() in ['yield']:
            rename_dict[col] = 'Yield'
            
    df = df.rename(columns=rename_dict)
    
    # If Yield isn't present but Production and Area are, compute it
    if 'Yield' not in df.columns and 'Production' in df.columns and 'Area' in df.columns:
        df['Yield'] = df['Production'] / df['Area']
        
    # Filter features
    required_cols = ['Crop', 'Season', 'Area', 'Rainfall', 'Fertilizer', 'Yield']
    df = df[required_cols].dropna()
    
    # Categorical encoders
    crop_encoder = LabelEncoder()
    season_encoder = LabelEncoder()
    
    df['Crop'] = crop_encoder.fit_transform(df['Crop'])
    df['Season'] = season_encoder.fit_transform(df['Season'])
    
    # Train test split
    X = df[['Crop', 'Season', 'Area', 'Rainfall', 'Fertilizer']]
    y = df['Yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Regressor model...")
    model = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f} tonnes/hectare")
    print(f"R² Score: {r2:.4f}")
    
    # Save the model package
    model_package = {
        'model': model,
        'crop_encoder': crop_encoder,
        'season_encoder': season_encoder,
        'features': ['Crop', 'Season', 'Area', 'Rainfall', 'Fertilizer']
    }
    
    model_path = os.path.join(os.path.dirname(__file__), "yield_prediction_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    print(f"Model package saved to {model_path}")
    
    return model_package

if __name__ == "__main__":
    train_model()
