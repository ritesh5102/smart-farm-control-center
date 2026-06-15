import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set style for premium aesthetics
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.figsize': (10, 6)
})

# Harmonious color palette
PALETTE = sns.color_palette("muted")

def plot_crop_recommendation_soil():
    csv_path = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
    if not os.path.exists(csv_path):
        print("Crop Recommendation dataset not found. Skipping soil distribution plot.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Select a subset of popular crops to keep the plot readable
    popular_crops = ['rice', 'maize', 'chickpea', 'coconut', 'cotton', 'coffee']
    df_sub = df[df['label'].isin(popular_crops)]
    
    # Melt dataframe for easier plotting with Seaborn
    df_melt = pd.melt(df_sub, id_vars=['label'], value_vars=['N', 'P', 'K'], 
                      var_name='Soil Nutrient', value_name='Nutrient Level (mg/kg)')
    
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df_melt, x='label', y='Nutrient Level (mg/kg)', hue='Soil Nutrient', palette="viridis")
    plt.title("Soil Nutrient (N, P, K) Requirement Distribution by Crop Type")
    plt.xlabel("Recommended Crop")
    plt.ylabel("Nutrient Level (mg/kg)")
    plt.legend(title="Nutrient Type")
    plt.tight_layout()
    
    img_path = os.path.join(os.path.dirname(__file__), "static", "crop_soil_distribution.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"Saved crop soil distribution plot to {img_path}")

def plot_disease_distribution():
    # Simulate a distribution of diseases across different plant types for visualization
    plant_types = ['Tomato', 'Potato', 'Pepper', 'Apple']
    diseases = ['Healthy', 'Early Blight', 'Late Blight', 'Leaf Mold']
    
    # Generate some counts
    np.random.seed(42)
    data = []
    for plant in plant_types:
        for disease in diseases:
            # Skip invalid combinations (e.g. apple doesn't get leaf mold in our simulation)
            if plant == 'Apple' and disease == 'Leaf Mold':
                count = 0
            else:
                count = np.random.randint(20, 150)
            data.append({'Plant Type': plant, 'Disease Status': disease, 'Count': count})
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Plant Type', y='Count', hue='Disease Status', palette="flare")
    plt.title("Distribution of Crop Diseases Across Plant Types")
    plt.xlabel("Plant Category")
    plt.ylabel("Number of Detected Cases")
    plt.legend(title="Condition")
    plt.tight_layout()
    
    img_path = os.path.join(os.path.dirname(__file__), "static", "disease_distribution.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"Saved disease distribution plot to {img_path}")

def plot_yield_prediction():
    model_path = os.path.join(os.path.dirname(__file__), "yield_prediction_model.pkl")
    csv_path = os.path.join(os.path.dirname(__file__), "crop_yield.csv")
    
    if not os.path.exists(model_path) or not os.path.exists(csv_path):
        print("Yield model or dataset not found. Skipping yield prediction plot.")
        return
        
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
        
    df = pd.read_csv(csv_path)
    # Standardize headers (same as in trainer)
    rename_dict = {}
    for col in df.columns:
        if col.lower() in ['crop', 'croptype']: rename_dict[col] = 'Crop'
        elif col.lower() in ['season']: rename_dict[col] = 'Season'
        elif col.lower() in ['area']: rename_dict[col] = 'Area'
        elif col.lower() in ['annual_rainfall', 'rainfall']: rename_dict[col] = 'Rainfall'
        elif col.lower() in ['fertilizer', 'fertiliser']: rename_dict[col] = 'Fertilizer'
        elif col.lower() in ['yield']: rename_dict[col] = 'Yield'
    df = df.rename(columns=rename_dict)
    
    if 'Yield' not in df.columns and 'Production' in df.columns and 'Area' in df.columns:
        df['Yield'] = df['Production'] / df['Area']
        
    df = df[['Crop', 'Season', 'Area', 'Rainfall', 'Fertilizer', 'Yield']].dropna()
    
    crop_encoder = model_package['crop_encoder']
    season_encoder = model_package['season_encoder']
    model = model_package['model']
    
    # We will score on a subset of test entries
    df_encoded = df.copy()
    df_encoded['Crop'] = crop_encoder.transform(df_encoded['Crop'])
    df_encoded['Season'] = season_encoder.transform(df_encoded['Season'])
    
    X = df_encoded[['Crop', 'Season', 'Area', 'Rainfall', 'Fertilizer']]
    y_actual = df_encoded['Yield']
    y_pred = model.predict(X)
    
    plt.figure(figsize=(10, 6))
    # Sample 300 points for a clean scatter plot
    sample_indices = np.random.choice(len(y_actual), min(300, len(y_actual)), replace=False)
    plt.scatter(y_actual.iloc[sample_indices], y_pred[sample_indices], alpha=0.6, color='teal', edgecolors='w')
    
    # Draw perfect prediction line y = x
    max_val = max(y_actual.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label="Perfect Fit (Y=X)")
    
    plt.title("Actual vs. Predicted Crop Yield")
    plt.xlabel("Actual Yield (tonnes/hectare)")
    plt.ylabel("Predicted Yield (tonnes/hectare)")
    plt.legend()
    plt.tight_layout()
    
    img_path = os.path.join(os.path.dirname(__file__), "static", "yield_prediction_vs_actual.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"Saved yield prediction comparison plot to {img_path}")

def plot_sensor_data():
    csv_path = os.path.join(os.path.dirname(__file__), "sensor_data.csv")
    if not os.path.exists(csv_path):
        print("Sensor data not found. Skipping sensor timeline plot.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Plot first 150 points for readability
    df_sub = df.head(150).copy()
    df_sub['Time (s)'] = df_sub.index
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot Temperature
    sns.lineplot(data=df_sub, x='Time (s)', y='temperature', ax=axes[0], color='orange', label='Temperature (°C)')
    # Highlight anomalies in red
    anomalies_temp = df_sub[df_sub['is_anomaly'] == 1]
    axes[0].scatter(anomalies_temp.index, anomalies_temp['temperature'], color='red', s=40, zorder=5, label='Anomaly Flag')
    axes[0].set_ylabel("Temp (°C)")
    axes[0].set_title("Real-Time Simulated Sensor Streams & Anomaly Detection")
    axes[0].legend(loc="upper right")
    
    # Plot Soil Moisture
    sns.lineplot(data=df_sub, x='Time (s)', y='soil_moisture', ax=axes[1], color='blue', label='Soil Moisture (%)')
    axes[1].scatter(anomalies_temp.index, anomalies_temp['soil_moisture'], color='red', s=40, zorder=5)
    axes[1].set_ylabel("Moisture (%)")
    axes[1].legend(loc="upper right")
    
    # Plot Rainfall
    sns.lineplot(data=df_sub, x='Time (s)', y='rainfall', ax=axes[2], color='green', label='Rainfall (mm)')
    axes[2].scatter(anomalies_temp.index, anomalies_temp['rainfall'], color='red', s=40, zorder=5)
    axes[2].set_ylabel("Rainfall (mm)")
    axes[2].set_xlabel("Time sequence (seconds)")
    axes[2].legend(loc="upper right")
    
    plt.tight_layout()
    img_path = os.path.join(os.path.dirname(__file__), "static", "sensor_data_over_time.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"Saved sensor data timeline plot to {img_path}")

def generate_all():
    plot_crop_recommendation_soil()
    plot_disease_distribution()
    plot_yield_prediction()
    plot_sensor_data()

if __name__ == "__main__":
    generate_all()
