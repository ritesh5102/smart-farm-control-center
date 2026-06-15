import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Classes configuration
CLASSES = ['Healthy', 'Early_Blight', 'Late_Blight', 'Leaf_Mold']
IMG_SIZE = 128
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset", "crop_disease")

def create_synthetic_leaf(label, save_path):
    # Create black canvas
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    # Draw a stem (brown line)
    cv2.line(img, (IMG_SIZE//2, IMG_SIZE), (IMG_SIZE//2, IMG_SIZE//4), (30, 50, 80), 3)
    
    # Draw leaf base (green ellipse)
    center = (IMG_SIZE//2, IMG_SIZE//2 + 10)
    axes = (35, 50)
    angle = 0
    # Base green color
    green_color = (40, 150, 40)
    cv2.ellipse(img, center, axes, angle, 0, 360, green_color, -1)
    
    # Add disease features
    if label == 'Healthy':
        # Just add some simple veins (light green lines)
        cv2.line(img, center, (center[0]-20, center[1]-20), (60, 200, 60), 2)
        cv2.line(img, center, (center[0]+20, center[1]-20), (60, 200, 60), 2)
        cv2.line(img, (center[0], center[1]-15), (center[0]-15, center[1]-35), (60, 200, 60), 2)
        cv2.line(img, (center[0], center[1]-15), (center[0]+15, center[1]-35), (60, 200, 60), 2)
        
    elif label == 'Early_Blight':
        # Early blight has small concentric brown circles
        np.random.seed(os.getpid() + int(save_path[-8:-4].replace('/', '1').replace('\\', '1').replace('.', '1') or '0') % 1000)
        num_spots = np.random.randint(4, 9)
        for _ in range(num_spots):
            spot_x = np.random.randint(center[0] - 20, center[0] + 20)
            spot_y = np.random.randint(center[1] - 30, center[1] + 30)
            # Brown spot
            cv2.circle(img, (spot_x, spot_y), 4, (10, 40, 70), -1) # Dark brown core
            cv2.circle(img, (spot_x, spot_y), 6, (30, 70, 110), 1)  # Light brown outer ring
            
    elif label == 'Late_Blight':
        # Late blight has large irregular dark-grey/black wet-looking lesions
        np.random.seed(os.getpid() + int(save_path[-8:-4].replace('/', '1').replace('\\', '1').replace('.', '1') or '0') % 1000)
        num_patches = np.random.randint(1, 4)
        for _ in range(num_patches):
            patch_x = np.random.randint(center[0] - 20, center[0] + 20)
            patch_y = np.random.randint(center[1] - 30, center[1] + 30)
            # Draw a dark gray blob
            cv2.circle(img, (patch_x, patch_y), np.random.randint(10, 18), (30, 30, 30), -1)
            # Draw slightly lighter outer ring
            cv2.circle(img, (patch_x, patch_y), np.random.randint(12, 20), (50, 50, 50), 1)
            
    elif label == 'Leaf_Mold':
        # Leaf mold has fuzzy yellow/pale-green spots
        np.random.seed(os.getpid() + int(save_path[-8:-4].replace('/', '1').replace('\\', '1').replace('.', '1') or '0') % 1000)
        num_molds = np.random.randint(3, 7)
        for _ in range(num_molds):
            mold_x = np.random.randint(center[0] - 20, center[0] + 20)
            mold_y = np.random.randint(center[1] - 30, center[1] + 30)
            # Draw diffuse yellowish spots (low opacity/blended look)
            overlay = img.copy()
            cv2.circle(overlay, (mold_x, mold_y), np.random.randint(8, 15), (100, 220, 220), -1) # Yellow-ish
            # Blend
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
            
    # Save image
    cv2.imwrite(save_path, img)

def generate_dataset(images_per_class=150):
    print("Generating synthetic crop disease leaf image dataset...")
    for label in CLASSES:
        class_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(images_per_class):
            filename = f"leaf_{i:04d}.jpg"
            save_path = os.path.join(class_dir, filename)
            create_synthetic_leaf(label, save_path)
    print("Dataset generated successfully.")

def preprocess_image(img_path):
    # Preprocess using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # 1. Resize to target size (128x128)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 2. Apply Gaussian Blur to smooth noise (demonstrating OpenCV preprocess)
    img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
    
    # 3. Normalize pixels (0-1)
    img_normalized = img_blurred.astype('float32') / 255.0
    return img_normalized

def load_data():
    X = []
    y = []
    
    for class_idx, label in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_DIR, label)
        files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        for file in files:
            img_path = os.path.join(class_dir, file)
            img = preprocess_image(img_path)
            if img is not None:
                X.append(img)
                y.append(class_idx)
                
    return np.array(X), np.array(y)

def train_cnn():
    # Make sure dataset is generated
    if not os.path.exists(DATASET_DIR):
        generate_dataset()
        
    X, y = load_data()
    print(f"Loaded {len(X)} images belonging to {len(CLASSES)} classes.")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define CNN architecture
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training CNN model...")
    history = model.fit(
        X_train, y_train,
        epochs=8,
        batch_size=32,
        validation_data=(X_val, y_val)
    )
    
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), "plant_disease_model.h5")
    model.save(model_path)
    print(f"CNN Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    train_cnn()
