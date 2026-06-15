import os
import cv2
import numpy as np

# Re-use the exact same drawing logic from disease_detection.py
IMG_SIZE = 128
CLASSES = ['Healthy', 'Early_Blight', 'Late_Blight', 'Leaf_Mold']
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "static", "test_leaves")

def create_synthetic_leaf(label, save_path):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    # Draw a stem
    cv2.line(img, (IMG_SIZE//2, IMG_SIZE), (IMG_SIZE//2, IMG_SIZE//4), (30, 50, 80), 3)
    
    # Draw leaf base (green ellipse)
    center = (IMG_SIZE//2, IMG_SIZE//2 + 10)
    axes = (35, 50)
    angle = 0
    green_color = (40, 150, 40)
    cv2.ellipse(img, center, axes, angle, 0, 360, green_color, -1)
    
    # Add disease features
    if label == 'Healthy':
        cv2.line(img, center, (center[0]-20, center[1]-20), (60, 200, 60), 2)
        cv2.line(img, center, (center[0]+20, center[1]-20), (60, 200, 60), 2)
        cv2.line(img, (center[0], center[1]-15), (center[0]-15, center[1]-35), (60, 200, 60), 2)
        cv2.line(img, (center[0], center[1]-15), (center[0]+15, center[1]-35), (60, 200, 60), 2)
        
    elif label == 'Early_Blight':
        # Concentric brown rings
        cv2.circle(img, (center[0]-10, center[1]-10), 4, (10, 40, 70), -1)
        cv2.circle(img, (center[0]-10, center[1]-10), 6, (30, 70, 110), 1)
        
        cv2.circle(img, (center[0]+12, center[1]+15), 5, (10, 40, 70), -1)
        cv2.circle(img, (center[0]+12, center[1]+15), 7, (30, 70, 110), 1)
        
        cv2.circle(img, (center[0]-8, center[1]+20), 4, (10, 40, 70), -1)
        cv2.circle(img, (center[0]-8, center[1]+20), 6, (30, 70, 110), 1)
        
    elif label == 'Late_Blight':
        # Large dark gray/black wet-looking spots
        cv2.circle(img, (center[0]-5, center[1]-15), 14, (30, 30, 30), -1)
        cv2.circle(img, (center[0]-5, center[1]-15), 16, (50, 50, 50), 1)
        
        cv2.circle(img, (center[0]+10, center[1]+10), 12, (30, 30, 30), -1)
        cv2.circle(img, (center[0]+10, center[1]+10), 14, (50, 50, 50), 1)
        
    elif label == 'Leaf_Mold':
        # Fuzzy yellowish mold patches
        for offset in [(-12, -15), (10, -10), (2, 12)]:
            overlay = img.copy()
            cv2.circle(overlay, (center[0]+offset[0], center[1]+offset[1]), 12, (100, 220, 220), -1)
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
            
    # Save image
    cv2.imwrite(save_path, img)

def generate_leaves():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating sample test leaves inside {OUTPUT_DIR}...")
    for label in CLASSES:
        filename = f"{label.lower()}_leaf.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        create_synthetic_leaf(label, save_path)
        print(f"Saved {filename}")
    print("Done!")

if __name__ == "__main__":
    generate_leaves()
