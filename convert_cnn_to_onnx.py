import subprocess
import sys
import os

base_dir = os.path.dirname(__file__)
model_h5_path = os.path.join(base_dir, "plant_disease_model.h5")
model_onnx_path = os.path.join(base_dir, "static", "plant_disease_model.onnx")

def run_conversion():
    # 1. Install tf2onnx if not present
    try:
        import tf2onnx
        print("tf2onnx is already installed.")
    except ImportError:
        print("Installing tf2onnx via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tf2onnx"])
        print("tf2onnx installed successfully.")

    # 2. Run tf2onnx CLI converter
    print(f"Converting {model_h5_path} to ONNX format...")
    # Command: python -m tf2onnx.convert --keras-model <path> --output <path>
    cmd = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--keras",
        model_h5_path,
        "--output",
        model_onnx_path
    ]
    
    try:
        subprocess.check_call(cmd)
        print(f"Successfully converted model to ONNX: {model_onnx_path}")
        if os.path.exists(model_onnx_path):
            size_mb = os.path.getsize(model_onnx_path) / (1024 * 1024)
            print(f"ONNX Model size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_conversion()
