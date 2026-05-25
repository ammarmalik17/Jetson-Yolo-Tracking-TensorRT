from ultralytics import YOLO
import os

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load a YOLO11n PyTorch model (check models directory first)
model_path = 'models/yolo11n.pt' if os.path.exists('models/yolo11n.pt') else 'yolo11n.pt'
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# Export the model to TensorRT with conservative settings
# Using static shapes and disabling problematic features
output_path = 'models/yolo11n.engine'
print(f"Exporting TensorRT engine to: {output_path}")
model.export(
    format="engine", 
    half=True, 
    simplify=True, 
    dynamic=False,      # Use static shapes to avoid dimension issues
    batch=1,           # Single batch to minimize memory issues
    workspace=8,       # Minimal workspace for Jetson Orin
    nms=True,
    device=0,          # Explicitly specify GPU device
    verbose=True       # Enable verbose output for debugging
)  # creates 'models/yolo11n.engine'

# Load the exported TensorRT model
print(f"Loading TensorRT engine from: {output_path}")
trt_model = YOLO(output_path)
print("✅ Export complete! Model saved to models/yolo11n.engine")