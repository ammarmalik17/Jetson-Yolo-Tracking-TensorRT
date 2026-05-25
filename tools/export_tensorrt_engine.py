from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt")

# Export the model to TensorRT with conservative settings
# Using static shapes and disabling problematic features
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
 )  # creates 'yolo11n.engine'

# Load the exported TensorRT model
trt_model = YOLO("yolo11n.engine")