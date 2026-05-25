import cv2
from ultralytics import YOLO
import time
import os

def get_jetson_camera():
    """Get camera using GStreamer pipeline for Jetson ARGUS camera"""
    
    # GStreamer pipelines based on working nvgstcapture (Camera index = 0)
    gstreamer_pipelines = [
        # Match nvgstcapture configuration - sensor-id=0 with scaling
        "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
        
        # Simpler version
        "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
        
        # Minimal version
        "nvarguscamerasrc sensor-id=0 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
    ]
    
    print("Trying GStreamer pipelines for Jetson ARGUS camera...")
    
    for i, pipeline in enumerate(gstreamer_pipelines):
        print(f"Trying pipeline {i+1}...")
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if cap.isOpened():
                # Test if we can read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print("✅ Found working ARGUS camera pipeline!")
                    print(f"   Resolution: {frame.shape}")
                    return cap
                else:
                    cap.release()
        except Exception as e:
            print(f"Pipeline {i+1} error: {e}")
    
    print("❌ No working ARGUS camera pipeline found")
    return None

def main():
    # First, make sure nvgstcapture is not running
    print("Make sure nvgstcapture-1.0 is not running in another terminal...")
    time.sleep(2)
    
    # Load the TensorRT engine model
    print("Loading TensorRT model...")
    # Check models directory first, then fall back to root
    model_path = 'models/yolo11n.engine' if os.path.exists('models/yolo11n.engine') else 'yolo11n.engine'
    try:
        model = YOLO(model_path, task='detect')  # Explicitly specify task
        print(f"✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure yolo11n.engine exists in models/ directory.")
        print("Run 'python tools/export_tensorrt_engine.py' first if needed.")
        return
    
    # Get Jetson ARGUS camera
    cap = get_jetson_camera()
    if cap is None:
        print("❌ No working camera found!")
        print("Troubleshooting:")
        print("1. Make sure nvgstcapture-1.0 is not running")
        print("2. Check OpenCV GStreamer support:")
        print("   python -c 'import cv2; print(cv2.getBuildInformation())' | grep -i gstreamer")
        print("3. Try running test_camera.py first")
        return
    
    print("✅ Camera initialized successfully!")
    print("Starting inference... Press 'q' to quit")
    
    # Initialize FPS counter and timing
    fps_counter = 0
    start_time = time.time()
    frame_count = 0
    inference_time_total = 0
    last_fps = 0.0
    last_avg_inf_time = 0.0
    
    # For smooth FPS calculation
    frame_times = []
    max_frame_history = 30  # Keep last 30 frame times for smoother FPS
    
    try:
        while True:
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not read frame, retrying...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            try:
                # Measure inference time
                inf_start = time.time()
                
                # Run inference using ultralytics predict
                results = model.predict(
                    frame,
                    conf=0.5,      # Confidence threshold
                    verbose=False,  # Disable verbose output
                    show=False,    # Don't show results automatically
                    device=0       # Use GPU
                )
                
                inf_time = time.time() - inf_start
                inference_time_total += inf_time
                
                # Draw results on frame
                annotated_frame = results[0].plot()
                
            except Exception as e:
                print(f"Inference error: {e}")
                annotated_frame = frame
                inf_time = 0
            
            # Calculate real-time FPS using frame times
            current_time = time.time()
            frame_times.append(current_time)
            
            # Keep only recent frame times for smooth FPS calculation
            if len(frame_times) > max_frame_history:
                frame_times.pop(0)
            
            # Calculate current FPS
            if len(frame_times) >= 2:
                time_span = frame_times[-1] - frame_times[0]
                current_fps = (len(frame_times) - 1) / time_span if time_span > 0 else 0
            else:
                current_fps = 0
            
            # Update periodic averages every second for stability
            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                last_fps = fps_counter / elapsed_time
                last_avg_inf_time = inference_time_total / fps_counter if fps_counter > 0 else 0
                fps_counter = 0
                start_time = time.time()
                inference_time_total = 0
            
            # Always display performance info on every frame
            cv2.putText(annotated_frame, f'FPS: {current_fps:.1f}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Avg FPS: {last_fps:.1f}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Inference: {inf_time*1000:.1f}ms', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Avg Inf: {last_avg_inf_time*1000:.1f}ms', 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Frame: {frame_count}', 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, 'Press Q to quit', 
                       (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('YOLO TensorRT Inference - Jetson', annotated_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera inference stopped")

if __name__ == "__main__":
    main() 