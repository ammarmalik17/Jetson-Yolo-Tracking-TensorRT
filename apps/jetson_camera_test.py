import cv2
import time

def test_camera():
    """Simple camera test to verify access"""
    print("Testing camera access...")
    
    # GStreamer pipelines based on working nvgstcapture configuration
    # nvgstcapture uses Camera index = 0 with ARGUS, so we'll match that
    gstreamer_pipelines = [
        # Match the working nvgstcapture configuration (Camera index = 0)
        "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
        
        # Simpler version with direct format
        "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
        
        # Even simpler version
        "nvarguscamerasrc sensor-id=0 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
        
        # Try with different sensor IDs just in case
        "nvarguscamerasrc sensor-id=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
    ]
    
    print("\n=== Testing GStreamer Pipelines (ARGUS Camera) ===")
    for i, pipeline in enumerate(gstreamer_pipelines):
        print(f"\nTrying GStreamer pipeline {i+1}...")
        print(f"Pipeline: {pipeline[:60]}...")
        
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if cap.isOpened():
                print(f"✅ Pipeline opened successfully")
                
                # Try to read frames
                success_count = 0
                for frame_num in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        success_count += 1
                        print(f"  Frame {frame_num+1}: ✅ Success ({frame.shape})")
                    else:
                        print(f"  Frame {frame_num+1}: ❌ Failed")
                    time.sleep(0.3)
                
                if success_count > 0:
                    print(f"✅ Successfully read {success_count}/3 frames!")
                    
                    # Show test video
                    print("Showing test video for 3 seconds... Press 'q' to skip")
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        ret, frame = cap.read()
                        if ret:
                            cv2.imshow('Camera Test - ARGUS', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                    cv2.destroyAllWindows()
                    cap.release()
                    
                    print(f"✅ FOUND WORKING PIPELINE!")
                    return pipeline
                
                cap.release()
            else:
                print(f"❌ Failed to open pipeline")
                
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
    
    print("\n❌ No working camera configuration found!")
    return None

if __name__ == "__main__":
    result = test_camera()
    if result:
        print(f"\n✅ Camera test passed!")
        print(f"Working pipeline found!")
        print("\nYou can now run jetson_camera_inference.py")
    else:
        print("\n❌ Camera test failed.")
        print("Troubleshooting:")
        print("1. Make sure nvgstcapture-1.0 --camsrc=1 is not running")
        print("2. Check OpenCV GStreamer support: python -c 'import cv2; print(cv2.getBuildInformation())'")
        print("3. Try: export GST_DEBUG=3 to get more GStreamer debug info") 