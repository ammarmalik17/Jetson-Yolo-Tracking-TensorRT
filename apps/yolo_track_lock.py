#!/usr/bin/env python
from ultralytics import YOLO
import cv2
import argparse
import os
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

# Define class names for the dataset
CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

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

def parse_classes(classes_arg):
    """
    Parse the classes argument to handle both integer indices and string class names.
    Returns a list of integers (class indices).
    """
    if not classes_arg:
        return None
        
    class_indices = []
    for cls in classes_arg:
        try:
            # Try to parse as integer
            class_idx = int(cls)
            if 0 <= class_idx < len(CLASSES):
                class_indices.append(class_idx)
        except ValueError:
            # Try to match as string
            if cls in CLASSES:
                class_idx = CLASSES.index(cls)
                class_indices.append(class_idx)
    
    return class_indices if class_indices else None

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO tracking with object locking capability")
    # Model and source parameters
    parser.add_argument('--model', type=str, default='models/yolo11n.engine', help='Path to the YOLO model (TensorRT engine preferred)')
    parser.add_argument('--source', type=str, required=True, help='Path to the video file, image, directory, URL, or device ID')
    
    # Detection and tracking parameters
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='Maximum number of detections per image')
    parser.add_argument('--classes', nargs='+', help='Filter by class: --classes 0, or --classes car truck')
    parser.add_argument('--agnostic-nms', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--tracker', type=str, default='botsort.yaml', help='Tracker config file (botsort.yaml or bytetrack.yaml)')
    
    # Image processing parameters
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='Image size as [height, width] or single int for square')
    parser.add_argument('--half', action='store_true', help='Use FP16 half-precision inference')
    parser.add_argument('--augment', action='store_true', help='Apply test-time augmentation')
    parser.add_argument('--visualize', action='store_true', help='Visualize model features')
    parser.add_argument('--retina-masks', action='store_true', help='Use high-resolution segmentation masks')
    
    # Video parameters
    parser.add_argument('--vid-stride', type=int, default=1, help='Video frame-rate stride')
    parser.add_argument('--stream-buffer', action='store_true', help='Buffer all streaming frames')
    parser.add_argument('--batch', type=int, default=32, help='Batch size for processing')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='tracking_output', help='Directory to save results')
    parser.add_argument('--save-vid', action='store_true', help='Save annotated video')
    parser.add_argument('--save-tracks', action='store_true', help='Save tracking data')
    
    # Tracking visualization parameters
    parser.add_argument('--track-history', type=int, default=30, help='Number of frames to keep in track history')
    parser.add_argument('--show-tracks', action='store_true', help='Show tracking lines')
    parser.add_argument('--track-thickness', type=int, default=2, help='Thickness of tracking lines')
    
    # Execution parameters
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--display', action='store_true', default=True, help='Display video during processing')
    parser.add_argument('--verbose', action='store_true', default=True, help='Print verbose output')
    parser.add_argument('--show-fps', action='store_true', default=True, help='Display FPS counter on video')
    
    # Object locking parameters
    parser.add_argument('--enable-lock', action='store_true', help='Enable object locking mode')
    parser.add_argument('--lock-id', type=int, default=None, help='Lock onto a specific object ID')
    parser.add_argument('--lock-class', type=str, default=None, help='Lock onto objects of this class')
    parser.add_argument('--highlight-color', type=str, default='255,0,0', help='RGB color for highlighting locked object (comma-separated)')
    
    args = parser.parse_args()
    # Process class names/indices
    args.classes = parse_classes(args.classes)
    
    # Parse highlight color
    if args.highlight_color:
        args.highlight_color = [int(c) for c in args.highlight_color.split(',')]
        # Convert to BGR (OpenCV format)
        args.highlight_color = args.highlight_color[::-1]
    
    return args

def main():
    args = parse_args()
    
    # Convert imgsz to proper format
    imgsz = args.imgsz
    if len(imgsz) == 1:
        imgsz = imgsz[0]
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {args.model}...")
    try:
        if args.model.endswith('.engine'):
            # TensorRT engine
            model = YOLO(args.model, task='detect')
            print("✅ TensorRT engine loaded successfully")
        else:
            # Regular PyTorch model
            model = YOLO(args.model)
            print("✅ PyTorch model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        if args.model.endswith('.engine'):
            print("Make sure the TensorRT engine exists. Run export_tensorrt_engine.py first if needed.")
        return
    
    # Process video file or camera
    if os.path.isfile(args.source) and args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Video file
        process_video_with_locking(model, args)
    elif args.source.lower() in ['camera', 'jetson', 'camsrc=1']:
        # Jetson ARGUS camera
        print("Using Jetson ARGUS camera (equivalent to nvgstcapture --camsrc=1)")
        process_video_with_locking(model, args, is_camera=True, use_jetson_camera=True)
    elif args.source.isdigit() or args.source.startswith(('rtsp://', 'http://', 'https://', 'rtmp://')):
        # Camera or streaming source
        print(f"Using camera/stream source: {args.source}")
        process_video_with_locking(model, args, is_camera=True)
    else:
        print("Source must be a video file (.mp4, .avi, .mov, .mkv), 'camera'/'jetson' for ARGUS camera, or camera ID (0, 1, etc.) or stream URL")

def process_video_with_locking(model, args, is_camera=False, use_jetson_camera=False):
    # Open the video or camera
    print(f"Processing {'camera/stream' if is_camera else 'video'}: {args.source}")
    
    if use_jetson_camera:
        # Use Jetson ARGUS camera with GStreamer pipeline
        print("Make sure nvgstcapture-1.0 is not running in another terminal...")
        time.sleep(2)
        cap = get_jetson_camera()
        if cap is None:
            print("❌ No working Jetson camera found!")
            print("Troubleshooting:")
            print("1. Make sure nvgstcapture-1.0 is not running")
            print("2. Check OpenCV GStreamer support:")
            print("   python -c 'import cv2; print(cv2.getBuildInformation())' | grep -i gstreamer")
            print("3. Try using regular camera source (0, 1, etc.) instead of 'camera'")
            return
        print("✅ Jetson ARGUS camera initialized successfully!")
    else:
        # Use regular camera or video file
        cap = cv2.VideoCapture(args.source if not args.source.isdigit() else int(args.source))
        if not cap.isOpened():
            print(f"Error: Could not open {'camera' if is_camera else 'video'} {args.source}")
            return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # For camera sources, fps might be 0, set a default
    if fps <= 0:
        fps = 30.0
        print(f"Camera FPS not available, using default: {fps}")
    
    # For cameras, total_frames is unknown
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else float('inf')
    
    # Apply vid_stride
    actual_fps = fps / args.vid_stride
    
    # Convert imgsz to proper format
    imgsz = args.imgsz
    if len(imgsz) == 1:
        imgsz = imgsz[0]
    
    # Setup video writer if saving
    output_writer = None
    output_path = None
    if args.save_vid:
        if is_camera:
            # For camera, use timestamp for output filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(args.output_dir, f"camera_{timestamp}_locked.mp4")
        else:
            # For video file, use source filename
            video_name = Path(args.source).stem
            output_path = os.path.join(args.output_dir, f"{video_name}_locked.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))
    
    # Setup track history for visualization
    track_history = defaultdict(lambda: [])
    
    # Setup track data file if saving tracks
    tracks_file = None
    if args.save_tracks:
        if is_camera:
            # For camera, use timestamp for output filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            tracks_path = os.path.join(args.output_dir, f"camera_{timestamp}_tracks.txt")
        else:
            # For video file, use source filename
            video_name = Path(args.source).stem
            tracks_path = os.path.join(args.output_dir, f"{video_name}_tracks.txt")
        
        tracks_file = open(tracks_path, 'w')
        # Write header
        tracks_file.write("frame,track_id,class_id,class_name,x,y,w,h,confidence,locked\n")
    
    # Process frames
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    # FPS calculation variables
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0
    
    # For smooth real-time FPS calculation like camera_inference.py
    frame_times = []
    max_frame_history = 30  # Keep last 30 frame times for smoother FPS
    real_time_fps = 0.0
    
    # Locking variables
    locked_id = args.lock_id
    locked_object_info = None
    locked_class_id = None
    
    # Mouse click variables
    mouse_x, mouse_y = 0, 0
    mouse_clicked = False
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y, mouse_clicked
        if event == cv2.EVENT_LBUTTONDOWN and args.enable_lock:
            mouse_x, mouse_y = x, y
            mouse_clicked = True
    
    # If a class name is provided for locking, convert it to a class ID
    if args.lock_class:
        if args.lock_class in CLASSES:
            locked_class_id = CLASSES.index(args.lock_class)
        else:
            try:
                locked_class_id = int(args.lock_class)
                if locked_class_id < 0 or locked_class_id >= len(CLASSES):
                    locked_class_id = None
            except ValueError:
                locked_class_id = None
    
    print(f"Object locking mode: {'Enabled' if args.enable_lock else 'Disabled'}")
    if locked_id is not None:
        print(f"Initially locked onto object ID: {locked_id}")
    if locked_class_id is not None:
        print(f"Looking to lock onto class: {CLASSES[locked_class_id]}")
    
    # Create a window with a trackbar for object selection
    if args.display:
        cv2.namedWindow("YOLO11n Tracking with Object Locking")
        # Set the mouse callback
        cv2.setMouseCallback("YOLO11n Tracking with Object Locking", mouse_callback)
    
    paused = False
    input_lock_id = ""  # Buffer for keyboard input of lock ID
    entering_id = False  # Flag to indicate if user is currently entering an ID
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Apply frame stride
            if frame_count % args.vid_stride != 0:
                continue
            
            processed_count += 1
            fps_counter += 1
        
        # Calculate real-time FPS using frame times (like camera_inference.py)
        current_time = time.time()
        frame_times.append(current_time)
        
        # Keep only recent frame times for smooth FPS calculation
        if len(frame_times) > max_frame_history:
            frame_times.pop(0)
        
        # Calculate current real-time FPS
        if len(frame_times) >= 2:
            time_span = frame_times[-1] - frame_times[0]
            real_time_fps = (len(frame_times) - 1) / time_span if time_span > 0 else 0
        else:
            real_time_fps = 0
        
        # Update average FPS every second
        if time.time() - fps_timer >= 1.0:
            current_fps = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()
            if args.verbose:
                print(f"Real-time FPS: {real_time_fps:.2f}, Average FPS: {current_fps:.2f}")
        
        if processed_count % 10 == 0 and args.verbose and not paused:
            progress = frame_count / total_frames * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Run tracking on current frame
        results = model.track(
            source=frame,
            conf=args.conf_thres,
            batch=args.batch,
            iou=args.iou_thres,
            imgsz=imgsz,
            half=args.half,
            device=args.device,
            max_det=args.max_det,
            visualize=args.visualize,
            augment=args.augment,
            agnostic_nms=args.agnostic_nms,
            classes=args.classes,
            retina_masks=args.retina_masks,
            tracker=args.tracker,
            persist=True,  # Enable persistent tracking
            verbose=False
        )
        
        # Create a clean frame for drawing (copy of original)
        annotated_frame = frame.copy()
        
        # Check if we have tracking results
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            
            # Handle mouse click to select object
            if mouse_clicked and args.enable_lock:
                # Find the object closest to the click point
                min_distance = float('inf')
                closest_id = None
                
                for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                    x, y, w, h = box
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    
                    # Check if click is inside bounding box
                    if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                        # Calculate distance to center of box
                        center_x, center_y = int(x), int(y)
                        distance = ((mouse_x - center_x) ** 2 + (mouse_y - center_y) ** 2) ** 0.5
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_id = track_id
                
                if closest_id is not None:
                    locked_id = closest_id
                    print(f"Locked onto object ID: {locked_id} (selected by mouse click)")
                
                mouse_clicked = False
            
            # Auto-lock on first object of specified class if no object is locked yet
            if args.enable_lock and locked_id is None and locked_class_id is not None:
                for i, (box, track_id, class_id) in enumerate(zip(boxes, track_ids, class_ids)):
                    if class_id == locked_class_id:
                        locked_id = track_id
                        print(f"Auto-locked onto object ID {locked_id} of class {CLASSES[class_id]}")
                        break
            
            # For each detected object
            locked_object_exists = False
            
            for i, (box, track_id, class_id, conf) in enumerate(zip(boxes, track_ids, class_ids, confidences)):
                x, y, w, h = box
                
                # Check if this is the locked object
                is_locked = args.enable_lock and locked_id is not None and track_id == locked_id
                
                if is_locked:
                    locked_object_exists = True
                    locked_object_info = {
                        'id': track_id,
                        'class_id': class_id,
                        'class_name': CLASSES[class_id] if class_id < len(CLASSES) else f"class_{class_id}",
                        'box': (int(x - w/2), int(y - h/2), int(w), int(h)),
                        'center': (int(x), int(y)),
                        'conf': conf
                    }
                
                # Only draw bounding box if not in lock mode or if this is the locked object
                if not args.enable_lock or locked_id is None or is_locked:
                    # Draw bounding box
                    color = args.highlight_color if is_locked else (0, 255, 0)  # Default green for unlocked, specified color for locked
                    thickness = 3 if is_locked else 2
                    
                    # Convert from center format to top-left format for drawing
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw label
                    class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"class_{class_id}"
                    label = f"{class_name} {track_id} {conf:.2f}"
                    label_color = (0, 0, 0)  # Black text
                    label_bg_color = color
                    
                    # Get text size for background rectangle
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    # Draw label background
                    cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), label_bg_color, -1)
                    
                    # Draw text
                    cv2.putText(annotated_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
                    
                    # Draw track history if requested
                    if args.show_tracks:
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        
                        # Remove old points
                        if len(track) > args.track_history:
                            track.pop(0)
                        
                        # Draw tracking line
                        if len(track) > 1:
                            # Convert track history to points array for polylines
                            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(
                                annotated_frame, 
                                [points], 
                                isClosed=False, 
                                color=(230, 230, 230) if not is_locked else (128, 0, 255),  # White for regular, purple for locked
                                thickness=args.track_thickness
                            )
                
                # Save track data if requested
                if args.save_tracks and tracks_file:
                    class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"class_{class_id}"
                    is_locked_int = 1 if is_locked else 0
                    tracks_file.write(f"{frame_count},{track_id},{class_id},{class_name},{x},{y},{w},{h},{conf},{is_locked_int}\n")
            
            # If we had a locked object but it's gone, display a message
            if args.enable_lock and locked_id is not None and not locked_object_exists:
                cv2.putText(
                    annotated_frame,
                    f"Lost locked object ID: {locked_id}",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),  # Red
                    2
                )
        
        # Add status info to the frame
        if args.enable_lock:
            lock_status = f"Locked: {locked_id if locked_id is not None else 'None'}"
            if locked_object_info:
                lock_status += f" ({locked_object_info['class_name']})"
            cv2.putText(
                annotated_frame,
                lock_status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),  # Yellow
                2
            )
            
            # Show current input if user is entering an ID
            if entering_id:
                cv2.putText(
                    annotated_frame,
                    f"Enter ID: {input_lock_id}_",
                    (width - 250, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 0),  # Cyan
                    2
                )
            else:
                # Show mouse click instruction
                cv2.putText(
                    annotated_frame,
                    "Click on object to lock",
                    (width - 250, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),  # Light gray
                    1
                )
        
        # Add FPS counter to the frame if requested
        if args.show_fps:
            fps_y = 80 if args.enable_lock else 40
            cv2.putText(
                annotated_frame,
                f"FPS: {real_time_fps:.1f}",
                (20, fps_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),  # Green
                2
            )
            cv2.putText(
                annotated_frame,
                f"Avg FPS: {current_fps:.1f}",
                (20, fps_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),  # Green
                2
            )
        
        # Add pause indicator if paused
        if paused:
            cv2.putText(
                annotated_frame,
                "PAUSED",
                (width - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # Red
                2
            )
        
        # Save frame to video if requested
        if args.save_vid and output_writer is not None and not paused:
            output_writer.write(annotated_frame)
        
        # Display if requested
        if args.display:
            cv2.imshow("YOLOv8 Tracking with Object Locking", annotated_frame)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('p') or key == 32:  # 'p' or space key
                paused = not paused
                print(f"Video {'paused' if paused else 'resumed'}")
            elif key == ord('l'):
                args.enable_lock = not args.enable_lock
                print(f"Object locking {'enabled' if args.enable_lock else 'disabled'}")
            elif key == ord('c'):
                # Clear the locked object
                locked_id = None
                print("Cleared locked object")
            elif key == ord('i'):
                # Start entering an ID
                entering_id = True
                input_lock_id = ""
                print("Enter the track ID to lock (press Enter when done)")
            elif entering_id:
                if key == 13:  # Enter key
                    # Finish entering ID
                    entering_id = False
                    if input_lock_id:
                        try:
                            new_id = int(input_lock_id)
                            if new_id in track_ids:
                                locked_id = new_id
                                print(f"Locked onto object ID: {locked_id}")
                            else:
                                print(f"ID {new_id} not found in current tracks")
                        except ValueError:
                            print("Invalid ID format")
                elif key == 27:  # Escape key
                    # Cancel entering ID
                    entering_id = False
                    print("Cancelled ID input")
                elif key == 8 or key == 127:  # Backspace or Delete
                    # Remove last character
                    input_lock_id = input_lock_id[:-1] if input_lock_id else ""
                elif 48 <= key <= 57:  # Number keys 0-9
                    # Add digit to ID
                    input_lock_id += chr(key)
            elif key == ord('s') and not paused:
                # Save the current frame as an image
                frame_path = os.path.join(args.output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, annotated_frame)
                print(f"Saved frame to {frame_path}")
    
    # Calculate performance
    elapsed_time = time.time() - start_time
    fps_processing = processed_count / elapsed_time
    print(f"Processed {processed_count} frames in {elapsed_time:.2f} seconds ({fps_processing:.2f} FPS)")
    
    # Save tracking results summary
    if is_camera:
        # For camera, use timestamp for output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(args.output_dir, f"camera_{timestamp}_tracking_summary.txt")
    else:
        # For video file, use source filename
        video_name = Path(args.source).stem
        results_path = os.path.join(args.output_dir, f"{video_name}_tracking_summary.txt")
    
    with open(results_path, 'w') as f:
        f.write(f"Source: {args.source}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Tracker: {args.tracker}\n")
        if not is_camera:
            f.write(f"Total frames: {total_frames}\n")
        f.write(f"Processed frames: {processed_count}\n")
        f.write(f"Frame stride: {args.vid_stride}\n")
        f.write(f"Processing time: {elapsed_time:.2f} seconds\n")
        f.write(f"Processing FPS: {fps_processing:.2f}\n")
        f.write(f"Object locking: {'Enabled' if args.enable_lock else 'Disabled'}\n")
        if locked_id is not None:
            f.write(f"Last locked object: ID {locked_id}\n")
    
    # Cleanup
    cap.release()
    if output_writer is not None:
        output_writer.release()
    if tracks_file:
        tracks_file.close()
    cv2.destroyAllWindows()
    
    print("Tracking complete!")
    if args.save_vid:
        print(f"Output video saved to: {output_path}")
    if args.save_tracks:
        print(f"Track data saved to: {tracks_path if 'tracks_path' in locals() else 'N/A'}")
    print(f"Tracking summary saved to: {results_path}")

if __name__ == "__main__":

    main() 
