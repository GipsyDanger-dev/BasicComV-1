"""
YOLOv8 Object Detection with GPU Support
High-performance real-time object detection using YOLOv8
"""

import cv2
import argparse
import time
import os
from ultralytics import YOLO
from statistics import DetectionStatistics
from tracker import CentroidTracker


def main():
    parser = argparse.ArgumentParser(
        description='High Performance Object Detection using YOLOv8',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLOv8 model (yolov8n/s/m/l/x.pt)')
    
    # Detection parameters
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IOU threshold for NMS')
    
    # Performance
    parser.add_argument('--device', type=str, default='',
                        help='Device to run on (cuda/cpu, empty for auto)')
    
    # Input/Output
    parser.add_argument('--input', type=str, default=None,
                        help='Input video file path (if not specified, uses camera)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (used if --input not specified)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera frame width (ignored for video input)')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera frame height (ignored for video input)')
    parser.add_argument('--output', type=str, default='output_yolov8.mp4',
                        help='Output video file path')
    
    # Statistics
    parser.add_argument('--stats-output', type=str, default=None,
                        help='Export statistics to CSV file (e.g., stats.csv)')
    parser.add_argument('--tracking', action='store_true',
                        help='Enable object tracking with persistent IDs')
    
    args = parser.parse_args()
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {args.model}")
    try:
        model = YOLO(args.model)
        
        # Set device
        if args.device:
            device = args.device
        else:
            # Auto-detect: use CUDA if available, else CPU
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {device.upper()}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Downloading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        device = 'cpu'
    
    # Initialize input source (video file or camera)
    is_video_input = args.input is not None
    total_frames = 0
    
    if is_video_input:
        # Video file input
        if not os.path.exists(args.input):
            print(f"Error: Input video file not found: {args.input}")
            return
        
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Cannot open video file: {args.input}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {args.input}")
        print(f"Total frames: {total_frames}")
    else:
        # Camera input
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {args.camera}")
            print("Please check if the camera is connected and not in use by another application.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f"Using camera {args.camera}")
    
    # Get actual video properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, actual_fps if actual_fps > 0 else 30, 
                          (actual_width, actual_height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video file: {args.output}")
        cap.release()
        return
    
    # Initialize statistics and tracking
    stats = DetectionStatistics() if args.stats_output else None
    tracker = CentroidTracker(max_disappeared=40) if args.tracking else None
    
    if is_video_input:
        print(f"Processing video... Press 'q' to quit")
    else:
        print("Starting object detection... Press 'q' to quit")
    
    if args.tracking:
        print("Object tracking: ENABLED")
    if args.stats_output:
        print(f"Statistics will be exported to: {args.stats_output}")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_video_input:
                    print("\nVideo processing completed")
                else:
                    print("\nError: Failed to read frame from camera")
                break
            
            # Run YOLOv8 inference
            results = model(frame, conf=args.confidence, iou=args.iou, 
                          device=device, verbose=False)
            
            # Get detections
            result = results[0]
            boxes = result.boxes
            
            # Extract detection data
            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
                confidences = boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = boxes.cls.cpu().numpy().astype(int)  # Class IDs
                class_names = [model.names[int(cls)] for cls in class_ids]
                
                # Update statistics
                if stats:
                    stats.update(class_ids, class_names, confidences)
                
                # Update tracker
                tracked_objects = {}
                if tracker:
                    # Convert to (startX, startY, endX, endY) format
                    rects = [(int(x1), int(y1), int(x2), int(y2)) 
                            for x1, y1, x2, y2 in xyxy]
                    tracked_objects = tracker.update(rects)
            else:
                if tracker:
                    tracked_objects = tracker.update([])
            
            # Draw detections (YOLOv8 has built-in plotting)
            annotated_frame = result.plot()
            
            # Draw tracking IDs
            if tracker and len(tracked_objects) > 0:
                for (object_id, centroid) in tracked_objects.items():
                    text = f"ID {object_id}"
                    cv2.putText(annotated_frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
            # Calculate and display FPS
            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            
            # Display FPS
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display statistics
            if stats:
                top_classes = stats.get_top_classes(3)
                y_offset = 70
                for i, (class_name, count, avg_conf) in enumerate(top_classes):
                    text = f"{class_name}: {count} ({avg_conf:.2f})"
                    cv2.putText(annotated_frame, text, (10, y_offset + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show progress for video input
            if is_video_input and total_frames > 0:
                progress = (frame_count / total_frames) * 100
                progress_text = f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)"
                y_pos = 160 if stats else 70
                cv2.putText(annotated_frame, progress_text, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # Console progress update every 30 frames
                if frame_count % 30 == 0:
                    print(f"\rProcessing: {frame_count}/{total_frames} frames ({progress:.1f}%) - FPS: {fps:.1f}", 
                          end="", flush=True)
            
            # Display frame
            cv2.imshow('YOLOv8 Object Detection - Press Q to quit', annotated_frame)
            out.write(annotated_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Export statistics if requested
        if stats and args.stats_output:
            try:
                stats.export_csv(args.stats_output)
                print(f"\nStatistics exported to: {args.stats_output}")
            except Exception as e:
                print(f"\nError exporting statistics: {e}")
        
        # Cleanup
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n\nDetection completed:")
        print(f"  Model: {args.model}")
        print(f"  Device: {device.upper()}")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Output saved to: {args.output}")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
