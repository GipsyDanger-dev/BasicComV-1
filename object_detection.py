import cv2
import numpy as np
import argparse
import logging
import os
import time
from modules.statistics import DetectionStatistics
from modules.tracker import CentroidTracker

class ObjectDetector:
    def __init__(self, weights_path, config_path, names_path, confidence=0.5, threshold=0.3, use_gpu=True):
        """
        Initialize YOLO Object Detector
        
        Args:
            weights_path: Path to YOLO weights file
            config_path: Path to YOLO config file
            names_path: Path to class names file
            confidence: Minimum confidence threshold for detections
            threshold: NMS threshold for removing duplicate detections
            use_gpu: Enable GPU acceleration if available (default: True)
        """
        self.confidence = confidence
        self.threshold = threshold
        
        # Logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Validate file existence
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(names_path):
            raise FileNotFoundError(f"Names file not found: {names_path}")

        try:
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

            # GPU acceleration setup
            self.gpu_enabled = False
            if use_gpu:
                try:
                    # Check if CUDA is available
                    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
                    if cuda_available:
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        self.gpu_enabled = True
                        self.logger.info("GPU acceleration ENABLED (CUDA)")
                    else:
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                        self.logger.info("GPU not available, using CPU")
                except:
                    # Fallback to CPU if CUDA check fails
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.logger.info("GPU acceleration not available, using CPU")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.logger.info("GPU disabled by user, using CPU")
            

            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            

            with open(names_path, 'r') as f:
                self.classes = f.read().strip().split('\n')
            
            self.logger.info("Model YOLO berhasil dimuat dengan optimasi")
        except Exception as e:
            self.logger.error(f"Gagal memuat model: {e}")
            raise
    
    def detect(self, frame):

        height, width, _ = frame.shape
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        #deteksi objek
        class_ids = []
        confidences = []
        boxes = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        
        return indices, boxes, confidences, class_ids
    
    def draw_detections(self, frame, indices, boxes, confidences, class_ids):

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                

                color = self._get_color(class_ids[i])

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                

                text = f"{label}: {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width, y), color, -1)
                cv2.putText(frame, text, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _get_color(self, class_id):

        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, 3).tolist())

def main():
    parser = argparse.ArgumentParser(
        description='High Performance Object Detection using YOLO',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--weights', type=str, default='yolov3.weights',
                        help='Path to YOLO weights file')
    parser.add_argument('--config', type=str, default='yolov3.cfg',
                        help='Path to YOLO config file')
    parser.add_argument('--names', type=str, default='coco.names',
                        help='Path to class names file')
    
    # Detection parameters
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum confidence threshold for detections')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='NMS threshold for removing duplicate detections')
    
    # Performance
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', default=True,
                        help='Enable GPU acceleration (default: enabled)')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                        help='Disable GPU acceleration, force CPU mode')
    
    # Input/Output
    parser.add_argument('--input', type=str, default=None,
                        help='Input video file path (if not specified, uses camera)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (used if --input not specified)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera frame width (ignored for video input)')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera frame height (ignored for video input)')
    parser.add_argument('--fps', type=int, default=60,
                        help='Target FPS for recording (ignored for video input)')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Output video file path')
    # Statistics
    parser.add_argument('--stats-output', type=str, default=None,
                        help='Export statistics to CSV file (e.g., stats.csv)')
    parser.add_argument('--tracking', action='store_true',
                        help='Enable object tracking with persistent IDs')
    
    args = parser.parse_args()
    
    # Initialize detector with validation
    try:
        detector = ObjectDetector(
            weights_path=args.weights,
            config_path=args.config,
            names_path=args.names,
            confidence=args.confidence,
            threshold=args.threshold,
            use_gpu=args.use_gpu
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all model files are in the correct location.")
        return
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
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
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        print(f"Using camera {args.camera}")
    
    # Get actual video properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
    print(f"GPU: {'ENABLED' if detector.gpu_enabled else 'DISABLED (CPU)'}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, actual_fps, (actual_width, actual_height))
    
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
            
            # Detect objects
            indices, boxes, confidences, class_ids = detector.detect(frame)
            
            # Update statistics
            if stats and len(indices) > 0:
                detected_classes = [detector.classes[class_ids[i]] for i in indices.flatten()]
                detected_confidences = [confidences[i] for i in indices.flatten()]
                stats.update(indices.flatten(), detected_classes, detected_confidences)
            
            # Update tracker
            tracked_objects = {}
            if tracker and len(indices) > 0:
                # Convert boxes to (startX, startY, endX, endY) format
                rects = []
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    rects.append((x, y, x + w, y + h))
                tracked_objects = tracker.update(rects)
            
            # Draw detections
            frame = detector.draw_detections(frame, indices, boxes, confidences, class_ids)
            
            # Draw tracking IDs
            if tracker and len(tracked_objects) > 0:
                for (object_id, centroid) in tracked_objects.items():
                    text = f"ID {object_id}"
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
            # Calculate and display FPS
            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display statistics
            if stats:
                top_classes = stats.get_top_classes(3)
                y_offset = 70
                for i, (class_name, count, avg_conf) in enumerate(top_classes):
                    text = f"{class_name}: {count} ({avg_conf:.2f})"
                    cv2.putText(frame, text, (10, y_offset + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show progress for video input
            if is_video_input and total_frames > 0:
                progress = (frame_count / total_frames) * 100
                progress_text = f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)"
                y_pos = 160 if stats else 70
                cv2.putText(frame, progress_text, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # Console progress update every 30 frames
                if frame_count % 30 == 0:
                    print(f"\rProcessing: {frame_count}/{total_frames} frames ({progress:.1f}%)", end="", flush=True)
            
            # Display frame
            cv2.imshow('Object Detection - Press Q to quit', frame)
            out.write(frame)
            
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
        
        print(f"\nDetection completed:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Output saved to: {args.output}")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()