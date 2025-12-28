import cv2
import numpy as np
import argparse
import logging
import os
import time

class ObjectDetector:
    def __init__(self, weights_path, config_path, names_path, confidence=0.5, threshold=0.3):
        """
        Initialize YOLO Object Detector
        
        Args:
            weights_path: Path to YOLO weights file
            config_path: Path to YOLO config file
            names_path: Path to class names file
            confidence: Minimum confidence threshold for detections
            threshold: NMS threshold for removing duplicate detections
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


            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            

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
    
    # Input/Output
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera frame width')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera frame height')
    parser.add_argument('--fps', type=int, default=60,
                        help='Target FPS for recording')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Output video file path')
    
    args = parser.parse_args()
    
    # Initialize detector with validation
    try:
        detector = ObjectDetector(
            weights_path=args.weights,
            config_path=args.config,
            names_path=args.names,
            confidence=args.confidence,
            threshold=args.threshold
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all model files are in the correct location.")
        return
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Initialize camera with validation
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        print("Please check if the camera is connected and not in use by another application.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, actual_fps, (actual_width, actual_height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video file: {args.output}")
        cap.release()
        return
    
    print("Starting object detection... Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            # Detect objects
            indices, boxes, confidences, class_ids = detector.detect(frame)
            
            # Draw detections
            frame = detector.draw_detections(frame, indices, boxes, confidences, class_ids)
            
            # Calculate and display FPS
            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Object Detection - Press Q to quit', frame)
            out.write(frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
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