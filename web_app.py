from flask import Flask, render_template, Response, jsonify, request
import cv2
import json
import threading
import time
from ultralytics import YOLO
from statistics import DetectionStatistics
from tracker import CentroidTracker
from zone_detector import ZoneDetector
from alert_manager import AlertManager


app = Flask(__name__)

class DetectionSystem:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model = YOLO(self.config['model']['name'])
        self.cap = None
        self.running = False
        self.current_frame = None
        self.lock = threading.Lock()
        
        self.stats = DetectionStatistics() if self.config['features']['statistics'] else None
        self.tracker = CentroidTracker() if self.config['features']['tracking'] else None
        self.zone_detector = ZoneDetector() if self.config['features']['zones'] else None
        self.alert_manager = AlertManager(self.config) if self.config['features']['alerts'] else None
        
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        self.process_every_n_frames = 2
    
    def start(self):
        if self.running:
            return False
        
        source = self.config['video']['source']
        if source.isdigit():
            source = int(source)
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            return False
        
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        thread = threading.Thread(target=self._detection_loop)
        thread.daemon = True
        thread.start()
        
        return True
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.current_frame = None
    
    def _detection_loop(self):
        process_frame_counter = 0
        last_annotated_frame = None
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            process_frame_counter += 1
            
            if process_frame_counter % self.process_every_n_frames == 0:
                results = self.model(frame, 
                                   conf=self.config['model']['confidence'],
                                   iou=self.config['model']['iou'],
                                   verbose=False)
                
                result = results[0]
                annotated_frame = result.plot()
                
                if len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    class_names = [self.model.names[int(cls)] for cls in class_ids]
                    
                    if self.stats:
                        self.stats.update(class_ids, class_names, confidences)
                    
                    if self.tracker:
                        rects = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]
                        tracked_objects = self.tracker.update(rects)
                        
                        for (object_id, centroid) in tracked_objects.items():
                            cv2.putText(annotated_frame, f"ID {object_id}", 
                                      (centroid[0] - 10, centroid[1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.circle(annotated_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                    
                    if self.zone_detector:
                        xyxy_boxes = [(int(x1), int(y1), int(x2-x1), int(y2-y1)) for x1, y1, x2, y2 in boxes]
                        self.zone_detector.check_detections(xyxy_boxes, class_ids)
                        annotated_frame = self.zone_detector.draw_zones(annotated_frame)
                
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if self.stats:
                    top_classes = self.stats.get_top_classes(3)
                    y_offset = 70
                    for i, (class_name, count, avg_conf) in enumerate(top_classes):
                        text = f"{class_name}: {count} ({avg_conf:.2f})"
                        cv2.putText(annotated_frame, text, (10, y_offset + i * 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if self.alert_manager and self.stats and self.frame_count % 30 == 0:
                    alerts = self.alert_manager.check_thresholds(self.stats)
                    if alerts:
                        self.alert_manager.process_alerts(alerts)
                
                last_annotated_frame = annotated_frame
                
                with self.lock:
                    self.current_frame = annotated_frame.copy()
            else:
                if last_annotated_frame is not None:
                    with self.lock:
                        self.current_frame = last_annotated_frame.copy()
    
    def get_frame(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_stats(self):
        data = {
            'fps': float(round(self.fps, 1)),
            'frame_count': int(self.frame_count),
            'running': bool(self.running)
        }
        
        if self.stats:
            top_classes = self.stats.get_top_classes(5)
            data['detections'] = [
                {'class': str(name), 'count': int(count), 'confidence': float(round(conf, 2))}
                for name, count, conf in top_classes
            ]
        
        if self.zone_detector:
            data['zones'] = {str(k): int(v) for k, v in self.zone_detector.get_zone_stats().items()}
        
        return data


detection_system = DetectionSystem()


@app.route('/')
def index():
    return render_template('index.html', config=detection_system.config)


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = detection_system.get_frame()
            if frame is not None:
                frame_resized = cv2.resize(frame, (640, 480))
                ret, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start', methods=['POST'])
def start_detection():
    success = detection_system.start()
    return jsonify({'success': success})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    detection_system.stop()
    return jsonify({'success': True})


@app.route('/api/stats')
def get_stats():
    return jsonify(detection_system.get_stats())


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        new_config = request.json
        detection_system.config.update(new_config)
        
        with open('config.json', 'w') as f:
            json.dump(detection_system.config, f, indent=2)
        
        return jsonify({'success': True})
    
    return jsonify(detection_system.config)


@app.route('/api/zones', methods=['POST'])
def add_zone():
    data = request.json
    if detection_system.zone_detector:
        detection_system.zone_detector.add_zone(
            data['name'],
            data['points'],
            data.get('type', 'inclusion')
        )
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Zones not enabled'})


if __name__ == '__main__':
    app.run(
        host=detection_system.config['web']['host'],
        port=detection_system.config['web']['port'],
        debug=detection_system.config['web']['debug'],
        threaded=True
    )
