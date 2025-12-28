import cv2
import numpy as np
from collections import defaultdict


class ZoneDetector:
    def __init__(self):
        self.zones = []
        self.zone_counts = defaultdict(int)
        self.zone_history = defaultdict(list)
    
    def add_zone(self, name, points, zone_type="inclusion"):
        zone = {
            'name': name,
            'points': np.array(points, dtype=np.int32),
            'type': zone_type,
            'count': 0
        }
        self.zones.append(zone)
    
    def clear_zones(self):
        self.zones = []
        self.zone_counts.clear()
        self.zone_history.clear()
    
    def point_in_zone(self, point, zone):
        return cv2.pointPolygonTest(zone['points'], point, False) >= 0
    
    def check_detections(self, boxes, class_ids):
        results = []
        
        for zone in self.zones:
            zone['count'] = 0
            
            for box, class_id in zip(boxes, class_ids):
                x, y, w, h = box
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                
                if self.point_in_zone((center_x, center_y), zone):
                    zone['count'] += 1
                    results.append({
                        'zone': zone['name'],
                        'class_id': class_id,
                        'position': (center_x, center_y)
                    })
            
            self.zone_counts[zone['name']] = zone['count']
        
        return results
    
    def draw_zones(self, frame):
        for zone in self.zones:
            color = (0, 255, 0) if zone['type'] == 'inclusion' else (0, 0, 255)
            
            cv2.polylines(frame, [zone['points']], True, color, 2)
            
            centroid = np.mean(zone['points'], axis=0).astype(int)
            text = f"{zone['name']}: {zone['count']}"
            cv2.putText(frame, text, tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def get_zone_stats(self):
        return dict(self.zone_counts)
