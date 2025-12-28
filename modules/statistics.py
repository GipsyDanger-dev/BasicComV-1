import csv
import json
from datetime import datetime
from collections import defaultdict


class DetectionStatistics:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.class_confidences = defaultdict(list)
        self.class_first_seen = {}
        self.class_last_seen = {}
        self.total_detections = 0
        self.start_time = datetime.now()
    
    def update(self, class_ids, class_names, confidences):
        current_time = datetime.now()
        
        for class_id, class_name, confidence in zip(class_ids, class_names, confidences):
            self.class_counts[class_name] += 1
            self.class_confidences[class_name].append(confidence)
            self.total_detections += 1
            
            if class_name not in self.class_first_seen:
                self.class_first_seen[class_name] = current_time
            self.class_last_seen[class_name] = current_time
    
    def get_summary(self):
        summary = {}
        for class_name in self.class_counts:
            avg_confidence = sum(self.class_confidences[class_name]) / len(self.class_confidences[class_name])
            summary[class_name] = {
                'count': self.class_counts[class_name],
                'avg_confidence': avg_confidence,
                'first_seen': self.class_first_seen[class_name],
                'last_seen': self.class_last_seen[class_name]
            }
        return summary
    
    def get_top_classes(self, n=3):
        summary = self.get_summary()
        sorted_classes = sorted(summary.items(), key=lambda x: x[1]['count'], reverse=True)
        
        result = []
        for class_name, stats in sorted_classes[:n]:
            result.append((class_name, stats['count'], stats['avg_confidence']))
        return result
    
    def export_csv(self, filepath):
        summary = self.get_summary()
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['class_name', 'total_count', 'avg_confidence', 'first_seen', 'last_seen']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for class_name, stats in sorted(summary.items(), key=lambda x: x[1]['count'], reverse=True):
                writer.writerow({
                    'class_name': class_name,
                    'total_count': stats['count'],
                    'avg_confidence': f"{stats['avg_confidence']:.4f}",
                    'first_seen': stats['first_seen'].strftime('%H:%M:%S'),
                    'last_seen': stats['last_seen'].strftime('%H:%M:%S')
                })
    
    def export_json(self, filepath):
        summary = self.get_summary()
        
        json_data = {}
        for class_name, stats in summary.items():
            json_data[class_name] = {
                'count': stats['count'],
                'avg_confidence': round(stats['avg_confidence'], 4),
                'first_seen': stats['first_seen'].strftime('%H:%M:%S'),
                'last_seen': stats['last_seen'].strftime('%H:%M:%S')
            }
        
        with open(filepath, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2)
