"""
Centroid Tracker Module
Simple object tracking using centroid distance
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, max_disappeared=50):
        """
        Initialize centroid tracker
        
        Args:
            max_disappeared: Maximum frames an object can disappear before deregistering
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
    
    def register(self, centroid):
        """
        Register a new object with next available ID
        
        Args:
            centroid: (x, y) coordinates of object centroid
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """
        Deregister an object ID
        
        Args:
            object_id: ID of object to remove
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, rects):
        """
        Update tracked objects with new detections
        
        Args:
            rects: List of bounding boxes as (startX, startY, endX, endY)
            
        Returns:
            OrderedDict: Mapping of object IDs to centroids
        """
        # If no detections, mark all as disappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Deregister if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        # Calculate centroids for new detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)
        
        # If no existing objects, register all new ones
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        
        # Otherwise, match existing objects to new detections
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance between each pair of object centroids and input centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Find smallest distance in each row and sort by distance
            rows = D.min(axis=1).argsort()
            
            # Find smallest distance in each column
            cols = D.argmin(axis=1)[rows]
            
            # Track which rows and columns we've already examined
            used_rows = set()
            used_cols = set()
            
            # Loop over the combination of (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # Ignore if already examined
                if row in used_rows or col in used_cols:
                    continue
                
                # Update centroid and reset disappeared counter
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Mark as examined
                used_rows.add(row)
                used_cols.add(col)
            
            # Compute row and column indices we have NOT examined
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)
            
            # If number of object centroids >= number of input centroids,
            # check if some objects have disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Otherwise, register new objects
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        return self.objects
