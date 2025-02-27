# YOLOv8 vehicle detection
from ultralytics import YOLO
from boxmot import ByteTrack
import numpy as np

class VehicleCounter:
    def __init__(self):
        self.model=YOLO('yolov8n.pt') #pretrained model
        self.tracker=ByteTrack() #tracker added
        self.allowed_class_ids={
            'car': 2,
            'motorcycle': 3,
            'bus': 5,
            'truck': 7
        }
        self.counts={cls: 0 for cls in self.allowed_class_ids.keys()} #only the mentioned classes will be counted

    def process_frame(self, frame):
        results = self.model(frame)
        filtered_detections = self._format_detections(results, frame.shape)
        tracks = self.tracker.update(filtered_detections, frame)
        
        self._update_counts(results)
        return results
    
    def _update_counts(self, results):
        self.counts={cls: 0 for cls in self.allowed_class_ids.keys()}

        boxes=results[0].boxes
        for box in boxes:
            class_id=int(box.cls)
            class_name=results[0].names[class_id]
            if class_name in self.allowed_class_ids:
                self.counts[class_name]+=1

    def _format_detections(self, results, img_shape):
        """Convert YOLO results to [x1, y1, x2, y2, conf, cls] format"""
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = results[0].names[class_id]
            
            if class_name in self.allowed_class_ids:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                img_height, img_width = img_shape[:2]

                detections.append([
                    x1 * img_width,   # Convert to absolute coordinates
                    y1 * img_height,  # Convert to absolute coordinates
                    x2 * img_width,   # Convert to absolute coordinates
                    y2 * img_height,  # Convert to absolute coordinates
                    box.conf.item(),   # Confidence score
                    class_id          # Class ID
                ])
        
        return np.array(detections) if detections else np.empty((0, 6))
