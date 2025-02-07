#  Functionality:
#  Uses YOLOv8 for detection
#  Uses DeepSORT for tracking across frames

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self):
        """ Load YOLO model & DeepSORT tracker """
        self.model = YOLO("yolov8n.pt")
        self.tracker = DeepSort(max_age=50)

    def detect_and_track(self, frame):
        """ Detect and track people in a frame """
        results = self.model(frame)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2, conf, cls = box.xyxy[0].tolist()
                if int(cls) == 0:  # Only detect 'person' class
                    detections.append([[x1, y1, x2, y2], conf])

        tracked_objects = self.tracker.update_tracks(detections, frame=frame)
        return tracked_objects

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Webcam feed
    tracker = ObjectTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracks = tracker.detect_and_track(frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
