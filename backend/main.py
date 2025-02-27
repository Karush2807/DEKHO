# main.py
import cv2
import numpy as np
from models.yolo_model import VehicleCounter
from models.homography import VirtualLineCounter
import os

def main():
    # Initialize components
    vehicle_counter = VehicleCounter()
    
    # Video input setup
    video_path = 'C:/Users/Piyush/Desktop/Personal Work/DEKHO/data/test1.mp4'
    print(f"Video path: {video_path}")  # Print the video path for verification

    if not os.path.exists(video_path):
        print("Video file does not exist")
        return

    cap = cv2.VideoCapture(video_path)
    
    # Get first frame to set line position
    ret, frame = cap.read()
    if not ret:
        print("Error reading video file")
        return
    
    # Initialize virtual line counter (middle of frame)
    line_y = frame.shape[0] // 2
    line_counter = VirtualLineCounter(line_y)
    
    # Rewind video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame through detection and tracking
        tracks = vehicle_counter.process_frame(frame)
        
        # Update line crossing counts
        line_counter.update(tracks)
        
        # Visualization
        # 1. Draw virtual line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)
        
        # 2. Display counts
        cv2.putText(frame, f"Northbound: {line_counter.counts['north']}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Southbound: {line_counter.counts['south']}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 3. Show tracking information
        for track in tracks:
            if track.shape[0]>=6:
                x1, y1, x2, y2, track_id, _ = track.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 4. Show FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Traffic Monitoring System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()