import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (replace with yolov12x.pt when available)
try:
    model = YOLO("yolov8x.pt")  # Temporarily using YOLOv8x; replace with "yolov12x.pt" when downloaded
    print("YOLO model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Please download yolov12x.pt from Ultralytics releases.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define the source (video file or webcam)
source = "C:/Users/Piyush/Desktop/Personal Work/DEKHO/backend/data/dayROI.mp4"  # Change to 0 for webcam
cap = cv2.VideoCapture(source)

# Debugging: Check if the video file is opened
if not cap.isOpened():
    print("Error: Could not open video file or webcam.")
    exit()
else:
    print("Video file opened successfully.")

# Shift the ROI upward to capture the traffic signal area
roi = [(50, 50), (600, 250)]  # Adjusted to move the region upward  # Adjusted to move the region downward
roi_area = (roi[1][0] - roi[0][0]) * (roi[1][1] - roi[0][1])  # Area in pixels

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video.")
        break
    
    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    # Run YOLO inference with confidence threshold
    results = model(frame, stream=True, conf=0.5)  # Confidence threshold of 0.5 to reduce misclassifications
    
    vehicle_count = 0
    vehicles_in_roi = 0
    
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            conf = result.boxes.conf[0]  # Confidence score
            label_idx = int(result.boxes.cls[0])
            label = result.names[label_idx]
            
            # Debug: Print detected labels to verify
            print(f"Detected: {label} with confidence {conf:.2f}")
            
            # Check if detected object is a vehicle
            if label in ["car", "bus", "truck", "motorbike"]:
                vehicle_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check if bounding box center is within ROI
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                x_min, y_min = roi[0]
                x_max, y_max = roi[1]
                if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
                    vehicles_in_roi += 1
    
    # Draw the precise ROI
    cv2.rectangle(frame, roi[0], roi[1], (0, 0, 255), 2)
    
    # Calculate traffic density (vehicles per 1000 pixels as an example metric)
    if roi_area > 0:
        density = (vehicles_in_roi / roi_area) * 1000  # Vehicles per 1000 pixels
    else:
        density = 0
    
    # Categorize traffic density
    if density == 0:
        density_level = "Low"
    elif density <= 0.05:  # Adjust threshold based on your needs
        density_level = "Medium"
    else:
        density_level = "High"
    
    # Display counts and density
    cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Vehicles in ROI: {vehicles_in_roi}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Traffic Density: {density_level} ({density:.2f}/1000px)", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show the frame
    cv2.imshow("Frame", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
