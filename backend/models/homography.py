# models/homography.py
class VirtualLineCounter:
    def __init__(self, line_y):
        self.line_y = line_y  # horizontal line positioning
        self.counts = {'north': 0, 'south': 0}  # counts of vehicles crossing the line
        self.track_history = {}  # stores the y-positions of tracks by ID

    def update(self, tracks):
        for track in tracks:
            # The track format is [x1, y1, x2, y2, id, confidence, class_id, ...]
            # Extract just what we need
            x1, y1, x2, y2, track_id = track[0:5]
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            # Calculate center position
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            
            # Initialize track history if this is a new track ID
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            # Check if the line is crossed
            if len(self.track_history[track_id]) > 0:
                prev_y = self.track_history[track_id][-1]
                
                # Southbound crossing (top to bottom)
                if prev_y < self.line_y and y_center >= self.line_y:
                    self.counts['south'] += 1
                
                # Northbound crossing (bottom to top)
                elif prev_y > self.line_y and y_center <= self.line_y:
                    self.counts['north'] += 1
            
            # Add current position to track history
            self.track_history[track_id].append(y_center)
            
            # Optional: Limit history length to prevent memory issues
            if len(self.track_history[track_id]) > 50:
                self.track_history[track_id] = self.track_history[track_id][-50:]