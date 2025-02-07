# 📌 Functionality:
# ✅ Converts CCTV pixel locations → real-world coordinates

import numpy as np
import cv2

# Define four reference points in the real world (meters)
real_world_pts = np.array([
    [0, 0], [100, 0], [100, 200], [0, 200]  # Example real-world locations
], dtype=np.float32)

# Corresponding four points in CCTV frame (pixel coordinates)
cctv_frame_pts = np.array([
    [320, 240], [640, 240], [640, 480], [320, 480]  # Example positions
], dtype=np.float32)

# Compute homography matrix
H, _ = cv2.findHomography(cctv_frame_pts, real_world_pts)

def map_to_real_world(x, y):
    """ Convert detected person's position to real-world coordinates """
    pixel_coords = np.array([[x, y, 1]], dtype=np.float32).T
    world_coords = H @ pixel_coords
    world_coords /= world_coords[2]  # Normalize
    return world_coords[0], world_coords[1]

# Example usage
if __name__ == "__main__":
    real_x, real_y = map_to_real_world(400, 300)
    print(f"Mapped Coordinates: ({real_x:.2f}, {real_y:.2f}) meters")
