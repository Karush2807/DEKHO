# Functionality:
# Manages CCTV feed / webcam input
# Supports multiple camera feeds

import cv2

class CameraSystem:
    def __init__(self, source=0):
        """ Initialize camera source (0 = webcam, URL = IP CCTV) """
        self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        """ Capture a single frame from camera """
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """ Release camera resources """
        self.cap.release()

# Example usage
if __name__ == "__main__":
    cam = CameraSystem(0)  # 0 for webcam, or replace with 'rtsp://...' for CCTV
    while True:
        frame = cam.get_frame()
        if frame is not None:
            cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
