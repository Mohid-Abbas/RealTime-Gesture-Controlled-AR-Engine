import cv2

class Camera:
    """Helper class for webcam access and frame acquisition."""
    def __init__(self, camera_id=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
    def get_frame(self):
        """Captures a frame and returns it."""
        success, frame = self.cap.read()
        if not success:
            return None
        return frame
    
    def release(self):
        """Releases the camera."""
        self.cap.release()

if __name__ == "__main__":
    cam = Camera()
    while True:
        frame = cam.get_frame()
        if frame is None:
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
