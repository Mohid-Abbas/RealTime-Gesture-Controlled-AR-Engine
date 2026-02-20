import cv2
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
except ImportError:
    import mediapipe as mp
    mp_pose = mp.solutions.pose

class PoseTracker:
    """MediaPipe Pose integration for full body tracking."""
    def __init__(self, static_image_mode=False, model_complexity=1, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp_pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame):
        """Processes the frame and returns pose landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results

    def get_shoulder_points(self, results, w, h):
        """Extracts left and right shoulder coordinates."""
        if not results.pose_landmarks:
            return None, None
        
        lm = results.pose_landmarks.landmark
        l_shoulder = (int(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                      int(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        r_shoulder = (int(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                      int(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        return l_shoulder, r_shoulder
