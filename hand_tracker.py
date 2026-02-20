import cv2
try:
    import mediapipe as mp
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
except ImportError:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

class HandTracker:
    """MediaPipe Hands integration for 21 landmarks per hand."""
    def __init__(self, static_image_mode=False, max_num_hands=2, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles

    def process(self, frame):
        """Processes the frame and returns hand landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results

    def draw_landmarks(self, frame, results):
        """Draws hand landmarks on the frame."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame
