import cv2
import numpy as np
from camera import Camera
from face_tracker import FaceTracker
from hand_tracker import HandTracker
from gesture_engine import GestureEngine
from face_swap_engine import FaceSwapEngine
from effects_engine import EffectsEngine
from utils import get_landmark_points, get_hand_center

def main():
    # Initialize components
    cam = Camera()
    face_tracker = FaceTracker()
    hand_tracker = HandTracker()
    gesture_engine = GestureEngine()
    face_swap_engine = FaceSwapEngine()
    effects_engine = EffectsEngine()

    # Load reference face
    src_face = cv2.imread("assets/reference_face.png")
    src_landmarks = None
    if src_face is not None:
        src_results = face_tracker.process(src_face)
        if src_results.multi_face_landmarks:
            src_landmarks = get_landmark_points(src_results.multi_face_landmarks[0], src_face.shape[1], src_face.shape[0])

    print("Project Saiyan AR is running. Press 'q' to quit.")

    is_transformed = False

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        h, w, _ = frame.shape

        # 1. Processing
        face_results = face_tracker.process(frame)
        hand_results = hand_tracker.process(frame)

        # 2. Gesture Detection
        gesture_engine.update(hand_results, face_results, w, h)
        if gesture_engine.is_swipe_triggered():
            is_transformed = not is_transformed # Toggle transformation
            gesture_engine.reset_swipe()

        # 3. Rendering Logic
        display_frame = frame.copy()

        # Handle Face Swap
        if is_transformed and face_results.multi_face_landmarks and src_landmarks is not None:
            dst_landmarks = get_landmark_points(face_results.multi_face_landmarks[0], w, h)
            display_frame = face_swap_engine.swap_face(display_frame, src_face, src_landmarks, dst_landmarks)

        # Handle Effects (Energy Ball)
        if gesture_engine.is_energy_triggered():
            if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) >= 2:
                # Calculate midpoint between hands
                c1 = get_hand_center(hand_results.multi_hand_landmarks[0], w, h)
                c2 = get_hand_center(hand_results.multi_hand_landmarks[1], w, h)
                mid_x = (c1[0] + c2[0]) // 2
                mid_y = (c1[1] + c2[1]) // 2
                display_frame = effects_engine.draw_energy_ball(display_frame, (mid_x, mid_y), 50)

        # Handle Face Swap (Visual Check)
        if True: # For testing, showing landmarks
            display_frame = face_tracker.draw_landmarks(display_frame, face_results)
            display_frame = hand_tracker.draw_landmarks(display_frame, hand_results)

        # 4. Display
        cv2.imshow("Project Saiyan AR", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
