import cv2
import numpy as np
from camera import Camera
from face_tracker import FaceTracker
from hand_tracker import HandTracker
from gesture_engine import GestureEngine
from effects_engine import EffectsEngine
from overlay_engine import OverlayEngine
from utils import get_landmark_points, get_hand_center

def main():
    # Initialize components
    cam = Camera()
    face_tracker = FaceTracker()
    hand_tracker = HandTracker()
    gesture_engine = GestureEngine()
    effects_engine = EffectsEngine()
    overlay_engine = OverlayEngine()

    # Load assets
    hair_path = "assets/Goku-Hair-PNG-Images.png"
    kame_path = "assets/kamehameha effect.png"
    
    # Load PNGs with alpha channel
    hair_overlay = cv2.imread(hair_path, -1)
    kame_overlay = cv2.imread(kame_path, -1)
    
    if hair_overlay is None: print(f"Warning: Could not load {hair_path}")
    if kame_overlay is None: print(f"Warning: Could not load {kame_path}")

    print("Project Saiyan AR is running. Press 's' to toggle, 'q' to quit.")

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
            print(f"Gesture Triggered: Face Swap {'Enabled' if is_transformed else 'Disabled'}")
            gesture_engine.reset_swipe()

        # 3. Rendering Logic
        display_frame = frame.copy()

        # Handle Transformation
        if is_transformed:
            # 1. Realistic Hair Overlay
            if face_results.multi_face_landmarks and hair_overlay is not None:
                pd = face_results.multi_face_landmarks[0].landmark
                # Target points for head
                pts_dst = [
                    (int(pd[21].x * w), int(pd[21].y * h)),   # Left side
                    (int(pd[251].x * w), int(pd[251].y * h)), # Right side
                    (int(pd[10].x * w), int(pd[10].y * h))    # Top
                ]
                
                oh, ow = hair_overlay.shape[:2]
                # Adjust these based on the new asset's proportions
                pts_src = [
                    (int(0.2 * ow), int(0.9 * oh)),
                    (int(0.8 * ow), int(0.9 * oh)),
                    (int(0.5 * ow), int(0.1 * oh))
                ]
                
                # Match lighting and warp
                y, x = pts_dst[2][1], pts_dst[2][0]
                roi = display_frame[max(0, y):min(h, y+50), max(0, x-25):min(w, x+25)]
                matched_hair = overlay_engine.match_lighting(hair_overlay, roi)
                display_frame = overlay_engine.perspective_overlay(display_frame, matched_hair, pts_src, pts_dst)

        # Handle Kamehameha Effect
        if gesture_engine.is_energy_triggered():
            if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) >= 2:
                c1 = get_hand_center(hand_results.multi_hand_landmarks[0], w, h)
                c2 = get_hand_center(hand_results.multi_hand_landmarks[1], w, h)
                mid_x, mid_y = (c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2
                display_frame = effects_engine.draw_kamehameha(display_frame, (mid_x, mid_y), 50, kame_overlay)

        # Draw hand landmarks (Movement marks)
        if hand_results.multi_hand_landmarks:
            display_frame = hand_tracker.draw_landmarks(display_frame, hand_results)

        # Handle UI Overlay
        cv2.putText(display_frame, f"Swap (S): {'ON' if is_transformed else 'OFF'}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Gesture: Swipe (Face Toggle) | Kamehameha (Energy)", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 4. Display
        cv2.imshow("Project Saiyan AR", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            is_transformed = not is_transformed
            print(f"Manual Toggle: Transformation {'Enabled' if is_transformed else 'Disabled'}")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
