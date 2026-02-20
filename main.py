import cv2
import numpy as np
from camera import Camera
from face_tracker import FaceTracker
from hand_tracker import HandTracker
from gesture_engine import GestureEngine
from effects_engine import EffectsEngine
from background_engine import BackgroundEngine
from utils import get_landmark_points, get_hand_center

def main():
    # Initialize components
    cam = Camera()
    face_tracker = FaceTracker()
    hand_tracker = HandTracker()
    gesture_engine = GestureEngine()
    effects_engine = EffectsEngine()
    background_engine = BackgroundEngine()

    power_asset = cv2.imread("assets/cinematic_kamehameha_ball.png", -1)
    if power_asset is None:
        power_asset = cv2.imread("assets/kamehameha effect.png", -1)
    
    # Load Cinematic Background
    cinematic_bg = cv2.imread("assets/alien_planet_background_saiyan.png")
    if cinematic_bg is None:
        # Create a procedural cinematic space if file missing
        cinematic_bg = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Add deep blue/purple nebula gradient
        for y in range(720):
            for x in range(1280):
                cinematic_bg[y, x] = [
                    int(40 + 20 * np.sin(x/400.0)), # Blue
                    int(10 + 10 * np.cos(y/200.0)), # Green
                    int(20 + 20 * np.sin((x+y)/300.0)) # Red
                ]
        # Add stars
        for _ in range(200):
            sx, sy = np.random.randint(0, 1280), np.random.randint(0, 720)
            cv2.circle(cinematic_bg, (sx, sy), 1, (255, 255, 255), -1)

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
            print(f"Gesture Triggered: Face Swap {'Enabled' if is_transformed else 'Disabled'}")
            gesture_engine.reset_swipe()

        # 3. Rendering Logic
        display_frame = frame.copy()

        # Handle Background Replacement (Phase 4)
        is_bursting = gesture_engine.is_burst_triggered()
        is_charging = gesture_engine.is_energy_triggered()
        
        if is_charging or is_bursting:
            display_frame, body_mask = background_engine.replace_background(display_frame, cinematic_bg)
            # Add body-hugging lightning
            display_frame = effects_engine.draw_body_lightning(display_frame, body_mask)

        # Handle Effects (Energy Ball)
        if gesture_engine.is_energy_triggered():
            if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) >= 2:
                # Calculate midpoint with base smoothing
                c1 = get_hand_center(hand_results.multi_hand_landmarks[0], w, h)
                c2 = get_hand_center(hand_results.multi_hand_landmarks[1], w, h)
                
                # Use current burst state
                is_bursting = gesture_engine.is_burst_triggered()
                
                mid_x = (c1[0] + c2[0]) // 2
                mid_y = (c1[1] + c2[1]) // 2
                
                # Render the cinematic energy effect
                display_frame = effects_engine.draw_energy_ball(
                    display_frame, (mid_x, mid_y), 60, 
                    asset=power_asset, burst=is_bursting
                )
                
                if is_bursting:
                    print("BOOM! Kamehameha Burst Fired!")

        # Draw hand landmarks (Movement marks)
        if hand_results.multi_hand_landmarks:
            display_frame = hand_tracker.draw_landmarks(display_frame, hand_results)

        # Handle UI Overlay
        cv2.putText(display_frame, "SUPER SAIYAN MODE: Palms Together to charge", (10, h - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(display_frame, "Push TOWARD Camera to BURST", (10, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 4. Display
        cv2.imshow("Project Saiyan AR", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
