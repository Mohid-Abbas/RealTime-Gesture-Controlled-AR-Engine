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

    # Load Power Asset
    power_asset = cv2.imread("assets/cinematic_kamehameha_ball.png", -1)
    if power_asset is None:
        power_asset = cv2.imread("assets/kamehameha effect.png", -1)

    # Load Cinematic Background Layers (Phase 6 & 8)
    # Priority: Video Background -> Mountain Layers
    video_bg_path = "assets/saiyan_background.mp4"
    background_engine.set_video_background(video_bg_path)
    
    bg_starfield = cv2.imread("assets/background_stars.png")
    bg_nebula = cv2.imread("assets/background_nebula.png")
    
    if bg_starfield is None or bg_nebula is None:
        # Create high-quality procedural layers (Mountainous Environment Phase 7)
        w_bg, h_bg = 1280, 720
        # Layer 0: Sky/Nebula (Distant)
        bg_starfield = np.zeros((h_bg, w_bg, 3), dtype=np.uint8)
        for y in range(h_bg):
            c = int(20 + 40 * (y / h_bg)) # Gradient
            bg_starfield[y, :] = (c + 10, c, c) # Dark red/brown sky
            
        # Layer 1: Mountains (Closer parallax)
        bg_nebula = np.zeros((h_bg, w_bg, 3), dtype=np.uint8)
        # Draw mountain silhouettes
        for x in range(0, w_bg, 2):
            m_h = int(200 + 150 * np.sin(x/100.0) + 50 * np.sin(x/25.0))
            cv2.line(bg_nebula, (x, h_bg), (x, h_bg - m_h), (30, 40, 50), 2)
        bg_nebula = cv2.GaussianBlur(bg_nebula, (5, 5), 0)

    cinematic_layers = [bg_starfield, bg_nebula]

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
            # We only need the mask for body lightning now, not replacing the background
            _, body_mask = background_engine.replace_background(display_frame, background_layers=None)
            if body_mask is not None:
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

        # 4. Final Polish (Screen Shake)
        display_frame = effects_engine.apply_screen_shake(display_frame)

        # 5. Display
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
