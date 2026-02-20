import cv2
import numpy as np
from camera import Camera
from face_tracker import FaceTracker
from hand_tracker import HandTracker
from pose_tracker import PoseTracker
from gesture_engine import GestureEngine
from face_swap_engine import FaceSwapEngine
from effects_engine import EffectsEngine
from overlay_engine import OverlayEngine
from utils import get_landmark_points, get_hand_center

def main():
    # Initialize components
    cam = Camera()
    face_tracker = FaceTracker()
    hand_tracker = HandTracker()
    pose_tracker = PoseTracker()
    gesture_engine = GestureEngine()
    face_swap_engine = FaceSwapEngine()
    effects_engine = EffectsEngine()
    overlay_engine = OverlayEngine()

    # Load reference face
    reference_path = "assets/reference_face.png"
    src_face = cv2.imread(reference_path)
    src_landmarks = None
    if src_face is not None:
        # Resize to a standard size to help MediaPipe detection
        src_face = cv2.resize(src_face, (640, 640))
        print(f"Successfully loaded reference face. Resized to: {src_face.shape}")
        
        # Use a specialized tracker for static reference images
        static_face_tracker = FaceTracker(static_image_mode=True, min_detection_confidence=0.1) # Extreme low confidence
        src_results = static_face_tracker.process(src_face)
        
        if src_results.multi_face_landmarks:
            print("SUCCESS: Landmarks detected on reference face!")
            src_landmarks = get_landmark_points(src_results.multi_face_landmarks[0], src_face.shape[1], src_face.shape[0])
        else:
            print("CRITICAL: MediaPipe still cannot find a face in 'assets/reference_face.png'.")
            print("NOTE: This usually happens if the anime style is too abstract. Try a more realistic face image.")
            # Show the image so the user can see it
            cv2.imshow("DEBUG: Reference Image Looked at by AI", src_face)
            cv2.waitKey(2000) # Show for 2 seconds
    else:
        print(f"Error: Could not find or read {reference_path}.")

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
        pose_results = pose_tracker.process(frame)

        # 2. Gesture Detection
        gesture_engine.update(hand_results, face_results, w, h)
        if gesture_engine.is_swipe_triggered():
            is_transformed = not is_transformed # Toggle transformation
            print(f"Gesture Triggered: Face Swap {'Enabled' if is_transformed else 'Disabled'}")
            gesture_engine.reset_swipe()

        # 3. Rendering Logic
        display_frame = frame.copy()

        # Handle Face Swap
        if is_transformed:
            if face_results.multi_face_landmarks and src_landmarks is not None:
                dst_landmarks = get_landmark_points(face_results.multi_face_landmarks[0], w, h)
                display_frame = face_swap_engine.swap_face(display_frame, src_face, src_landmarks, dst_landmarks)
                
                # Full Character Transformation (Phase 2)
                # Drawing Hair & Clothes
                if face_results.multi_face_landmarks:
                    face_lm = face_results.multi_face_landmarks[0].landmark
                    # Head position (using nose as reference)
                    nose_p = (int(face_lm[1].x * w), int(face_lm[1].y * h))
                    
                    # 1. Draw Hair (Procedural spikey hair)
                    # We use a yellow/gold color for Super Saiyan
                    hair_color = (0, 255, 255) # Gold
                    # Draw some spikes
                    spikes = [(-50, -100), (-20, -150), (0, -180), (20, -150), (50, -100)]
                    pts = []
                    for s in spikes:
                        pts.append([nose_p[0] + s[0], nose_p[1] + s[1]])
                    cv2.fillPoly(display_frame, [np.array(pts)], hair_color)

                # 2. Draw Gi/Armor using Pose
                l_shoulder, r_shoulder = pose_tracker.get_shoulder_points(pose_results, w, h)
                if l_shoulder and r_shoulder:
                    gi_color = (0, 165, 255) # Orange
                    chest_mid = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
                    # Draw a trapezoid for the shirt
                    shirt_pts = [
                        [l_shoulder[0] - 20, l_shoulder[1]],
                        [r_shoulder[0] + 20, r_shoulder[1]],
                        [r_shoulder[0] + 40, r_shoulder[1] + 200],
                        [l_shoulder[0] - 40, l_shoulder[1] + 200]
                    ]
                    cv2.fillPoly(display_frame, [np.array(shirt_pts)], gi_color)
                    # Draw a blue undershirt V-neck
                    v_neck = [
                        [chest_mid[0] - 30, l_shoulder[1]],
                        [chest_mid[0] + 30, l_shoulder[1]],
                        [chest_mid[0], l_shoulder[1] + 60]
                    ]
                    cv2.fillPoly(display_frame, [np.array(v_neck)], (255, 0, 0)) # Blue
            else:
                if src_landmarks is None:
                    cv2.putText(display_frame, "Error: Reference face landmarks not loaded", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not face_results.multi_face_landmarks:
                    cv2.putText(display_frame, "Error: No face detected", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Handle Effects (Energy Ball)
        if gesture_engine.is_energy_triggered():
            if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) >= 2:
                # Calculate midpoint between hands
                c1 = get_hand_center(hand_results.multi_hand_landmarks[0], w, h)
                c2 = get_hand_center(hand_results.multi_hand_landmarks[1], w, h)
                mid_x = (c1[0] + c2[0]) // 2
                mid_y = (c1[1] + c2[1]) // 2
                display_frame = effects_engine.draw_energy_ball(display_frame, (mid_x, mid_y), 50)

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
            print(f"Manual Toggle: Face Swap {'Enabled' if is_transformed else 'Disabled'}")
        elif key == ord('r'):
            print("Reloading reference face...")
            src_face = cv2.imread(reference_path)
            if src_face is not None:
                src_results = static_face_tracker.process(src_face)
                if src_results.multi_face_landmarks:
                    src_landmarks = get_landmark_points(src_results.multi_face_landmarks[0], src_face.shape[1], src_face.shape[0])
                    print("Reload Successful!")
                else:
                    src_landmarks = None
                    print("Reload Failed: Could not find landmarks on the new file.")
            else:
                print(f"Reload Failed: Could not find {reference_path}")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
