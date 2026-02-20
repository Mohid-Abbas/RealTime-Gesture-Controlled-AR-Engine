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

    # Load assets
    reference_path = "assets/reference_face.png"
    hair_path = "assets/goku_hair.png"
    gi_path = "assets/goku_gi.png"
    
    src_face = cv2.imread(reference_path)
    # Load PNGs with alpha channel
    hair_overlay = cv2.imread(hair_path, -1)
    gi_overlay = cv2.imread(gi_path, -1)
    
    src_landmarks = None
    if src_face is not None:
        src_face = cv2.resize(src_face, (640, 640))
        static_face_tracker = FaceTracker(static_image_mode=True, min_detection_confidence=0.1)
        src_results = static_face_tracker.process(src_face)
        if src_results.multi_face_landmarks:
            src_landmarks = get_landmark_points(src_results.multi_face_landmarks[0], src_face.shape[1], src_face.shape[0])
    
    print("Project Saiyan AR is running. Press 's' to toggle, 'r' to reload, 'q' to quit.")

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
                # 1. Realistic Hair Overlay
                if face_results.multi_face_landmarks and hair_overlay is not None:
                    face_lm = face_results.multi_face_landmarks[0].landmark
                    # Define 3 points for mapping hair to head
                    # Landmarks: 21 (Right Temple), 251 (Left Temple), 10 (Top of Face)
                    pd = face_results.multi_face_landmarks[0].landmark
                    pts_dst = [
                        (int(pd[21].x * w), int(pd[21].y * h)),   # Left side of head (tracker right)
                        (int(pd[251].x * w), int(pd[251].y * h)), # Right side of head (tracker left)
                        (int(pd[10].x * w), int(pd[10].y * h))    # Top of face
                    ]
                    
                    oh, ow = hair_overlay.shape[:2]
                    # Source points in the asset (roughly bottom-left, bottom-right, top-center)
                    pts_src = [
                        (int(0.2 * ow), int(0.9 * oh)),
                        (int(0.8 * ow), int(0.9 * oh)),
                        (int(0.5 * ow), int(0.1 * oh))
                    ]
                    
                    # Apply lighting match before overlay
                    # Get a safe ROI for lighting estimation
                    y, x = pts_dst[2][1], pts_dst[2][0]
                    roi = display_frame[max(0, y):min(h, y+50), max(0, x-25):min(w, x+25)]
                    matched_hair = overlay_engine.match_lighting(hair_overlay, roi)
                    display_frame = overlay_engine.perspective_overlay(display_frame, matched_hair, pts_src, pts_dst)

                # 2. Realistic Gi Overlay using Pose
                l_shoulder, r_shoulder = pose_tracker.get_shoulder_points(pose_results, w, h)
                if l_shoulder and r_shoulder and gi_overlay is not None:
                    # Target points: shoulders and chest center
                    chest_center = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2 + 100)
                    pts_dst = [l_shoulder, r_shoulder, chest_center]
                    
                    oh, ow = gi_overlay.shape[:2]
                    # Source points in the asset: left shoulder, right shoulder, bottom mid
                    pts_src = [
                        (int(0.2 * ow), int(0.1 * oh)),
                        (int(0.8 * ow), int(0.1 * oh)),
                        (int(0.5 * ow), int(0.9 * oh))
                    ]
                    
                    # Get a safe ROI for lighting estimation
                    roi_y1, roi_y2 = max(0, l_shoulder[1]), min(h, l_shoulder[1]+100)
                    roi_x1, roi_x2 = max(0, min(l_shoulder[0], r_shoulder[0])), min(w, max(l_shoulder[0], r_shoulder[0]))
                    roi = display_frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    
                    matched_gi = overlay_engine.match_lighting(gi_overlay, roi)
                    display_frame = overlay_engine.perspective_overlay(display_frame, matched_gi, pts_src, pts_dst)
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
