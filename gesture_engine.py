import time
from utils import get_hand_center, calculate_velocity

class GestureEngine:
    """State machine for detecting face swipes and Kamehameha poses."""
    def __init__(self):
        self.last_hand_pos = [None, None]
        self.last_hand_areas = [0, 0]
        self.swipe_triggered = False
        self.energy_triggered = False
        self.burst_triggered = False
        self.last_time = time.time()
        self.tick = 0

    def update(self, hand_results, face_results, width, height):
        """Updates gesture states based on new tracking data."""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.tick += 1

        # Reset per frame
        self.energy_triggered = False
        self.burst_triggered = False

        if not hand_results.multi_hand_landmarks:
            self.last_hand_pos = [None, None]
            return

        hand_centers = []
        for hand_landmarks in hand_results.multi_hand_landmarks:
            center = get_hand_center(hand_landmarks, width, height)
            hand_centers.append(center)

        # Detect Face Swipe (Hand moving across face)
        if face_results.multi_face_landmarks and len(hand_centers) > 0:
            face_landmarks = face_results.multi_face_landmarks[0]
            # Get face center (nose tip is usually index 1)
            nose_tip = face_landmarks.landmark[1]
            nose_x = int(nose_tip.x * width)
            
            for i, hand_center in enumerate(hand_centers):
                prev_pos = self.last_hand_pos[i] if i < len(self.last_hand_pos) else None
                if prev_pos:
                    # Check if hand moved horizontally across the nose x-coordinate
                    # (Quick swipe detection logic)
                    if (prev_pos[0] < nose_x < hand_center[0]) or (hand_center[0] < nose_x < prev_pos[0]):
                        velocity = calculate_velocity(prev_pos, hand_center, dt)
                        if velocity > 300: # Lowered from 500 for better sensitivity
                            self.swipe_triggered = True
                            print(f"Swipe detected! Velocity: {int(velocity)}")

        # Detect Kamehameha Pose (Hands close together)
        if len(hand_centers) >= 2:
            dist = calculate_velocity(hand_centers[0], hand_centers[1], dt=1.0)
            if dist < 200: # Slightly more lenient distance
                self.energy_triggered = True
                
                # Calculate current combined hand area
                current_total_area = 0
                for hl in hand_results.multi_hand_landmarks:
                    xs = [lm.x for lm in hl.landmark]
                    ys = [lm.y for lm in hl.landmark]
                    current_total_area += (max(xs) - min(xs)) * (max(ys) - min(ys))
                
                # Initialize base area if it's the first frame hands are together
                if sum(self.last_hand_areas) == 0:
                    self.last_hand_areas = [current_total_area, 0] # Use first slot for total base
                    print(f"Charge Initiated. Base Area: {current_total_area:.4f}")
                
                base_area = self.last_hand_areas[0]
                growth_ratio = current_total_area / base_area if base_area > 0 else 1.0
                
                # Debug print for the user
                if self.tick % 10 == 0:
                    print(f"Energy Charging... Area Growth: {growth_ratio:.2f}x")

                # If hands grow by 25% relative to when they first met, OR if absolute area is huge (> 10% of frame)
                # This makes the pose in the user's image trigger the burst much more easily
                if growth_ratio > 1.25 or current_total_area > 0.12:
                    self.burst_triggered = True
                    if self.tick % 5 == 0:
                        print(f"BURST TRIGGERED! Area: {current_total_area:.2f}")
            else:
                # Reset base area when hands separate
                if sum(self.last_hand_areas) > 0:
                    print("Hands separated. Resetting charge.")
                self.last_hand_areas = [0, 0]

        # Update last positions for velocity next frame
        for i in range(min(len(hand_centers), 2)):
            self.last_hand_pos[i] = hand_centers[i]

    def is_swipe_triggered(self):
        return self.swipe_triggered

    def is_energy_triggered(self):
        return self.energy_triggered

    def is_burst_triggered(self):
        return self.burst_triggered

    def reset_swipe(self):
        self.swipe_triggered = False
