import time
import numpy as np
from utils import get_hand_center, calculate_velocity

class GestureEngine:
    """State machine for detecting face swipes and Kamehameha poses."""
    def __init__(self):
        self.last_hand_pos = [None, None]
        self.last_hand_areas = [0, 0]
        self.swipe_triggered = False
        self.energy_triggered = False
        self.burst_triggered = False
        self.hand_openness = [0, 0] # 0 = Fist, 1 = Palm
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

        # Detect Kamehameha Pose (Hands close together) - CHECK THIS FIRST to block swipes
        in_kamehameha_zone = False
        if len(hand_centers) >= 2:
            dist = calculate_velocity(hand_centers[0], hand_centers[1], dt=1.0)
            
            # Detect Hand Openness (Fist vs Palm)
            self.hand_openness = []
            for hl in hand_results.multi_hand_landmarks:
                p_center = hl.landmark[0]
                m_tip = hl.landmark[12]
                tip_dist = np.sqrt((p_center.x - m_tip.x)**2 + (p_center.y - m_tip.y)**2)
                # Normalize: fist is around 0.1, full palm around 0.3+
                openness = np.clip((tip_dist - 0.1) / 0.2, 0, 1)
                self.hand_openness.append(openness)
            
            avg_openness = sum(self.hand_openness) / len(self.hand_openness) if self.hand_openness else 0

            # Increased distance threshold to 400 for better stability
            if dist < 400:
                in_kamehameha_zone = True
                self.energy_triggered = True
                
                # Calculate current combined hand area
                current_total_area = 0
                for hl in hand_results.multi_hand_landmarks:
                    xs = [lm.x for lm in hl.landmark]
                    ys = [lm.y for lm in hl.landmark]
                    current_total_area += (max(xs) - min(xs)) * (max(ys) - min(ys))
                
                # Continuous Burst Hysteresis:
                # Trigger at 0.4, but stay bursting until it drops below 0.25
                threshold = 0.25 if self.burst_triggered else 0.4
                if sum(self.last_hand_areas) > 0 and avg_openness > threshold:
                    self.burst_triggered = True
                
                # Maintain base area for charge tracking
                if sum(self.last_hand_areas) == 0 and avg_openness < 0.5:
                    self.last_hand_areas = [current_total_area, 0]
                    print(f"Charge Initiated. Base Area: {current_total_area:.4f}")

                if self.tick % 5 == 0:
                    state = "BURSTING" if self.burst_triggered else "CHARGING"
                    print(f"SJ Mode: {state} | O: {avg_openness:.2f}")
            else:
                self.last_hand_areas = [0, 0]

        # Detect Face Swipe - ONLY if not trying to do a Kamehameha
        if not in_kamehameha_zone and face_results.multi_face_landmarks and len(hand_centers) > 0:
            face_landmarks = face_results.multi_face_landmarks[0]
            nose_tip = face_landmarks.landmark[1]
            nose_x = int(nose_tip.x * width)
            
            for i, hand_center in enumerate(hand_centers):
                prev_pos = self.last_hand_pos[i] if i < len(self.last_hand_pos) else None
                if prev_pos:
                    if (prev_pos[0] < nose_x < hand_center[0]) or (hand_center[0] < nose_x < prev_pos[0]):
                        velocity = calculate_velocity(prev_pos, hand_center, dt)
                        if velocity > 300:
                            self.swipe_triggered = True
                            print(f"Swipe detected! Velocity: {int(velocity)}")

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
