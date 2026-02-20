import cv2
import numpy as np

class EffectsEngine:
    """Cinematic AR effects including energy balls and beams."""
    def __init__(self):
        self.particles = []

    def draw_energy_ball(self, frame, center, radius):
        """Draws a glowing energy ball at the specified center."""
        if center is None:
            return frame

        # Inner core
        cv2.circle(frame, center, radius, (255, 255, 255), -1)
        
        # Outer glow (using blurring)
        glow = np.zeros_like(frame)
        cv2.circle(glow, center, int(radius * 1.5), (255, 100, 0), -1)
        glow = cv2.GaussianBlur(glow, (51, 51), 0)
        
        frame = cv2.addWeighted(frame, 1.0, glow, 0.8, 0)
        return frame

    def draw_beam(self, frame, start_point, end_point, thickness):
        """Draws an energy beam between two points."""
        cv2.line(frame, start_point, end_point, (255, 250, 200), thickness)
        # Add glow to beam...
        return frame
