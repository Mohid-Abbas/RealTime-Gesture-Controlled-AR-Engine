import cv2
import numpy as np

class EffectsEngine:
    """Cinematic AR effects including energy balls and beams."""
    def __init__(self):
        self.particles = []

    def draw_energy_ball(self, frame, center, radius):
        """Procedural energy ball (fallback)."""
        if center is None:
            return frame
        cv2.circle(frame, center, radius, (255, 255, 255), -1)
        glow = np.zeros_like(frame)
        cv2.circle(glow, center, int(radius * 1.5), (255, 100, 0), -1)
        glow = cv2.GaussianBlur(glow, (51, 51), 0)
        return cv2.addWeighted(frame, 1.0, glow, 0.8, 0)

    def draw_kamehameha(self, frame, center, radius, asset):
        """Draws a high-quality Kamehameha PNG effect."""
        if asset is None or center is None:
            return self.draw_energy_ball(frame, center, radius)
        
        # Calculate size
        size = int(radius * 4) 
        x, y = center[0] - size // 2, center[1] - size // 2
        
        # Region of interest
        h, w = frame.shape[:2]
        y1, y2 = max(0, y), min(h, y + size)
        x1, x2 = max(0, x), min(w, x + size)
        
        if y1 >= y2 or x1 >= x2:
            return frame
            
        overlay = cv2.resize(asset, (size, size))
        crop_y1, crop_y2 = max(0, -y), min(size, h - y)
        crop_x1, crop_x2 = max(0, -x), min(size, w - x)
        
        overlay_crop = overlay[crop_y1:crop_y2, crop_x1:crop_x2]
        alpha = overlay_crop[:, :, 3] / 255.0
        
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (alpha * overlay_crop[:, :, c] + (1 - alpha) * frame[y1:y2, x1:x2, c])
            
        return frame
