import cv2
import numpy as np

class OverlayEngine:
    """Engine for applying PNG overlays (hair, clothes) onto the frame using landmarks."""
    def __init__(self):
        pass

    def overlay_alpha(self, img, overlay, pos, scale=1.0, angle=0):
        """Overlays a PNG with alpha channel onto an image."""
        h, w = overlay.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        overlay_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Handle rotation
        if angle != 0:
            M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
            overlay_resized = cv2.warpAffine(overlay_resized, M, (new_w, new_h), 
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                           borderValue=(0,0,0,0))

        # Calculate top-left position
        x, y = pos[0] - new_w // 2, pos[1] - new_h // 2

        # Region of interest in the background image
        y1, y2 = max(0, y), min(img.shape[0], y + new_h)
        x1, x2 = max(0, x), min(img.shape[1], x + new_w)

        # Region of interest in the overlay image
        oy1, oy2 = max(0, -y), min(new_h, img.shape[0] - y)
        ox1, ox2 = max(0, -x), min(new_w, img.shape[1] - x)

        if y1 >= y2 or x1 >= x2:
            return img

        overlay_crop = overlay_resized[oy1:oy2, ox1:ox2]
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha

        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha * overlay_crop[:, :, c] +
                                    alpha_inv * img[y1:y2, x1:x2, c])

        return img

    def draw_aura(self, frame, pose_results):
        """Draws a glowing aura around the user."""
        if not pose_results or not pose_results.pose_landmarks:
            return frame
        
        # Simple procedural aura: use a blurred version of the silhouette
        # In a real app, we'd use body segmentation
        # For now, we'll draw centered circles or highlights
        return frame
