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

    def match_lighting(self, src, dst):
        """Adjusts src image brightness/contrast to match dst ROI."""
        if dst is None or dst.size == 0 or src is None or src.size == 0:
            return src
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
        dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        
        src_mean = np.mean(src_gray[src[:, :, 3] > 0])
        dst_mean = np.mean(dst_gray)
        
        if src_mean > 0:
            ratio = dst_mean / src_mean
            # Apply gain with limits to avoid extreme distortions
            gain = np.clip(ratio, 0.5, 1.5)
            # Apply to BGR channels only
            matched = src.copy()
            matched[:, :, :3] = np.clip(src[:, :, :3] * gain, 0, 255).astype(np.uint8)
            return matched
        return src

    def perspective_overlay(self, img, overlay, src_pts, dst_pts):
        """Warps overlay using perspective transform to match body movement."""
        h, w = img.shape[:2]
        oh, ow = overlay.shape[:2]
        
        # Calculate Homography or Affine depending on point count
        if len(src_pts) >= 4:
            M, _ = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts))
            warped = cv2.warpPerspective(overlay, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        else:
            M = cv2.getAffineTransform(np.float32(src_pts[:3]), np.float32(dst_pts[:3]))
            warped = cv2.warpAffine(overlay, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        
        # Blend
        alpha = warped[:, :, 3] / 255.0
        for c in range(3):
            img[:, :, c] = (alpha * warped[:, :, c] + (1 - alpha) * img[:, :, c]).astype(np.uint8)
        return img
