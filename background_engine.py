import cv2
import mediapipe as mp
import numpy as np

class BackgroundEngine:
    """Manages real-time background segmentation and replacement."""
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentor = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def replace_background(self, frame, background_img):
        """Replaces frame background and returns composite + mask."""
        if background_img is None:
            return frame, None

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmentor.process(rgb_frame)

        # Create binary mask (1 for person, 0 for background)
        mask = results.segmentation_mask > 0.5
        mask_raw = mask.astype(np.float32)
        
        # Refine mask: smoothing (Feathering)
        mask_sm = (mask_raw * 255).astype(np.uint8)
        mask_sm = cv2.GaussianBlur(mask_sm, (7, 7), 0) / 255.0
        mask_blend = mask_sm[:, :, np.newaxis]

        # Resize background to match frame
        h, w = frame.shape[:2]
        bg_resized = cv2.resize(background_img, (w, h))

        # Blend original frame and background using the mask
        composite = (mask_blend * frame + (1 - mask_blend) * bg_resized).astype(np.uint8)
        
        return composite, mask_raw
