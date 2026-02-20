import cv2
import mediapipe as mp
import numpy as np

class BackgroundEngine:
    """Manages real-time background segmentation and replacement with animated layers."""
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentor = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.tick = 0
        self.cap = None
        self.video_path = None

    def set_video_background(self, video_path):
        """Sets a video file as the background asset."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video background {video_path}")
            self.cap = None

    def get_animated_background(self, layers, width, height):
        """Creates a composite animated background from multi-layer assets."""
        self.tick += 1
        composite_bg = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i, layer in enumerate(layers):
            if layer is None: continue
            
            # Different speeds for parallax effect
            speed = (i + 1) * 2
            offset = (self.tick * speed) % width
            
            # Resize layer to match frame
            layer_resized = cv2.resize(layer, (width, height))
            
            # Loop/Scroll layer
            scrolled = np.roll(layer_resized, offset, axis=1)
            
            # Blend layer into composite
            if i == 0:
                composite_bg = scrolled
            else:
                # Additive or alpha blend depending on layer type
                # For simplicity, assuming these are nebula/star layers with transparency or black bg
                composite_bg = cv2.add(composite_bg, scrolled)
                
        return composite_bg

    def replace_background(self, frame, background_layers=None):
        """Replaces frame background with animated layers and returns composite + mask."""
        h, w = frame.shape[:2]
        
        # 1. Get/Create the animated background (only if layers provided)
        bg_img = None
        if background_layers is not None or self.cap is not None:
            if self.cap is not None:
                ret, bg_img = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, bg_img = self.cap.read()
                if bg_img is not None:
                    bg_img = cv2.resize(bg_img, (w, h))

            if bg_img is None and background_layers is not None:
                if isinstance(background_layers, list):
                    bg_img = self.get_animated_background(background_layers, w, h)
                else:
                    bg_img = cv2.resize(background_layers, (w, h))

        # 2. Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmentor.process(rgb_frame)

        # 3. Create binary mask
        if results.segmentation_mask is None:
            return frame, None
            
        mask = results.segmentation_mask > 0.5
        mask_raw = mask.astype(np.float32)
        
        # 4. Final Blend (only if we have a background image)
        if bg_img is not None:
            mask_sm = (mask_raw * 255).astype(np.uint8)
            mask_sm = cv2.GaussianBlur(mask_sm, (7, 7), 0) / 255.0
            mask_blend = mask_sm[:, :, np.newaxis]
            composite = (mask_blend * frame + (1 - mask_blend) * bg_img).astype(np.uint8)
            return composite, mask_raw
        
        return frame, mask_raw
