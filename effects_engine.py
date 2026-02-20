import cv2
import numpy as np
import random

class Particle:
    """Represents a single energy fragment."""
    def __init__(self, pos, color):
        self.pos = np.array(pos, dtype=float)
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(2, 8)
        self.vel = np.array([np.cos(angle) * speed, np.sin(angle) * speed])
        self.life = 1.0  # 1.0 to 0.0
        self.decay = random.uniform(0.02, 0.05)
        self.color = color

class EffectsEngine:
    """Advanced AR effects engine with cinematic lighting and physics-based visuals."""
    def __init__(self):
        self.tick = 0
        self.particles = []
        self.dust_particles = []
        self.rock_particles = []
        self.burst_timer = 0
        self.shake_offset = (0, 0)
        # Color Palette: Yellow/Gold/White
        self.color_outer = (20, 150, 255) # Golden yellow in BGR
        self.color_mid = (50, 220, 255)   # Bright yellow
        self.color_spark = (200, 255, 255) # White-yellow

    def additive_blend(self, background, overlay):
        """Standard Linear Dodge (Add) blending with dynamic scene exposure."""
        # Brighten background slightly based on overlay intensity (Exposure)
        exposure = cv2.addWeighted(background, 1.0, background, 0.05, 0)
        added = cv2.add(exposure, overlay)
        return added

    def draw_fractal_lightning(self, frame, start_p, end_p, color, thickness=2, noise=20):
        """Recursively draws jagged, branching electrical arcs."""
        if np.linalg.norm(np.array(start_p) - np.array(end_p)) < 10:
            cv2.line(frame, start_p, end_p, color, thickness)
            return

        # Calculate midpoint with random offset
        mid_x = (start_p[0] + end_p[0]) // 2 + random.randint(-noise, noise)
        mid_y = (start_p[1] + end_p[1]) // 2 + random.randint(-noise, noise)
        mid_p = (mid_x, mid_y)

        # Draw main branches
        self.draw_fractal_lightning(frame, start_p, mid_p, color, thickness, noise // 2)
        self.draw_fractal_lightning(frame, mid_p, end_p, color, thickness, noise // 2)

        # Occasional side branches
        if random.random() < 0.2:
            branch_end = (mid_x + random.randint(-noise*2, noise*2), 
                          mid_y + random.randint(-noise*2, noise*2))
            self.draw_fractal_lightning(frame, mid_p, branch_end, color, max(1, thickness-1), noise // 2)

    def apply_heat_distortion(self, frame, center, radius):
        """Simulates air refraction/heat haze around the energy ball."""
        h, w = frame.shape[:2]
        distort_r = int(radius * 2.2)
        
        y1, y2 = max(0, center[1]-distort_r), min(h, center[1]+distort_r)
        x1, x2 = max(0, center[0]-distort_r), min(w, center[0]+distort_r)
        
        if y2 <= y1 or x2 <= x1: return frame
        
        roi = frame[y1:y2, x1:x2].copy()
        rows, cols = roi.shape[:2]
        
        # Create a wavy displacement map
        map_x = np.zeros((rows, cols), np.float32)
        map_y = np.zeros((rows, cols), np.float32)
        
        # Performance optimized waviness
        time_factor = self.tick * 0.5
        for i in range(rows):
            for j in range(cols):
                # Only distort within a certain radius
                dist = np.sqrt((i - rows//2)**2 + (j - cols//2)**2)
                if dist < distort_r:
                    map_x[i, j] = j + 3 * np.sin(i / 10.0 + time_factor)
                    map_y[i, j] = i + 3 * np.cos(j / 10.0 + time_factor)
                else:
                    map_x[i, j] = j
                    map_y[i, j] = i
                    
        distorted_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = distorted_roi
        return frame

    def draw_burst(self, frame, center):
        """Creates an intense, forward-expanding energy blast."""
        if center is None: return frame
        h, w = frame.shape[:2]
        layer = np.zeros_like(frame)
        
        # Expanding concentric rings (Shockwaves)
        for i in range(3):
            r = int((self.tick % 30) * 10) + (i * 50)
            alpha = max(0, 255 - r)
            cv2.circle(layer, center, r, (255, 255, 255), 10 - i*2)
        
        # Massive Bloom Burst (Yellow)
        cv2.circle(layer, center, 400, (0, 180, 255), -1)
        layer = cv2.GaussianBlur(layer, (99, 99), 0)
        
        # Branching lightning firing everywhere (Yellow-White)
        for _ in range(8):
            angle = random.uniform(0, 2 * np.pi)
            end_p = (int(center[0] + 600 * np.cos(angle)), int(center[1] + 600 * np.sin(angle)))
            self.draw_fractal_lightning(layer, center, end_p, (220, 255, 255), 4, noise=100)
            
        return self.additive_blend(frame, layer)

    def draw_energy_ball(self, frame, center, radius, asset=None, burst=False):
        """Main rendering pipeline for the cinematic energy ball."""
        if center is None: return frame
        self.tick += 1
        
        # 0. Handle Screen Shake and Burst Timing
        if burst:
            self.burst_timer = max(self.burst_timer, 10) # Maintain at least 10 frames of burst
        
        intensity = 0
        if self.burst_timer > 0:
            intensity = 15 # Strong shake during burst
            frame = self.draw_burst(frame, center)
            # Add scene flash (stronger at start of burst)
            flash_intensity = (self.burst_timer / 10.0) * 0.4
            flash = np.full_like(frame, 255)
            frame = cv2.addWeighted(frame, (1.0 - flash_intensity), flash, flash_intensity, 0)
            self.burst_timer -= 1
            # When bursting, we skip the normal energy ball drawing
            # But we update shake_offset
            self.shake_offset = (random.randint(-intensity, intensity), 
                                 random.randint(-intensity, intensity))
            return frame
        
        # Charging shake
        if radius > 50:
            intensity = 3
        
        if intensity > 0:
            self.shake_offset = (random.randint(-intensity, intensity), 
                                 random.randint(-intensity, intensity))
        else:
            self.shake_offset = (0, 0)

        h, w = frame.shape[:2]

        # 1. Apply Heat Haze, Dust, and Rocks
        frame = self.apply_heat_distortion(frame, center, radius)
        frame = self.draw_dust(frame)
        frame = self.draw_rocks(frame, center, radius)
        
        # 2. Create a transparent black overlay for additive blending
        effect_layer = np.zeros_like(frame)
        
        # 3. Dynamic Scale (Pulse)
        pulse = 1.0 + 0.15 * np.sin(self.tick * 0.4)
        r_dyn = int(radius * pulse)

        # 4. Multi-Layer Bloom (Outer Halos)
        # Deep Gold Outer Glow
        cv2.circle(effect_layer, center, int(r_dyn * 2.5), (0, 120, 200), -1)
        # Bright Yellow Mid Glow
        cv2.circle(effect_layer, center, int(r_dyn * 1.8), (50, 200, 255), -1)
        
        # Apply heavy blur to halos
        effect_layer = cv2.GaussianBlur(effect_layer, (99, 99), 0)

        # 5. Fractal Lightning Arcs (Yellow)
        for _ in range(4):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(radius * 0.5, radius * 2.5)
            end_p = (int(center[0] + dist * np.cos(angle)), int(center[1] + dist * np.sin(angle)))
            self.draw_fractal_lightning(effect_layer, center, end_p, (200, 255, 255), 2, noise=30)

        # 6. Particle System (Energy Fragments)
        if random.random() < 0.4:
            self.particles.append(Particle(center, (255, 255, 200)))

        for p in self.particles[:]:
            p.pos += p.vel
            p.life -= p.decay
            if p.life <= 0:
                self.particles.remove(p)
                continue
            p_radius = max(1, int(4 * p.life))
            p_color = tuple(int(c * p.life) for c in p.color)
            cv2.circle(effect_layer, (int(p.pos[0]), int(p.pos[1])), p_radius, p_color, -1)

        # 7. Core Asset or Procedural Core
        if asset is not None:
            overlay_r = int(radius * 1.5)
            y1, y2 = max(0, center[1]-overlay_r), min(h, center[1]+overlay_r)
            x1, x2 = max(0, center[0]-overlay_r), min(w, center[0]+overlay_r)
            if y2 > y1 and x2 > x1:
                asset_res = cv2.resize(asset, (x2-x1, y2-y1))
                if asset_res.shape[2] == 4:
                    mask = (asset_res[:,:,3] / 255.0)[:,:,None]
                    effect_layer[y1:y2, x1:x2] = (mask * asset_res[:,:,:3] + (1-mask) * effect_layer[y1:y2, x1:x2]).astype(np.uint8)
        
        # White hot core
        cv2.circle(effect_layer, center, int(r_dyn * 0.6), (255, 255, 255), -1)

        # 8. Final Additive Merge
        return self.additive_blend(frame, effect_layer)
    def draw_dust(self, frame):
        """Draws moving dust/debris across the screen."""
        if not self.dust_particles:
            for _ in range(30):
                self.dust_particles.append({
                    'pos': [random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0])],
                    'vel': [random.uniform(-2, -5), random.uniform(-1, 1)], # Moving left
                    'size': random.randint(1, 3),
                    'alpha': random.uniform(0.1, 0.4)
                })

        overlay = frame.copy()
        for p in self.dust_particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            if p['pos'][0] < 0: p['pos'][0] = frame.shape[1]
            if p['pos'][1] < 0: p['pos'][1] = frame.shape[0]
            if p['pos'][1] > frame.shape[0]: p['pos'][1] = 0
            
            cv2.circle(overlay, (int(p['pos'][0]), int(p['pos'][1])), p['size'], (100, 150, 200), -1) # Dust color

        return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0) # Thicker dust

    def draw_rocks(self, frame, center, radius):
        """Draws flying debris/rocks that lift off the ground."""
        h, w = frame.shape[:2]
        # Emit rocks if charging or bursting
        if radius > 40 and random.random() < 0.2:
            self.rock_particles.append({
                'pos': [center[0] + random.randint(-400, 400), h], # Start from bottom
                'vel': [random.uniform(-1, 1), random.uniform(-5, -15)], # Fly up
                'size': random.randint(5, 15),
                'angle': 0,
                'rot_speed': random.uniform(-10, 10)
            })

        overlay = frame.copy()
        for p in self.rock_particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['angle'] += p['rot_speed']
            
            if p['pos'][1] < -50:
                self.rock_particles.remove(p)
                continue
            
            # Draw rock as a simple polygon
            pts = np.array([
                [p['pos'][0]-p['size'], p['pos'][1]],
                [p['pos'][0], p['pos'][1]-p['size']],
                [p['pos'][0]+p['size'], p['pos'][1]],
                [p['pos'][0], p['pos'][1]+p['size']]
            ], np.int32)
            cv2.fillPoly(overlay, [pts], (40, 60, 80)) # Dark rock color

        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    def apply_screen_shake(self, frame):
        """Applies the calculated shake offset to the entire frame."""
        if self.shake_offset == (0, 0): return frame
        h, w = frame.shape[:2]
        matrix = np.float32([[1, 0, self.shake_offset[0]], [0, 1, self.shake_offset[1]]])
        return cv2.warpAffine(frame, matrix, (w, h))

    def draw_body_lightning(self, frame, mask):
        """Draws electric arcs crawling around the user's silhouette."""
        if mask is None: return frame
        
        h, w = frame.shape[:2]
        layer = np.zeros_like(frame)
        
        # Find edges of the mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        edges = cv2.Canny(mask_uint8, 100, 200)
        edge_pts = np.argwhere(edges > 0)
        
        if len(edge_pts) > 10:
            for _ in range(5):
                # Pick two random points on the edge
                idx1 = random.randint(0, len(edge_pts)-1)
                idx2 = random.randint(0, len(edge_pts)-1)
                p1 = (edge_pts[idx1][1], edge_pts[idx1][0])
                p2 = (edge_pts[idx2][1], edge_pts[idx2][0])
                
                # If they are close enough, draw a crawling arc
                if np.linalg.norm(np.array(p1) - np.array(p2)) < 150:
                    self.draw_fractal_lightning(layer, p1, p2, (200, 255, 255), 1, noise=15)

        return self.additive_blend(frame, layer)
