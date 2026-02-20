import cv2
import numpy as np

class FaceSwapEngine:
    """Landmark-based face transformation engine using Delaunay triangulation."""
    def __init__(self):
        # MediaPipe FaceMesh indices for triangulation (simplified or full)
        # For a professional look, we usually use a predefined set of triangles
        self.triangulation_indices = None 

    def get_triangulation_indices(self, points):
        """Returns triangulation indices based on points."""
        rect = (0, 0, 1280, 720) # Default, should be updated
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert(p)
        triangleList = subdiv.getTriangleList()
        # This is a bit complex for real-time; usually we precompute indices
        # based on the 468 landmarks once.
        return triangleList

    def apply_affine_transform(self, src, src_tri, dst_tri, size):
        """Warp a triangle from src to dst."""
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, 
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return dst

    def warp_triangle(self, img1, img2, t1, t2):
        """Warps a triangular region from img1 to img2."""
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        t1_rect = []
        t2_rect = []
        t2_rect_int = []

        for i in range(3):
            t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

        img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        size = (r2[2], r2[3])
        img2_rect = self.apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

        img2_rect = img2_rect * mask
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ( (1.0, 1.0, 1.0) - mask )
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

    def warp_face(self, src_img, dst_img, src_points, dst_points, tri_indices):
        """Warps the entire face using Delaunay triangulation."""
        img_w, img_h = dst_img.shape[1], dst_img.shape[0]
        
        # Output image for face warp
        warped_face = np.zeros_like(dst_img)
        
        for tri in tri_indices:
            # Get triangle points for src and dst
            t1 = [src_points[tri[0]], src_points[tri[1]], src_points[tri[2]]]
            t2 = [dst_points[tri[0]], dst_points[tri[1]], dst_points[tri[2]]]
            
            # Warp current triangle
            self.warp_triangle(src_img, warped_face, t1, t2)
            
        return warped_face

    def swap_face(self, frame, src_face, src_landmarks, dst_landmarks):
        """Performs face swap from src_face to frame using landmarks."""
        if src_landmarks is None or dst_landmarks is None:
            return frame

        # We need triangulation indices. For 468 landmarks, we usually pre-calculate these.
        # Here we'll generate them for the convex hull of the landmarks to simplify.
        points_dst = np.array(dst_landmarks, np.int32)
        hull_idx = cv2.convexHull(points_dst, returnPoints=False)
        
        hull8bit = []
        for i in range(len(hull_idx)):
            hull8bit.append(dst_landmarks[hull_idx[i][0]])
            
        # Draw the target mask
        mask = np.zeros(frame.shape, dtype=frame.dtype)
        cv2.fillConvexPoly(mask, np.int32(hull8bit), (255, 255, 255))
        
        # Perform Delaunay Triangulation on the hull
        rect = cv2.boundingRect(np.array(hull8bit))
        subdiv = cv2.Subdiv2D(rect)
        for p in dst_landmarks:
            subdiv.insert(p)
        
        triangle_list = subdiv.getTriangleList()
        tri_indices = []
        for t in triangle_list:
            pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            indices = []
            for p in pt:
                # Find the index of the landmark closest to the triangle vertex
                dist = np.linalg.norm(np.array(dst_landmarks) - p, axis=1)
                indices.append(np.argmin(dist))
            tri_indices.append(indices)

        # Warp the face
        warped_face = self.warp_face(src_face, frame, src_landmarks, dst_landmarks, tri_indices)
        
        # Professional Blending (Seamless Clone)
        # Find center of face for seamless clone
        r = cv2.boundingRect(np.int32(hull8bit))
        center = ((r[0] + int(r[2]/2), r[1] + int(r[3]/2)))
        
        # Prepare mask for seamless clone
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        output = cv2.seamlessClone(warped_face, frame, mask, center, cv2.NORMAL_CLONE)
        return output
