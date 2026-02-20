import cv2
import numpy as np

def extract_goku_hair(image_path, output_path):
    """Semi-automated hair extraction using color masking and cropping."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Goku hair in the reference is black/dark
    # We'll use a rough crop first (top part of the head)
    h, w = img.shape[:2]
    hair_crop = img[0:int(h*0.4), int(w*0.3):int(w*0.7)]
    
    # Convert to BGRA
    hair_bgra = cv2.cvtColor(hair_crop, cv2.COLOR_BGR2BGRA)
    
    # Simple thresholding: make anything that isn't dark/black transparent
    # In the reference, the background is sky/clouds
    gray = cv2.cvtColor(hair_crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    hair_bgra[:, :, 3] = mask
    cv2.imwrite(output_path, hair_bgra)
    return True

def extract_goku_gi(image_path, output_path):
    """Semi-automated gi (chest) extraction."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    h, w = img.shape[:2]
    # Crop chest area
    gi_crop = img[int(h*0.3):int(h*0.6), int(w*0.2):int(w*0.8)]
    
    gi_bgra = cv2.cvtColor(gi_crop, cv2.COLOR_BGR2BGRA)
    
    # Masking for orange colors (HSV)
    hsv = cv2.cvtColor(gi_crop, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([20, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Masking for blue colors (undershirt)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    combined_mask = cv2.bitwise_or(mask_orange, mask_blue)
    
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    gi_bgra[:, :, 3] = combined_mask
    cv2.imwrite(output_path, gi_bgra)
    return True

if __name__ == "__main__":
    ref_path = "assets/full_body_reference.png"
    print("Extracting hair...")
    if extract_goku_hair(ref_path, "assets/goku_hair.png"):
        print("Hair extracted to assets/goku_hair.png")
    
    print("Extracting gi...")
    if extract_goku_gi(ref_path, "assets/goku_gi.png"):
        print("Gi extracted to assets/goku_gi.png")
