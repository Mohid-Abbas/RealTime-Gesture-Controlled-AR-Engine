import numpy as np

def get_landmark_points(landmarks, width, height):
    """Converts MediaPipe landmarks to a list of (x, y) coordinates."""
    points = []
    for landmark in landmarks.landmark:
        points.append((int(landmark.x * width), int(landmark.y * height)))
    return points

def get_hand_center(hand_landmarks, width, height):
    """Calculates the average center of hand landmarks."""
    x_sum = sum(lm.x for lm in hand_landmarks.landmark)
    y_sum = sum(lm.y for lm in hand_landmarks.landmark)
    count = len(hand_landmarks.landmark)
    return (int(x_sum / count * width), int(y_sum / count * height))

def calculate_velocity(pos1, pos2, dt=1.0):
    """Calculates velocity between two points."""
    if pos1 is None or pos2 is None:
        return 0.0
    dist = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    return dist / dt
