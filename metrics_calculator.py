import numpy as np
from tensorflow.keras import backend as K

# Compute dice coefficient
def compute_dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice_coefficient = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice_coefficient

def extract_foreground_points(binary_mask):
    # Extracts coordinates of the foreground points from a binary mask."""
    return np.argwhere(binary_mask)

def hausdorff_distance_calculator(y_true, y_pred):
    set_a = extract_foreground_points(y_true)
    set_b = extract_foreground_points(y_pred)

    # Calculates the Hausdorff distance between two sets of points.
    set_a = np.array(set_a)
    set_b = np.array(set_b)

    # Compute the pairwise distances
    dists = np.linalg.norm(set_a[:, np.newaxis] - set_b, axis=2)

    # Minimum distances from set_a to set_b and vice versa
    min_dist_a_to_b = np.min(dists, axis=1)
    min_dist_b_to_a = np.min(dists, axis=0)

    # The Hausdorff distance
    return max(np.max(min_dist_a_to_b), np.max(min_dist_b_to_a))

def accuracy_metric(y_true, y_pred):
    m1 = 0.6*hausdorff_distance_calculator(y_true, y_pred)
    m2 = 0.4*compute_dice_coefficient(y_true, y_pred)
    return m1 + m2