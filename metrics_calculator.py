import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# Compute dice coefficient
def compute_dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice_coefficient = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice_coefficient

def extract_foreground_points_3d(binary_mask):
    """
    Extracts coordinates of the foreground points from a binary mask and adds a Z-axis.
    The Z-axis depth for all slices is set to 1, following competition guidelines.
    """
    points = tf.where(tf.equal(binary_mask, 1))
    z_axis = tf.ones((tf.shape(points)[0], 1), dtype=tf.int64)  # Add Z-axis with depth 1
    return tf.cast(tf.concat([points, z_axis], axis=1), tf.float32)

def hausdorff_distance_calculator(y_true, y_pred):
    # Extract points for each binary mask and convert to float
    set_a = extract_foreground_points_3d(y_true)
    set_b = extract_foreground_points_3d(y_pred)

    def compute_distance():
        # Compute pairwise distances
        expanded_a = tf.expand_dims(set_a, 1)
        expanded_b = tf.expand_dims(set_b, 0)
        dists = tf.norm(expanded_a - expanded_b, axis=2)

        # Minimum distances from set_a to set_b and vice versa
        min_dist_a_to_b = tf.reduce_min(dists, axis=1)
        min_dist_b_to_a = tf.reduce_min(dists, axis=0)

        # The Hausdorff distance
        return tf.maximum(tf.reduce_max(min_dist_a_to_b), tf.reduce_max(min_dist_b_to_a))

    # Use tf.cond to check if either set is empty
    hausdorff_distance = tf.cond(
        tf.logical_or(tf.equal(tf.size(set_a), 0), tf.equal(tf.size(set_b), 0)),
        lambda: tf.constant(1.0, dtype=tf.float32),  # Return max distance if one set is empty
        compute_distance  # Otherwise compute the distance
    )
    return hausdorff_distance

def accuracy_metric(y_true, y_pred):
    dice = compute_dice_coefficient(y_true, y_pred)
    hausdorff_dist = hausdorff_distance_calculator(y_true, y_pred)
    
    # Combine the metrics with the competition's weighting scheme
    accuracy = 0.4 * dice + 0.6 * (1 - hausdorff_dist)  # Ensure Hausdorff is normalized as expected
    return accuracy
