import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# Compute dice coefficient
def compute_dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice_coefficient = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice_coefficient

def extract_foreground_points(binary_mask):
    # Extracts coordinates of the foreground points from a binary mask."""
    return tf.where(tf.equal(binary_mask, 1))

def hausdorff_distance_calculator(y_true, y_pred):
    set_a = extract_foreground_points(y_true)
    set_b = extract_foreground_points(y_pred)

    # Compute pairwise distances
    expanded_a = tf.expand_dims(set_a, 1)
    expanded_b = tf.expand_dims(set_b, 0)
    dists = tf.norm(expanded_a - expanded_b, axis=2)

    # Minimum distances from set_a to set_b and vice versa
    min_dist_a_to_b = tf.reduce_min(dists, axis=1)
    min_dist_b_to_a = tf.reduce_min(dists, axis=0)

    # The Hausdorff distance
    hausdorff_distance = tf.maximum(tf.reduce_max(min_dist_a_to_b), tf.reduce_max(min_dist_b_to_a))
    return hausdorff_distance

def accuracy_metric(y_true, y_pred):
    dice = compute_dice_coefficient(y_true, y_pred)
    hausdorff_dist = hausdorff_distance_calculator(y_true, y_pred)
    accuracy = 0.4 * dice + 0.6 * (1 - hausdorff_dist)  # Adjust as necessary
    return accuracy
