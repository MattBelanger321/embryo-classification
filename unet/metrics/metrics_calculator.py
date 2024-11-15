import tensorflow as tf
import numpy as np
from tensorflow.keras.saving import register_keras_serializable

def hausdorff_distance_2d(y_true, y_pred, image_shape):
    y_true_coords = tf.where(y_true == 1.0)
    y_pred_coords = tf.where(y_pred == 1.0)
    
    no_foreground_true = tf.less(tf.size(y_true_coords), 1)
    no_foreground_pred = tf.less(tf.size(y_pred_coords), 1)
    
    hausdorff_dist = tf.cond(
        tf.logical_or(no_foreground_true, no_foreground_pred), 
        lambda: 0.0, 
        lambda: calculate_hausdorff_distance(y_true_coords, y_pred_coords, image_shape)
    )
    
    return hausdorff_dist

def calculate_hausdorff_distance(y_true_coords, y_pred_coords, image_shape):
    # Normalize coordinates by image size (height, width)
    y_true_coords = tf.cast(y_true_coords, tf.float32) / tf.cast(image_shape, tf.float32)
    y_pred_coords = tf.cast(y_pred_coords, tf.float32) / tf.cast(image_shape, tf.float32)
    
    # Expand dimensions to compute pairwise squared distances
    y_true_coords_exp = tf.expand_dims(y_true_coords, axis=1)
    y_pred_coords_exp = tf.expand_dims(y_pred_coords, axis=0)
    
    # Compute squared Euclidean distances between all point pairs
    squared_distances = tf.reduce_sum(tf.square(y_true_coords_exp - y_pred_coords_exp), axis=-1)
    d_true_to_pred = tf.reduce_min(squared_distances, axis=1)
    d_pred_to_true = tf.reduce_min(squared_distances, axis=0)
    
    # Compute Hausdorff distance as the max of minimum distances
    hausdorff_dist = tf.sqrt(tf.maximum(tf.reduce_max(d_true_to_pred), tf.reduce_max(d_pred_to_true)))
    
    # Compute the maximum possible distance in normalized coordinates (diagonal of the unit square)
    max_distance = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(image_shape, tf.float32))))
    
    # Normalize Hausdorff distance to fall within the range [0, 1]
    normalized_hausdorff_dist = hausdorff_dist / max_distance
    
    return normalized_hausdorff_dist


def dice_coefficient(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + tf.keras.backend.epsilon())

@register_keras_serializable(name="combined_metric")
def combined_metric(y_true, y_pred, image_shape=(128, 128, 1)):
    dice_score = 0.0
    hausdorff_score = 0.0
    
    for i in range(3):  # Assuming three classes
        y_true_class = y_true[..., i]
        y_pred_class = y_pred[..., i]

        dice = dice_coefficient(y_true_class, y_pred_class)
        hausdorff = hausdorff_distance_2d(y_true_class, y_pred_class, image_shape)

        dice_score += dice
        hausdorff_score += hausdorff

    combined = 0.4 * dice_score + 0.6 * (1 - hausdorff_score)
    return combined / 3.0