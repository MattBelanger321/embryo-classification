import tensorflow as tf
from tensorflow.keras import backend as K

def compute_dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice_coefficient = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice_coefficient

def extract_foreground_points_3d(binary_mask, img_shape):
    points = tf.where(tf.equal(binary_mask, 1))
    z_axis = tf.ones((tf.shape(points)[0], 1), dtype=tf.int64)
    points_3d = tf.concat([points, z_axis], axis=1)

    # Normalize the points by the image size to bound coordinates between 0 and 1
    normalized_points = tf.cast(points_3d, tf.float32) / tf.constant(img_shape + [1], dtype=tf.float32)
    return normalized_points

def hausdorff_distance_calculator(y_true, y_pred, img_shape, max_distance=1.0):
    set_a = extract_foreground_points_3d(y_true, img_shape)
    set_b = extract_foreground_points_3d(y_pred, img_shape)

    def compute_distance():
        expanded_a = tf.expand_dims(set_a, 1)
        expanded_b = tf.expand_dims(set_b, 0)
        dists = tf.norm(expanded_a - expanded_b, axis=2)
        min_dist_a_to_b = tf.reduce_min(dists, axis=1)
        min_dist_b_to_a = tf.reduce_min(dists, axis=0)
        return tf.maximum(tf.reduce_max(min_dist_a_to_b), tf.reduce_max(min_dist_b_to_a))

    hausdorff_distance = tf.cond(
        tf.logical_or(tf.equal(tf.size(set_a), 0), tf.equal(tf.size(set_b), 0)),
        lambda: tf.constant(max_distance, dtype=tf.float32),
        compute_distance
    )
    return hausdorff_distance

def accuracy_metric(y_true, y_pred, img_shape):
    dice = compute_dice_coefficient(y_true, y_pred)
    hausdorff_dist = hausdorff_distance_calculator(y_true, y_pred, img_shape)
    accuracy = 0.4 * dice + 0.6 * hausdorff_dist
    return accuracy
