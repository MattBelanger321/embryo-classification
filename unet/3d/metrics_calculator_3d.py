import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice coefficient for 3D data in TensorFlow/Keras.
    
    Args:
        y_true (tensor): Ground truth segmentation (binary), shape (batch_size, depth, height, width, channels)
        y_pred (tensor): Predicted segmentation (binary), shape (batch_size, depth, height, width, channels)
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        float: Dice coefficient score
    """

    # Convert to TensorFlow tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Flatten the tensors using tf.reshape (instead of PyTorch's .view())
    y_true_flat = tf.reshape(y_true, (-1,))
    y_pred_flat = tf.reshape(y_pred, (-1,))
    
    # Compute the intersection and union
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    
    # Calculate the Dice coefficient
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return dice_score

# --- Hausdorff Distance ---
def hausdorff_distance(set1, set2):
    """Compute Hausdorff distance between two sets of points."""
    dists = tf.norm(tf.expand_dims(set1, 1) - tf.expand_dims(set2, 0), axis=-1)
    # Find the minimum distance for each point in set1 to the points in set2
    min_dists_set1_to_set2 = tf.reduce_min(dists, axis=1)
    # Find the minimum distance for each point in set2 to the points in set1
    min_dists_set2_to_set1 = tf.reduce_min(dists, axis=0)
    # The Hausdorff distance is the maximum of these minimum distances
    return tf.maximum(tf.reduce_max(min_dists_set1_to_set2), tf.reduce_max(min_dists_set2_to_set1))

def compute_3d_hausdorff_distance(pred_mask, gt_mask, num_channels=5):
    """Compute the Hausdorff distance for each channel in a multi-channel 3D U-Net output."""
    # Ensure masks are of the same shape
    assert pred_mask.shape == gt_mask.shape, "Prediction and ground truth masks must have the same shape."

    # Compute Hausdorff distance for each channel
    hausdorff_distances = []
    for i in range(num_channels):
        # Extract non-zero indices for the current channel in the prediction and ground truth masks
        pred_surface_points = tf.where(pred_mask != 0)
        gt_surface_points = tf.where(gt_mask != 0)

        # If there are no non-zero points in either mask, we can set the Hausdorff distance to zero
        if pred_surface_points.shape[0] == 0 or gt_surface_points.shape[0] == 0:
            hausdorff_distances.append(0.0)
        else:
            # Compute Hausdorff distance for this channel
            pred_surface_points = tf.cast(pred_surface_points, tf.float32)
            gt_surface_points = tf.cast(gt_surface_points, tf.float32)
            hd = hausdorff_distance(pred_surface_points, gt_surface_points)
            hausdorff_distances.append(hd)

    return sum(hausdorff_distances)/len(hausdorff_distances)