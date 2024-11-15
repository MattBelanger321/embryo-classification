import tensorflow as tf

def find_channel_dimension(tensor, num_channels):
    """
    Identifies the dimension in the tensor that matches the expected number of channels.
    """
    for dim, size in enumerate(tensor.shape):
        if size == num_channels:
            return dim
    raise ValueError(f"Could not find a dimension with size {num_channels}.")

def split_channels(tensor, channel_dim):
    """
    Splits the tensor along the specified channel dimension.
    """
    return tf.split(tensor, num_or_size_splits=tensor.shape[channel_dim], axis=channel_dim)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_flat = tf.reshape(y_true, (-1,))
    y_pred_flat = tf.reshape(y_pred, (-1,))
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return dice_score

def hausdorff_distance(set1, set2):
    dists = tf.norm(tf.expand_dims(set1, 1) - tf.expand_dims(set2, 0), axis=-1)
    min_dists_set1_to_set2 = tf.reduce_min(dists, axis=1)
    min_dists_set2_to_set1 = tf.reduce_min(dists, axis=0)
    return tf.maximum(tf.reduce_max(min_dists_set1_to_set2), tf.reduce_max(min_dists_set2_to_set1))

def compute_accuracy3d(gt_mask, pred_mask, num_channels=5):
    assert pred_mask.shape == gt_mask.shape, "Prediction and ground truth masks must have the same shape."

    # Find and split along the channel dimension
    channel_dim = find_channel_dimension(gt_mask, num_channels)
    gt_channels = split_channels(gt_mask, channel_dim)
    pred_channels = split_channels(pred_mask, channel_dim)

    hausdorff_distances = []
    dice_coefficients = []

    for gt_channel, pred_channel in zip(gt_channels, pred_channels):
        # Extract non-zero indices for Hausdorff distance
        gt_surface_points = tf.where(gt_channel != 0)
        pred_surface_points = tf.where(pred_channel != 0)

        if tf.shape(pred_surface_points)[0] == 0 or tf.shape(gt_surface_points)[0] == 0:
            hausdorff_distances.append(0.0)
        else:
            gt_surface_points = tf.cast(gt_surface_points, tf.float32)
            pred_surface_points = tf.cast(pred_surface_points, tf.float32)
            hd = hausdorff_distance(pred_surface_points, gt_surface_points)
            hausdorff_distances.append(hd)
        
        # Compute Dice coefficient for this channel
        dc = dice_coefficient(gt_channel, pred_channel)
        dice_coefficients.append(dc)

    # Weighted combination of the averaged Hausdorff distances and Dice coefficients
    accuracy = 0.6 * tf.reduce_mean(hausdorff_distances) + 0.4 * tf.reduce_mean(dice_coefficients)
    return accuracy
