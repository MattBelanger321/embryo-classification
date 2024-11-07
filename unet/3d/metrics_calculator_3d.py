import tensorflow as tf
import numpy as np

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