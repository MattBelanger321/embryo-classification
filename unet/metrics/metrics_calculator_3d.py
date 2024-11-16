import tensorflow as tf
import metrics_calculator as mc
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(name="accuracy_metric_3d")
def accuracy_metric_3d(y_true, y_pred):
    # Initialize list to store individual metric results
    accuracy_scores = []

    # Iterate over the second dimension (the 5 slices) and apply accuracy_metric to each
    for i in range(5):
        # Extract the slice along the second dimension
        y_true_slice = y_true[:, i, :, :, :]
        y_pred_slice = y_pred[:, i, :, :, :]
        # Calculate accuracy for the slice and store it
        score = mc.combined_metric(y_true_slice, y_pred_slice)
        accuracy_scores.append(score)  # Assuming we're interested in "combined_accuracy"

    # Average the results across the 5 slices
    avg_accuracy = tf.reduce_mean(accuracy_scores)
    
    return avg_accuracy