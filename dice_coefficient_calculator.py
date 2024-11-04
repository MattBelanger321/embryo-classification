from tensorflow.keras import backend as K

# Compute dice coefficient
def compute_dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice_coefficient = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice_coefficient