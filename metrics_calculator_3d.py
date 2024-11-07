import torch
import torch.nn.functional as F

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Compute the Dice coefficient for 3D data.

    Args:
        pred (torch.Tensor): The predicted segmentation (binary) tensor, shape (N, C, D, H, W)
        target (torch.Tensor): The ground truth segmentation (binary) tensor, shape (N, C, D, H, W)
        smooth (float): Small constant to avoid division by zero.
    
    Returns:
        float: The Dice coefficient score.
    """
    # Flatten the tensors to treat all elements as a 1D array
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Compute Dice score with a small constant to avoid division by zero
    dice_score = (2. * intersection + smooth) / (union + smooth)
    
    return dice_score


