import torch

def calculate_iou_box(pred_boxes, true_boxes):
    """
    Calculate the Intersection over Union (IoU) for bounding boxes.
    
    Args:
        pred_boxes: Predicted boxes (N, 4), format [x_min, y_min, x_max, y_max]
        true_boxes: Ground truth boxes (N, 4), format [x_min, y_min, x_max, y_max]
        
    Returns:
        float: The average IoU value for the batch
    """
    # Extract coordinates (ensure validity)
    x1_pred = pred_boxes[:, 0]
    y1_pred = pred_boxes[:, 1]
    x2_pred = pred_boxes[:, 2]
    y2_pred = pred_boxes[:, 3]
    
    x1_true = true_boxes[:, 0]
    y1_true = true_boxes[:, 1]
    x2_true = true_boxes[:, 2]
    y2_true = true_boxes[:, 3]
    
    # Calculate intersection coordinates
    # Intersection top-left = max(pred_x1, true_x1)
    x1_inter = torch.max(x1_pred, x1_true)
    y1_inter = torch.max(y1_pred, y1_true)
    # Intersection bottom-right = min(pred_x2, true_x2)
    x2_inter = torch.min(x2_pred, x2_true)
    y2_inter = torch.min(y2_pred, y2_true)
    
    # Calculate intersection area (clamp(min=0) ensures area is 0 when no overlap, rather than negative)
    inter_area = (x2_inter - x1_inter).clamp(min=0) * (y2_inter - y1_inter).clamp(min=0)
    
    # Calculate individual areas
    pred_area = (x2_pred - x1_pred).clamp(min=0) * (y2_pred - y1_pred).clamp(min=0)
    true_area = (x2_true - x1_true).clamp(min=0) * (y2_true - y1_true).clamp(min=0)
    
    # Calculate union area = Area_A + Area_B - Intersection
    union_area = pred_area + true_area - inter_area
    
    # Calculate IoU = Intersection / Union
    # Add epsilon (1e-6) to prevent division by zero
    iou = inter_area / (union_area + 1e-6)
    return iou.mean().item()

def calculate_detection_accuracy_at_iou(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Calculate Detection Accuracy where a detection is considered correct 
    if its IoU with the ground truth is >= threshold.
    
    Args:
        pred_boxes: Predicted boxes (N, 4), format [x_min, y_min, x_max, y_max]
        true_boxes: Ground truth boxes (N, 4), format [x_min, y_min, x_max, y_max]
        iou_threshold: Minimum IoU required to consider a detection successful (default: 0.5)
        
    Returns:
        float: The proportion of boxes in the batch that meet the IoU threshold
    """
    x1_pred = pred_boxes[:, 0]
    y1_pred = pred_boxes[:, 1]
    x2_pred = pred_boxes[:, 2]
    y2_pred = pred_boxes[:, 3]
    
    x1_true = true_boxes[:, 0]
    y1_true = true_boxes[:, 1]
    x2_true = true_boxes[:, 2]
    y2_true = true_boxes[:, 3]
    
    x1_inter = torch.max(x1_pred, x1_true)
    y1_inter = torch.max(y1_pred, y1_true)
    x2_inter = torch.min(x2_pred, x2_true)
    y2_inter = torch.min(y2_pred, y2_true)
    
    inter_area = (x2_inter - x1_inter).clamp(min=0) * (y2_inter - y1_inter).clamp(min=0)
    
    pred_area = (x2_pred - x1_pred).clamp(min=0) * (y2_pred - y1_pred).clamp(min=0)
    true_area = (x2_true - x1_true).clamp(min=0) * (y2_true - y1_true).clamp(min=0)
    
    union_area = pred_area + true_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    # Calculate accuracy: ratio of samples where IoU >= threshold
    correct_detections = (iou >= iou_threshold).float()
    return correct_detections.mean().item()

def calculate_iou_mask(pred_masks, true_masks, threshold=0.5):
    """
    Calculate the Intersection over Union (IoU) for segmentation masks (also known as Jaccard Index).
    
    Args:
        pred_masks: Predicted mask logits (N, 1, H, W)
        true_masks: Ground truth binary masks (N, 1, H, W)
        threshold: Binarization threshold, default 0.5
        
    Returns:
        float: The average IoU value for the batch
    """
    # Convert logits to probabilities using Sigmoid, and binarize based on threshold
    pred_bin = (torch.sigmoid(pred_masks) > threshold).float()
    
    # Calculate Intersection: number of pixels where both pred and true are 1
    # Sum over dimensions (1, 2, 3) which are (C, H, W)
    intersection = (pred_bin * true_masks).sum(dim=(1, 2, 3))
    
    # Calculate Union: number of pixels where either pred or true is 1
    # Union = A + B - Intersection
    union = (pred_bin + true_masks).sum(dim=(1, 2, 3)) - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)
    return iou.mean().item()

def calculate_dice_mask(pred_masks, true_masks, threshold=0.5):
    """
    Calculate the Dice Coefficient for segmentation masks.
    
    Args:
        pred_masks: Predicted mask logits (N, 1, H, W)
        true_masks: Ground truth binary masks (N, 1, H, W)
        threshold: Binarization threshold, default 0.5
        
    Returns:
        float: The average Dice Coefficient value for the batch
    """
    # Convert logits to probabilities using Sigmoid, and binarize based on threshold
    pred_bin = (torch.sigmoid(pred_masks) > threshold).float()
    
    # Calculate Intersection: number of pixels where both pred and true are 1
    # Sum over dimensions (1, 2, 3) which are (C, H, W)
    intersection = (pred_bin * true_masks).sum(dim=(1, 2, 3))
    
    # Calculate sum of elements in both masks
    pred_sum = pred_bin.sum(dim=(1, 2, 3))
    true_sum = true_masks.sum(dim=(1, 2, 3))
    
    # Calculate Dice
    # Dice = 2 * Intersection / (Sum of pred + Sum of true)
    dice = (2.0 * intersection) / (pred_sum + true_sum + 1e-6)
    return dice.mean().item()
