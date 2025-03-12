import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_binary_masks(gt_masks, pred_masks, iou_threshold=0.5):
    """
    Calculate evaluation statistics between ground truth and predicted binary masks.
    
    Parameters:
    -----------
    gt_masks : list of np.ndarray
        List of ground truth binary masks (0 or 1)
    pred_masks : list of np.ndarray
        List of predicted binary masks (0 or 1)
    iou_threshold : float, optional
        IoU threshold to consider a mask as successfully detected (default: 0.5)
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics including both pixel-level and mask-level metrics
    """
    total_gt_masks = len(gt_masks)
    total_pred_masks = len(pred_masks)
    
    # Calculate IoU between all pairs of GT and predicted masks
    iou_matrix = np.zeros((total_gt_masks, total_pred_masks))
    
    for i, gt_mask in enumerate(gt_masks):
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        for j, pred_mask in enumerate(pred_masks):
            if gt_binary.shape != pred_mask.shape:
                raise ValueError(f"Shape mismatch: GT mask {i} shape {gt_binary.shape} vs Pred mask {j} shape {pred_mask.shape}")
            
            pred_binary = (pred_mask > 0).astype(np.uint8)
            
            # Calculate IoU
            intersection = np.logical_and(gt_binary, pred_binary).sum()
            union = np.logical_or(gt_binary, pred_binary).sum()
            iou = intersection / union if union > 0 else 0
            iou_matrix[i, j] = iou
    
    # For each GT mask, find the predicted mask with highest IoU
    best_ious = np.max(iou_matrix, axis=1) if total_pred_masks > 0 else np.zeros(total_gt_masks)
    
    # Count detected masks (those with IoU >= threshold)
    detected_masks = np.sum(best_ious >= iou_threshold)
    
    # Calculate TP, FP, FN for mask-level metrics
    true_positives = detected_masks
    false_negatives = total_gt_masks - detected_masks
    false_positives = total_pred_masks - detected_masks
    
    # Calculate mask-level metrics
    mask_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    mask_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    mask_f1 = 2 * (mask_precision * mask_recall) / (mask_precision + mask_recall) if (mask_precision + mask_recall) > 0 else 0
    
    # Calculate average IoU for detected masks only
    avg_iou_detected = np.mean(best_ious[best_ious >= iou_threshold]) if np.any(best_ious >= iou_threshold) else 0
    
    # Calculate average IoU across all GT masks (including undetected ones)
    avg_iou_all = np.mean(best_ious)
    
    # Create metrics dictionary
    metrics = {
        'mask_precision': mask_precision,
        'mask_recall': mask_recall,
        'mask_f1': mask_f1,
        'detected_masks': int(detected_masks),
        'total_gt_masks': total_gt_masks,
        'total_pred_masks': total_pred_masks,
        'avg_iou_detected': avg_iou_detected,
        'avg_iou_all': avg_iou_all,
        'best_ious': best_ious.tolist(),  # IoU of each GT mask with its best matching prediction
    }
    
    return metrics