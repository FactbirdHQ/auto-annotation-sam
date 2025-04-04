import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def evaluate_binary_masks(gt_masks, pred_masks, iou_threshold=0.5):
    """
    Optimized version of evaluate_binary_masks for better performance.
    
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
        Dictionary with evaluation metrics
    """
    total_gt_masks = len(gt_masks)
    total_pred_masks = len(pred_masks)
    
    # Early return for empty cases
    if total_gt_masks == 0 or total_pred_masks == 0:
        return {
            'mask_precision': 0 if total_pred_masks > 0 else 1.0,  # Precision is 1.0 if no predictions (no false positives)
            'mask_recall': 0 if total_gt_masks > 0 else 1.0,       # Recall is 1.0 if no ground truth (no false negatives)
            'mask_f1': 0,
            'detected_masks': 0,
            'total_gt_masks': total_gt_masks,
            'total_pred_masks': total_pred_masks,
            'avg_iou_detected': 0,
            'avg_iou_all': 0,
            'best_ious': []
        }
    
    # Pre-compute binary masks once
    gt_binary = [(mask > 0).astype(np.uint8) for mask in gt_masks]
    pred_binary = [(mask > 0).astype(np.uint8) for mask in pred_masks]
    
    # Create IoU matrix more efficiently
    iou_matrix = np.zeros((total_gt_masks, total_pred_masks))
    
    # Optimize IoU calculation with vectorization where possible
    for i, gt in enumerate(gt_binary):
        gt_sum = gt.sum()  # Pre-compute sum of GT mask
        if gt_sum == 0:
            continue  # Skip empty GT masks
            
        for j, pred in enumerate(pred_binary):
            # Use fast bitwise operations
            intersection = np.logical_and(gt, pred).sum()
            if intersection == 0:
                continue  # Skip if no intersection (IoU will be 0)
                
            # Union = sum of both masks - intersection
            pred_sum = pred.sum()  # Pre-compute sum of pred mask
            union = gt_sum + pred_sum - intersection
            
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Process the IoU matrix with numpy operations
    best_ious = np.max(iou_matrix, axis=1)
    detected_mask_indices = best_ious >= iou_threshold
    detected_masks = np.sum(detected_mask_indices)
    
    # Calculate precision, recall, F1
    true_positives = detected_masks
    false_negatives = total_gt_masks - detected_masks
    false_positives = total_pred_masks - detected_masks
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate average IoU statistics
    avg_iou_detected = np.mean(best_ious[detected_mask_indices]) if np.any(detected_mask_indices) else 0
    avg_iou_all = np.mean(best_ious)
    
    # Create metrics dictionary
    metrics = {
        'mask_precision': precision,
        'mask_recall': recall,
        'mask_f1': f1,
        'detected_masks': int(detected_masks),
        'total_gt_masks': total_gt_masks,
        'total_pred_masks': total_pred_masks,
        'avg_iou_detected': avg_iou_detected,
        'avg_iou_all': avg_iou_all,
        'best_ious': best_ious.tolist()  # Convert to list for JSON serialization
    }
    
    return metrics

def plot_precision_recall_curve(test_masks_gt, all_masks, sorted_similarities):
    """
    Plot a precision-recall curve by evaluating masks at different similarity thresholds.
    
    Parameters:
    -----------
    test_masks_gt : list
        List of ground truth mask arrays
    all_masks : list
        List of all predicted mask arrays (should be in the same order as sorted_similarities)
    sorted_similarities : list
        Sorted list of similarity scores for each predicted mask
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the precision-recall curve
    """
    # Get unique similarity values to use as thresholds
    # Add a value slightly higher than max to ensure we include the point (0,1)
    unique_thresholds = np.unique(sorted_similarities)
    thresholds = np.append(unique_thresholds, max(sorted_similarities) + 0.01)
    thresholds = np.sort(thresholds)
    
    # Initialize lists to store precision and recall values
    precisions = []
    recalls = []
    
    # Evaluate masks at each threshold
    for threshold in thresholds:
        # Get masks with similarity above threshold
        selected_masks = [all_masks[i] for i in range(len(sorted_similarities)) 
                         if sorted_similarities[i] >= threshold]
        
        # If no masks selected, precision is 0
        if len(selected_masks) == 0:
            precisions.append(0)
            recalls.append(0)
            continue
        
        # Evaluate selected masks
        results = evaluate_binary_masks(test_masks_gt, selected_masks)
        
        # Store precision and recall
        precisions.append(results['mask_precision'])
        recalls.append(results['mask_recall'])
    
    # Create precision-recall curve
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Plot the curve
    ax.plot(recalls, precisions, 'b-', lw=2)
    
    # Add points for each threshold
    ax.scatter(recalls, precisions, c='blue', s=30, zorder=10)
    
    # Calculate AUC (Area Under Curve)
    pr_auc = auc(recalls, precisions)
    
    # Add labels and title
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(f'Precision-Recall Curve (AUC: {pr_auc:.4f})', fontsize=16)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    # Add a few threshold annotations (to avoid cluttering, only show a few)
    num_annotations = min(5, len(thresholds))
    annotation_indices = np.linspace(0, len(thresholds)-1, num_annotations, dtype=int)
    
    for idx in annotation_indices:
        ax.annotate(f'{thresholds[idx]:.2f}', 
                   (recalls[idx], precisions[idx]),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=10)
    
    # Add F1 score curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1, num=100)  # Use 100 points for better control
        
        # Handle division by zero: when x = f_score/2, denominator is zero
        # For each x, calculate y safely
        y = np.zeros_like(x)
        for i, x_val in enumerate(x):
            # Skip the point where denominator would be zero
            if abs(2 * x_val - f_score) < 1e-10:
                continue
            y[i] = f_score * x_val / (2 * x_val - f_score)
        
        # Only keep valid y values (not inf or nan) and y <= 1.05
        mask = np.logical_and(np.isfinite(y), y <= 1.05)
        
        if np.any(mask):  # Only plot if we have valid points
            ax.plot(x[mask], y[mask], color='gray', alpha=0.3, linestyle='--')
            
            # Find appropriate position for the annotation near the end of the curve
            valid_indices = np.where(mask)[0]
            if len(valid_indices) > 10:
                idx = valid_indices[-10]  # 10th point from the end of the visible curve
                ax.annotate(f'F1={f_score:0.1f}', xy=(x[idx], y[idx]), alpha=0.5)
    
    plt.tight_layout()
    
    return fig

def plot_threshold_vs_metrics(test_masks_gt, all_masks, sorted_similarities):
    """
    Plot precision, recall, and F1 score against similarity thresholds.
    
    Parameters:
    -----------
    test_masks_gt : list
        List of ground truth mask arrays
    all_masks : list
        List of all predicted mask arrays (should be in the same order as sorted_similarities)
    sorted_similarities : list
        Sorted list of similarity scores for each predicted mask
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the threshold vs metrics curves
    """
    # Get unique similarity values to use as thresholds
    unique_thresholds = np.unique(sorted_similarities)
    thresholds = np.sort(unique_thresholds)
    
    # Initialize lists to store precision, recall and F1 values
    precisions = []
    recalls = []
    f1_scores = []
    avg_ious = []
    
    # Evaluate masks at each threshold
    for threshold in thresholds:
        # Get masks with similarity above threshold
        selected_masks = [all_masks[i] for i in range(len(sorted_similarities)) 
                         if sorted_similarities[i] >= threshold]
        
        # If no masks selected, all metrics are 0
        if len(selected_masks) == 0:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
            avg_ious.append(0)
            continue
        
        # Evaluate selected masks
        results = evaluate_binary_masks(test_masks_gt, selected_masks)
        
        # Store metrics
        precision = results['mask_precision']
        recall = results['mask_recall']
        
        # Calculate F1 score - safely handle division by zero
        if (precision + recall) < 1e-10:  # Avoid division by zero with a small epsilon
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        avg_ious.append(results['avg_iou_detected'])
    
    # Create figure with multiple plots
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot metrics against thresholds
    ax.plot(thresholds, precisions, 'r-', label='Precision', lw=2)
    ax.plot(thresholds, recalls, 'b-', label='Recall', lw=2)
    ax.plot(thresholds, f1_scores, 'g-', label='F1 Score', lw=2)
    ax.plot(thresholds, avg_ious, 'm-', label='Avg IoU', lw=2)
    
    # Find best F1 threshold
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    # Mark the best F1 score point
    ax.scatter([best_threshold], [best_f1], c='green', s=100, zorder=10, 
               label=f'Best F1: {best_f1:.4f} at threshold {best_threshold:.4f}')
    
    # Add labels and title
    ax.set_xlabel('Similarity Threshold', fontsize=14)
    ax.set_ylabel('Metric Value', fontsize=14)
    ax.set_title('Precision, Recall, F1 Score, and IoU vs Similarity Threshold', fontsize=16)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=12)
    
    # Set axis limits
    ax.set_xlim([min(thresholds)-0.05, max(thresholds)+0.05])
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    
    return fig