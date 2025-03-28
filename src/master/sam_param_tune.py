import os
import json
import random
import itertools
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def load_param_grid(param_grid_path):
    """
    Load parameter grid from JSON file
    
    Args:
        param_grid_path: Path to the parameter grid file
        
    Returns:
        Parameter grid dictionary
    """
    with open(param_grid_path, 'r') as f:
        return json.load(f)

def sample_dataset_images(dataset_dir, img_path, num_samples=25):
    """
    Sample a subset of images from a dataset
    
    Args:
        dataset_dir: Base directory for the dataset
        img_path: Relative path to images
        num_samples: Number of images to sample
        
    Returns:
        List of sampled image paths
    """
    # Handle paths properly - img_path might be a string starting with '/'
    if isinstance(img_path, str) and img_path.startswith('/'):
        img_path = img_path[1:]  # Remove leading slash
    
    full_img_path = dataset_dir / img_path
    print(f"Looking for images in: {full_img_path}")
    all_images = list(full_img_path.glob("*.PNG"))
    
    if len(all_images) <= num_samples:
        return all_images
    
    return random.sample(all_images, num_samples)

def load_ground_truth(gt_path, img_name, img_width=640, img_height=480):
    """
    Load ground truth masks for an image
    
    Args:
        gt_path: Path to ground truth masks
        img_name: Image name (without extension)
        img_width: Width of the image for normalizing coordinates
        img_height: Height of the image for normalizing coordinates
        
    Returns:
        List of ground truth masks (binary numpy arrays)
    """
    # Handle paths properly
    if isinstance(gt_path, str) and gt_path.startswith('/'):
        gt_path = gt_path[1:]  # Remove leading slash
    
    gt_file = Path(gt_path) / f"{img_name}.txt"
    
    if not gt_file.exists():
        return []
    
    return load_yolo_masks_as_binary_list(str(gt_file), img_width, img_height)

def load_yolo_masks_as_binary_list(txt_path, img_width, img_height):
    """
    Load YOLO segmentation masks from a .txt file and convert each line to a separate binary mask.
    
    Args:
        txt_path (str): Path to the YOLO segmentation .txt file
        img_width (int): Width of the original image
        img_height (int): Height of the original image
        
    Returns:
        list: List of binary masks (np.ndarray), one for each line in the txt file
    """
    try:
        # Read the YOLO segmentation file
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        # List to store individual masks
        masks = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Create an empty mask for this line
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # Split the line into values
            values = line.strip().split()
            
            # In YOLO format, the first value is the class ID, which we'll ignore
            # The rest are x,y coordinates in normalized form (0-1)
            polygon_points = values[1:]
            
            # Convert to numpy array and reshape to pairs of x,y coordinates
            polygon_points = np.array(polygon_points, dtype=float).reshape(-1, 2)
            
            # Convert normalized coordinates (0-1) to pixel coordinates
            polygon_points[:, 0] *= img_width  # x coordinates
            polygon_points[:, 1] *= img_height  # y coordinates
            
            # Convert to integer points for drawing
            polygon_points = polygon_points.astype(np.int32)
            
            # Draw the filled polygon onto the mask
            cv2.fillPoly(mask, [polygon_points], 1)
            
            # Add this mask to our list
            masks.append(mask)

        # Add this debugging
        for i, mask in enumerate(masks):
            if 0 in mask.shape:
                print(f"Warning: Mask {i} has invalid dimensions: {mask.shape}")
                # Fix masks with zero dimensions by creating small valid mask
                masks[i] = np.zeros((img_height, img_width), dtype=np.uint8)
        
        return masks
    except Exception as e:
        print(f"Error loading YOLO masks from {txt_path}: {e}")
        return []

def evaluate_masks(pred_masks, gt_masks, iou_threshold=0.5):
    """
    Evaluate predicted masks against ground truth
    
    Args:
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks
        iou_threshold: IoU threshold for considering a detection
        
    Returns:
        Dictionary with recall, precision, and F1 score
    """
    if not gt_masks:
        return {"recall": 0, "precision": 0, "f1": 0}
    
    if not pred_masks:
        return {"recall": 0, "precision": 0, "f1": 0}
    
    # Match each ground truth mask to the best predicted mask
    matches = 0
    
    for gt_mask in gt_masks:
        best_iou = 0
        
        for pred_mask in pred_masks:
            # Calculate IoU
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = intersection / union if union > 0 else 0
            
            best_iou = max(best_iou, iou)
        
        if best_iou >= iou_threshold:
            matches += 1
    
    recall = matches / len(gt_masks)
    precision = matches / len(pred_masks) if pred_masks else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1
    }

def tune_parameters(config, param_grid, mask_generator_class, model, data_base_path, num_samples=25):
    """
    Tune SAM parameters using grid search
    
    Args:
        config: Base configuration
        param_grid: Grid of parameters to search
        mask_generator_class: SAM mask generator class
        model: SAM model
        data_base_path: Base path for data
        num_samples: Number of images to sample per dataset
        
    Returns:
        Best parameters and evaluation results
    """
    # Create parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    results = []
    
    # Convert base_path to Path object
    data_base = Path(data_base_path)
    
    # Sample images from datasets
    sampled_images = {}
    for dataset_config in config['datasets']:
        dataset_name = dataset_config['name']
        dataset_dir = data_base / dataset_name
        
        # Handle paths that start with '/' by removing the leading slash
        img_path = dataset_config['img_path']
        if isinstance(img_path, str) and img_path.startswith('/'):
            img_path = img_path[1:]
            
        print(f"Dataset: {dataset_name}, Directory: {dataset_dir}, Image path: {img_path}")
        sampled_images[dataset_name] = sample_dataset_images(dataset_dir, img_path, num_samples)
    
    print(f"Sampled {sum(len(imgs) for imgs in sampled_images.values())} images for parameter tuning")
    
    # Evaluate each parameter combination
    for i, param_combination in enumerate(tqdm(param_combinations, desc="Parameter combinations")):
        # Update configuration
        tuning_config = config.copy()
        for j, param_name in enumerate(param_names):
            # Split the parameter name by dots
            parts = param_name.split('.')
            
            # Navigate to the nested dict
            current = tuning_config
            for part in parts[:-1]:
                current = current[part]
            
            # Set the value
            current[parts[-1]] = param_combination[j]
        
        # Initialize mask generator with the current parameters
        mask_generator = mask_generator_class(
            model,
            points_per_side=tuning_config['inference']['points_per_side'],
            pred_iou_thresh=tuning_config['inference']['pred_iou_thresh'],
            stability_score_thresh=tuning_config['inference']['stability_score_thresh'],
            box_nms_thresh=tuning_config['inference']['box_nms_thresh'],
        )
        
        # Evaluate on sampled images
        dataset_results = {}
        for dataset_config in config['datasets']:
            dataset_name = dataset_config['name']
            dataset_dir = data_base / dataset_name
            # Handle paths that start with '/' by removing the leading slash
            gt_path_str = dataset_config['gt_path']
            if isinstance(gt_path_str, str) and gt_path_str.startswith('/'):
                gt_path_str = gt_path_str[1:]
                
            gt_path = dataset_dir / gt_path_str
            print(f"Ground truth path: {gt_path}")
            
            dataset_metrics = {
                "recall": [],
                "precision": [],
                "f1": [],
                "masks_per_image": []
            }
            
            for img_file in tqdm(sampled_images[dataset_name], desc=f"Evaluating {dataset_name}", leave=False):
                try:
                    # Get image name without extension
                    img_name = os.path.splitext(os.path.basename(img_file))[0]
                    
                    # Load image
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    # Run inference
                    masks = mask_generator.generate(img)
                    pred_masks = [mask_data["segmentation"] for mask_data in masks]
                    
                    # Load ground truth
                    # First, get image dimensions from the loaded image
                    img_height, img_width = img.shape[:2]
                    gt_masks = load_ground_truth(gt_path, img_name, img_width, img_height)
                    
                    # Skip evaluation if no ground truth masks
                    if not gt_masks:
                        continue
                        
                    # Evaluate
                    metrics = evaluate_masks(pred_masks, gt_masks)
                    dataset_metrics["recall"].append(metrics["recall"])
                    dataset_metrics["precision"].append(metrics["precision"])
                    dataset_metrics["f1"].append(metrics["f1"])
                    dataset_metrics["masks_per_image"].append(len(masks))
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue
            
            # Average metrics for this dataset
            for metric in ["recall", "precision", "f1", "masks_per_image"]:
                if dataset_metrics[metric]:
                    dataset_metrics[metric] = np.mean(dataset_metrics[metric])
                else:
                    dataset_metrics[metric] = 0
            
            dataset_results[dataset_name] = dataset_metrics
        
        # Calculate overall metrics
        overall_metrics = {
            "recall": np.mean([res["recall"] for res in dataset_results.values()]),
            "precision": np.mean([res["precision"] for res in dataset_results.values()]),
            "f1": np.mean([res["f1"] for res in dataset_results.values()]),
            "masks_per_image": np.mean([res["masks_per_image"] for res in dataset_results.values()])
        }
        
        # Store result
        results.append({
            "parameters": {param_name: value for param_name, value in zip(param_names, param_combination)},
            "dataset_results": dataset_results,
            "overall_metrics": overall_metrics
        })
        
        print(f"Combination {i+1}/{len(param_combinations)}: Recall = {overall_metrics['recall']:.4f}, "
              f"Precision = {overall_metrics['precision']:.4f}, "
              f"F1 = {overall_metrics['f1']:.4f}, "
              f"Masks per image = {overall_metrics['masks_per_image']:.2f}")
    
    # Sort results by recall (highest first)
    results.sort(key=lambda x: x["overall_metrics"]["recall"], reverse=True)
    
    # Return best parameters (highest recall)
    best_result = results[0]
    
    # Save all results
    results_path = os.path.join(data_base_path, "parameter_tuning_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nParameter tuning completed. Results saved to {results_path}")
    print(f"Best parameters (highest recall):")
    for param, value in best_result["parameters"].items():
        print(f"  {param}: {value}")
    print(f"Overall metrics:")
    for metric, value in best_result["overall_metrics"].items():
        print(f"  {metric}: {value}")
    
    return best_result