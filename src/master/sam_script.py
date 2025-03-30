import os
import glob
import json
import time
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
from pathlib import Path

# Import SAM2 modules
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.utils.misc import variant_to_config_mapping

# Import parameter tuning functions
from sam_param_tune import load_param_grid, tune_parameters

def binary_mask_to_polygon(mask, simplify=True, epsilon=1.0):
    """
    Convert a binary mask to a polygon
    
    Args:
        mask: Binary mask array
        simplify: Whether to simplify the polygon (reduce points)
        epsilon: Simplification parameter (higher = more simplification)
        
    Returns:
        List of polygon points in [x1, y1, x2, y2, ...] format
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour (typically there would be only one for a single mask)
    if not contours:
        return []
    
    contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour if requested
    if simplify:
        contour = cv2.approxPolyDP(contour, epsilon, True)
    
    # Convert contour to a flattened list [x1, y1, x2, y2, ...]
    polygon = contour.flatten().tolist()
    
    return polygon

def load_config(config_path):
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def save_runtime_stats(runtime_stats, output_path):
    """
    Save runtime statistics to a JSON file
    
    Args:
        runtime_stats: Dictionary of runtime statistics
        output_path: Path to save the statistics
    """
    with open(output_path, 'w') as f:
        json.dump(runtime_stats, f, indent=2)


def process_dataset(dataset_config, mask_generator, data_base_path):
    """
    Process a single dataset with SAM2
    
    Args:
        dataset_config: Configuration for the dataset
        mask_generator: SAM2 mask generator
        data_base_path: Base path for data
        
    Returns:
        Dictionary with runtime statistics
    """
    dataset_name = dataset_config['name']
    
    data_base = Path(data_base_path)
    dataset_dir = data_base / dataset_name
    
    # Handle paths that start with '/' by removing the leading slash
    img_path_str = dataset_config.get('img_path', '')
    if img_path_str.startswith('/'):
        img_path_str = img_path_str[1:]
    
    out_path_str = dataset_config.get('out_path', '')
    if out_path_str.startswith('/'):
        out_path_str = out_path_str[1:]
    
    img_path = dataset_dir / img_path_str
    out_path = dataset_dir / out_path_str
    
    print(f"Image path: {img_path}")
    print(f"Output path: {out_path}")
    
    # Ensure output directory exists
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {dataset_name} dataset...")

    img_files = glob.glob(str(img_path / "*.PNG"))
    
    if not img_files:
        print(f"No images found in {img_path}")
        return {}
    
    print(f"Found {len(img_files)} images.")
    
    # Initialize statistics
    stats = {
        "dataset": dataset_name,
        "num_images": len(img_files),
        "total_masks": 0,
        "inference_times": [],
        "masks_per_image": [],
        "start_time": time.time(),
    }
    
    # Process each image
    for img_file in tqdm(img_files, desc=f"Processing {dataset_name}"):
        # Get image name without extension
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        
        # Load image
        img = cv2.imread(img_file)
        if img is None:
            print(f"Error: Could not read image {img_file}")
            continue
        
        # Get image dimensions for normalization
        img_height, img_width = img.shape[:2]
        
        # Run SAM2 inference with timing
        start_time = time.time()
        masks = mask_generator.generate(img)
        inference_time = time.time() - start_time
        
        # Update statistics
        stats["inference_times"].append(inference_time)
        stats["masks_per_image"].append(len(masks))
        stats["total_masks"] += len(masks)
        
        # Save all masks for this image in a single file
        mask_output_path = out_path / f"{img_name}.txt"
        with open(mask_output_path, 'w') as f:
            for i, mask_data in enumerate(masks):
                # Extract the segmentation mask
                mask = mask_data["segmentation"]

                # Get polygon points
                poly_mask = binary_mask_to_polygon(mask)
                
                # Skip if no valid polygon was found
                if not poly_mask:
                    continue
                
                # Convert polygon to numpy array and reshape to (N, 2)
                poly_points = np.array(poly_mask).reshape(-1, 2)
                
                # Normalize coordinates (YOLO format)
                poly_points_normalized = poly_points.copy().astype(float)
                poly_points_normalized[:, 0] /= img_width  # Normalize X
                poly_points_normalized[:, 1] /= img_height  # Normalize Y
                
                # Class ID (using 0 for all masks)
                class_id = 0
                
                # Format in YOLO style: class_id x1 y1 x2 y2...
                yolo_string = f"{class_id}"
                for x, y in poly_points_normalized:
                    yolo_string += f" {x:.6f} {y:.6f}"
                
                # Write to file
                f.write(f"{yolo_string}\n")
    
    # Calculate and save summary statistics
    stats["end_time"] = time.time()
    stats["total_duration"] = stats["end_time"] - stats["start_time"]
    stats["avg_inference_time"] = np.mean(stats["inference_times"])
    stats["avg_masks_per_image"] = np.mean(stats["masks_per_image"])
    
    # Save statistics
    stats_path = Path(data_base_path) / f"{dataset_name}_runtime_stats.json"
    save_runtime_stats(stats, stats_path)
    
    print(f"Completed processing {dataset_name}:")
    print(f"  Total images: {stats['num_images']}")
    print(f"  Total masks: {stats['total_masks']}")
    print(f"  Avg. masks per image: {stats['avg_masks_per_image']:.2f}")
    print(f"  Avg. inference time: {stats['avg_inference_time']:.4f} seconds")
    print(f"  Total runtime: {stats['total_duration']:.2f} seconds")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="SAM2 Inference on one or more Dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory")
    parser.add_argument("--param-grid", type=str, help="Path to parameter grid JSON file for tuning")
    parser.add_argument("--tune", action="store_true", help="Perform parameter tuning")
    parser.add_argument("--tune-samples", type=int, default=25, help="Number of samples per dataset for tuning")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Save config for reference 
    data_base = Path(args.data)
    with open(data_base / "used_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load SAM2 model
    print(f"Loading SAM2 model ({config['model']['variant']})...")
    model_path = config['model']['checkpoint_path']
    
    try:
        model = build_sam2(
            variant_to_config_mapping[config['model']['variant']],
            model_path,
            device=device,
        )
        
        # Parameter tuning if requested
        if args.tune:
            if not args.param_grid:
                print("Error: --param-grid must be specified when --tune is used")
                return
            
            print("\nPerforming parameter tuning...")
            param_grid = load_param_grid(args.param_grid)
            
            # Tune parameters
            best_result = tune_parameters(
                config,
                param_grid,
                SAM2AutomaticMaskGenerator,
                model,
                args.data,
                num_samples=args.tune_samples
            )
            
            # Update config with best parameters
            for param_name, value in best_result["parameters"].items():
                parts = param_name.split('.')
                current = config
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            
            # Save updated config
            with open(data_base / "tuned_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Updated configuration saved to {data_base / 'tuned_config.json'}")
        
        # Initialize mask generator with (potentially updated) config
        mask_generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=config['inference']['points_per_side'],
            pred_iou_thresh=config['inference']['pred_iou_thresh'],
            stability_score_thresh=config['inference']['stability_score_thresh'],
            box_nms_thresh=config['inference']['box_nms_thresh'],
        )
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process each dataset
    all_stats = {}
    for dataset_config in config['datasets']:
        dataset_stats = process_dataset(
            dataset_config, 
            mask_generator, 
            args.data,
        )
        all_stats[dataset_config['name']] = dataset_stats
    
    # Save overall statistics
    overall_stats_path = os.path.join(args.data, "all_runtime_stats.json")
    with open(overall_stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("\nSAM2 inference completed for all datasets!")
    print(f"Results saved to: {args.data}")


if __name__ == "__main__":
    main()