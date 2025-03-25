
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


def setup_directories(base_path, dataset_name):
    """
    Create necessary directories for storing inference results
    
    Args:
        base_path: Base output path
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with created directory paths
    """
    output_path = Path(base_path) / dataset_name
    
    dirs = {
        "main": output_path,
        "masks": output_path / "masks",
        "images": output_path / "images",
        "metadata": output_path / "metadata",
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def crop_image(image, crop_coords):
    """
    Crop image using provided coordinates
    
    Args:
        image: Input image array
        crop_coords: List/tuple of [y_start, y_end, x_start, x_end]
        
    Returns:
        Cropped image
    """
    y_start, y_end, x_start, x_end = crop_coords
    return image[y_start:y_end, x_start:x_end]


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

def save_mask(mask, output_path, format="txt", img_width=None, img_height=None, class_id=0):
    """
    Save a binary mask to a file
    
    Args:
        mask: Binary mask array
        output_path: Path to save the mask
        format: Format to save ("txt", "png", or "yolo")
        img_width: Width of original image (for YOLO format)
        img_height: Height of original image (for YOLO format)
        class_id: Class ID (for YOLO format)
    """
    if format == "txt":
        np.savetxt(output_path, mask.astype(np.uint8), fmt='%d', delimiter=',')
    elif format == "png":
        cv2.imwrite(output_path, mask.astype(np.uint8) * 255)
    elif format == "yolo":
        # Convert mask to polygon points
        polygon = binary_mask_to_polygon(mask)
        
        if not polygon:  # Skip if no valid polygon
            return
            
        # Normalize coordinates to YOLO format [0-1]
        normalized_points = []
        for i in range(0, len(polygon), 2):
            if i+1 < len(polygon):
                x, y = polygon[i], polygon[i+1]
                normalized_points.append(x / img_width)
                normalized_points.append(y / img_height)
        
        # Write to file in YOLO format: class_id x1 y1 x2 y2 ...
        with open(output_path, 'w') as f:
            f.write(f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_points]))

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


def process_dataset(dataset_config, mask_generator, output_base_path, save_format="txt", yolo_dir=None):
    """
    Process a single dataset with SAM2
    
    Args:
        dataset_config: Configuration for the dataset
        mask_generator: SAM2 mask generator
        output_base_path: Base path for output
        save_format: Format to save masks ("txt", "png", or "yolo")
        yolo_dir: Directory to save YOLO format masks (if save_format is "yolo")
        
    Returns:
        Dictionary with runtime statistics
    """
    dataset_name = dataset_config['name']
    crop_coords = dataset_config['crop_coords']
    data_path = dataset_config.get('data_path', '')
    
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Setup directories
    dirs = setup_directories(output_base_path, dataset_name)
    
    # Find images
    images_path = Path(data_path) / "images" / "train"
    img_files = glob.glob(str(images_path / "*.PNG"))
    if not img_files:
        img_files = glob.glob(str(images_path / "*.png"))
    
    if not img_files:
        print(f"No images found in {images_path}")
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
        
        # Load and crop image
        img = cv2.imread(img_file)
        if img is None:
            print(f"Error: Could not read image {img_file}")
            continue
            
        cropped_img = crop_image(img, crop_coords)
        
        # Save cropped image
        cropped_img_path = dirs["images"] / f"{img_name}.png"
        cv2.imwrite(str(cropped_img_path), cropped_img)
        
        # Run SAM2 inference with timing
        start_time = time.time()
        masks = mask_generator.generate(cropped_img)
        inference_time = time.time() - start_time
        
        # Update statistics
        stats["inference_times"].append(inference_time)
        stats["masks_per_image"].append(len(masks))
        stats["total_masks"] += len(masks)
        
        # Save all masks for this image in a single file
        if save_format == "txt":
            mask_output_path = dirs["masks"] / f"{img_name}.txt"
            with open(mask_output_path, 'w') as f:
                for i, mask_data in enumerate(masks):
                    # Extract the segmentation mask
                    mask = mask_data["segmentation"]
                    # Convert to string representation
                    mask_str = ','.join(map(str, mask.astype(np.uint8).flatten()))
                    # Write to file with mask index
                    f.write(f"Mask {i}: {mask_str}\n")
    
    # Calculate and save summary statistics
    stats["end_time"] = time.time()
    stats["total_duration"] = stats["end_time"] - stats["start_time"]
    stats["avg_inference_time"] = np.mean(stats["inference_times"])
    stats["avg_masks_per_image"] = np.mean(stats["masks_per_image"])
    
    # Save statistics
    stats_path = dirs["main"] / "runtime_stats.json"
    save_runtime_stats(stats, stats_path)
    
    print(f"Completed processing {dataset_name}:")
    print(f"  Total images: {stats['num_images']}")
    print(f"  Total masks: {stats['total_masks']}")
    print(f"  Avg. masks per image: {stats['avg_masks_per_image']:.2f}")
    print(f"  Avg. inference time: {stats['avg_inference_time']:.4f} seconds")
    print(f"  Total runtime: {stats['total_duration']:.2f} seconds")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="SAM2 Inference on HPC Cluster")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--output", type=str, default="sam2_inference_results", help="Base output directory")
    parser.add_argument("--save_format", type=str, default="txt", choices=["txt", "png", "yolo"], 
                        help="Format to save masks")
    parser.add_argument("--yolo_dir", type=str, help="Directory to save YOLO format masks (optional)")
    args = parser.parse_args()
    

    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save config for reference 
    with open(os.path.join(args.output, "used_config.json"), 'w') as f:
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
        
        # Initialize mask generator
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
            args.output,
            save_format=args.save_format,
            yolo_dir=args.yolo_dir
        )
        all_stats[dataset_config['name']] = dataset_stats
    
    # Save overall statistics
    overall_stats_path = os.path.join(args.output, "all_runtime_stats.json")
    with open(overall_stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("\nSAM2 inference completed for all datasets!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()