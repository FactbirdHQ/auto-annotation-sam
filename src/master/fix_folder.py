import os
import json
import glob
import numpy as np
import cv2
from pathlib import Path
import shutil

def load_config(config_path):
    """Load the configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Successfully loaded config from: {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find config file at '{config_path}'.")

def consolidate_mask_files(sam_inference_dir, dataset_name):
    """
    Consolidate multiple mask .txt files per frame into a single file per frame.
    
    Args:
        sam_inference_dir: Directory containing SAM inference results
        dataset_name: Name of the dataset to process
    """
    print(f"Processing {dataset_name} dataset...")
    
    # Define paths
    dataset_dir = os.path.join(sam_inference_dir, dataset_name)
    masks_dir = os.path.join(dataset_dir, 'masks')
    consolidated_dir = os.path.join(dataset_dir, 'consolidated_masks')
    
    # Create consolidated masks directory if it doesn't exist
    os.makedirs(consolidated_dir, exist_ok=True)
    
    # Check if masks directory exists
    if not os.path.exists(masks_dir):
        print(f"Warning: Masks directory {masks_dir} not found")
        return consolidated_dir
        
    print(f"Looking for mask files in: {masks_dir}")
    mask_files = glob.glob(os.path.join(masks_dir, '*.txt'))
    print(f"Found {len(mask_files)} mask files")
    
    # Get unique frame IDs based on the naming format "frame_XXXXXX_mask_Y"
    frame_ids = set()
    for mask_file in mask_files:
        base_name = os.path.basename(mask_file)
        # Extract the frame ID part (e.g., "frame_000000" from "frame_000000_mask_0")
        parts = base_name.split('_mask_')
        if len(parts) >= 2:
            frame_id = parts[0]  # This gives us "frame_000000"
            frame_ids.add(frame_id)
    
    print(f"Found {len(frame_ids)} unique frame IDs")
    
    # Process each frame
    for frame_id in sorted(frame_ids):
        # Find all mask files for this frame
        frame_mask_files = glob.glob(os.path.join(masks_dir, f"{frame_id}_mask_*.txt"))
        
        if not frame_mask_files:
            print(f"No mask files found for frame ID: {frame_id}")
            continue
        
        # Consolidate masks
        all_masks = []
        for mask_file in frame_mask_files:
            with open(mask_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Only add non-empty lines
                        all_masks.append(line)
        
        # Write consolidated masks to a new file - use just the frame number for the output
        # Extract frame number (e.g., "000000" from "frame_000000")
        frame_number = frame_id.split('_')[-1]
        consolidated_file = os.path.join(consolidated_dir, f"frame_{frame_number}.txt")
        
        with open(consolidated_file, 'w') as f:
            f.write('\n'.join(all_masks))
        
        print(f"Consolidated {len(frame_mask_files)} mask files for {frame_id} -> {len(all_masks)} masks")
    
    return consolidated_dir

def convert_yolo_to_sam_format(gt_dir, dataset_config, output_dir):
    """
    Convert YOLO segment format to SAM format and apply cropping.
    
    Args:
        gt_dir: Directory containing ground truth YOLO segment format annotations
        dataset_config: Configuration for the dataset including crop coordinates
        output_dir: Directory to save converted ground truth masks
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get crop coordinates
    crop_coords = dataset_config['crop_coords']
    x1, x2, y1, y2 = crop_coords
    
    # Process each YOLO annotation file
    for yolo_file in glob.glob(os.path.join(gt_dir, '*.txt')):
        frame_id = os.path.basename(yolo_file).split('.')[0]
        
        # Read YOLO segment format (normalized coordinates)
        with open(yolo_file, 'r') as f:
            yolo_annotations = f.readlines()
        
        # Find corresponding image to get dimensions
        image_path = find_image_path(frame_id, dataset_config['data_path'])
        if not image_path:
            print(f"Could not find image for {frame_id}, skipping...")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_path}, skipping...")
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Convert YOLO segments to SAM format and apply cropping
        converted_masks = []
        for anno in yolo_annotations:
            parts = anno.strip().split()
            if len(parts) < 5:
                continue  # Skip invalid format
                
            class_id = int(parts[0])
            
            # Get polygon points (convert from normalized to absolute)
            polygon_points = []
            for i in range(1, len(parts), 2):
                if i+1 < len(parts):
                    x_norm = float(parts[i])
                    y_norm = float(parts[i+1])
                    
                    # Convert to absolute coordinates
                    x_abs = int(x_norm * img_width)
                    y_abs = int(y_norm * img_height)
                    
                    # Apply cropping
                    if x1 <= x_abs <= x2 and y1 <= y_abs <= y2:
                        # Adjust coordinates to cropped image space
                        x_cropped = x_abs - x1
                        y_cropped = y_abs - y1
                        polygon_points.append((x_cropped, y_cropped))
            
            if polygon_points:
                # Convert polygon points to SAM format (flattened list)
                sam_points = [str(coord) for point in polygon_points for coord in point]
                mask_line = f"{class_id} {' '.join(sam_points)}"
                converted_masks.append(mask_line)
        
        # Write converted masks to output file
        output_file = os.path.join(output_dir, f"{frame_id}.txt")
        with open(output_file, 'w') as f:
            f.write('\n'.join(converted_masks))
        
        print(f"Converted ground truth for {frame_id}")

def find_image_path(frame_id, data_path):
    """Find the corresponding image path for a given frame ID."""
    # Look for common image extensions
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(data_path, f"{frame_id}{ext}")
        if os.path.exists(path):
            return path
    return None

def process_all_datasets(config, sam_inference_dir, gt_base_dir):
    """Process all datasets defined in the config."""
    # Create a directory structure for the merged dataset
    merged_base_dir = os.path.join(sam_inference_dir, 'processed_data')
    os.makedirs(merged_base_dir, exist_ok=True)
    
    # Create a 'gt' folder in sam_inference directory to store all ground truth masks
    gt_folder = os.path.join(sam_inference_dir, 'gt')
    os.makedirs(gt_folder, exist_ok=True)
    
    for dataset in config['datasets']:
        dataset_name = dataset['name']
        print(f"\n=== Processing {dataset_name} dataset ===")
        
        # 1. Consolidate SAM masks
        consolidated_dir = consolidate_mask_files(sam_inference_dir, dataset_name)
        
        # 2. Convert ground truth masks and save to dataset-specific folder within 'gt'
        gt_dataset_dir = os.path.join(gt_base_dir, dataset_name, 'labels')
        gt_output_dir = os.path.join(gt_folder, dataset_name)  # New path in the gt folder
        
        if os.path.exists(gt_dataset_dir):
            convert_yolo_to_sam_format(gt_dataset_dir, dataset, gt_output_dir)
        else:
            print(f"Warning: Ground truth directory {gt_dataset_dir} not found")
        
        # 3. Create dataset directory in the merged structure
        dataset_dir = os.path.join(merged_base_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Copy images
        images_dir = os.path.join(sam_inference_dir, dataset_name, 'images')
        merged_images_dir = os.path.join(dataset_dir, 'images')
        if os.path.exists(images_dir):
            if not os.path.exists(merged_images_dir):
                shutil.copytree(images_dir, merged_images_dir)
            else:
                print(f"Images directory {merged_images_dir} already exists, skipping copy")
        
        # Copy consolidated masks
        merged_masks_dir = os.path.join(dataset_dir, 'sam_masks')
        if not os.path.exists(merged_masks_dir):
            shutil.copytree(consolidated_dir, merged_masks_dir)
        else:
            print(f"Masks directory {merged_masks_dir} already exists, skipping copy")
        
        # Copy ground truth masks to the processed_data structure too
        merged_gt_dir = os.path.join(dataset_dir, 'gt_masks')
        if os.path.exists(gt_output_dir):
            if not os.path.exists(merged_gt_dir):
                shutil.copytree(gt_output_dir, merged_gt_dir)
            else:
                print(f"GT masks directory {merged_gt_dir} already exists, skipping copy")

def main():
    # Import sys for potential early exit
    import sys
    
    # Load configuration from the configs folder
    config_path = os.path.join('./configs', 'sam2_config.json')
    # If running from src/master, adjust path
    if not os.path.exists(config_path):
        config_path = os.path.join('../../configs', 'sam2_config.json')
    
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please specify the correct path to sam2_config.json")
        sys.exit(1)
    
    # Define directories with respect to the parent of the repo
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '../..'))  # Assumes script is in src/master
    parent_dir = os.path.abspath(os.path.join(repo_root, '..'))  # One level up from repo
    
    # Data directories are one level up from the repo
    sam_inference_dir = os.path.join(parent_dir, 'data/sam_inference')
    gt_base_dir = os.path.join(parent_dir, 'data/processed')
    
    print(f"Using sam_inference_dir: {sam_inference_dir}")
    print(f"Using gt_base_dir: {gt_base_dir}")
    
    # Create directories if they don't exist
    os.makedirs(sam_inference_dir, exist_ok=True)
    
    # Process all datasets
    process_all_datasets(config, sam_inference_dir, gt_base_dir)
    
    print("\nProcessing complete. The organized data is available in the 'sam_inference/processed_data' directory.")
    print("Ground truth masks are also saved in the 'sam_inference/gt' directory.")

if __name__ == "__main__":
    main()