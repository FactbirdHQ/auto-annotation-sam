import os
import glob
import numpy as np
import cv2
from PIL import Image
import torch
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import sys


# Import from the provided files
from src.master.model import CLIPEmbedding, KNNClassifier
from src.master.data import load_yolo_masks_as_binary_list, process_segmentation_masks, sort_masks
from src.master.evaluate import batch_calculate_iou_yolo

class AutoAnnotator:
    def __init__(self, input_dir, output_dir=None, sam_model=None):
        """
        Initialize the auto-annotator for YOLO segmentation data
        
        Args:
            input_dir: Path to the input directory containing YOLO segmentation data
            output_dir: Path to save the annotations (defaults to same as input_dir)
            sam_model: SAM model for generating masks
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir
        self.sam_model = sam_model
        
        # Track images with and without ground truth
        self.images_with_gt = []
        self.images_without_gt = []
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def find_data(self):
        """
        Find all images and label files in the dataset structure:
        - images/Train, images/Validation, images/Test
        - labels/Train, labels/Validation, labels/Test
        
        Returns:
            Dictionary mapping image paths to label paths (or None if no label exists)
        """
        print("Finding data in input directory...")
        
        # Check if we have the main structure with images and labels directories
        images_dir = self.input_dir / "images"
        labels_dir = self.input_dir / "labels"
        
        if not images_dir.exists():
            print(f"Warning: Could not find 'images' directory at {images_dir}")
            return {}
        
        # Look for Train, Validation, Test subfolders in images directory
        subfolders = ["Train", "Validation", "Test"]
        available_subfolders = [subfolder for subfolder in subfolders if (images_dir / subfolder).exists()]
        
        if not available_subfolders:
            print(f"Warning: Could not find any of {subfolders} subfolders in images directory")
            return {}
        
        # Collect all image and label files
        image_files = []
        label_files = []
        
        for subfolder in available_subfolders:
            # Find image files in this subfolder
            images_subfolder_path = images_dir / subfolder
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_files.extend(glob.glob(str(images_subfolder_path / ext)))
            
            # Find label files in this subfolder (if labels directory exists)
            if labels_dir.exists():
                labels_subfolder_path = labels_dir / subfolder
                if labels_subfolder_path.exists():
                    label_files.extend(glob.glob(str(labels_subfolder_path / "*.txt")))
        
        # Map image files to label files
        data_map = {}
        for image_path in image_files:
            image_path = Path(image_path)
            image_name = image_path.name
            base_name = image_path.stem
            
            # Determine which subfolder this image is in
            subfolder = image_path.parent.name  # Should be "Train", "Validation", or "Test"
            
            # Look for corresponding label file in the matching labels subfolder
            expected_label_path = labels_dir / subfolder / f"{base_name}.txt"
            label_path = str(expected_label_path) if expected_label_path.exists() else None
            
            data_map[str(image_path)] = label_path
            
            # Track which images have ground truth
            if label_path:
                self.images_with_gt.append(str(image_path))
            else:
                self.images_without_gt.append(str(image_path))
        
        print(f"Found {len(data_map)} images - {len(self.images_with_gt)} with labels, {len(self.images_without_gt)} without labels")
        return data_map

    def generate_sam_masks(self, image):
        """
        Generate masks using SAM model
        
        Args:
            image: Input image array
            
        Returns:
            List of binary masks
        """
        
        # Run SAM inference 
        masks_info = self.sam_model.generate(image)
        
        # Convert the SAM2 output to binary masks
        binary_masks = []
        
        # Check if masks_info is a dictionary or list of dictionaries
        if isinstance(masks_info, dict):
            # Handle single mask case
            if 'segmentation' in masks_info:
                binary_masks.append(masks_info['segmentation'].astype(np.uint8))
        elif isinstance(masks_info, list):
            # Handle list of masks case
            for mask_info in masks_info:
                if isinstance(mask_info, dict) and 'segmentation' in mask_info:
                    binary_masks.append(mask_info['segmentation'].astype(np.uint8))
        
        return binary_masks

    def train_model(self):
        """
        Train a model using existing images with ground truth labels
        
        Returns:
            Trained classifier model
        """
        print("Training the model on labeled images...")
        
        # Prepare training data
        training_images = []
        training_masks = []
        training_labels = []
        
        for image_path in tqdm(self.images_with_gt, desc="Preparing training data"):
            try:
                # Convert string path to Path object
                image_path_obj = Path(image_path)
                
                # Load image
                image = np.array(Image.open(image_path).convert("RGB"))
                
                # Extract subfolder name (Train, Validation, Test)
                subfolder = image_path_obj.parent.name
                
                # Construct the correct path to the label file
                label_path = self.input_dir / "labels" / subfolder / f"{image_path_obj.stem}.txt"
                
                if not label_path.exists():
                    print(f"Warning: Could not find label file at {label_path}")
                    continue
                
                gt_masks = load_yolo_masks_as_binary_list(str(label_path), image.shape[1], image.shape[0])
                if not gt_masks:
                    continue
                    
                # Generate SAM masks for this image
                sam_masks = self.generate_sam_masks(image)
                
                # Process masks to get training data
                combined_masks, combined_labels = process_segmentation_masks(gt_masks, sam_masks)
                
                # Add to training data
                if combined_masks and combined_labels:
                    training_images.append(image)
                    training_masks.append(combined_masks)
                    training_labels.append(combined_labels)
            except Exception as e:
                print(f"Error processing training image {image_path}: {e}")
                continue
        
        if not training_images:
            raise ValueError("No valid training data could be prepared. Check your labeled images and paths.")
            
        print(f"Prepared {len(training_images)} training samples")
        
        # Create and train the model
        embedding = CLIPEmbedding(config={'clip_model': 'ViT-B/32'})
        classifier = KNNClassifier(config={}, embedding=embedding)
        
        # Train the classifier
        start_time = time.time()
        classifier.fit(training_images, training_masks, training_labels)
        training_time = time.time() - start_time
        
        print(f"Model trained in {training_time:.2f} seconds")
        return classifier

    def save_masks_as_yolo(self, predicted_masks, img_width, img_height, output_path):
        """
        Save binary masks in YOLO segmentation format
        
        Args:
            predicted_masks: List of binary masks
            img_width: Image width
            img_height: Image height
            output_path: Path to save the YOLO format masks
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for mask in predicted_masks:
                # Find contours in the mask
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours and len(contours) > 0:
                    # Use the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Simplify contfour
                    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    # Skip if too few points
                    if len(approx) < 3:
                        continue
                    
                    # Flatten and normalize the points
                    points = approx.reshape(-1, 2)
                    norm_points = points.astype(float)
                    norm_points[:, 0] /= img_width  # Normalize x coordinates
                    norm_points[:, 1] /= img_height  # Normalize y coordinates
                    
                    # Write to file in YOLO format: class_id x1 y1 x2 y2 ...
                    point_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in norm_points])
                    f.write(f"0 {point_str}\n")  # Class ID 0 for predicted masks

    def annotate_images(self, classifier):
        """
        Annotate images without ground truth using the trained classifier
        
        Args:
            classifier: Trained classifier model
            
        Returns:
            Number of successfully annotated images
        """
        print(f"Annotating {len(self.images_without_gt)} unlabeled images...")
        annotated_count = 0
        
        for image_path in tqdm(self.images_without_gt, desc="Generating annotations"):
            try:
                # Load image
                image = np.array(Image.open(image_path).convert("RGB"))
                
                # Generate SAM masks for this image
                sam_masks = self.generate_sam_masks(image)
                
                if not sam_masks:
                    print(f"No SAM masks generated for {image_path}")
                    continue
                
                # Predict on SAM masks
                pred_results, probabilities = classifier.predict(image, sam_masks)
                
                # Extract predicted masks (class 1)
                predicted_masks = []
                predicted_probs = []

                for (mask, pred_class), prob in zip(pred_results, probabilities):
                    if pred_class == 1:  # Positive class
                        predicted_masks.append(mask)
                        predicted_probs.append(prob[1] if len(prob) > 1 else prob[0])

                # Apply NMS to remove redundant masks
                if predicted_masks and predicted_probs:
                    # Convert masks to bounding boxes in YOLO format [center_x, center_y, width, height]
                    boxes = []
                    for mask in predicted_masks:
                        # Find bounding box of mask
                        y_indices, x_indices = np.where(mask > 0)
                        if len(y_indices) == 0 or len(x_indices) == 0:
                            continue
                            
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        
                        # Convert to YOLO format (normalized)
                        center_x = (x_min + x_max) / (2 * image.shape[1])
                        center_y = (y_min + y_max) / (2 * image.shape[0])
                        width = (x_max - x_min) / image.shape[1]
                        height = (y_max - y_min) / image.shape[0]
                        
                        boxes.append([center_x, center_y, width, height])
                    
                    if boxes:
                        boxes = np.array(boxes)
                        # Apply NMS
                        keep_indices = self.apply_nms(boxes, np.array(predicted_probs), iou_threshold=0.45)
                        
                        # Keep only the masks that survived NMS
                        filtered_masks = [predicted_masks[i] for i in keep_indices]
                        
                        # Sort remaining masks by probability if needed
                        filtered_probs = [predicted_probs[i] for i in keep_indices]
                        filtered_masks, _ = sort_masks(filtered_masks, filtered_probs)
                        
                        # Update the predicted masks to the filtered ones
                        predicted_masks = filtered_masks
                    
                    if predicted_masks:  # Now contains only the filtered masks
                        # Determine output path for the label file
                        image_path_obj = Path(image_path)
                        image_rel_path = image_path_obj.relative_to(self.input_dir)

                        # Extract subfolder (Train, Validation, Test)
                        subfolder = image_path_obj.parent.name

                        # Construct the correct output path
                        output_dir_path = self.output_dir / "labels" / subfolder

                        # Check if the directory exists - if not, this is an error
                        if not output_dir_path.exists():
                            print(f"Error: Output directory {output_dir_path} does not exist. Please check your dataset structure.")
                            print(f"Skipping annotation for {image_path}")
                            continue

                        # Create the output path
                        output_path = output_dir_path / f"{image_path_obj.stem}.txt"

                        # Save the filtered masks as YOLO format
                        self.save_masks_as_yolo(predicted_masks, image.shape[1], image.shape[0], output_path)
                        annotated_count += 1
                    
                    # Optionally save visualization for debugging
                    if hasattr(self, 'save_visualizations') and self.save_visualizations:
                        self._visualize_prediction(image, predicted_masks, os.path.splitext(os.path.basename(image_path))[0])
            
            except Exception as e:
                print(f"Error annotating image {image_path}: {e}")
                continue
        
        print(f"Successfully annotated {annotated_count} images")
        return annotated_count

    def _visualize_prediction(self, image, predicted_masks, base_name):
        """
        Visualize prediction results (for debugging)
        
        Args:
            image: Input image
            predicted_masks: List of predicted masks
            base_name: Base name for saving the visualization
        """
        # Create a copy of the image for visualization
        viz_image = image.copy()
        
        # Draw predicted masks
        for i, mask in enumerate(predicted_masks):
            # Create random color for this mask
            color = np.random.randint(0, 255, 3).tolist()
            
            # Apply color overlay
            mask_overlay = np.zeros_like(viz_image)
            mask_overlay[mask > 0] = color
            
            # Blend with original image
            alpha = 0.5
            viz_image = cv2.addWeighted(viz_image, 1, mask_overlay, alpha, 0)
            
            # Draw contour
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz_image, contours, -1, color, 2)
        
        # Create debug directory if needed
        debug_dir = self.output_dir / "debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save visualization
        output_path = debug_dir / f"{base_name}_prediction.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))

    def apply_nms(self, boxes, scores, iou_threshold=0.45):
        """
        Apply Non-Maximum Suppression to bounding boxes.
        
        Parameters:
        boxes: Array of shape (N, 4) with boxes in YOLO format [center_x, center_y, width, height]
        scores: Array of shape (N) with confidence scores
        iou_threshold: IoU threshold for considering boxes as duplicates (default: 0.45)
        score_threshold: Minimum score threshold for considering boxes (default: 0.25)
        
        Returns:
        Array of indices of selected boxes
        """
        # Convert to numpy arrays if not already
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # If no boxes remain after filtering
        if len(boxes) == 0:
            return []
        
        # Sort boxes by decreasing scores
        indices = np.argsort(-scores)
        boxes = boxes[indices]
        
        keep_indices = []
        
        while len(boxes) > 0:
            # Keep the box with highest score
            keep_indices.append(indices[0])
            
            # If only one box is left, we're done
            if len(boxes) == 1:
                break
            
            # Calculate IoU of the first box with all remaining boxes
            first_box = boxes[0:1]
            remaining_boxes = boxes[1:]
            ious = batch_calculate_iou_yolo(first_box, remaining_boxes)[0]
            
            # Keep boxes with IoU less than threshold
            mask = ious < iou_threshold
            boxes = remaining_boxes[mask]
            indices = indices[1:][mask]
        
        return np.array(keep_indices)

    def run(self, save_visualizations=False):
        """
        Run the complete auto-annotation pipeline
        
        Args:
            save_visualizations: Whether to save visualizations of the predictions
            
        Returns:
            Dictionary with results
        """
        print("Starting auto-annotation process...")
        self.save_visualizations = save_visualizations
        
        # Step 1: Find all data
        data_map = self.find_data()
        
        if not data_map:
            print("No data found in the specified directory")
            return {"success": False, "error": "No data found"}
        
        if not self.images_with_gt:
            print("No images with ground truth found, cannot train the model")
            return {"success": False, "error": "No labeled images for training"}
        
        # Step 2: Train the model
        try:
            classifier = self.train_model()
        except Exception as e:
            print(f"Error training model: {e}")
            return {"success": False, "error": f"Training error: {str(e)}"}
        
        # Step 3: Annotate unlabeled images
        if self.images_without_gt:
            annotated_count = self.annotate_images(classifier)
            
            results = {
                "success": True,
                "total_images": len(data_map),
                "labeled_images": len(self.images_with_gt),
                "unlabeled_images": len(self.images_without_gt),
                "annotated_images": annotated_count
            }
            
            print(f"Auto-annotation completed. Added labels for {annotated_count} images.")
            return results
        else:
            print("No unlabeled images found to annotate")
            return {
                "success": True,
                "total_images": len(data_map),
                "labeled_images": len(self.images_with_gt),
                "unlabeled_images": 0,
                "annotated_images": 0
            }

def convert_polygon_to_bbox(polygon_file, output_file=None):
    """
    Convert YOLO polygon format to Ultralytics YOLO detection format
    
    YOLO polygon: class_id x1 y1 x2 y2 x3 y3 ... xn yn
    Ultralytics YOLO: class_id x_center y_center width height
    (All values normalized between 0-1)
    """
    if output_file is None:
        output_file = polygon_file
        
    with open(polygon_file, 'r') as f:
        lines = f.readlines()
    
    bbox_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:  # Need at least class_id and two points
            continue
            
        class_id = parts[0]
        # Extract all x,y coordinates
        coords = [float(coord) for coord in parts[1:]]
        
        # Group into x,y pairs
        points = np.array(coords).reshape(-1, 2)
        
        # Calculate bounding box from polygon points
        min_x = np.min(points[:, 0])
        min_y = np.min(points[:, 1])
        max_x = np.max(points[:, 0])
        max_y = np.max(points[:, 1])
        
        # Convert to YOLO format (center_x, center_y, width, height)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
        bbox_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    # Write the converted lines to the output file
    with open(output_file, 'w') as f:
        f.writelines(bbox_lines)
    
    return True

def convert_all_label_files(labels_dir, backup=False):
    """Convert all label files in a directory from polygon to bbox format"""
    if isinstance(labels_dir, str):
        labels_dir = Path(labels_dir)
    
    # Get all subdirectories (Train, Validation, Test)
    subfolders = [d for d in labels_dir.iterdir() if d.is_dir()]
    
    count = 0
    for subfolder in subfolders:
        polygon_files = glob.glob(str(subfolder / "*.txt"))
        
        for polygon_file in polygon_files:
            # Use the same filename for output
            polygon_file_path = Path(polygon_file)
            
            # Create backup of original file if requested
            if backup:
                backup_file = str(polygon_file) + '.polygon_backup'
                if not os.path.exists(backup_file):
                    os.rename(polygon_file, backup_file)
                    # Now convert from backup to original filename
                    if convert_polygon_to_bbox(backup_file, polygon_file):
                        count += 1
            else:
                # Convert in place without backup
                if convert_polygon_to_bbox(polygon_file):
                    count += 1
    
    return count

def main():
    """Main function to run the auto-annotator"""
    parser = argparse.ArgumentParser(description="YOLO Segmentation Auto-Annotator")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing YOLO segmentation data")
    parser.add_argument("--output_dir", help="Path to save the annotations (defaults to same as input_dir)")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations of predictions for debugging")
    parser.add_argument("--convert_to_bbox", action="store_true", help="Convert polygon segmentation to bounding box detection format")
    args = parser.parse_args()
    
    # Try to import and initialize SAM2
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        import os.path as osp
        
        # Define model path
        project_root = os.getcwd()  # This will give you the repo root directory

        # Construct the model path relative to the project root
        model_path = osp.join(project_root, "models", "sam2.1_hiera_small.pt")  # Include .1 in filename
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        
        # Initialize SAM model
        print("Initializing SAM2 model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = build_sam2(model_cfg, model_path, device=device)
        
        sam_model = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=32,
            pred_iou_thresh=0.6,
            stability_score_thresh=0.8,
            box_nms_thresh=0.6,
            device=device,
        )
        print("SAM2 model initialized successfully")
    except ImportError as e:
        print(f"SAM2 is required but not available: {e}")
        print("Please install SAM2 and its dependencies. Exiting.")
        sys.exit(1)
    
    # Create and run the auto-annotator
    annotator = AutoAnnotator(
        args.input_dir,
        args.output_dir,
        sam_model=sam_model
    )
    
    result = annotator.run(save_visualizations=args.visualize)
    
    # Print final result summary
    if result["success"]:
        print("\nSummary:")
        print(f"Total images: {result['total_images']}")
        print(f"Initially labeled: {result['labeled_images']}")
        print(f"Initially unlabeled: {result['unlabeled_images']}")
        print(f"Newly annotated: {result['annotated_images']}")
        print(f"Annotation success rate: {result['annotated_images'] / max(1, result['unlabeled_images']) * 100:.1f}%")
        
        # Add conversion step if requested
        if args.convert_to_bbox:
            output_dir = args.output_dir if args.output_dir else args.input_dir
            labels_dir = Path(output_dir) / "labels"
            if labels_dir.exists():
                print("\nConverting segmentation masks to detection format...")
                converted_count = convert_all_label_files(labels_dir)
                print(f"Converted {converted_count} label files from polygon to detection format")
    else:
        print(f"\nAnnotation failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()