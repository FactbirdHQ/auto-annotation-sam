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

# Import from the provided files
from src.master.model import CLIPEmbedding, KNNClassifier
from src.master.data import load_yolo_masks_as_binary_list, process_segmentation_masks, sort_masks

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
        Find all images and label files in train, val, and test folders
        
        Returns:
            Dictionary mapping image paths to label paths (or None if no label exists)
        """
        print("Finding data in input directory...")
        
        # Look for images and labels in train, val, and test folders
        subfolders = ["train", "val", "test"]
        if not any(os.path.exists(self.input_dir / subfolder) for subfolder in subfolders):
            # If no train/val/test structure, look in root directory
            subfolders = [""]
        
        # Collect all image and label files
        image_files = []
        label_files = []
        
        for subfolder in subfolders:
            subfolder_path = self.input_dir / subfolder
            if not os.path.exists(subfolder_path):
                continue
                
            # Find image files in this subfolder
            images_path = subfolder_path / "images" if (subfolder_path / "images").exists() else subfolder_path
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_files.extend(glob.glob(str(images_path / ext)))
            
            # Find label files in this subfolder
            labels_path = subfolder_path / "labels" if (subfolder_path / "labels").exists() else subfolder_path
            label_files.extend(glob.glob(str(labels_path / "*.txt")))
        
        # Map image files to label files
        data_map = {}
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]
            
            # Look for corresponding label file
            label_path = None
            for l_path in label_files:
                if os.path.splitext(os.path.basename(l_path))[0] == base_name:
                    label_path = l_path
                    break
            
            data_map[image_path] = label_path
            
            # Track which images have ground truth
            if label_path:
                self.images_with_gt.append(image_path)
            else:
                self.images_without_gt.append(image_path)
        
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
        if self.sam_model is None:
            # Create placeholder masks (for testing without SAM)
            print("Warning: No SAM model provided, using placeholder masks")
            return self._generate_placeholder_masks(image)
        
        # Run SAM inference and convert to binary masks
        masks = self.sam_model.generate(image)
        return masks
    
    def _generate_placeholder_masks(self, image, num_masks=5):
        """
        Generate placeholder masks for testing without SAM
        
        Args:
            image: Input image array
            num_masks: Number of masks to generate
            
        Returns:
            List of binary masks
        """
        height, width = image.shape[:2]
        masks = []
        
        for _ in range(num_masks):
            # Create a random rectangle mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Random rectangle parameters
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            x2 = np.random.randint(width // 2, width)
            y2 = np.random.randint(height // 2, height)
            
            # Draw the rectangle
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
            masks.append(mask)
        
        return masks

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
                # Load image
                image = np.array(Image.open(image_path).convert("RGB"))
                
                # Load ground truth masks from corresponding label file
                label_path = image_path.parent.parent / "labels" / f"{os.path.splitext(os.path.basename(image_path))[0]}.txt"
                if not os.path.exists(label_path):
                    # Try alternate path formats
                    alt_path = Path(str(image_path).replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt"))
                    if os.path.exists(alt_path):
                        label_path = alt_path
                    else:
                        continue
                
                gt_masks = load_yolo_masks_as_binary_list(label_path, image.shape[1], image.shape[0])
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
                    
                    # Simplify contour
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
                
                # Sort masks by probability
                if predicted_masks and predicted_probs:
                    predicted_masks, predicted_probs = sort_masks(predicted_masks, predicted_probs)
                    
                    # Determine output path for the label file
                    image_rel_path = os.path.relpath(image_path, self.input_dir)
                    
                    # Replace 'images' with 'labels' in the path
                    if "images" in image_rel_path:
                        label_rel_path = image_rel_path.replace("images", "labels")
                    else:
                        # If no 'images' folder, use the same structure but with .txt extension
                        label_rel_path = os.path.splitext(image_rel_path)[0] + ".txt"
                    
                    # Ensure we have the correct extension
                    label_rel_path = os.path.splitext(label_rel_path)[0] + ".txt"
                    
                    # Full output path
                    output_path = self.output_dir / label_rel_path
                    
                    # Save the predicted masks as YOLO format
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


def main():
    """Main function to run the auto-annotator"""
    parser = argparse.ArgumentParser(description="YOLO Segmentation Auto-Annotator")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing YOLO segmentation data")
    parser.add_argument("--output_dir", help="Path to save the annotations (defaults to same as input_dir)")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations of predictions for debugging")
    args = parser.parse_args()
    
    # Try to import SAM2 if available
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        from sam2.utils.misc import variant_to_config_mapping
        
        # Initialize SAM model
        print("Initializing SAM2 model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = build_sam2(
            variant_to_config_mapping["small"],
            "../models/sam2_hiera_small.pt",
        )
        
        sam_model = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=32,
            pred_iou_thresh=0.6,
            stability_score_thresh=0.8,
            box_nms_thresh=0.6,
            device=device,
        )
        print("SAM2 model initialized successfully")
    except ImportError:
        print("SAM2 model not available, will use placeholder masks")
        sam_model = None
    
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
    else:
        print(f"\nAnnotation failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()