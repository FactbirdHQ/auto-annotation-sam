from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from typing import List, Tuple, Union
import numpy as np
from sklearn.model_selection import KFold
import os
import torch
import random

def custom_collate_fn(batch):
        """Custom collate function to properly handle lists of dictionaries."""
        batch_dict = {}
        
        # Initialize keys based on first item
        for key in batch[0].keys():
            batch_dict[key] = []
        
        # Add each item's data to the batch
        for item in batch:
            for key, value in item.items():
                batch_dict[key].append(value)
        
        return batch_dict

class SegmentationDataset(Dataset):
    """
    Dataset for loading images and masks in YOLO segmentation format,
    specifically designed to work with the embedding-classifier framework.
    """
    
    def __init__(self, dataset_path, class_id=1):
        """
        Initialize dataset for a specific object class.
        
        Args:
            dataset_path (str): Path to the specific dataset directory
            class_id (int): The class ID for this dataset
        """
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / 'images'
        self.gt_path = self.dataset_path / 'gt_masks'
        self.sam_masks_path = self.dataset_path / 'sam_masks'
        self.class_id = class_id
        
        # Verify folders exist
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.gt_path.exists():
            raise FileNotFoundError(f"GT masks directory not found: {self.gt_path}")
        if not self.sam_masks_path.exists():
            raise FileNotFoundError(f"SAM masks directory not found: {self.sam_masks_path}")
        
        # Get sorted list of image filenames
        self.image_files = sorted([f for f in os.listdir(self.images_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Store the dataset name
        self.dataset_name = self.dataset_path.name
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.image_files)
    
    def _parse_yolo_segmentation(self, txt_path, is_sam_mask=False):
        """
        Parse YOLO segmentation format.
        
        YOLO format: class_id x1 y1 x2 y2 ... xn yn
        Where x and y are normalized coordinates (0-1)
        
        Args:
            txt_path: Path to the YOLO format text file
            is_sam_mask: Whether this is a SAM mask (class_id is placeholder)
            
        Returns:
            List of dictionaries, each containing 'class_id' and 'polygon' keys
        """
        masks = []
        
        # Check if file exists (some frames might not have annotations)
        if not txt_path.exists():
            return masks
            
        # Parse each line in the txt file
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:  # Need at least class_id and 2 points
                    continue
                    
                # For SAM masks, class_id is a placeholder
                # For GT masks, use the dataset's class_id
                class_id = 0 if is_sam_mask else self.class_id
                
                # Parse all x,y pairs (every two values after class_id)
                polygon = []
                for i in range(1, len(parts), 2):
                    if i+1 < len(parts):
                        x, y = float(parts[i]), float(parts[i+1])
                        polygon.append((x, y))
                
                masks.append({
                    'class_id': class_id,
                    'polygon': polygon
                })
                
        return masks
    
    def _polygons_to_binary_mask(self, polygons, img_size):
        """
        Convert YOLO normalized polygons to a binary mask.
        
        Args:
            polygons: List of polygon dicts with 'class_id' and 'polygon' keys
            img_size: Tuple of (width, height) of the image
            
        Returns:
            np.ndarray: Binary mask with 1s where objects are present
        """
        width, height = img_size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for poly_data in polygons:
            polygon = poly_data['polygon']
            
            # Convert normalized coordinates to pixel coordinates
            pixel_polygon = [(int(x * width), int(y * height)) for x, y in polygon]
            
            # Convert to format for OpenCV drawing
            points = np.array(pixel_polygon, dtype=np.int32).reshape((-1, 1, 2))
            
            # Fill polygon
            cv2.fillPoly(mask, [points], 1)
                
        return mask
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the given index.
        """
        # Get filenames
        img_filename = self.image_files[idx]
        base_filename = os.path.splitext(img_filename)[0]
        
        # Load image
        img_path = str(self.images_path / img_filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_height, img_width = image.shape[:2]
            
        # Load GT annotations
        gt_path = self.gt_path / f"{base_filename}.txt"
        gt_masks_data = self._parse_yolo_segmentation(gt_path, is_sam_mask=False)
        
        # Load SAM masks
        sam_mask_path = self.sam_masks_path / f"{base_filename}.txt"
        sam_masks_data = self._parse_yolo_segmentation(sam_mask_path, is_sam_mask=True)
        
        # Convert polygon annotations to INDIVIDUAL binary masks (not combined)
        gt_binary_masks = []
        for mask_data in gt_masks_data:
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            polygon = mask_data['polygon']
            pixel_polygon = [(int(x * img_width), int(y * img_height)) for x, y in polygon]
            points = np.array(pixel_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [points], 1)
            gt_binary_masks.append(mask)
            
        sam_binary_masks = []
        for mask_data in sam_masks_data:
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            polygon = mask_data['polygon']
            pixel_polygon = [(int(x * img_width), int(y * img_height)) for x, y in polygon]
            points = np.array(pixel_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [points], 1)
            sam_binary_masks.append(mask)
            
        # Process segmentation masks
        combined_masks, combined_labels = [], []
        if gt_binary_masks and sam_binary_masks:
            combined_masks, combined_labels = process_segmentation_masks(
                gt_binary_masks, sam_binary_masks, iou_threshold=0.5
            )
        elif gt_binary_masks:
            combined_masks = gt_binary_masks
            combined_labels = [1] * len(gt_binary_masks)
        elif sam_binary_masks:
            combined_masks = sam_binary_masks
            combined_labels = [0] * len(sam_binary_masks)
        
        # Return sample with all information
        return {
            'image': image,
            'gt_masks': gt_masks_data,  # Original polygon data
            'sam_masks': sam_masks_data,  # Original polygon data
            'gt_binary_masks': gt_binary_masks,  # List of individual GT binary masks
            'sam_binary_masks': sam_binary_masks,  # List of individual SAM binary masks
            'combined_masks': combined_masks,  # List of combined masks (GT + non-overlapping SAM)
            'combined_labels': combined_labels,  # Corresponding labels (1 for GT, 0 for SAM)
            'has_gt': len(gt_masks_data) > 0,
            'filename': img_filename,
            'dataset': self.dataset_name
        }


class KFoldSegmentationManager:
    """
    Manager for K-fold cross-validation with the segmentation dataset,
    specifically adapted for the embedding-classifier framework.
    """
    
    def __init__(self, dataset_path, class_id=0):
        """
        Initialize dataset manager with K-fold CV support.
        
        Args:
            dataset_path (str): Path to the dataset directory
            class_id (int): The class ID for this dataset
        """
        self.dataset_path = Path(dataset_path)
        self.class_id = class_id
        
        # Create the dataset
        self.dataset = SegmentationDataset(
            dataset_path=dataset_path,
            class_id=class_id
        )
        
        # Store dataset name for reference
        self.dataset_name = self.dataset.dataset_name
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataset)
    
    def get_dataset_info(self):
        """Return information about the dataset."""
        return {
            'dataset_name': self.dataset_name,
            'class_id': self.class_id,
            'total_samples': len(self.dataset)
        }
    
    def get_training_data(self, dataloader):
        """
        Get training data from dataloader in a format suitable for your classifier.
        
        Returns:
            tuple: (images, masks, labels) ready for classifier.fit()
        """
        images = []
        masks_per_image = []  # Will be a list of lists
        labels_per_image = []  # Will be a list of lists
        
        for batch in dataloader:
            batch_images = batch['image']
            batch_combined_masks = batch['combined_masks']
            batch_combined_labels = batch['combined_labels']
            batch_has_gt = batch['has_gt']
            
            # Process each item in the batch
            for i in range(len(batch_images)):
                # Only include samples with ground truth
                if batch_has_gt[i]:
                    # Get image as numpy array
                    image = batch_images[i].numpy() if isinstance(batch_images[i], torch.Tensor) else batch_images[i]
                    
                    # Get masks and labels
                    masks = batch_combined_masks[i]
                    labels = batch_combined_labels[i]
                    
                    # Add to training data
                    images.append(image)
                    masks_per_image.append(masks)
                    labels_per_image.append(labels)
                
        return images, masks_per_image, labels_per_image
        
    def get_prediction_data(self, dataloader, return_combined=False):
        """
        Get prediction data from dataloader, including ground truth masks for validation.
        
        Args:
            dataloader: PyTorch DataLoader with dataset samples
            return_combined: If True, returns combined candidate masks (both GT and SAM),
                            otherwise returns only SAM masks (default: False)
            
        Returns:
            list: List of (image, candidate_masks, gt_masks) tuples for prediction and validation
        """
        prediction_data = []
        
        for batch in dataloader:
            batch_images = batch['image']
            batch_gt_binary_masks = batch['gt_binary_masks']  # Ground truth masks for validation
            
            # Choose which candidate masks to use based on the mode
            if return_combined:
                # Use combined masks (both GT and SAM) as candidates
                batch_candidate_masks = batch['combined_masks']
            else:
                # Use only SAM masks as candidates
                batch_candidate_masks = batch['sam_binary_masks']
            
            # Process each item in the batch
            for i in range(len(batch_images)):
                # Get image as numpy array
                image = batch_images[i].numpy() if isinstance(batch_images[i], torch.Tensor) else batch_images[i]
                
                # Get candidate masks for this image
                candidate_masks = batch_candidate_masks[i]
                
                # Get ground truth masks for this image
                gt_masks = batch_gt_binary_masks[i]
                
                # Only add to prediction data if there are candidate masks
                if candidate_masks and len(candidate_masks) > 0:
                    prediction_data.append((image, candidate_masks, gt_masks))
                
        return prediction_data
    
    def get_kfold_dataloaders(self, k=5, batch_size=8, shuffle=True, random_state=42):
        """
        Generate K-fold cross-validation splits for this dataset.
        
        Args:
            k (int): Number of folds
            batch_size (int): Batch size for dataloaders
            shuffle (bool): Whether to shuffle the data
            random_state (int): Random seed for reproducibility
            
        Returns:
            A list of (train_dataloader, val_dataloader) pairs for each fold
        """
        # Initialize random number generator for reproducibility
        np.random.seed(random_state)
        random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Create K-fold splitter
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        
        # Store dataloaders for each fold
        fold_dataloaders = []
        
        # Generate indices for each fold
        indices = list(range(len(self.dataset)))
        for train_idx, val_idx in kf.split(indices):
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            val_loader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                collate_fn=custom_collate_fn
            )
            
            train_loader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                collate_fn=custom_collate_fn
            )
            
            fold_dataloaders.append((train_loader, val_loader))
        
        return fold_dataloaders


def load_and_crop_data(img_path, label_path, crop_coords=None):
    """
    Load image and YOLO masks, then optionally crop them to specified region
    
    Args:
        img_path: Path to the input image
        label_path: Path to the YOLO format label file
        crop_coords: Optional tuple of (y_start, y_end, x_start, x_end) for cropping.
                    If None, uses the entire image.
    
    Returns:
        tuple: (image, masks) where image is a numpy array and masks is a list of binary masks
    """
    # Load full image for original dimensions
    full_img = np.array(Image.open(img_path).convert("RGB"))
    original_h, original_w, _ = full_img.shape
    
    # Generate masks using original dimensions
    masks = load_yolo_masks_as_binary_list(label_path, original_w, original_h)
    
    # If crop coordinates are provided, crop the image and masks
    if crop_coords is not None:
        y_start, y_end, x_start, x_end = crop_coords
        
        # Validate crop coordinates
        y_start = max(0, y_start)
        y_end = min(original_h, y_end)
        x_start = max(0, x_start)
        x_end = min(original_w, x_end)
        
        # Crop image and masks
        cropped_img = full_img[y_start:y_end, x_start:x_end]
        cropped_masks = [mask[y_start:y_end, x_start:x_end] for mask in masks]
        
        return cropped_img, cropped_masks
    else:
        # Return the full image and masks without cropping
        return full_img, masks


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

        # Add this debugging before returning masks
        for i, mask in enumerate(masks):
            if 0 in mask.shape:
                print(f"Warning: Mask {i} has invalid dimensions: {mask.shape}")
                # Optionally fix masks with zero dimensions by creating small valid mask
                masks[i] = np.zeros((img_height, img_width), dtype=np.uint8)
    
    return masks


def process_segmentation_masks(
    gt_masks: List[np.ndarray], 
    full_masks: List[np.ndarray], 
    iou_threshold: float = 0.5
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Process ground truth and full segmentation masks by:
    1. Identifying which full masks correspond to ground truth masks
    2. Removing those from the full segmentation masks
    3. Returning a combined list of all masks with corresponding labels
    
    Args:
        gt_masks: List of ground truth binary segmentation masks (np.ndarray)
        full_masks: List of all segmentation masks (np.ndarray)
        iou_threshold: IoU threshold to determine if a full mask matches a GT mask
        
    Returns:
        Tuple containing:
        - combined_masks: List of all masks (GT masks + remaining full masks)
        - labels: List of labels (1 for GT masks, 0 for remaining full masks)
    """
    # Check if inputs are valid
    if not gt_masks or not full_masks:
        raise ValueError("Both gt_masks and full_masks must be non-empty lists")
    
    # Ensure all masks have the same shape
    gt_shape = gt_masks[0].shape
    for mask in gt_masks + full_masks:
        if mask.shape != gt_shape:
            raise ValueError(f"All masks must have the same shape. Expected {gt_shape}, got {mask.shape}")
    
    # Calculate IoU between each GT mask and full mask
    mask_matches = []
    for full_idx, full_mask in enumerate(full_masks):
        matched = False
        for gt_mask in gt_masks:
            iou = calculate_iou(gt_mask, full_mask)
            if iou >= iou_threshold:
                matched = True
                break
        mask_matches.append(matched)
    
    # Create combined list of masks and labels
    combined_masks = []
    labels = []
    
    # Add all GT masks first (label 1)
    for gt_mask in gt_masks:
        combined_masks.append(gt_mask)
        labels.append(1)
    
    # Add remaining full masks (label 0)
    for idx, full_mask in enumerate(full_masks):
        if not mask_matches[idx]:
            combined_masks.append(full_mask)
            labels.append(0)
    
    return combined_masks, labels

def get_positive_negative_masks(combined_masks: List[np.ndarray], labels: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Separate the combined masks list into positive (GT) and negative (remaining) masks
    based on their corresponding labels.
    
    Args:
        combined_masks: List of all masks (GT + remaining)
        labels: List of labels (1 for GT masks, 0 for remaining masks)
        
    Returns:
        Tuple containing:
        - positive_masks: List of GT masks (label 1)
        - negative_masks: List of remaining masks (label 0)
    """
    positive_masks = []
    negative_masks = []
    
    for mask, label in zip(combined_masks, labels):
        if label == 1:
            positive_masks.append(mask)
        else:
            negative_masks.append(mask)
    
    return positive_masks, negative_masks

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1: First binary mask (numpy array where positive values indicate mask)
        mask2: Second binary mask (numpy array where positive values indicate mask)
        
    Returns:
        IoU score (float between 0 and 1)
    """
    # Convert masks to binary (0 or 1)
    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()
    
    # Handle edge case
    if union == 0:
        return 0.0
    
    return intersection / union

def sort_masks(filtered_masks, filtered_probs):
    # Create pairs of (mask, score)
    mask_score_pairs = list(zip(filtered_masks, filtered_probs))

    # Sort the pairs based on scores in descending order (highest score first)
    sorted_pairs = sorted(mask_score_pairs, key=lambda x: x[1], reverse=True)

    # Unzip the sorted pairs
    sorted_masks, sorted_probs = zip(*sorted_pairs) if sorted_pairs else ([], [])

    # Convert back to lists if needed
    sorted_masks = list(sorted_masks)
    sorted_probs = list(sorted_probs)

    return sorted_masks, sorted_probs
