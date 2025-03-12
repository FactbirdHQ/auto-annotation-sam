from pathlib import Path
import cv2
import numpy as np
import typer
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple, Union


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)

def load_and_crop_data(img_path, label_path, crop_coords):
    """Load image and YOLO masks, then crop them to specified region"""
    y_start, y_end, x_start, x_end = crop_coords
    
    # Load full image for original dimensions
    full_img = np.array(Image.open(img_path).convert("RGB"))
    original_h, original_w, _ = full_img.shape
    
    # Generate masks using original dimensions
    masks = load_yolo_masks_as_binary_list(label_path, original_w, original_h)
    
    # Crop image and masks
    cropped_img = full_img[y_start:y_end, x_start:x_end]
    cropped_masks = [mask[y_start:y_end, x_start:x_end] for mask in masks]
    
    return cropped_img, cropped_masks


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

if __name__ == "__main__":
    typer.run(preprocess)
