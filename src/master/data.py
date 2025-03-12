from pathlib import Path
import cv2
import numpy as np
import typer
from PIL import Image
from torch.utils.data import Dataset


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


if __name__ == "__main__":
    typer.run(preprocess)
