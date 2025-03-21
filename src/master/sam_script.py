import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import json
from src.master.data import load_and_crop_data
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.utils.misc import variant_to_config_mapping

def setup_directories(base_path):
    """Create necessary directories for SAM inference results."""
    sam_path = os.path.join(base_path, "sam_inference")
    masks_path = os.path.join(sam_path, "masks")
    images_path = os.path.join(sam_path, "images")
    
    os.makedirs(sam_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    
    return sam_path, masks_path, images_path

def crop_image(image, crop_coords):
    """Crop image using the provided coordinates."""
    y_start, y_end, x_start, x_end = crop_coords
    return image[y_start:y_end, x_start:x_end]

def save_mask_to_txt(mask, output_path):
    """Save a binary mask as a text file."""
    np.savetxt(output_path, mask.astype(np.uint8), fmt='%d', delimiter=',')

def process_theme(img_path, crop_coords, mask_generator):
    """Process images for a specific theme."""
    # Extract theme name from path
    theme_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
    print(f"Processing {theme_name} images...")
    
    # Setup directories
    base_path = os.path.dirname(os.path.dirname(img_path))
    sam_path, masks_path, images_path = setup_directories(base_path)
    
    # Get all PNG images
    png_files = glob.glob(os.path.join(img_path, "*.PNG"))
    
    for img_file in tqdm(png_files):
        # Get image name without extension
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        
        # Load and crop image
        img = cv2.imread(img_file)
        cropped_img = crop_image(img, crop_coords)
        
        # Save cropped image
        cropped_img_path = os.path.join(images_path, f"{img_name}.png")
        cv2.imwrite(cropped_img_path, cropped_img)
        
        # Run SAM inference
        masks = mask_generator.generate(cropped_img)
        
        # Save mask data as txt files
        for i, mask_data in enumerate(masks):
            # Extract the segmentation mask
            mask = mask_data["segmentation"]
            
            # Save each mask as a separate text file
            mask_output_path = os.path.join(masks_path, f"{img_name}_mask_{i}.txt")
            save_mask_to_txt(mask, mask_output_path)
        
        # Save metadata about all masks in this image
        mask_meta_output_path = os.path.join(masks_path, f"{img_name}_metadata.json")
        with open(mask_meta_output_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_masks = []
            for mask_data in masks:
                serializable_mask = {}
                for k, v in mask_data.items():
                    if isinstance(v, np.ndarray):
                        serializable_mask[k] = v.tolist()
                    else:
                        serializable_mask[k] = v
                serializable_masks.append(serializable_mask)
            
            json.dump(serializable_masks, f)

def main():
    # Initialize SAM model
    print("Loading SAM model...")
    model = build_sam2(
        variant_to_config_mapping["small"],
        "models/sam2_hiera_small.pt",
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=32,  
        pred_iou_thresh=0.7,  
        stability_score_thresh=0.85,  
        box_nms_thresh=0.6, 
    )
    
    # Define paths and crop coordinates
    themes = [
        {
            "name": "meatballs",
            "path": "C:/Users/gtoft/OneDrive/DTU/4_Semester_AS/Master_Thesis/data/processed/meatballs/images/train",
            "crop": (30, 300, 200, 450)
        },
        {
            "name": "cans",
            "path": "C:/Users/gtoft/OneDrive/DTU/4_Semester_AS/Master_Thesis/data/processed/cans/images/train",
            "crop": (40, 590, 100, 640)
        },
        {
            "name": "doughs",
            "path": "C:/Users/gtoft/OneDrive/DTU/4_Semester_AS/Master_Thesis/data/processed/doughs/images/train",
            "crop": (50, 640, 30, 450)
        },
        {
            "name": "bottles",
            "path": "C:/Users/gtoft/OneDrive/DTU/4_Semester_AS/Master_Thesis/data/processed/bottles/images/train",
            "crop": (100, 480, 20, 640)
        }
    ]
    
    # Process each theme
    for theme in themes:
        process_theme(theme["path"], theme["crop"], mask_generator)
    
    print("SAM inference completed for all themes!")

if __name__ == "__main__":
    main()