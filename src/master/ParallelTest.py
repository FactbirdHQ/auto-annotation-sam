from src.master.evaluate import evaluate_binary_masks
from src.master.model import (
    RandomForestClassifier, KNNClassifier, LogRegClassifier, SVMClassifier,
    CLIPEmbedding, HoGEmbedding, ResNET18Embedding
)

from src.master.data import KFoldSegmentationManager


import concurrent.futures
import traceback
import time
import numpy as np
import os
import sys

def detailed_process_single_sample(sample_info_and_meta):
    """
    Global function for processing a single sample with extensive logging
    
    Args:
        sample_info_and_meta (tuple): Contains 
            (image, candidate_masks, gt_masks, classifier, sample_index)
    
    Returns:
        dict: Processing results or None if processing fails
    """
    # Unpack the input
    image, candidate_masks, gt_masks, classifier, sample_index = sample_info_and_meta
    
    try:
        # Validate inputs
        if not isinstance(image, np.ndarray):
            print(f"Sample {sample_index}: Invalid image type: {type(image)}")
            return None
        
        if not candidate_masks or not gt_masks:
            print(f"Sample {sample_index}: No candidate or ground truth masks")
            return None
        
        # Detailed logging about input sizes
        print(f"Sample {sample_index}: Processing...")
        print(f"  Image shape: {image.shape}")
        print(f"  Candidate masks: {len(candidate_masks)}")
        print(f"  GT masks: {len(gt_masks)}")
        
        # Attempt to process
        results_with_classes, probs = classifier.predict(
            image, 
            candidate_masks, 
            return_probabilities=True
        )
        
        # Filter positive masks
        positive_masks = [mask for mask, class_label in results_with_classes if class_label == 1]
        
        # Evaluate 
        metrics = evaluate_binary_masks(gt_masks, positive_masks)
        
        return {
            'sample_index': sample_index,
            'f1_score': metrics['mask_f1'],
            'detected_masks': len(positive_masks),
            'total_candidate_masks': len(candidate_masks)
        }
    
    except Exception as e:
        print(f"Sample {sample_index}: Processing error")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        print("Full traceback:")
        traceback.print_exc(file=sys.stdout)
        return None

def diagnose_parallel_processing(validation_data, classifier, max_samples=None):
    """
    Comprehensive diagnostic for parallel processing issues
    
    Args:
        validation_data (list): List of tuples (image, candidate_masks, gt_masks)
        classifier: Classifier object to use for predictions
        max_samples (int, optional): Limit number of samples to process
    
    Returns:
        tuple: (list of processing results, mean F1 score)
    """
    print("Diagnostic Parallel Processing")
    print("=" * 40)
    
    # System and environment information
    print(f"Python Version: {sys.version}")
    print(f"CPU Count: {os.cpu_count()}")
    
    # Limit samples if specified
    if max_samples is not None:
        validation_data = validation_data[:max_samples]
    
    print(f"Total validation samples: {len(validation_data)}")
    
    # Check if data is valid
    if not validation_data:
        print("ERROR: No validation data provided")
        return [], 0
    
    # Prepare data for processing with index
    processing_data = [
        (*sample, classifier, idx) for idx, sample in enumerate(validation_data)
    ]
    
    # Timing and tracking
    start_time = time.time()
    
    # Attempt parallel processing with detailed tracking
    results_all = []
    try:
        # Use ProcessPoolExecutor for true parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            # Map processing across all samples
            futures = list(executor.map(
                detailed_process_single_sample, 
                processing_data
            ))
            
            # Collect results
            results_all = [f for f in futures if f is not None]
    
    except Exception as e:
        print("Catastrophic parallel processing failure:")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        print("Full traceback:")
        traceback.print_exc(file=sys.stdout)
    
    # Final timing and summary
    end_time = time.time()
    print("\nProcessing Summary")
    print("=" * 20)
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Samples processed: {len(results_all)}/{len(validation_data)}")
    
    # Extract F1 scores
    f1_scores = [result['f1_score'] for result in results_all if 'f1_score' in result]
    
    # Calculate mean F1
    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    print(f"Mean F1 score: {mean_f1:.4f}")
    
    # Detailed results logging
    print("\nDetailed Processing Results:")
    for result in results_all:
        if result:
            print(f"Sample {result.get('sample_index', 'N/A')}: "
                  f"F1 = {result.get('f1_score', 'N/A'):.4f}, "
                  f"Detected Masks: {result.get('detected_masks', 0)}")
    
    return results_all, mean_f1

# Note: Ensure evaluate_binary_masks is imported or defined


def main():

    # Define the dataset path
    dataset_path = "C:/Users/GustavToft/OneDrive/DTU/4_Semester_AS/Master_Thesis/data/sam_inference/processed_data/meatballs"

    # Create dataset manager for this specific dataset
    dataset_manager = KFoldSegmentationManager(
        dataset_path=dataset_path,
        class_id=1
    )

    # Print dataset information
    print(f"Dataset info: {dataset_manager.get_dataset_info()}")

    # Get 5-fold cross validation dataloaders
    folds = dataset_manager.get_kfold_dataloaders(k=5, batch_size=1)

    # Example: Using with your embedding-classifier framework
    print("\nTraining example with first fold:")
    train_loader, val_loader = folds[0]

    # Get training data directly in the format for classifier.fit()
    train_images, train_masks, train_labels = dataset_manager.get_training_data(train_loader)

    print(f"Training data prepared:")
    print(f"  Images: {len(train_images)}")
    print(f"  Shape: {train_images[0].shape}")
    print(f"  GT masks: {len(train_masks)}")
    print(f"  Shape: {train_masks[0][0].shape}")
    print(f"  Labels: {len(train_labels)}")
    print(f"  Sample: {train_labels[0]}")

    # 1. Create embedding and classifier
    config = {
        'clip_model': 'ViT-B/32',
        'use_PCA': True,
        'PCA_var': 0.95
    }
    embedding = CLIPEmbedding(config)
    classifier = KNNClassifier(config, embedding)

    print('Fitting classifier')
    # 2. Train classifier directly with the data
    classifier.fit(train_images, train_masks, train_labels)

    # 3. Get validation data for prediction
    print('Getting validation data')
    validation_data = dataset_manager.get_prediction_data(val_loader)

    print(validation_data[0][0].shape)
    print(len(validation_data[0][0]))

    print('Trying inference')

    # Usage
    try:
        results_all, mean_f1 = diagnose_parallel_processing(
            validation_data, 
            classifier,
        )
        print(f"\nMean F1 score: {mean_f1:.4f}")
    except Exception as e:
        print(f"Diagnostic failed: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()