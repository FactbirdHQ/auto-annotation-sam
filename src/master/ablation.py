import os
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import time
from sklearn.model_selection import ParameterGrid
from scipy import stats
import logging
import concurrent.futures
from pathlib import Path

from src.master.report_generator import SegmentationReportGenerator
from src.master.evaluate import evaluate_binary_masks, plot_precision_recall_curve, plot_threshold_vs_metrics
from src.master.data import KFoldSegmentationManager
from src.master.model import (
    RandomForestClassifier, KNNClassifier, LogRegClassifier, SVMClassifier,
    CLIPEmbedding, HoGEmbedding, ResNET18Embedding
)

class AblationStudy:
    def __init__(self, config):
        """
        Initialize the ablation study with configuration
        
        Args:
            config: Dictionary containing study parameters including:
                - dataset_paths: Paths to datasets
                - embeddings: List of embeddings to test
                - classifiers: List of classifiers to test
                - metrics: List of metrics to track
                - output_dir: Directory to save results
                - k_folds: Number of cross-validation folds
        """
        self.config = config
        self.datasets = {}
        self.dataset_managers = {}
        self.results = defaultdict(dict)
        self.output_dir = config.get('output_dir', 'ablation_results')
        self.k_folds = config.get('k_folds', 5)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure file and console handlers
        self.logger = logging.getLogger('ablation_study')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(os.path.join(log_dir, 'ablation.log'))
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format for logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Initializing Ablation Study")

        # Set the maximum number of workers for concurrent execution
        self.max_workers = config.get('max_workers', os.cpu_count())
        self.logger.info(f"Using up to {self.max_workers} concurrent workers")
        
        # Initialize embedding and classifier factories
        self.embedding_factory = {
            'CLIP': lambda cfg: CLIPEmbedding(cfg),
            'HOG': lambda cfg: HoGEmbedding(cfg),
            'ResNet18': lambda cfg: ResNET18Embedding(cfg)
        }
        
        self.classifier_factory = {
            'KNN': lambda cfg, emb: KNNClassifier(cfg, emb),
            'SVM': lambda cfg, emb: SVMClassifier(cfg, emb),
            'RF': lambda cfg, emb: RandomForestClassifier(cfg, emb),
            'LR': lambda cfg, emb: LogRegClassifier(cfg, emb)
        }
    
    def load_datasets(self):
        """
        Load all datasets specified in the configuration
        
        Returns:
            Dictionary mapping dataset names to dataset manager objects
        """
        dataset_paths = self.config.get('dataset_paths', {})
        
        for name, path in dataset_paths.items():
            self.logger.info(f"Loading dataset: {name} from {path}")
            try:
                # Initialize dataset manager with proper class ID
                class_id = self.config.get('class_ids', {}).get(name, 1)
                dataset_manager = KFoldSegmentationManager(path, class_id=class_id)
                self.dataset_managers[name] = dataset_manager
                
                # Print dataset info
                info = dataset_manager.get_dataset_info()
                self.logger.info(f"  - Total samples: {info['total_samples']}")
                self.logger.info(f"  - Class ID: {info['class_id']}")
                
            except Exception as e:
                self.logger.error(f"Error loading dataset {name}: {e}")
        
        return self.dataset_managers
    
    def _evaluate_model(self, model, train_data, val_data, return_predictions=False):
        """
        Core model evaluation logic used across different analysis methods
        """
        train_images, train_masks, train_labels = train_data
        val_prediction_data, val_gt_masks = val_data
        
        # Time the training process
        start_time = time.perf_counter()
        model.fit(train_images, train_masks, train_labels)
        train_time = time.perf_counter() - start_time
        
        # Initialize metrics
        all_metrics = []
        all_pred_masks = []
        all_pred_probs = []
        
        # Time the inference process
        total_inference_time = 0
        num_predictions = 0
        
        # Process each validation sample with immediate evaluation
        for i, (image, candidate_masks) in enumerate(val_prediction_data):
            if not candidate_masks:
                continue
                
            # Get ground truth masks for this image
            # Assuming val_gt_masks is organized the same way as val_prediction_data
            image_gt_masks = val_gt_masks[i] if i < len(val_gt_masks) else []
                
            # Time the prediction process for this sample
            start_time = time.perf_counter()
            pred_results, probs = model.predict(image, candidate_masks)
            inference_time = time.perf_counter() - start_time
            total_inference_time += inference_time
            num_predictions += 1
            
            # Extract positive predictions
            image_pred_masks = []
            image_pred_probs = []
            for (mask, pred_class), prob in zip(pred_results, probs):
                # Only keep masks predicted as positive (class 1)
                if pred_class == 1:
                    image_pred_masks.append(mask)
                    image_pred_probs.append(prob[1] if len(prob) > 1 else prob[0])
            
            # Save predictions for return if needed
            all_pred_masks.extend(image_pred_masks)
            all_pred_probs.extend(image_pred_probs)
            
            # Evaluate this image immediately
            if image_gt_masks and image_pred_masks:
                image_metrics = evaluate_binary_masks(image_gt_masks, image_pred_masks)
                all_metrics.append(image_metrics)
        
        # Calculate average inference time
        avg_inference_time = total_inference_time / max(1, num_predictions)
        
        # Aggregate metrics across all images
        if all_metrics:
            aggregated_metrics = {
                'mask_precision': np.mean([m['mask_precision'] for m in all_metrics]),
                'mask_recall': np.mean([m['mask_recall'] for m in all_metrics]),
                'mask_f1': np.mean([m['mask_f1'] for m in all_metrics]),
                'detected_masks': sum(m['detected_masks'] for m in all_metrics),
                'total_gt_masks': sum(m['total_gt_masks'] for m in all_metrics),
                'avg_iou_detected': np.mean([m['avg_iou_detected'] for m in all_metrics if m['avg_iou_detected'] > 0]),
                'avg_iou_all': np.mean([m['avg_iou_all'] for m in all_metrics]),
                'training_time': train_time,
                'inference_time': avg_inference_time
            }
        else:
            # Return empty metrics if no predictions
            aggregated_metrics = {
                'mask_precision': 0, 
                'mask_recall': 0,
                'mask_f1': 0,
                'detected_masks': 0,
                'total_gt_masks': sum(len(gt) for gt in val_gt_masks) if val_gt_masks else 0,
                'avg_iou_detected': 0,
                'avg_iou_all': 0,
                'training_time': train_time,
                'inference_time': avg_inference_time
            }
        
        if return_predictions:
            return aggregated_metrics, all_pred_masks, all_pred_probs
        else:
            return aggregated_metrics
    
    def _evaluate_config(self, combo_idx, param_combinations):
        """Worker function to evaluate a specific hyperparameter configuration"""
        emb_config, cls_config = param_combinations[combo_idx]
        try:
            metrics, model = self._create_and_evaluate_model(
                self.current_dataset, self.current_embedding, self.current_classifier,
                emb_config, cls_config
            )
            
            # Clean up model regardless of result
            if model is not None:
                self._cleanup_model(model)
            
            # Return results if evaluation was successful
            if metrics is not None:
                return {
                    'embedding_config': emb_config,
                    'classifier_config': cls_config,
                    'metrics': metrics,
                    'combo_idx': combo_idx
                }
            return None
        except Exception as e:
            self.logger.error(f"Error evaluating config {combo_idx}: {e}")
            return None

    
    def create_model(self, embedding_name, classifier_name, embedding_config=None, classifier_config=None):
        """
        Create model with specified embedding and classifier
        
        Args:
            embedding_name: Name of the embedding
            classifier_name: Name of the classifier
            embedding_config: Configuration for the embedding
            classifier_config: Configuration for the classifier
            
        Returns:
            Initialized classifier model with embedding
        """
        if embedding_name not in self.embedding_factory:
            raise ValueError(f"Unknown embedding: {embedding_name}")
        if classifier_name not in self.classifier_factory:
            raise ValueError(f"Unknown classifier: {classifier_name}")
        
        self.logger.info(f"Creating model: {embedding_name} + {classifier_name}")
        self.logger.debug(f"Embedding config: {embedding_config}")
        self.logger.debug(f"Classifier config: {classifier_config}")
        # Use empty dict if configs are None
        embedding_config = embedding_config or {}
        classifier_config = classifier_config or {}
        
        # Create embedding and classifier
        embedding = self.embedding_factory[embedding_name](embedding_config)
        classifier = self.classifier_factory[classifier_name](classifier_config, embedding)
        
        return classifier
    
    def _create_and_evaluate_model(self, dataset_name, embedding_name, classifier_name, 
                              embedding_config=None, classifier_config=None,
                              train_loader=None, val_loader=None):
        """
        Create, train and evaluate a model with specified configuration
        
        Args:
            dataset_name: Name of the dataset
            embedding_name: Name of the embedding
            classifier_name: Name of the classifier
            embedding_config: Configuration for the embedding
            classifier_config: Configuration for the classifier
            train_loader: DataLoader for training data (if None, uses first fold)
            val_loader: DataLoader for validation data (if None, uses first fold)
            
        Returns:
            Tuple of (metrics, model) where metrics is a dictionary with evaluation results
            and model is the trained model
        """

        import psutil
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        self.logger.info(f"Memory before model creation: {mem_before:.2f} MB")

        # Get dataset manager
        if dataset_name not in self.dataset_managers:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset_manager = self.dataset_managers[dataset_name]
        
        # Get data loaders if not provided
        if train_loader is None or val_loader is None:
            fold_dataloaders = dataset_manager.get_kfold_dataloaders(k=self.k_folds, batch_size=1)
            train_loader, val_loader = fold_dataloaders[0]
        
        # Get training and validation data
        train_images, train_masks, train_labels = dataset_manager.get_training_data(train_loader)
        val_prediction_data = dataset_manager.get_prediction_data(val_loader)
        
        # Get ground truth masks for validation
        val_gt_masks = []
        for batch in val_loader:
            batch_gt_masks = batch['gt_binary_masks']
            batch_has_gt = batch.get('has_gt', [True] * len(batch_gt_masks))
            
            for i, has_gt in enumerate(batch_has_gt):
                if has_gt:
                    gt_mask = batch_gt_masks[i].numpy() if hasattr(batch_gt_masks[i], 'numpy') else batch_gt_masks[i]
                    val_gt_masks.append(gt_mask)
        
        # Create and evaluate model
        model = None
        metrics = None
        try:
            model = self.create_model(embedding_name, classifier_name, embedding_config, classifier_config)
            
            # Evaluate model
            metrics = self._evaluate_model(model, 
                                        (train_images, train_masks, train_labels),
                                        (val_prediction_data, val_gt_masks))
            
            mem_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            self.logger.info(f"Memory after evaluation: {mem_after:.2f} MB (diff: {mem_after-mem_before:.2f} MB)")

            return metrics, model
        except Exception as e:
            self.logger.error(f"Error evaluating {embedding_name}-{classifier_name} on {dataset_name}: {e}")
            if model is not None:
                self._cleanup_model(model)
            return None, None
        
    def _cleanup_model(self, model):
        """Safely clean up model resources"""
        try:
            # Clean up any hooks if the embedding has a cleanup method
            if model is not None and hasattr(model.embedding, 'cleanup'):
                model.embedding.cleanup()
            
            # Delete model reference
            del model
            gc.collect()
        except Exception as e:
            self.logger.error(f"Error cleaning up model: {e}")
    
    def _save_to_json(self, data, filename):
        """
        Save data to JSON file with proper serialization of numpy types
        
        Args:
            data: Data to serialize
            filename: Path to save the JSON file
        """
        # Ensure path exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert numpy values to Python native types
        def numpy_converter(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, default=numpy_converter, indent=4)

    def _get_train_val_data(self, dataset_name, fold_idx=0, return_combined = False):
        """
        Get training and validation data for a specific dataset and fold
        
        Args:
            dataset_name: Name of the dataset
            fold_idx: Index of the fold to use
            
        Returns:
            Tuple of (train_data, val_data) where
            train_data is (train_images, train_masks, train_labels) and
            val_data is (val_prediction_data, val_gt_masks)
        """
        if dataset_name not in self.dataset_managers:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset_manager = self.dataset_managers[dataset_name]
        
        # Get k-fold data loaders
        fold_dataloaders = dataset_manager.get_kfold_dataloaders(k=self.k_folds, batch_size=1)
        if fold_idx >= len(fold_dataloaders):
            raise ValueError(f"Fold {fold_idx} is out of range for dataset {dataset_name} with {len(fold_dataloaders)} folds")
        
        train_loader, val_loader = fold_dataloaders[fold_idx]
        
        # Get training data
        train_images, train_masks, train_labels = dataset_manager.get_training_data(train_loader)
        
        # Get validation data
        val_prediction_data = dataset_manager.get_prediction_data(val_loader, return_combined)
        
        # Get ground truth masks for validation
        val_gt_masks = []
        for batch in val_loader:
            batch_gt_masks = batch['gt_binary_mask']
            batch_has_gt = batch.get('has_gt', [True] * len(batch_gt_masks))
            
            for i, has_gt in enumerate(batch_has_gt):
                if has_gt:
                    gt_mask = batch_gt_masks[i].numpy() if hasattr(batch_gt_masks[i], 'numpy') else batch_gt_masks[i]
                    val_gt_masks.append(gt_mask)
        
        return (train_images, train_masks, train_labels), (val_prediction_data, val_gt_masks)
    
    def run_pipeline_comparison(self):
        """
        Compare performance between ideal (full recall) and realistic (SAM2-based) pipelines.
        - Reuses existing cross-validation results for the realistic track
        - Reuses existing best configurations from hyperparameter tuning
        - Only runs the ideal track with GT + SAM masks as candidates
        
        Returns:
            Dictionary with comparison results between ideal and realistic pipelines
        """
        # Ensure we have all_results with best_configs and per_dataset results
        if not hasattr(self, 'all_results'):
            raise ValueError("No ablation results found. Run full ablation study first.")
        
        if 'best_configs' not in self.all_results:
            raise ValueError("No best configurations found. Run hyperparameter tuning first.")
        
        if 'per_dataset' not in self.all_results:
            raise ValueError("No cross-validation results found. Run full ablation study first.")
        
        # Get the best configurations and realistic results
        best_configs = self.all_results['best_configs']
        realistic_results = self.all_results['per_dataset']
        
        # Initialize results structure for ideal track
        ideal_results = {}
        
        print("\n" + "="*80)
        print("Running Pipeline Comparison: Ideal vs. Realistic")
        print("="*80)
        print("Using existing cross-validation results for realistic track")
        print("Using existing best configurations from hyperparameter tuning")
        
        # Process datasets
        for dataset_name, dataset_dict in realistic_results.items():
            if dataset_name not in self.dataset_managers:
                self.logger.warning(f"Warning: Dataset {dataset_name} not loaded. Skipping.")
                continue
                
            ideal_results[dataset_name] = {}
            dataset_manager = self.dataset_managers[dataset_name]
            
            # Get k-fold data loaders for this dataset
            fold_dataloaders = dataset_manager.get_kfold_dataloaders(k=self.k_folds, batch_size=1)
            
            for embedding_name, embedding_dict in dataset_dict.items():
                ideal_results[dataset_name][embedding_name] = {}
                
                for classifier_name, _ in embedding_dict.items():
                    # Check if we have best configs for this combination
                    if (dataset_name not in best_configs or
                        embedding_name not in best_configs[dataset_name] or
                        classifier_name not in best_configs[dataset_name][embedding_name]):
                        self.logger.info(f"Skipping {dataset_name}, {embedding_name}, {classifier_name} - no best config")
                        continue
                    
                    self.logger.info(f"\nProcessing {dataset_name}, {embedding_name}, {classifier_name}")
                    
                    # Get best configs
                    best_config = best_configs[dataset_name][embedding_name][classifier_name]
                    embedding_config = best_config.get('embedding', {})
                    classifier_config = best_config.get('classifier', {})
                    
                    # Track metrics across all folds
                    fold_metrics = []
                    
                    # Process each fold (skipping fold 0, which is for hyperparameter tuning)
                    for fold_idx in range(1, self.k_folds):
                        self.logger.info(f"  Processing fold {fold_idx}/{self.k_folds-1}")
                        
                        # Get training and validation data for this fold, with return_combined=True for ideal track
                        train_data, val_data = self._get_train_val_data(
                            dataset_name, fold_idx, return_combined=True
                        )
                        
                        train_images, train_masks, train_labels = train_data
                        val_prediction_data, val_gt_masks = val_data
                        
                        if not train_images:
                            self.logger.debug(f"  No training data for fold {fold_idx}")
                            continue
                        
                        # Create and evaluate model
                        model = None
                        try:
                            model = self.create_model(embedding_name, classifier_name, embedding_config, classifier_config)
                            
                            # Evaluate model
                            metrics = self._evaluate_model(
                                model,
                                (train_images, train_masks, train_labels),
                                (val_prediction_data, val_gt_masks)
                            )
                            
                            # Add fold index to metrics
                            metrics['fold'] = fold_idx
                            fold_metrics.append(metrics)
                            
                        except Exception as e:
                            self.logger.error(f"  Error evaluating {embedding_name}-{classifier_name} on fold {fold_idx}: {e}")
                            continue
                        finally:
                            # Clean up model
                            self._cleanup_model(model)
                    
                    # Aggregate metrics across folds
                    if fold_metrics:
                        aggregated_metrics = self._aggregate_fold_metrics(fold_metrics)
                        ideal_results[dataset_name][embedding_name][classifier_name] = aggregated_metrics
        
        # Save results
        comparison_results = {
            'ideal': ideal_results,
            'realistic': realistic_results
        }
        
        self._save_to_json(comparison_results, os.path.join(self.output_dir, 'pipeline_comparison_results.json'))
        
        return comparison_results
    
    # First, add this as a new method to the AblationStudy class
    def _process_fold(self, fold_idx, dataset_name, embedding_name, classifier_name, 
                    embedding_config, classifier_config):
        """Worker function to process a specific fold during cross-validation"""
        if fold_idx == 0:
            return None  # Skip fold 0 (reserved for hyperparameter tuning)
            
        self.logger.info(f"\nProcessing fold {fold_idx+1}/{self.k_folds}")
        
        # Create model for this fold
        model = None
        try:
            # Get training and validation data
            train_data, val_data = self._get_train_val_data(dataset_name, fold_idx)
            train_images, train_masks, train_labels = train_data
            val_prediction_data, val_gt_masks = val_data
            
            if not train_images:
                self.logger.info(f"No training data for fold {fold_idx+1}")
                return None
                
            if not val_prediction_data:
                self.logger.info(f"No validation data for fold {fold_idx+1}")
                return None
                
            # Create and evaluate model
            model = self.create_model(embedding_name, classifier_name, embedding_config, classifier_config)
            
            # Evaluate model using the helper method
            metrics, val_pred_masks, val_pred_probs = self._evaluate_model(
                model,
                (train_images, train_masks, train_labels),
                (val_prediction_data, val_gt_masks),
                return_predictions=True
            )
            
            # Add fold index to metrics
            metrics['fold'] = fold_idx
            
            # Save precision-recall curve for this fold
            self._save_pr_curve(val_gt_masks, val_pred_masks, val_pred_probs, 
                            dataset_name, embedding_name, classifier_name, fold_idx)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating {embedding_name}-{classifier_name} on fold {fold_idx}: {e}")
            return None
        finally:
            # Use dedicated cleanup helper
            self._cleanup_model(model)

    # Then modify the perform_cross_validation method
    def perform_cross_validation(self, dataset_name, embedding_name, classifier_name, 
                                embedding_config=None, classifier_config=None):
        """
        Perform k-fold cross-validation using concurrent execution
        
        Args:
            dataset_name: Dataset name
            embedding_name: Embedding name 
            classifier_name: Classifier name
            embedding_config: Configuration for embedding
            classifier_config: Configuration for classifier
            
        Returns:
            Dictionary with aggregated metrics across folds
        """
        if dataset_name not in self.dataset_managers:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset_manager = self.dataset_managers[dataset_name]
        
        # Get k-fold data loaders
        fold_dataloaders = dataset_manager.get_kfold_dataloaders(k=self.k_folds, batch_size=1)
        
        # Track metrics across all folds
        fold_metrics = []
        
        # Use ThreadPoolExecutor for I/O-bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all folds for processing (skip fold 0)
            future_to_fold = {
                executor.submit(self._process_fold, fold_idx, dataset_name, embedding_name, 
                            classifier_name, embedding_config, classifier_config): fold_idx 
                for fold_idx in range(1, self.k_folds)
            }
            
            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(future_to_fold), 
                total=len(future_to_fold),
                desc=f"CV {embedding_name}-{classifier_name}"
            ):
                metrics = future.result()
                if metrics is not None:
                    fold_metrics.append(metrics)
        
        # Aggregate metrics across folds
        if fold_metrics:
            aggregated_metrics = self._aggregate_fold_metrics(fold_metrics)
            # Add average timing information
            aggregated_metrics['avg_training_time'] = np.mean([m['training_time'] for m in fold_metrics])
            aggregated_metrics['avg_inference_time'] = np.mean([m['inference_time'] for m in fold_metrics])
            return aggregated_metrics
        else:
            return None
    
    def analyze_cross_dataset_consistency(self):
        """
        Analyze how consistently different model combinations perform across datasets.
        
        Returns:
            Dictionary with consistency metrics for each embedding-classifier pair
        """
        # First ensure we have results from all dataset-embedding-classifier combinations
        if not hasattr(self, 'all_results') or not self.all_results:
            self.run_full_ablation()
        
        # Extract per-dataset results
        per_dataset_results = self.all_results['per_dataset']
        
        # Create a DataFrame to organize results
        rows = []
        for dataset_name, dataset_results in per_dataset_results.items():
            for embedding_name, embedding_results in dataset_results.items():
                for classifier_name, metrics in embedding_results.items():
                    # Use F1 score as the primary performance metric
                    f1_score = metrics.get('mask_f1', 0)
                    rows.append({
                        'Dataset': dataset_name,
                        'Embedding': embedding_name,
                        'Classifier': classifier_name,
                        'F1_Score': f1_score
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Calculate consistency metrics
        consistency_metrics = {}
        
        # Get all embedding-classifier combinations
        model_combinations = df[['Embedding', 'Classifier']].drop_duplicates()
        
        for _, combo in model_combinations.iterrows():
            embedding_name = combo['Embedding']
            classifier_name = combo['Classifier']
            
            # Get F1 scores for this combination across all datasets
            combo_df = df[(df['Embedding'] == embedding_name) & 
                        (df['Classifier'] == classifier_name)]
            
            if len(combo_df) < 2:  # Need at least 2 datasets for comparison
                continue
                
            # Calculate performance metrics
            f1_scores = combo_df['F1_Score'].values
            
            # 1. Performance Variability (coefficient of variation)
            cv = np.std(f1_scores) / np.mean(f1_scores) if np.mean(f1_scores) > 0 else float('inf')
            
            # Store the metrics
            key = f"{embedding_name}_{classifier_name}"
            consistency_metrics[key] = {
                'embedding': embedding_name,
                'classifier': classifier_name,
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'cv': cv,  # Coefficient of variation
                'min_f1': np.min(f1_scores),
                'max_f1': np.max(f1_scores),
                'range': np.max(f1_scores) - np.min(f1_scores)
            }
        
        # Calculate Ranking Correlation
        # For each dataset, rank the model combinations
        datasets = df['Dataset'].unique()
        rankings = {}
        
        for dataset in datasets:
            dataset_df = df[df['Dataset'] == dataset]
            # Rank by F1 score (higher is better, so ascending=False)
            dataset_df['Rank'] = dataset_df['F1_Score'].rank(ascending=False)
            
            for _, row in dataset_df.iterrows():
                combo = f"{row['Embedding']}_{row['Classifier']}"
                if combo not in rankings:
                    rankings[combo] = {}
                rankings[combo][dataset] = row['Rank']
        
        # Calculate Spearman's rank correlation between datasets
        rank_correlations = {}
        for i, dataset1 in enumerate(datasets):
            for dataset2 in datasets[i+1:]:
                # Extract ranks for each model combination
                d1_ranks = []
                d2_ranks = []
                combos = []
                
                for combo in rankings:
                    if dataset1 in rankings[combo] and dataset2 in rankings[combo]:
                        d1_ranks.append(rankings[combo][dataset1])
                        d2_ranks.append(rankings[combo][dataset2])
                        combos.append(combo)
                
                if d1_ranks and d2_ranks:
                    # Calculate Spearman's rank correlation
                    rank_corr, _ = stats.spearmanr(d1_ranks, d2_ranks)
                    rank_correlations[f"{dataset1}_vs_{dataset2}"] = {
                        'correlation': rank_corr,
                        'model_combinations': combos
                    }
        
        # Add the ranking correlations to the results
        consistency_results = {
            'model_consistency': consistency_metrics,
            'ranking_correlations': rank_correlations
        }
        
        # Save results to JSON
        output_path = os.path.join(self.output_dir, 'model_consistency_analysis.json')
        self._save_to_json(consistency_results, output_path)
        
        return consistency_results
    
    def hyperparameter_grid_search(self, dataset_name, embedding_name, classifier_name, grid_config):
        """
        Perform grid search over hyperparameters using concurrent execution
        
        Args:
            dataset_name: Dataset name
            embedding_name: Embedding name
            classifier_name: Classifier name
            grid_config: Dictionary with hyperparameter grids
            
        Returns:
            Dictionary with best configuration and metrics
        """
        # Store current dataset, embedding and classifier names for the worker function
        self.current_dataset = dataset_name
        self.current_embedding = embedding_name
        self.current_classifier = classifier_name
        
        # Get parameter grids
        embedding_grid = grid_config.get(embedding_name, {})
        classifier_grid = grid_config.get(classifier_name, {})
        
        if not embedding_grid and not classifier_grid:
            raise ValueError(f"No hyperparameter grid specified for {embedding_name} and {classifier_name}")
        
        # Get all possible combinations
        embedding_param_grid = list(ParameterGrid(embedding_grid)) if embedding_grid else [{}]
        classifier_param_grid = list(ParameterGrid(classifier_grid)) if classifier_grid else [{}]
        
        # Create all parameter combinations
        param_combinations = []
        for emb_config in embedding_param_grid:
            for cls_config in classifier_param_grid:
                param_combinations.append((emb_config, cls_config))
        
        # Calculate total combinations
        total_configs = len(param_combinations)
        print(f"Grid search with {len(embedding_param_grid)} embedding configs and {len(classifier_param_grid)} classifier configs")
        print(f"Total configurations to try: {total_configs}")
        
        # Track best configuration and performance
        best_config = {'embedding': {}, 'classifier': {}}
        best_f1 = -1
        all_results = []
        
        # Use a ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_idx = {
                executor.submit(self._evaluate_config, i, param_combinations): i
                for i in range(total_configs)
            }
            
            # Process results as they complete
            with tqdm(total=total_configs, desc=f"Grid search {embedding_name}-{classifier_name}") as pbar:
                for future in concurrent.futures.as_completed(future_to_idx):
                    result = future.result()
                    pbar.update(1)
                    
                    if result is not None:
                        all_results.append(result)
                        
                        # Update best configuration if better F1 score
                        metrics = result['metrics']
                        if metrics['mask_f1'] > best_f1:
                            best_f1 = metrics['mask_f1']
                            best_config = {
                                'embedding': result['embedding_config'],
                                'classifier': result['classifier_config'],
                                'metrics': metrics
                            }
                            pbar.set_postfix({"Best F1": f"{best_f1:.4f}"})
        
        # Clean up after grid search
        self.current_dataset = None
        self.current_embedding = None
        self.current_classifier = None
        
        # Save all grid search results
        self._save_grid_search_results(dataset_name, embedding_name, classifier_name, all_results)
        
        return best_config
    
    # First, add this as a new method to the AblationStudy class
    def _process_fold_and_size(self, task, dataset_name, embedding_name, classifier_name,
                            embedding_config, classifier_config):
        """Worker function to process a specific fold and training size"""
        fold_idx, train_size, subset_idx = task
        
        # Get training and validation data
        train_data, val_data = self._get_train_val_data(dataset_name, fold_idx)
        all_train_images, all_train_masks, all_train_labels = train_data
        val_prediction_data, val_gt_masks = val_data
        
        if not all_train_images:
            self.logger.info(f"No training data for fold {fold_idx}")
            return None
        
        # Calculate information content (number of GT masks) for each training image
        image_info_scores = []
        for i, masks in enumerate(all_train_masks):
            # Count the number of ground truth masks (where label is 1)
            gt_mask_count = sum(1 for j, label in enumerate(all_train_labels[i]) 
                            if label == 1)
            image_info_scores.append((i, gt_mask_count))
        
        # Sort by information content (highest number of GT masks first)
        image_info_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [item[0] for item in image_info_scores]
        
        # For the first subset, always use the most informative images
        if subset_idx == 0:
            # Take the top train_size most informative images
            indices = sorted_indices[:train_size]
        else:
            # For additional subsets, select randomly from remaining images
            available_indices = list(set(range(len(all_train_images))) - set(sorted_indices[:train_size]))
            if len(available_indices) < train_size:
                # If not enough remaining images, use random sampling with replacement
                indices = np.random.choice(range(len(all_train_images)), train_size, replace=True)
            else:
                # Otherwise, sample without replacement from remaining images
                indices = np.random.choice(available_indices, train_size, replace=False)
        
        # Create the subset
        subset_images = [all_train_images[i] for i in indices]
        subset_masks = [all_train_masks[i] for i in indices]
        subset_labels = [all_train_labels[i] for i in indices]
        
        # Create model
        model = None
        try:
            model = self.create_model(embedding_name, classifier_name, embedding_config, classifier_config)
            
            # Evaluate model using the helper method
            metrics, val_pred_masks, val_pred_probs = self._evaluate_model(
                model,
                (subset_images, subset_masks, subset_labels),
                (val_prediction_data, val_gt_masks),
                return_predictions=True
            )
            
            # Add metadata
            metrics['fold'] = fold_idx
            metrics['subset'] = subset_idx
            metrics['train_size'] = train_size
            
            # Save precision-recall curve for this fold and subset
            self._save_pr_curve(
                val_gt_masks, val_pred_masks, val_pred_probs,
                f"{dataset_name}_size{train_size}", embedding_name, classifier_name, 
                f"{fold_idx}_{subset_idx}"
            )
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error processing fold {fold_idx}, size {train_size}, subset {subset_idx}: {e}")
            return None
        finally:
            # Use dedicated cleanup helper
            self._cleanup_model(model)

    # Then modify the perform_training_size_analysis method
    def perform_training_size_analysis(self, dataset_name, embedding_name, classifier_name, 
                                    embedding_config=None, classifier_config=None,
                                    training_sizes=None):
        """
        Perform analysis of training set size using concurrent execution
        
        Args:
            dataset_name: Dataset name
            embedding_name: Embedding name
            classifier_name: Classifier name
            embedding_config: Configuration for embedding
            classifier_config: Configuration for classifier
            training_sizes: List of training set sizes to test
            
        Returns:
            Dictionary with metrics for each training size
        """
        training_sizes = training_sizes or [1, 2, 3, 5, 10]
        
        if dataset_name not in self.dataset_managers:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset_manager = self.dataset_managers[dataset_name]
        
        # Get k-fold data loaders 
        fold_dataloaders = dataset_manager.get_kfold_dataloaders(k=self.k_folds, batch_size=1)
        
        # Track metrics across all folds and training sizes
        training_size_metrics = {size: [] for size in training_sizes}
        
        # Create list of all tasks
        tasks = []
        
        # Calculate total iterations
        for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders[1:], 1):
            train_images, _, _ = dataset_manager.get_training_data(train_loader)
            if train_images:
                for train_size in training_sizes:
                    if train_size <= len(train_images):
                        num_subsets = min(5, len(train_images) // train_size)
                        for subset_idx in range(num_subsets):
                            tasks.append((fold_idx, train_size, subset_idx))
        
        # Process tasks with concurrent execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_fold_and_size, task, dataset_name, embedding_name, 
                            classifier_name, embedding_config, classifier_config): task
                for task in tasks
            }
            
            # Process results as they complete
            with tqdm(total=len(tasks), desc=f"Training size analysis {embedding_name}-{classifier_name}") as pbar:
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    fold_idx, train_size, subset_idx = task
                    metrics = future.result()
                    
                    if metrics is not None:
                        training_size_metrics[train_size].append(metrics)
                    
                    pbar.update(1)
                    if metrics:
                        pbar.set_postfix({"F1": f"{metrics.get('mask_f1', 0):.4f}"})
        
        # Aggregate metrics for each training size
        final_metrics = {}
        for train_size, metrics_list in training_size_metrics.items():
            if metrics_list:
                # Group by fold and subset
                subsets_by_fold = defaultdict(list)
                for m in metrics_list:
                    subsets_by_fold[m['fold']].append(m)
                
                # Aggregate subsets for each fold
                fold_metrics = []
                for fold_idx, subset_metrics in subsets_by_fold.items():
                    if subset_metrics:
                        avg_fold_metrics = self._aggregate_subset_metrics(subset_metrics)
                        avg_fold_metrics['fold'] = fold_idx
                        avg_fold_metrics['train_size'] = train_size
                        fold_metrics.append(avg_fold_metrics)
                
                # Aggregate metrics across folds
                if fold_metrics:
                    final_metrics[train_size] = self._aggregate_fold_metrics(fold_metrics)
        
        # Save aggregated metrics
        self._save_training_size_metrics(dataset_name, embedding_name, classifier_name, final_metrics)
        
        # Plot learning curve
        self._plot_learning_curve(dataset_name, embedding_name, classifier_name, final_metrics)
        
        return final_metrics

    def _aggregate_subset_metrics(self, subset_metrics):
        """Aggregate metrics across subsets by taking the mean"""
        # Similar to _aggregate_fold_metrics but for subsets
        if not subset_metrics:
            return None
        
        # Get all metric keys except non-aggregatable ones
        keys = [k for k in subset_metrics[0].keys() if k not in ['fold', 'subset', 'train_size']]
        
        # Calculate mean for each metric
        aggregated = {}
        for key in keys:
            if isinstance(subset_metrics[0][key], list):
                all_values = []
                for metrics in subset_metrics:
                    all_values.extend(metrics[key])
                aggregated[key] = all_values
            else:
                values = [metrics[key] for metrics in subset_metrics]
                aggregated[key] = sum(values) / len(values)
        
        # Add subset count
        aggregated['num_subsets'] = len(subset_metrics)
        
        return aggregated

    def _save_training_size_metrics(self, dataset_name, embedding_name, classifier_name, metrics):
        """Save training size metrics to a JSON file"""
        output_path = os.path.join(
            self.output_dir, 
            f"{dataset_name}_{embedding_name}_{classifier_name}_training_size_metrics.json"
        )
        
        # Convert to serializable format with training size keys as strings
        serializable_metrics = {}
        for train_size, metrics_dict in metrics.items():
            serializable_metrics[str(train_size)] = metrics_dict
        
        # Use the generic JSON saving helper
        self._save_to_json(serializable_metrics, output_path)

    def _plot_learning_curve(self, dataset_name, embedding_name, classifier_name, metrics):
        """Plot learning curve showing how performance scales with training set size"""
        plt.figure(figsize=(10, 6))
        
        # Extract train sizes and F1 scores
        train_sizes = sorted(metrics.keys())
        f1_scores = [metrics[size].get('mask_f1', 0) for size in train_sizes]
        
        # Plot learning curve
        plt.plot(train_sizes, f1_scores, 'o-', linewidth=2)
        plt.xlabel('Training Set Size (Number of Images)')
        plt.ylabel('F1 Score')
        plt.title(f'Learning Curve: {dataset_name} with {embedding_name}-{classifier_name}')
        plt.grid(True)
        
        # Add annotations for each point
        for x, y in zip(train_sizes, f1_scores):
            plt.annotate(f"{y:.3f}", 
                    (x, y), 
                    textcoords="offset points",
                    xytext=(0, 10), 
                    ha='center')
        
        # Save figure
        output_path = os.path.join(
            self.output_dir, 
            f"{dataset_name}_{embedding_name}_{classifier_name}_learning_curve.png"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    
    def run_integrated_ablation(self, grid_config):
        """
        Run integrated ablation study with hyperparameter tuning
        
        Args:
            grid_config: Hyperparameter grid configuration
            
        Returns:
            Dictionary with all results
        """
        # Ensure datasets are loaded
        if not self.dataset_managers:
            self.load_datasets()
        
        # Get configuration components to test
        embeddings = self.config.get('embeddings', ['CLIP', 'HOG', 'ResNet18'])
        classifiers = self.config.get('classifiers', ['KNN', 'SVM', 'RF', 'LR'])
        training_sizes = self.config.get('training_sizes', [1, 2, 3, 5, 10])
        
        # Initialize results structures
        results = {}
        training_size_results = {}
        best_configs = {}
        
        
        # Process one complete combination at a time
        for dataset_name in self.dataset_managers.keys():
            results[dataset_name] = {}
            training_size_results[dataset_name] = {}
            best_configs[dataset_name] = {}
            
            for embedding_name in embeddings:
                results[dataset_name][embedding_name] = {}
                training_size_results[dataset_name][embedding_name] = {}
                best_configs[dataset_name][embedding_name] = {}
                
                for classifier_name in classifiers:
                    print(f"\n{'='*80}")
                    print(f"Processing {dataset_name}, {embedding_name}, {classifier_name}")
                    print(f"{'='*80}")
                    
                    # Step 1: Hyperparameter tuning on fold 0
                    print(f"\nPerforming hyperparameter tuning on fold 0...")
                    best_config = self.hyperparameter_grid_search(
                        dataset_name, embedding_name, classifier_name, grid_config
                    )
                    
                    best_configs[dataset_name][embedding_name][classifier_name] = best_config
                    self._save_best_config(dataset_name, embedding_name, classifier_name, best_config)
                    
                    # Extract the best parameters
                    best_embedding_config = best_config['embedding']
                    best_classifier_config = best_config['classifier']
                    
                    # Free memory after tuning
                    gc.collect()
                    
                    # Step 2: Standard cross-validation with best parameters
                    print(f"\nRunning full-fold ablation with optimal parameters...")
                    cv_metrics = self.perform_cross_validation(
                        dataset_name, embedding_name, classifier_name,
                        embedding_config=best_embedding_config,
                        classifier_config=best_classifier_config
                    )
                    
                    if cv_metrics:
                        results[dataset_name][embedding_name][classifier_name] = cv_metrics
                        self._save_metrics(dataset_name, embedding_name, classifier_name, cv_metrics)
                    
                    # Free memory between steps
                    gc.collect()
                    
                    # Step 3: Training size analysis with best parameters
                    print(f"\nRunning training size analysis with optimal parameters...")
                    size_metrics = self.perform_training_size_analysis(
                        dataset_name, embedding_name, classifier_name,
                        embedding_config=best_embedding_config,
                        classifier_config=best_classifier_config,
                        training_sizes=training_sizes
                    )
                    
                    if size_metrics:
                        training_size_results[dataset_name][embedding_name][classifier_name] = size_metrics
                    
                    # Free memory after completing this combination
                    gc.collect()
                    print(f"Completed {dataset_name}, {embedding_name}, {classifier_name}")
        
        # Store all results for later analysis
        self.all_results = {
            'per_dataset': results,
            'training_size': training_size_results,
            'best_configs': best_configs
        }
        
        
        # Analyze cross-dataset consistency
        print("\nAnalyzing cross-dataset consistency...")
        consistency_results = self.analyze_cross_dataset_consistency()
        

        # Run pipeline comparison
        print("\nRunning pipeline comparison (ideal vs. realistic)...")
        comparison_results = self.run_pipeline_comparison()
        
        # Update all_results to include comparison
        all_results['pipeline_comparison'] = comparison_results
        
        # Complete results
        all_results = {
            'per_dataset': results,
            'training_size': training_size_results,
            'best_configs': best_configs,
            'consistency': consistency_results,
            'pipeline_comparison': comparison_results
        }
        
        return all_results
    
    def run_hyperparameter_study(self, grid_config):
        """
        Run hyperparameter study for specified configurations
        
        Args:
            grid_config: Dictionary mapping embedding/classifier names to parameter grids
            
        Returns:
            Dictionary with best configurations for each combination
        """
        # Ensure datasets are loaded
        if not self.dataset_managers:
            self.load_datasets()
        
        # Get what to test
        hp_study_config = self.config.get('hyperparameter_study', {})
        datasets = hp_study_config.get('datasets', list(self.dataset_managers.keys()))
        embeddings = hp_study_config.get('embeddings', ['CLIP', 'HOG', 'ResNet18'])
        classifiers = hp_study_config.get('classifiers', ['KNN', 'SVM', 'RF', 'LR'])
        
        # Track best configurations
        best_configs = {}
        
        # Run grid search for each combination
        for dataset_name in datasets:
            best_configs[dataset_name] = {}
            
            for embedding_name in embeddings:
                best_configs[dataset_name][embedding_name] = {}
                
                for classifier_name in classifiers:
                    print(f"\nRunning hyperparameter grid search for {dataset_name}, {embedding_name}, {classifier_name}")
                    
                    best_config = self.hyperparameter_grid_search(
                        dataset_name, embedding_name, classifier_name, grid_config
                    )
                    
                    best_configs[dataset_name][embedding_name][classifier_name] = best_config
                    
                    # Save best config
                    self._save_best_config(dataset_name, embedding_name, classifier_name, best_config)
        
        return best_configs
    
    def _aggregate_fold_metrics(self, fold_metrics):
        """Aggregate metrics across folds by taking the mean"""
        if not fold_metrics:
            return None
            
        # Get all metric keys except 'fold'
        keys = [k for k in fold_metrics[0].keys() if k != 'fold']
        
        # Calculate mean for each metric
        aggregated = {}
        for key in keys:
            # Handle metrics that are lists (like best_ious)
            if isinstance(fold_metrics[0][key], list):
                # Flatten all lists
                all_values = []
                for metrics in fold_metrics:
                    all_values.extend(metrics[key])
                aggregated[key] = all_values
            else:
                values = [metrics[key] for metrics in fold_metrics]
                aggregated[key] = sum(values) / len(values)
        
        # Add fold count
        aggregated['num_folds'] = len(fold_metrics)
        
        return aggregated
    
    def _sort_by_probability(self, masks, probabilities):
        """Sort masks by their probabilities in descending order"""
        pairs = sorted(zip(masks, probabilities), key=lambda x: x[1], reverse=True)
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    
    def _save_metrics(self, dataset_name, embedding_name, classifier_name, metrics):
        """Save metrics to a JSON file"""
        output_path = os.path.join(
            self.output_dir, 
            f"{dataset_name}_{embedding_name}_{classifier_name}_metrics.json"
        )
        self._save_to_json(metrics, output_path)

    def _save_grid_search_results(self, dataset_name, embedding_name, classifier_name, results):
        """Save grid search results to a JSON file"""
        output_path = os.path.join(
            self.output_dir, 
            f"{dataset_name}_{embedding_name}_{classifier_name}_grid_search.json"
        )
        self._save_to_json(results, output_path)

    
    def _save_best_config(self, dataset_name, embedding_name, classifier_name, best_config):
        """Save best configuration to a JSON file"""
        output_path = os.path.join(
            self.output_dir, 
            f"{dataset_name}_{embedding_name}_{classifier_name}_best_config.json"
        )
        self._save_to_json(best_config, output_path)
    
    def _save_pr_curve(self, gt_masks, pred_masks, pred_probs, dataset_name, embedding_name, classifier_name, fold_idx):
        """Save precision-recall curve for a specific fold"""
        if not gt_masks or not pred_masks or not pred_probs:
            return
        
        # Create PR curve
        pr_curve = plot_precision_recall_curve(gt_masks, pred_masks, pred_probs)
        
        # Save figure
        output_path = os.path.join(
            self.output_dir, 
            f"{dataset_name}_{embedding_name}_{classifier_name}_fold{fold_idx}_pr_curve.png"
        )
        pr_curve.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(pr_curve)
        
        # Create threshold vs metrics curve
        threshold_curve = plot_threshold_vs_metrics(gt_masks, pred_masks, pred_probs)
        
        # Save figure
        output_path = os.path.join(
            self.output_dir, 
            f"{dataset_name}_{embedding_name}_{classifier_name}_fold{fold_idx}_threshold_curve.png"
        )
        threshold_curve.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(threshold_curve)


def get_project_root():
    # Path to the directory containing the script being run
    current_file = Path(os.path.abspath(__file__))
    # Go up from src/master/script.py to reach the project root
    return current_file.parent.parent.parent

# Create configuration with proper paths
def create_default_config():
    project_root = get_project_root()
    processed_data_dir = project_root / "data" / "processed"
    
    config = {
        'dataset_paths': {
            'meatballs': str(processed_data_dir / 'meatballs'),
            'cans': str(processed_data_dir / 'cans'),
            'doughs': str(processed_data_dir / 'doughs'),
            'bottles': str(processed_data_dir / 'bottles'),
        },
        'class_ids': {
            'meatballs': 1,
            'cans': 1,
            'doughs': 1,
            'bottles': 1,
        },
        'embeddings': ['CLIP', 'HOG', 'ResNet18'],
        'classifiers': ['KNN', 'SVM', 'RF', 'LR'],
        'output_dir': str(project_root / 'ablation_results'),
        'k_folds': 5,
        'hyperparameter_study': {
            'datasets': ['meatballs', 'cans', 'doughs', 'bottles'],
            'embeddings': ['CLIP', 'HOG', 'ResNet18'],
            'classifiers': ['KNN', 'SVM', 'RF', 'LR'],
        }
    }
    return config

def create_hyperparameter_grid():
    """Create a hyperparameter grid for grid search"""
    grid_config = {
        # Embedding hyperparameters
        'CLIP': {
            'padding': [0, 5, 10],
            'clip_model': ['ViT-B/32']
        },
        'HOG': {
            'padding': [0, 5, 10],
            'hog_cell_size': [(8, 8), (16, 16)],
            'hog_block_size': [(2, 2), (3, 3)]
        },
        'ResNet18': {
            'padding': [0, 5, 10],
            'layers': [[2, 4, 6, 8]]
        },
        
        # Classifier hyperparameters
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'metric': ['cosine', 'manhattan'],
            'use_PCA': [True],
            'PCA_var': [0.9, 0.95, 0.99]
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'use_PCA': [True],
            'PCA_var': [0.95, 0.99]
        },
        'RF': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None],
            'min_samples_split': [2, 5],
            'criterion': ['gini'],
        },
        'LR': {
            'C': [0.01, 0.1, 1.0],
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2'],
            'max_iter': [1000],
            'class_weight': [None, 'balanced'],
        }
    }
    return grid_config

def main():
    """Main function to run the ablation study"""
    # Create configuration
    config = create_default_config()
    
    # Create ablation study instance
    study = AblationStudy(config)
    
    # Load datasets
    study.load_datasets()
    
    # Create hyperparameter grid
    grid_config = create_hyperparameter_grid()
    
    # Run integrated ablation study with hyperparameter tuning
    print("Running integrated ablation study...")
    results = study.run_integrated_ablation(grid_config)
    
    # Generate comprehensive report
    print("\nGenerating final report...")
    report_generator = SegmentationReportGenerator(config['output_dir'])
    report_generator.generate_full_report(results, study.dataset_managers)
    
    print(f"Ablation study complete. Results saved to {config['output_dir']}")
    
    return results

if __name__ == "__main__":
    main()