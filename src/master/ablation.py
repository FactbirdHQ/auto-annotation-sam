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
            print(f"Loading dataset: {name} from {path}")
            try:
                # Initialize dataset manager with proper class ID
                class_id = self.config.get('class_ids', {}).get(name, 1)
                dataset_manager = KFoldSegmentationManager(path, class_id=class_id)
                self.dataset_managers[name] = dataset_manager
                
                # Print dataset info
                info = dataset_manager.get_dataset_info()
                print(f"  - Total samples: {info['total_samples']}")
                print(f"  - Class ID: {info['class_id']}")
                
            except Exception as e:
                print(f"Error loading dataset {name}: {e}")
        
        return self.dataset_managers
    
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
        
        # Use empty dict if configs are None
        embedding_config = embedding_config or {}
        classifier_config = classifier_config or {}
        
        # Create embedding and classifier
        embedding = self.embedding_factory[embedding_name](embedding_config)
        classifier = self.classifier_factory[classifier_name](classifier_config, embedding)
        
        return classifier
    
    def perform_cross_validation(self, dataset_name, embedding_name, classifier_name, 
                        embedding_config=None, classifier_config=None):
        """
        Perform k-fold cross-validation for a specific dataset, embedding, and classifier
        """
        if dataset_name not in self.dataset_managers:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset_manager = self.dataset_managers[dataset_name]
        
        # Get k-fold data loaders
        fold_dataloaders = dataset_manager.get_kfold_dataloaders(k=self.k_folds, batch_size=1)
        
        # Track metrics across all folds
        fold_metrics = []
        training_times = []
        inference_times = []
        
        # Add tqdm progress bar for fold processing
        from tqdm import tqdm
        for fold_idx, (train_loader, val_loader) in tqdm(enumerate(fold_dataloaders), 
                                                        total=self.k_folds, 
                                                        desc=f"CV {embedding_name}-{classifier_name}"):
            print(f"\nProcessing fold {fold_idx+1}/{self.k_folds}")
            
            # Create model for this fold
            model = None
            try:
                model = self.create_model(embedding_name, classifier_name, embedding_config, classifier_config)
                
                # Extract training data
                train_images, train_masks, train_labels = dataset_manager.get_training_data(train_loader)
                
                if not train_images:
                    print(f"No training data for fold {fold_idx+1}")
                    continue
                
                # Time the training process
                start_time = time.perf_counter()
                model.fit(train_images, train_masks, train_labels)
                train_time = time.perf_counter() - start_time
                training_times.append(train_time)
                print(f"Training completed in {train_time:.2f} seconds")
                
                # Get validation data
                val_prediction_data = dataset_manager.get_prediction_data(val_loader)
                
                if not val_prediction_data:
                    print(f"No validation data for fold {fold_idx+1}")
                    continue
                
                # Collect ground truth masks for validation
                val_gt_masks = []
                val_pred_masks = []
                val_pred_probs = []
                
                # Time the inference process
                total_inference_time = 0
                num_predictions = 0
                
                # Process each validation sample
                for val_idx, (image, candidate_masks) in enumerate(val_prediction_data):
                    # Get ground truth masks (get them from the validation loader)
                    for batch in val_loader:
                        batch_images = batch['image']
                        batch_gt_masks = batch['gt_binary_mask']
                        batch_filenames = batch['filename']
                        
                        for i, filename in enumerate(batch_filenames):
                            if i == val_idx:  # Simple matching for this example
                                gt_mask = batch_gt_masks[i].numpy() if hasattr(batch_gt_masks[i], 'numpy') else batch_gt_masks[i]
                                val_gt_masks.append(gt_mask)
                                break
                    
                    # Time the prediction process for this sample
                    if candidate_masks:
                        start_time = time.perf_counter()
                        pred_results, probs = model.predict(image, candidate_masks)
                        inference_time = time.perf_counter() - start_time
                        total_inference_time += inference_time
                        num_predictions += 1
                        
                        for (mask, pred_class), prob in zip(pred_results, probs):
                            # Only keep masks predicted as positive (class 1)
                            if pred_class == 1:
                                val_pred_masks.append(mask)
                                val_pred_probs.append(prob[1] if len(prob) > 1 else prob[0])
                
                # Calculate average inference time
                avg_inference_time = total_inference_time / max(1, num_predictions)
                inference_times.append(avg_inference_time)
                print(f"Average inference time per sample: {avg_inference_time:.4f} seconds")
                
                # Sort masks by probability
                if val_pred_masks and val_pred_probs:
                    val_pred_masks, val_pred_probs = self._sort_by_probability(val_pred_masks, val_pred_probs)
                
                # Evaluate predictions
                if val_gt_masks and val_pred_masks:
                    metrics = evaluate_binary_masks(val_gt_masks, val_pred_masks)
                    metrics['fold'] = fold_idx
                    metrics['training_time'] = train_time
                    metrics['inference_time'] = avg_inference_time
                    fold_metrics.append(metrics)
                    
                    # Save precision-recall curve for this fold
                    self._save_pr_curve(val_gt_masks, val_pred_masks, val_pred_probs, 
                                    dataset_name, embedding_name, classifier_name, fold_idx)
                else:
                    print(f"No ground truth or predicted masks for evaluation in fold {fold_idx+1}")
            finally:
                # Clean up resources even if an exception occurred
                if model is not None:
                    # Clean up any hooks if the embedding has a cleanup method
                    if hasattr(model.embedding, 'cleanup'):
                        model.embedding.cleanup()
                    
                    # Delete model and force garbage collection
                    del model
                    gc.collect()
        
        # Aggregate metrics across folds
        if fold_metrics:
            aggregated_metrics = self._aggregate_fold_metrics(fold_metrics)
            # Add average timing information
            aggregated_metrics['avg_training_time'] = np.mean(training_times)
            aggregated_metrics['avg_inference_time'] = np.mean(inference_times)
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
        with open(output_path, 'w') as f:
            # Convert numpy values to native Python types
            serializable_results = json.loads(
                json.dumps(consistency_results, default=lambda x: x.item() if isinstance(x, (np.integer, np.floating)) else x)
            )
            json.dump(serializable_results, f, indent=4)
        
        return consistency_results
    
    def hyperparameter_grid_search(self, dataset_name, embedding_name, classifier_name, grid_config):
        """
        Perform grid search over hyperparameters for a specific dataset, embedding, and classifier
        
        Args:
            dataset_name: Name of the dataset
            embedding_name: Name of the embedding
            classifier_name: Name of the classifier
            grid_config: Dictionary with parameter grids for embedding and classifier
        
        Returns:
            Best configuration based on average F1 score
        """
        if dataset_name not in self.dataset_managers:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset_manager = self.dataset_managers[dataset_name]
        
        # Get parameter grids
        embedding_grid = grid_config.get(embedding_name, {})
        classifier_grid = grid_config.get(classifier_name, {})
        
        if not embedding_grid and not classifier_grid:
            raise ValueError(f"No hyperparameter grid specified for {embedding_name} and {classifier_name}")
        
        # Get all possible combinations
        embedding_param_grid = list(ParameterGrid(embedding_grid)) if embedding_grid else [{}]
        classifier_param_grid = list(ParameterGrid(classifier_grid)) if classifier_grid else [{}]
        
        # Calculate total combinations
        total_configs = len(embedding_param_grid) * len(classifier_param_grid)
        print(f"Grid search with {len(embedding_param_grid)} embedding configs and {len(classifier_param_grid)} classifier configs")
        print(f"Total configurations to try: {total_configs}")
        
        # Track best configuration and performance
        best_config = {'embedding': {}, 'classifier': {}}
        best_f1 = -1
        all_results = []
        
        # Split dataset for grid search (use first fold for simplicity)
        fold_dataloaders = dataset_manager.get_kfold_dataloaders(k=self.k_folds, batch_size=1)
        train_loader, val_loader = fold_dataloaders[0]
        
        # Get training and validation data
        train_images, train_masks, train_labels = dataset_manager.get_training_data(train_loader)
        val_prediction_data = dataset_manager.get_prediction_data(val_loader)
        
        # Get ground truth masks for validation
        val_gt_masks = []
        for batch in val_loader:
            batch_gt_masks = batch['gt_binary_mask']
            batch_has_gt = batch['has_gt']
            
            for i, has_gt in enumerate(batch_has_gt):
                if has_gt:
                    gt_mask = batch_gt_masks[i].numpy() if hasattr(batch_gt_masks[i], 'numpy') else batch_gt_masks[i]
                    val_gt_masks.append(gt_mask)

        # Create tqdm progress bar for the grid search
        with tqdm(total=total_configs, desc=f"Grid search {embedding_name}-{classifier_name}") as pbar:
            # Try each combination of parameters
            for emb_config in embedding_param_grid:
                for cls_config in classifier_param_grid:
                    # Create model with current configuration
                    model = None
                    try:
                        model = self.create_model(embedding_name, classifier_name, emb_config, cls_config)
                        
                        # Train model
                        model.fit(train_images, train_masks, train_labels)
                        
                        # Collect predictions
                        val_pred_masks = []
                        val_pred_probs = []
                        
                        for image, candidate_masks in val_prediction_data:
                            if candidate_masks:
                                pred_results, probs = model.predict(image, candidate_masks)
                                
                                for (mask, pred_class), prob in zip(pred_results, probs):
                                    if pred_class == 1:
                                        val_pred_masks.append(mask)
                                        val_pred_probs.append(prob[1] if len(prob) > 1 else prob[0])
                        
                        # Evaluate predictions
                        if val_pred_masks:
                            metrics = evaluate_binary_masks(val_gt_masks, val_pred_masks)
                            
                            # Save results
                            result = {
                                'embedding_config': emb_config,
                                'classifier_config': cls_config,
                                'metrics': metrics
                            }
                            all_results.append(result)
                            
                            # Update best configuration if better F1 score
                            if metrics['mask_f1'] > best_f1:
                                best_f1 = metrics['mask_f1']
                                best_config = {
                                    'embedding': emb_config,
                                    'classifier': cls_config,
                                    'metrics': metrics
                                }
                                
                                # Update progress bar with best F1 score
                                pbar.set_postfix({"Best F1": f"{best_f1:.4f}"})
                                
                    except Exception as e:
                        print(f"\nError with config - embedding: {emb_config}, classifier: {cls_config}")
                        print(f"Error: {e}")
                    finally:
                        # Clean up resources even if an exception occurred
                        if model is not None:
                            # Clean up any hooks if the embedding has a cleanup method
                            if hasattr(model.embedding, 'cleanup'):
                                model.embedding.cleanup()
                            
                            # Delete model and force garbage collection
                            del model
                            gc.collect()
                        
                        # Update progress bar
                        pbar.update(1)
        
        # Save all grid search results
        self._save_grid_search_results(dataset_name, embedding_name, classifier_name, all_results)
        
        return best_config
    
    def perform_training_size_analysis(self, dataset_name, embedding_name, classifier_name, 
                              embedding_config=None, classifier_config=None,
                              training_sizes=[1, 2, 3, 5, 10]):
        """
        Perform analysis of how model performance scales with training set size
        
        Args:
            dataset_name: Name of the dataset
            embedding_name: Name of the embedding
            classifier_name: Name of the classifier
            embedding_config: Configuration for the embedding
            classifier_config: Configuration for the classifier
            training_sizes: List of training set sizes to evaluate
            
        Returns:
            Dictionary of evaluation metrics for each training size
        """
        
        if dataset_name not in self.dataset_managers:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset_manager = self.dataset_managers[dataset_name]
        
        # Get k-fold data loaders
        fold_dataloaders = dataset_manager.get_kfold_dataloaders(k=self.k_folds, batch_size=1)
        
        # Track metrics across all folds and training sizes
        training_size_metrics = {size: [] for size in training_sizes}
        
        # Calculate total iterations
        total_iterations = 0
        for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders[1:], 1):
            train_images, _, _ = dataset_manager.get_training_data(train_loader)
            if train_images:
                for train_size in training_sizes:
                    if train_size <= len(train_images):
                        num_subsets = min(5, len(train_images) // train_size)
                        total_iterations += num_subsets
        
        # Create progress bar for overall progress
        with tqdm(total=total_iterations, desc=f"Training size analysis {embedding_name}-{classifier_name}") as pbar:
            # For each fold (except one reserved for hyperparameter tuning)
            for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders[1:], 1):
                print(f"\nProcessing fold {fold_idx}/{self.k_folds}")
                
                # Extract all training data from this fold
                all_train_images, all_train_masks, all_train_labels = dataset_manager.get_training_data(train_loader)
                
                if not all_train_images:
                    print(f"No training data for fold {fold_idx}")
                    continue
                
                # Get validation data
                val_prediction_data = dataset_manager.get_prediction_data(val_loader)
                val_gt_masks = []
                
                # Extract ground truth masks for validation
                for batch in val_loader:
                    batch_gt_masks = batch['gt_binary_mask']
                    batch_has_gt = batch.get('has_gt', [True] * len(batch_gt_masks))
                    
                    for i, has_gt in enumerate(batch_has_gt):
                        if has_gt:
                            gt_mask = batch_gt_masks[i].numpy() if hasattr(batch_gt_masks[i], 'numpy') else batch_gt_masks[i]
                            val_gt_masks.append(gt_mask)
                
                # For each training size
                for train_size in training_sizes:
                    if train_size > len(all_train_images):
                        print(f"Training size {train_size} exceeds available data ({len(all_train_images)})")
                        continue
                        
                    # Create multiple subsets for each training size
                    num_subsets = min(5, len(all_train_images) // train_size)
                    subset_metrics = []
                    
                    for subset_idx in range(num_subsets):
                        # Create random subset of specified size
                        indices = np.random.choice(len(all_train_images), train_size, replace=False)
                        subset_images = [all_train_images[i] for i in indices]
                        subset_masks = [all_train_masks[i] for i in indices]
                        subset_labels = [all_train_labels[i] for i in indices]
                        
                        # Create and train model on this subset
                        model = None
                        try:
                            model = self.create_model(embedding_name, classifier_name, embedding_config, classifier_config)
                            
                            # Update progress bar description with current details
                            pbar.set_description(f"Size: {train_size} | Subset: {subset_idx+1}/{num_subsets}")
                            
                            # Train model
                            model.fit(subset_images, subset_masks, subset_labels)
                            
                            # Evaluate on validation data
                            val_pred_masks = []
                            val_pred_probs = []
                            
                            for image, candidate_masks in val_prediction_data:
                                if candidate_masks:
                                    pred_results, probs = model.predict(image, candidate_masks)
                                    
                                    for (mask, pred_class), prob in zip(pred_results, probs):
                                        if pred_class == 1:
                                            val_pred_masks.append(mask)
                                            val_pred_probs.append(prob[1] if len(prob) > 1 else prob[0])
                            
                            # Sort masks by probability
                            if val_pred_masks and val_pred_probs:
                                val_pred_masks, val_pred_probs = self._sort_by_probability(val_pred_masks, val_pred_probs)
                            
                            # Evaluate predictions
                            if val_gt_masks and val_pred_masks:
                                metrics = evaluate_binary_masks(val_gt_masks, val_pred_masks)
                                metrics['fold'] = fold_idx
                                metrics['subset'] = subset_idx
                                metrics['train_size'] = train_size
                                subset_metrics.append(metrics)
                                
                                # Update progress bar with F1 score
                                pbar.set_postfix({"F1": f"{metrics['mask_f1']:.4f}"})
                                
                                # Save precision-recall curve for this fold and subset
                                self._save_pr_curve(
                                    val_gt_masks, val_pred_masks, val_pred_probs,
                                    f"{dataset_name}_size{train_size}", embedding_name, classifier_name, 
                                    f"{fold_idx}_{subset_idx}"
                                )
                        finally:
                            # Clean up resources even if an exception occurred
                            if model is not None:
                                # Clean up any hooks if the embedding has a cleanup method
                                if hasattr(model.embedding, 'cleanup'):
                                    model.embedding.cleanup()
                                
                                # Delete model and force garbage collection
                                del model
                                gc.collect()
                            
                            # Update progress bar
                            pbar.update(1)
                    
                    # Aggregate metrics across subsets for this training size and fold
                    if subset_metrics:
                        avg_metrics = self._aggregate_subset_metrics(subset_metrics)
                        avg_metrics['fold'] = fold_idx
                        avg_metrics['train_size'] = train_size
                        training_size_metrics[train_size].append(avg_metrics)
            
            # Aggregate metrics across folds for each training size
            final_metrics = {}
            for train_size, metrics_list in training_size_metrics.items():
                if metrics_list:
                    final_metrics[train_size] = self._aggregate_fold_metrics(metrics_list)
            
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
        
        # Convert to serializable format
        serializable_metrics = {}
        for train_size, metrics_dict in metrics.items():
            serializable_metrics[str(train_size)] = {}
            for key, value in metrics_dict.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[str(train_size)][key] = value.item()
                elif isinstance(value, list) and value and isinstance(value[0], (np.integer, np.floating)):
                    serializable_metrics[str(train_size)][key] = [v.item() for v in value]
                else:
                    serializable_metrics[str(train_size)][key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

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
        
        # Create the report generator once
        report_generator = SegmentationReportGenerator(self.output_dir)
        
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
        
        # Generate the full report
        print("\nGenerating report...")
        report_generator.generate_full_report(self.all_results, self.dataset_managers)
        
        # Analyze cross-dataset consistency
        print("\nAnalyzing cross-dataset consistency...")
        consistency_results = self.analyze_cross_dataset_consistency()
        
        # Add consistency to the report
        report_generator.plot_consistency_metrics(consistency_results)
        report_generator.add_consistency_section(consistency_results)
        
        # Complete results
        all_results = {
            'per_dataset': results,
            'training_size': training_size_results,
            'best_configs': best_configs,
            'consistency': consistency_results
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
        
        # Convert numpy values to Python native types
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = value.item()
            elif isinstance(value, list) and value and isinstance(value[0], (np.integer, np.floating)):
                serializable_metrics[key] = [v.item() for v in value]
            else:
                serializable_metrics[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

    def _save_grid_search_results(self, dataset_name, embedding_name, classifier_name, results):
        """Save grid search results to a JSON file"""
        output_path = os.path.join(
            self.output_dir, 
            f"{dataset_name}_{embedding_name}_{classifier_name}_grid_search.json"
        )
        
        # Convert numpy values and make serializable
        serializable_results = []
        for result in results:
            serializable_result = {
                'embedding_config': {k: v for k, v in result['embedding_config'].items()},
                'classifier_config': {k: v for k, v in result['classifier_config'].items()},
                'metrics': {}
            }
            
            for key, value in result['metrics'].items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_result['metrics'][key] = value.item()
                elif isinstance(value, list) and value and isinstance(value[0], (np.integer, np.floating)):
                    serializable_result['metrics'][key] = [v.item() for v in value]
                else:
                    serializable_result['metrics'][key] = value
            
            serializable_results.append(serializable_result)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    def _save_best_config(self, dataset_name, embedding_name, classifier_name, best_config):
        """Save best configuration to a JSON file"""
        output_path = os.path.join(
            self.output_dir, 
            f"{dataset_name}_{embedding_name}_{classifier_name}_best_config.json"
        )
        
        # Convert numpy values and make serializable
        serializable_config = {
            'embedding': {k: v for k, v in best_config['embedding'].items()},
            'classifier': {k: v for k, v in best_config['classifier'].items()},
            'metrics': {}
        }
        
        for key, value in best_config['metrics'].items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_config['metrics'][key] = value.item()
            elif isinstance(value, list) and value and isinstance(value[0], (np.integer, np.floating)):
                serializable_config['metrics'][key] = [v.item() for v in value]
            else:
                serializable_config['metrics'][key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_config, f, indent=4)
    
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


def create_default_config():
    """Create a default configuration for ablation study"""
    config = {
        'dataset_paths': {
            'meatballs': 'C:/Users/gtoft/OneDrive/DTU/4_Semester_AS/Master_Thesis/data/sam_inference/processed_data/meatballs',
            'cans': 'C:/Users/gtoft/OneDrive/DTU/4_Semester_AS/Master_Thesis/data/sam_inference/processed_data/cans',
            'doughs': 'C:/Users/gtoft/OneDrive/DTU/4_Semester_AS/Master_Thesis/data/sam_inference/processed_data/doughs',
            'bottles': 'C:/Users/gtoft/OneDrive/DTU/4_Semester_AS/Master_Thesis/data/sam_inference/processed_data/bottles',
        },
        'class_ids': {
            'meatballs': 1,
            'cans': 1,
            'doughs': 1,
            'bottles': 1,
        },
        'embeddings': ['CLIP', 'HOG', 'ResNet18'],
        'classifiers': ['KNN', 'SVM', 'RF', 'LR'],
        'output_dir': './ablation_results',
        'k_folds': 6,
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
            'padding': [0, 5, 10, 20],
            'clip_model': ['ViT-B/32', 'ViT-B/16']
        },
        'HOG': {
            'padding': [0, 5, 10, 20],
            'hog_cell_size': [(8, 8), (16, 16)],
            'hog_block_size': [(2, 2), (3, 3)]
        },
        'ResNet18': {
            'padding': [0, 5, 10, 20],
            'layers': [[2, 4, 6, 8], [4, 8]]
        },
        
        # Classifier hyperparameters
        'KNN': {
            'n_neighbors': [3, 5, 7, 9],
            'metric': ['cosine', 'euclidean', 'manhattan'],
            'use_PCA': [True, False],
            'PCA_var': [0.9, 0.95, 0.99]
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['rbf', 'linear', 'poly'],
            'use_PCA': [True, False],
            'PCA_var': [0.9, 0.95, 0.99]
        },
        'RF': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'use_PCA': [True, False],
            'PCA_var': [0.9, 0.95, 0.99]
        },
        'LR': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced'],
            'use_PCA': [True, False],
            'PCA_var': [0.9, 0.95, 0.99]
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
    
    print(f"Ablation study complete. Results saved to {config['output_dir']}")
    
    return results

if __name__ == "__main__":
    main()