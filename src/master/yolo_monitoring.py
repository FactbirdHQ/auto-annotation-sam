import numpy as np
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch.nn.functional as F

class MCDropoutTracker:
    def __init__(self, model_path, dropout_rate=0.05):
        self.model_path = model_path
        self.dropout_rate = dropout_rate
        # Initialize model for immediate use if needed
        self.model = self._create_fresh_model()
        
    def _create_fresh_model(self):
        """Create a fresh model instance with dropout applied"""
        # Load a new model instance
        model = YOLO(self.model_path)
        
        # Find detection head
        detect_head = None
        for module in model.model.modules():
            if 'Detect' in str(type(module)):
                detect_head = module
                break
                
        if detect_head is None:
            raise ValueError("Could not find detection head")
            
        # Store original forward method
        original_forward = detect_head.forward
        
        # Define new forward with dropout
        def forward_with_dropout(x, *args, **kwargs):
            # Apply dropout to feature maps
            if isinstance(x, list):
                x = [F.dropout(xi, p=self.dropout_rate, training=True) for xi in x]
            else:
                x = F.dropout(x, p=self.dropout_rate, training=True)
            
            # Call original forward
            return original_forward(x, *args, **kwargs)
            
        # Replace forward method
        detect_head.forward = forward_with_dropout
        
        # Set to train mode to enable dropout
        model.model.train()
        
        return model
    
    def fine_tune(self, data_yaml, epochs=5, batch_size=16, reset_model=False, **kwargs):
        """Fine-tune the model with dropout enabled
        
        Args:
            data_yaml: Path to data YAML file
            epochs: Number of training epochs
            batch_size: Batch size for training
            reset_model: Whether to create a fresh model before fine-tuning
            **kwargs: Additional arguments for YOLO training
        """
        # Create a fresh model if requested or if model doesn't exist
        if reset_model or not hasattr(self, 'model'):
            self.model = self._create_fresh_model()
        
        # Make sure dropout is active during training
        self.model.model.train()
        
        # Start fine-tuning
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            **kwargs
        )
        
        return results
    
    def track(self, source, fresh_model=True, **kwargs):
        """Run tracking with MC dropout enabled
        
        Args:
            source: Source for tracking (video, image, etc.)
            fresh_model: Whether to create a fresh model for this tracking session
            **kwargs: Additional arguments for YOLO tracking
        """
        # Create a fresh model if requested
        if fresh_model:
            tracking_model = self._create_fresh_model()
        else:
            # Use existing model (ensure it's in train mode for dropout)
            if not hasattr(self, 'model'):
                self.model = self._create_fresh_model()
            tracking_model = self.model
            tracking_model.model.train()
        
        # Run tracking
        return tracking_model.track(source, **kwargs)

def run_mc_tracking_analysis(model_path, video_path, num_runs=10, dropout_rate=0.01):
    """
    Run MC dropout tracking multiple times and collect statistics
    
    Args:
        model_path: Path to the YOLOv8 model
        video_path: Path to the video file
        num_runs: Number of MC samples to collect
        dropout_rate: Dropout rate to use
        
    Returns:
        DataFrame with statistics on confidence and unique IDs,
        raw confidence scores, track lifespans, and frame confidence variances
    """
    print(f"Starting MC tracking analysis with {num_runs} runs...")
    
    # Storage for results
    all_confidences = []
    all_unique_ids = []
    all_track_lifespans = defaultdict(list)  # Track how long each ID persists
    all_frame_confidence_data = defaultdict(lambda: defaultdict(list))  # Store confidences by frame and run
    
    # Run tracking multiple times
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}...")
        
        # Create tracker with dropout
        mc_tracker = MCDropoutTracker(model_path, dropout_rate=dropout_rate)
        
        # Run tracking
        results = mc_tracker.track(
            video_path,
            conf=0.3,  # Lower to capture more detections for analysis
            iou=0.45,
            persist=True,
            verbose=False
        )
        
        # Process results
        confidences = []
        track_ids = set()
        track_appearances = defaultdict(int)  # Count frames each ID appears in
        
        for i, frame_result in enumerate(results):
            if frame_result.boxes.id is not None:
                # Extract track IDs for this frame
                frame_ids = frame_result.boxes.id.cpu().numpy().tolist()
                frame_confs = frame_result.boxes.conf.cpu().numpy().tolist()
                
                # Update tracked IDs
                track_ids.update(frame_ids)
                
                # Count frame appearances for each ID
                for track_id in frame_ids:
                    track_appearances[track_id] += 1
                    
                # Add confidence scores
                confidences.extend(frame_confs)
                
                # Store confidences by frame for variance analysis
                all_frame_confidence_data[run][i].extend(frame_confs)
        
        # Store results from this run
        all_confidences.append(confidences)
        all_unique_ids.append(len(track_ids))
        
        # Store track lifespans (how many frames each ID appears in)
        for track_id, frames in track_appearances.items():
            all_track_lifespans[run].append(frames)
            
        print(f"  Run {run+1} - Unique IDs: {len(track_ids)}, Avg Conf: {np.mean(confidences):.4f}")
    
    # Calculate frame-by-frame confidence variances
    frame_confidence_variances = []
    for run in range(num_runs):
        # For each frame that has at least 2 detections, calculate variance of confidence scores
        run_frame_variances = [np.var(confs) for frame_idx, confs in all_frame_confidence_data[run].items() 
                              if len(confs) > 1]
        if run_frame_variances:
            frame_confidence_variances.append(run_frame_variances)
    
    # Compile statistics with confidence intervals
    stats = {
        'run': list(range(1, num_runs+1)),
        'unique_ids': all_unique_ids,
    }
    
    # Confidence score statistics
    stats['avg_confidence'] = [np.mean(confs) if confs else 0 for confs in all_confidences]
    stats['confidence_std'] = [np.std(confs) if len(confs) > 1 else 0 for confs in all_confidences]
    
    # 95% confidence intervals for average confidence
    stats['confidence_ci_lower'] = [
        mean - 1.96 * (std / np.sqrt(len(confs))) if len(confs) > 1 else mean 
        for mean, std, confs in zip(stats['avg_confidence'], stats['confidence_std'], all_confidences)
    ]
    stats['confidence_ci_upper'] = [
        mean + 1.96 * (std / np.sqrt(len(confs))) if len(confs) > 1 else mean 
        for mean, std, confs in zip(stats['avg_confidence'], stats['confidence_std'], all_confidences)
    ]
    
    # Track lifespan statistics
    stats['avg_lifespan'] = [np.mean(all_track_lifespans[run]) if all_track_lifespans[run] else 0 
                           for run in range(num_runs)]
    stats['lifespan_std'] = [np.std(all_track_lifespans[run]) if len(all_track_lifespans[run]) > 1 else 0 
                           for run in range(num_runs)]
    
    # 95% confidence intervals for average lifespan
    stats['lifespan_ci_lower'] = [
        mean - 1.96 * (std / np.sqrt(len(all_track_lifespans[run]))) if len(all_track_lifespans[run]) > 1 else mean 
        for mean, std, run in zip(stats['avg_lifespan'], stats['lifespan_std'], range(num_runs))
    ]
    stats['lifespan_ci_upper'] = [
        mean + 1.96 * (std / np.sqrt(len(all_track_lifespans[run]))) if len(all_track_lifespans[run]) > 1 else mean 
        for mean, std, run in zip(stats['avg_lifespan'], stats['lifespan_std'], range(num_runs))
    ]
    
    # Frame confidence variance statistics
    if frame_confidence_variances:
        stats['avg_frame_variance'] = [np.mean(vars) if vars else 0 for vars in frame_confidence_variances]
        stats['frame_variance_std'] = [np.std(vars) if len(vars) > 1 else 0 for vars in frame_confidence_variances]
        
        # 95% confidence intervals for average frame variance
        stats['frame_variance_ci_lower'] = [
            mean - 1.96 * (std / np.sqrt(len(vars))) if len(vars) > 1 else mean 
            for mean, std, vars in zip(stats['avg_frame_variance'], stats['frame_variance_std'], frame_confidence_variances)
        ]
        stats['frame_variance_ci_upper'] = [
            mean + 1.96 * (std / np.sqrt(len(vars))) if len(vars) > 1 else mean 
            for mean, std, vars in zip(stats['avg_frame_variance'], stats['frame_variance_std'], frame_confidence_variances)
        ]
    
    # Convert to DataFrame
    df_stats = pd.DataFrame(stats)
    
    print("Analysis complete! Statistics summary:")
    print(df_stats)
    
    return df_stats, all_confidences, all_track_lifespans, frame_confidence_variances

class YOLOPerformanceMonitor:
    def __init__(self, buffer_factor=0.95, min_deviation_threshold=0.2):
        """
        Initialize YOLOv8 performance monitor with likelihood-based approach
        
        Parameters:
        - buffer_factor: Factor applied to minimum baseline likelihood (lower = more sensitive)
                        Default 0.95 means threshold is set 5% below minimum baseline likelihood
        - min_deviation_threshold: minimum relative deviation to be considered practically significant
        """
        self.buffer_factor = buffer_factor
        self.min_deviation_threshold = min_deviation_threshold
        self.distribution_params = None
        self.baseline_clips_raw = []
        self.baseline_likelihoods = []
        self.likelihood_threshold = None
        self.baseline_mean_variance = None
    
    def establish_baseline(self, baseline_clips, visualize=False):
        """
        Establish baseline using all frame-level data from good clips
        
        Parameters:
        - baseline_clips: list of frame variance arrays from good clips
        - visualize: whether to visualize the baseline distributions
        
        Returns:
        - baseline_info: dictionary with baseline statistics
        """
        # Store raw baseline clips
        self.baseline_clips_raw = baseline_clips
        
        # 1. Combine all frame variances from all baseline clips
        all_frame_variances = []
        for clip in baseline_clips:
            if isinstance(clip, (list, np.ndarray)):
                all_frame_variances.extend(clip)
            else:
                all_frame_variances.extend([clip])  # Handle scalar case
        
        # Calculate baseline mean variance (needed for practical significance)
        self.baseline_mean_variance = np.mean(all_frame_variances)
        
        # 2. Fit a gamma distribution to the combined data
        # (Gamma is appropriate for variance data as it's always positive)
        try:
            self.distribution_params = gamma.fit(all_frame_variances)
            self.distribution_type = "gamma"
        except Exception as e:
            print(f"Error fitting gamma distribution: {e}")
            print("Using normal distribution instead")
            # Fallback to normal distribution if gamma fit fails
            self.mean = np.mean(all_frame_variances)
            self.std = np.std(all_frame_variances)
            self.distribution_type = "normal"
        
        # 3. Calculate likelihood scores for each baseline clip
        self.baseline_likelihoods = []
        for clip in baseline_clips:
            if self.distribution_type == "gamma":
                # Calculate log-likelihood for numerical stability
                log_likelihoods = gamma.logpdf(clip, *self.distribution_params)
                mean_log_likelihood = np.mean(log_likelihoods)
                self.baseline_likelihoods.append(mean_log_likelihood)
            else:
                # Use normal distribution as fallback
                log_likelihoods = -0.5 * ((np.array(clip) - self.mean) / self.std) ** 2
                mean_log_likelihood = np.mean(log_likelihoods)
                self.baseline_likelihoods.append(mean_log_likelihood)
        
        # 4. Set threshold as buffer_factor * minimum baseline likelihood
        min_baseline_likelihood = min(self.baseline_likelihoods)
        self.likelihood_threshold = min_baseline_likelihood * self.buffer_factor
        
        if visualize:
            self.visualize_baseline()
        
        return {
            'distribution_params': self.distribution_params,
            'distribution_type': self.distribution_type,
            'likelihood_threshold': self.likelihood_threshold,
            'baseline_mean_variance': self.baseline_mean_variance,
            'baseline_likelihoods': self.baseline_likelihoods,
            'min_baseline_likelihood': min_baseline_likelihood,
            'buffer_factor': self.buffer_factor
        }
    
    def check_performance(self, clip_variances):
        """
        Check if clip indicates performance issues using likelihood-based approach
        
        Parameters:
        - clip_variances: frame confidence variances from current clip
        
        Returns:
        - result: detection result with detailed metrics
        """
        if self.distribution_params is None:
            return {'status': 'error', 'message': 'Baseline not established'}
        
        # Calculate mean variance for practical significance
        mean_variance = np.mean(clip_variances)
        
        # Calculate relative deviation for practical significance
        relative_deviation = (mean_variance - self.baseline_mean_variance) / self.baseline_mean_variance
        
        # Practical significance check
        is_practically_significant = relative_deviation > self.min_deviation_threshold
        
        # Direction check - positive deviation means worse performance
        is_worse_direction = relative_deviation > 0
        
        # Calculate likelihood score for statistical anomaly check
        if self.distribution_type == "gamma":
            # Calculate log-likelihood for numerical stability
            log_likelihoods = gamma.logpdf(clip_variances, *self.distribution_params)
            clip_log_likelihood = np.mean(log_likelihoods)
        else:
            # Use normal distribution as fallback
            log_likelihoods = -0.5 * ((np.array(clip_variances) - self.mean) / self.std) ** 2
            clip_log_likelihood = np.mean(log_likelihoods)
        
        # Statistical anomaly if log-likelihood is below threshold
        is_statistical_anomaly = clip_log_likelihood < self.likelihood_threshold
        
        # Calculate likelihood percentile for reporting
        likelihood_percentile = 100 * np.mean(np.array(self.baseline_likelihoods) < clip_log_likelihood)
        
        # Combined decision: must be statistically anomalous, practically significant, and in worse direction
        has_issue = is_statistical_anomaly and is_practically_significant and is_worse_direction
        
        return {
            'has_issue': has_issue,
            'is_statistical_anomaly': is_statistical_anomaly,
            'is_practically_significant': is_practically_significant,
            'is_worse_direction': is_worse_direction,
            'log_likelihood': clip_log_likelihood,
            'likelihood_threshold': self.likelihood_threshold,
            'likelihood_percentile': likelihood_percentile,
            'mean_variance': mean_variance,
            'baseline_mean_variance': self.baseline_mean_variance,
            'relative_deviation': relative_deviation * 100,  # Present as percentage
            'min_deviation_threshold': self.min_deviation_threshold * 100  # Present as percentage
        }
    
    def visualize_baseline(self):
        """Visualize baseline distributions and thresholds"""
        plt.figure(figsize=(12, 8))
        
        # 1. Plot all frame variances from baseline clips
        plt.subplot(2, 1, 1)
        
        # Combine all variances for histogram
        all_variances = []
        for clip in self.baseline_clips_raw:
            if isinstance(clip, (list, np.ndarray)):
                all_variances.extend(clip)
            else:
                all_variances.append(clip)
        
        plt.hist(all_variances, bins=50, alpha=0.7, density=True, color='green', 
                label='Baseline frame variances')
        
        # Plot the fitted distribution
        x = np.linspace(min(all_variances), max(all_variances), 1000)
        if self.distribution_type == "gamma":
            pdf = gamma.pdf(x, *self.distribution_params)
            plt.plot(x, pdf, 'r-', lw=2, label=f'Fitted gamma distribution')
        else:
            pdf = 1/(self.std * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-self.mean)/self.std)**2)
            plt.plot(x, pdf, 'r-', lw=2, label=f'Fitted normal distribution')
        
        plt.axvline(x=self.baseline_mean_variance, color='blue', linestyle='--', 
                   label=f'Mean variance: {self.baseline_mean_variance:.6f}')
        
        plt.title('Baseline Frame Variance Distribution')
        plt.xlabel('Variance')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Plot baseline clip likelihoods
        plt.subplot(2, 1, 2)
        plt.hist(self.baseline_likelihoods, bins=20, alpha=0.7, color='blue', 
                label='Baseline clip likelihoods')
        
        min_baseline = min(self.baseline_likelihoods)
        plt.axvline(x=min_baseline, color='green', linestyle='--', 
                label=f'Min baseline: {min_baseline:.2f}')
        
        plt.axvline(x=self.likelihood_threshold, color='red', linestyle='--', 
                label=f'Threshold ({self.buffer_factor:.2%} of min): {self.likelihood_threshold:.2f}')
        
        plt.title('Baseline Log-Likelihood Distribution')
        plt.xlabel('Mean Log-Likelihood')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_results(self, test_clips, labels):
        """
        Visualize test results using likelihood-based approach
        
        Parameters:
        - test_clips: list of clips to visualize
        - labels: binary labels (1=issue, 0=good)
        """
        # Calculate metrics for each clip
        likelihoods = []
        deviations = []
        mean_variances = []
        
        for clip in test_clips:
            result = self.check_performance(clip)
            likelihoods.append(result['log_likelihood'])
            deviations.append(result['relative_deviation'])
            mean_variances.append(result['mean_variance'])
        
        # Create scatter plot
        plt.figure(figsize=(15, 6))
        
        # 1. Statistical Anomaly vs Practical Significance
        plt.subplot(1, 2, 1)
        colors = ['green' if label == 0 else 'red' for label in labels]
        markers = ['o' if label == 0 else 'x' for label in labels]
        
        for i, (ll, dev, c, m) in enumerate(zip(likelihoods, deviations, colors, markers)):
            plt.scatter(ll, dev, color=c, marker=m, s=100)
        
        plt.axvline(x=self.likelihood_threshold, color='purple', linestyle='--', 
                   label=f'Likelihood threshold: {self.likelihood_threshold:.2f}')
        plt.axhline(y=self.min_deviation_threshold * 100, color='orange', linestyle='--',
                   label=f'Practical threshold: {self.min_deviation_threshold * 100:.1f}%')
        
        plt.title('Statistical Anomaly vs Practical Significance')
        plt.xlabel('Log-Likelihood (lower = more anomalous)')
        plt.ylabel('Relative Deviation (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for the quadrants
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        
        # Ensure threshold lines are within the plot
        plt.xlim(min(xmin, self.likelihood_threshold*1.1), max(xmax, self.likelihood_threshold*0.9))
        plt.ylim(min(ymin, 0), max(ymax, self.min_deviation_threshold*150))
        
        # Add text labels for quadrants
        plt.text(self.likelihood_threshold*0.95, self.min_deviation_threshold*120, 
                "FALSE NEGATIVES", color='gray', ha='right')
        plt.text(self.likelihood_threshold*0.95, self.min_deviation_threshold*50, 
                "TRUE NEGATIVES", color='green', fontweight='bold', ha='right')
        plt.text(self.likelihood_threshold*1.05, self.min_deviation_threshold*120, 
                "TRUE POSITIVES", color='red', fontweight='bold', ha='left')
        plt.text(self.likelihood_threshold*1.05, self.min_deviation_threshold*50, 
                "FALSE POSITIVES", color='gray', ha='left')
        
        # 2. Variance Values
        plt.subplot(1, 2, 2)
        
        plt.scatter(range(len(test_clips)), mean_variances, c=colors, marker='o', s=100)
        plt.axhline(y=self.baseline_mean_variance, color='blue', linestyle='-', 
                   label=f'Baseline mean: {self.baseline_mean_variance:.6f}')
        plt.axhline(y=self.baseline_mean_variance * (1 + self.min_deviation_threshold), 
                   color='red', linestyle='--',
                   label=f'Threshold (+{self.min_deviation_threshold*100:.1f}%)')
        
        # Add clip labels
        for i, (var, label) in enumerate(zip(mean_variances, labels)):
            marker = 'x' if label == 1 else 'o'
            plt.text(i, var, f"{i}", fontsize=9, ha='center', va='bottom')
        
        plt.title('Mean Variance by Clip')
        plt.xlabel('Clip Index')
        plt.ylabel('Mean Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Create an additional plot showing likelihood vs. baseline
        plt.figure(figsize=(10, 6))
        
        # Sort clips by likelihood for better visualization
        sorted_indices = np.argsort(likelihoods)
        sorted_likelihoods = [likelihoods[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        
        plt.bar(range(len(sorted_likelihoods)), sorted_likelihoods, color=sorted_colors)
        plt.axhline(y=self.likelihood_threshold, color='red', linestyle='--',
                   label=f'Threshold: {self.likelihood_threshold:.2f}')
        
        # Add clip indices as labels
        for i, idx in enumerate(sorted_indices):
            plt.text(i, sorted_likelihoods[i], f"{idx}", fontsize=9, 
                    ha='center', va='bottom' if sorted_likelihoods[i] > 0 else 'top')
        
        plt.title('Log-Likelihood by Clip (Sorted)')
        plt.xlabel('Sorted Clip Index')
        plt.ylabel('Log-Likelihood (lower = more anomalous)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def tune_parameters(self, validation_clips, validation_labels, 
                  buffer_range=(0.85, 0.99), buffer_steps=10,
                  deviation_range=(0.1, 0.5), deviation_steps=5):
        """
        Tune parameters for optimal performance
        
        Parameters:
        - validation_clips: list of clips for validation
        - validation_labels: binary labels (1=issue, 0=good)
        - buffer_range: range of buffer factors to try (e.g., 0.85 to 0.99)
        - buffer_steps: number of buffer factor steps
        - deviation_range: range of deviation thresholds
        - deviation_steps: number of deviation threshold steps
        
        Returns:
        - best_params: dictionary with best parameters
        """
        # Generate parameter grid
        buffer_factors = np.linspace(buffer_range[0], buffer_range[1], buffer_steps)
        deviations = np.linspace(deviation_range[0], deviation_range[1], deviation_steps)
        
        min_baseline_likelihood = min(self.baseline_likelihoods)
        best_f1 = 0
        best_params = {
            'buffer_factor': self.buffer_factor,
            'deviation': self.min_deviation_threshold
        }
        
        results = []
        
        # Test all parameter combinations
        for buffer in buffer_factors:
            for deviation in deviations:
                # Update parameters
                self.buffer_factor = buffer
                self.min_deviation_threshold = deviation
                
                # Update likelihood threshold based on buffer factor
                self.likelihood_threshold = min_baseline_likelihood * buffer
                
                # Test all validation clips
                predictions = []
                for clip in validation_clips:
                    result = self.check_performance(clip)
                    predictions.append(result['has_issue'])
                
                # Calculate metrics
                true_positives = sum(pred and label for pred, label in zip(predictions, validation_labels))
                false_positives = sum(pred and not label for pred, label in zip(predictions, validation_labels))
                true_negatives = sum(not pred and not label for pred, label in zip(predictions, validation_labels))
                false_negatives = sum(not pred and label for pred, label in zip(predictions, validation_labels))
                
                # Calculate F1 score
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                results.append({
                    'buffer_factor': buffer,
                    'deviation': deviation,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'true_negatives': true_negatives,
                    'false_negatives': false_negatives
                })
                
                # Update best parameters
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {'buffer_factor': buffer, 'deviation': deviation}
        
        # Set to best parameters
        self.buffer_factor = best_params['buffer_factor']
        self.min_deviation_threshold = best_params['deviation']
        self.likelihood_threshold = min_baseline_likelihood * self.buffer_factor
        
        # Sort results by F1 score
        sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
        
        # Visualize parameter impact
        self.visualize_parameter_impact(results, buffer_factors, deviations)
        
        return {
            'best_params': best_params,
            'best_f1': best_f1,
            'all_results': sorted_results[:10]  # Top 10 results
        }

    def visualize_parameter_impact(self, results, buffer_factors, deviations):
        """
        Visualize impact of parameters on performance metrics
        
        Parameters:
        - results: list of results from parameter tuning
        - buffer_factors: list of buffer factors tried
        - deviations: list of deviation thresholds tried
        """
        # Create heatmap data
        buffer_map = {b: i for i, b in enumerate(buffer_factors)}
        deviation_map = {d: i for i, d in enumerate(deviations)}
        
        f1_matrix = np.zeros((len(deviations), len(buffer_factors)))
        precision_matrix = np.zeros((len(deviations), len(buffer_factors)))
        recall_matrix = np.zeros((len(deviations), len(buffer_factors)))
        
        for result in results:
            b_idx = buffer_map[result['buffer_factor']]
            d_idx = deviation_map[result['deviation']]
            f1_matrix[d_idx, b_idx] = result['f1']
            precision_matrix[d_idx, b_idx] = result['precision']
            recall_matrix[d_idx, b_idx] = result['recall']
        
        # Plot heatmaps
        plt.figure(figsize=(15, 5))
        
        # F1 score
        plt.subplot(1, 3, 1)
        plt.imshow(f1_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='F1 Score')
        plt.title('F1 Score by Parameters')
        plt.xlabel('Buffer Factor')
        plt.ylabel('Deviation Threshold')
        plt.xticks(np.arange(len(buffer_factors)), [f'{b:.2f}' for b in buffer_factors], rotation=45)
        plt.yticks(np.arange(len(deviations)), [f'{d:.2f}' for d in deviations])
        
        # Precision
        plt.subplot(1, 3, 2)
        plt.imshow(precision_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Precision')
        plt.title('Precision by Parameters')
        plt.xlabel('Buffer Factor')
        plt.ylabel('Deviation Threshold')
        plt.xticks(np.arange(len(buffer_factors)), [f'{b:.2f}' for b in buffer_factors], rotation=45)
        plt.yticks(np.arange(len(deviations)), [f'{d:.2f}' for d in deviations])
        
        # Recall
        plt.subplot(1, 3, 3)
        plt.imshow(recall_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Recall')
        plt.title('Recall by Parameters')
        plt.xlabel('Buffer Factor')
        plt.ylabel('Deviation Threshold')
        plt.xticks(np.arange(len(buffer_factors)), [f'{b:.2f}' for b in buffer_factors], rotation=45)
        plt.yticks(np.arange(len(deviations)), [f'{d:.2f}' for d in deviations])
        
        plt.tight_layout()
        plt.show()
        
        # Create an additional figure showing parameter combinations and their F1 scores
        plt.figure(figsize=(10, 6))
        
        # Extract top N parameter combinations
        top_n = min(10, len(results))
        top_results = sorted(results, key=lambda x: x['f1'], reverse=True)[:top_n]
        
        # Create labels for x-axis
        labels = [f"B:{r['buffer_factor']:.2f}, D:{r['deviation']:.2f}" for r in top_results]
        
        # Plot F1, precision, recall for top combinations
        f1_values = [r['f1'] for r in top_results]
        precision_values = [r['precision'] for r in top_results]
        recall_values = [r['recall'] for r in top_results]
        
        x = np.arange(len(labels))
        width = 0.25
        
        plt.bar(x - width, f1_values, width, label='F1 Score', color='purple')
        plt.bar(x, precision_values, width, label='Precision', color='blue')
        plt.bar(x + width, recall_values, width, label='Recall', color='green')
        
        plt.xlabel('Parameter Combinations')
        plt.ylabel('Score')
        plt.title('Top Parameter Combinations Performance')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    
    def run_test(self, test_clips, labels):
        """
        Run test on labeled clips to evaluate monitor performance
        
        Parameters:
        - test_clips: list of clips to test
        - labels: binary labels (1=issue, 0=good)
        
        Returns:
        - results: test results with metrics
        """
        if self.distribution_params is None:
            return {'status': 'error', 'message': 'Baseline not established'}
        
        # Test each clip
        results = []
        for clip, label in zip(test_clips, labels):
            result = self.check_performance(clip)
            result['true_label'] = label
            result['correct'] = (result['has_issue'] == bool(label))
            results.append(result)
        
        # Calculate metrics
        true_positives = sum(1 for r in results if r['has_issue'] and r['true_label'])
        false_positives = sum(1 for r in results if r['has_issue'] and not r['true_label'])
        true_negatives = sum(1 for r in results if not r['has_issue'] and not r['true_label'])
        false_negatives = sum(1 for r in results if not r['has_issue'] and r['true_label'])
        
        accuracy = (true_positives + true_negatives) / len(results) if len(results) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'results': results,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            }
        }