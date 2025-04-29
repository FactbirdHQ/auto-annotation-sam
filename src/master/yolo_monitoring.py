import numpy as np
import os
import time
import json
import torch
import cv2
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO
import torch.nn.functional as F

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


class UncertaintyMonitor:
    """
    Monitor YOLOv8 model performance using uncertainty estimates from MC Dropout.
    
    This class handles:
    1. Running MC Dropout on video clips
    2. Matching objects across multiple passes
    3. Calculating aleatoric and epistemic uncertainties
    4. Alerting when uncertainties deviate from baseline
    5. Diagnosing potential issues (grease, recipe changes)
    """
    
    def __init__(self, model_path, 
                 conf_threshold=0.5, 
                 iou_threshold=0.5,
                 mc_passes=10,
                 alert_thresholds=None):
        """
        Initialize the uncertainty monitor.
        
        Args:
            model_path (str): Path to YOLOv8 model
            conf_threshold (float): Confidence threshold for detection
            iou_threshold (float): IoU threshold for matching detections across MC runs
            mc_passes (int): Number of MC Dropout passes to run
            alert_thresholds (dict): Thresholds for alerting on uncertainty values
        """
        self.model = MCDropoutTracker(model_path, dropout_rate=0.01)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.mc_passes = mc_passes
        self.baseline_established = False
        
        # Default alert thresholds (standard deviations from baseline)
        if alert_thresholds is None:
            self.alert_thresholds = {
                'aleatoric_z_threshold': 3.0,  # Z-score threshold for aleatoric uncertainty
                'epistemic_z_threshold': 3.0,  # Z-score threshold for epistemic uncertainty
                'ratio_threshold': 2.0,        # Threshold for ratio change (current/baseline)
            }
        else:
            self.alert_thresholds = alert_thresholds
        
        # Baseline statistics
        self.baseline_stats = {
            'aleatoric': {'mean': None, 'std': None},
            'epistemic': {'mean': None, 'std': None},
            'ratio': {'mean': None, 'std': None},
            'samples': 0
        }
        
        # History tracking
        self.history = []
        

        
    def process_clip(self, clip_path, save_results=False, output_dir=None):
        """
        Process a video clip through MC Dropout and calculate uncertainties.
        
        Args:
            clip_path (str): Path to video clip
            save_results (bool): Whether to save visualization results
            output_dir (str): Directory to save results in (if save_results=True)
            
        Returns:
            dict: Dictionary of uncertainty metrics and analysis results
        """
        # Run MC Dropout passes
        mc_results = self._run_mc_dropout(clip_path)
        
        # Match objects across MC runs
        matched_objects = self._match_objects_across_runs(mc_results)
        
        # Calculate uncertainties
        uncertainties = self._calculate_uncertainties(matched_objects)
        
        # Check for issues if baseline established
        issues = None
        if self.baseline_established:
            issues = self._detect_issues(uncertainties)
            uncertainties['issues'] = issues
        
        # Save history
        self.history.append({
            'timestamp': time.time(),
            'clip_path': clip_path,
            'uncertainties': uncertainties,
            'issues': issues
        })
        
        # Optionally save visualization
        if save_results and output_dir is not None:
            self._save_visualization(clip_path, mc_results, matched_objects, uncertainties, output_dir)
        
        return uncertainties
    
    def _run_mc_dropout(self, clip_path):
        """
        Run multiple passes of YOLOv8 with dropout enabled.
        
        Args:
            clip_path (str): Path to video clip
            
        Returns:
            list: Results from each MC Dropout pass
        """
        # For simplicity, we'll use the first frame of the clip
        # In a real implementation, you'd process the entire clip or select frames
        cap = cv2.VideoCapture(clip_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to read clip: {clip_path}")
        
        # Run MC Dropout passes
        mc_results = []
        
        for _ in range(self.mc_passes):
            # Run YOLOv8 with dropout enabled
            results = self.model(frame, verbose=False)
            
            # Extract detection results
            detections = []
            if len(results) > 0:
                for r in results[0].boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls_id = r
                    
                    # Only include detections above threshold
                    if conf >= self.conf_threshold:
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(cls_id)
                        })
            
            mc_results.append(detections)
            
        return mc_results
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def _match_objects_across_runs(self, mc_results):
        """
        Match the same objects across different MC Dropout runs.
        
        Args:
            mc_results: List of detection results from MC Dropout passes
            
        Returns:
            list: List of matched object groups
        """
        # Start with first run's detections
        if not mc_results or len(mc_results) == 0 or len(mc_results[0]) == 0:
            return []
            
        matched_groups = [[det] for det in mc_results[0]]
        
        # For each subsequent run
        for run_idx in range(1, len(mc_results)):
            current_detections = mc_results[run_idx]
            unmatched_current = list(range(len(current_detections)))
            
            # For each existing group
            for group in matched_groups:
                last_det = group[-1]
                best_match_idx = -1
                best_match_iou = self.iou_threshold  # Minimum threshold
                
                # Find best match in current run
                for i in unmatched_current:
                    current_det = current_detections[i]
                    iou = self._calculate_iou(last_det['box'], current_det['box'])
                    if iou > best_match_iou:
                        best_match_iou = iou
                        best_match_idx = i
                
                # If match found, add to group and mark as matched
                if best_match_idx >= 0:
                    group.append(current_detections[best_match_idx])
                    unmatched_current.remove(best_match_idx)
            
            # Add any unmatched detections as new groups
            for i in unmatched_current:
                matched_groups.append([current_detections[i]])
        
        # Filter groups to only include those that appear in at least half the runs
        min_runs = len(mc_results) / 2
        consistent_groups = [group for group in matched_groups if len(group) >= min_runs]
        
        return consistent_groups
    
    def _calculate_uncertainties(self, matched_objects):
        """
        Calculate aleatoric and epistemic uncertainties from matched objects.
        
        Args:
            matched_objects: List of matched object groups
            
        Returns:
            dict: Dictionary of uncertainty metrics
        """
        if not matched_objects:
            return {
                'detection_count': 0,
                'mean_confidence': 0,
                'mean_aleatoric': 0,
                'mean_epistemic': 0,
                'mean_position_variance': 0,
                'aleatoric_epistemic_ratio': 0
            }
        
        # Calculate per-object uncertainties
        object_metrics = []
        
        for obj_group in matched_objects:
            # Confidence scores for this object across MC runs
            confidence_scores = [det['confidence'] for det in obj_group]
            
            # Calculate mean confidence
            mean_confidence = np.mean(confidence_scores)
            
            # Calculate aleatoric uncertainty (predictive entropy)
            # For classification/detection, related to confidence score: p*(1-p)
            aleatoric = np.mean([(1 - score) * score for score in confidence_scores])
            
            # Calculate epistemic uncertainty (variance of predictions)
            epistemic = np.var(confidence_scores)
            
            # Calculate position variance
            box_centers = [
                ((det['box'][0] + det['box'][2]) / 2, (det['box'][1] + det['box'][3]) / 2) 
                for det in obj_group
            ]
            position_var = np.var(box_centers, axis=0).mean()
            
            object_metrics.append({
                'mean_confidence': mean_confidence,
                'aleatoric': aleatoric,
                'epistemic': epistemic,
                'position_variance': position_var
            })
        
        # Aggregate metrics across all objects
        mean_confidence = np.mean([m['mean_confidence'] for m in object_metrics])
        mean_aleatoric = np.mean([m['aleatoric'] for m in object_metrics])
        mean_epistemic = np.mean([m['epistemic'] for m in object_metrics])
        mean_position_var = np.mean([m['position_variance'] for m in object_metrics])
        
        # Calculate ratio of aleatoric to epistemic uncertainty
        # Useful for diagnosing issue type
        ratio = mean_aleatoric / mean_epistemic if mean_epistemic > 0 else float('inf')
        
        return {
            'detection_count': len(matched_objects),
            'mean_confidence': mean_confidence,
            'mean_aleatoric': mean_aleatoric,
            'mean_epistemic': mean_epistemic,
            'mean_position_variance': mean_position_var,
            'aleatoric_epistemic_ratio': ratio
        }
    
    def establish_baseline(self, clip_paths):
        """
        Establish baseline statistics from known good clips.
        
        Args:
            clip_paths (list): List of paths to good video clips
            
        Returns:
            dict: Baseline statistics
        """
        if not clip_paths:
            raise ValueError("No clips provided for baseline establishment")
        
        print(f"Establishing baseline from {len(clip_paths)} clips...")
        
        # Process each clip and collect metrics
        all_metrics = []
        
        for clip_path in clip_paths:
            results = self.process_clip(clip_path)
            if results['detection_count'] > 0:  # Only include clips with detections
                all_metrics.append(results)
        
        if not all_metrics:
            raise ValueError("No valid metrics found in baseline clips")
        
        # Calculate baseline statistics
        aleatoric_values = [m['mean_aleatoric'] for m in all_metrics]
        epistemic_values = [m['mean_epistemic'] for m in all_metrics]
        ratio_values = [m['aleatoric_epistemic_ratio'] for m in all_metrics]
        
        self.baseline_stats = {
            'aleatoric': {
                'mean': np.mean(aleatoric_values),
                'std': np.std(aleatoric_values)
            },
            'epistemic': {
                'mean': np.mean(epistemic_values),
                'std': np.std(epistemic_values)
            },
            'ratio': {
                'mean': np.mean(ratio_values),
                'std': np.std(ratio_values)
            },
            'samples': len(all_metrics)
        }
        
        self.baseline_established = True
        
        print("Baseline established:")
        print(f"  Aleatoric: {self.baseline_stats['aleatoric']['mean']:.6f} ± {self.baseline_stats['aleatoric']['std']:.6f}")
        print(f"  Epistemic: {self.baseline_stats['epistemic']['mean']:.6f} ± {self.baseline_stats['epistemic']['std']:.6f}")
        print(f"  Ratio: {self.baseline_stats['ratio']['mean']:.2f} ± {self.baseline_stats['ratio']['std']:.2f}")
        
        return self.baseline_stats
    
    def _detect_issues(self, uncertainties):
        """
        Detect issues based on uncertainty values.
        
        Args:
            uncertainties (dict): Current uncertainty metrics
            
        Returns:
            dict: Issue detection results
        """
        if not self.baseline_established:
            return {'error': 'Baseline not established'}
        
        if uncertainties['detection_count'] == 0:
            return {'error': 'No detections in current clip'}
        
        # Calculate z-scores
        aleatoric_z = ((uncertainties['mean_aleatoric'] - self.baseline_stats['aleatoric']['mean']) / 
                      self.baseline_stats['aleatoric']['std']) if self.baseline_stats['aleatoric']['std'] > 0 else 0
        
        epistemic_z = ((uncertainties['mean_epistemic'] - self.baseline_stats['epistemic']['mean']) / 
                      self.baseline_stats['epistemic']['std']) if self.baseline_stats['epistemic']['std'] > 0 else 0
        
        ratio = uncertainties['aleatoric_epistemic_ratio']
        baseline_ratio = self.baseline_stats['ratio']['mean']
        ratio_change = ratio / baseline_ratio if baseline_ratio > 0 else float('inf')
        
        # Check for issues
        aleatoric_issue = abs(aleatoric_z) > self.alert_thresholds['aleatoric_z_threshold']
        epistemic_issue = abs(epistemic_z) > self.alert_thresholds['epistemic_z_threshold']
        ratio_issue = ratio_change > self.alert_thresholds['ratio_threshold'] or ratio_change < (1 / self.alert_thresholds['ratio_threshold'])
        
        has_issue = aleatoric_issue or epistemic_issue
        
        # Diagnose issue type
        issue_type = None
        if has_issue:
            if aleatoric_issue and not epistemic_issue:
                issue_type = 'camera_issue'  # Likely grease on lens
            elif epistemic_issue and not aleatoric_issue:
                issue_type = 'recipe_change'  # Likely recipe change
            else:
                issue_type = 'multiple_issues'  # Multiple issues detected
        
        return {
            'has_issue': has_issue,
            'issue_type': issue_type,
            'aleatoric_issue': aleatoric_issue,
            'epistemic_issue': epistemic_issue,
            'ratio_issue': ratio_issue,
            'metrics': {
                'aleatoric_z': aleatoric_z,
                'epistemic_z': epistemic_z,
                'ratio_change': ratio_change
            }
        }
    
    def _save_visualization(self, clip_path, mc_results, matched_objects, uncertainties, output_dir):
        """
        Save visualization of uncertainty results for debugging/monitoring.
        
        Args:
            clip_path (str): Path to original clip
            mc_results (list): MC Dropout results
            matched_objects (list): Matched objects across runs
            uncertainties (dict): Calculated uncertainties
            output_dir (str): Directory to save results in
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract filename from path
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save uncertainty metrics to JSON
        metrics_path = os.path.join(output_dir, f"{clip_name}_{timestamp}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(uncertainties, f, indent=2)
        
        # Read first frame for visualization
        cap = cv2.VideoCapture(clip_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return
        
        # Draw bounding boxes from all MC runs with transparency
        vis_frame = frame.copy()
        for run_idx, detections in enumerate(mc_results):
            alpha = 0.3  # Transparency
            overlay = vis_frame.copy()
            
            for det in detections:
                x1, y1, x2, y2 = map(int, det['box'])
                confidence = det['confidence']
                
                # Draw box with random color based on run index
                color = [(run_idx * 50) % 255, (run_idx * 80) % 255, (run_idx * 110) % 255]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Draw confidence
                cv2.putText(overlay, f"{confidence:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Blend overlay with original
            cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)
        
        # Draw matched objects with thicker lines
        for obj_idx, obj_group in enumerate(matched_objects):
            # Calculate average box
            avg_box = np.mean([[det['box'][0], det['box'][1], det['box'][2], det['box'][3]] 
                              for det in obj_group], axis=0).astype(int)
            
            # Draw average box
            x1, y1, x2, y2 = avg_box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get object metrics
            conf_scores = [det['confidence'] for det in obj_group]
            mean_conf = np.mean(conf_scores)
            aleatoric = np.mean([(1 - score) * score for score in conf_scores])
            epistemic = np.var(conf_scores)
            
            # Draw metrics
            cv2.putText(vis_frame, f"#{obj_idx}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"A: {aleatoric:.3f} E: {epistemic:.3f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add summary text
        summary_text = [
            f"Detections: {uncertainties['detection_count']}",
            f"Aleatoric: {uncertainties['mean_aleatoric']:.4f}",
            f"Epistemic: {uncertainties['mean_epistemic']:.4f}",
            f"A/E Ratio: {uncertainties['aleatoric_epistemic_ratio']:.2f}"
        ]
        
        # Add issue alert if baseline established
        if 'issues' in uncertainties and uncertainties['issues']:
            issues = uncertainties['issues']
            if issues['has_issue']:
                issue_color = (0, 0, 255)  # Red for issues
                issue_text = f"ISSUE: {issues['issue_type']}"
                summary_text.append(issue_text)
            else:
                issue_color = (0, 255, 0)  # Green for normal
                summary_text.append("Status: Normal")
        
        # Draw summary
        for i, text in enumerate(summary_text):
            cv2.putText(vis_frame, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"{clip_name}_{timestamp}_vis.jpg")
        cv2.imwrite(vis_path, vis_frame)
        
        print(f"Saved visualization to {vis_path}")
    
    def run_test(self, normal_clips, problem_clips, problem_labels=None):
        """
        Run performance test on labeled test set.
        
        Args:
            normal_clips (list): Paths to normal operation clips
            problem_clips (list): Paths to problem clips
            problem_labels (list): Optional problem type labels
            
        Returns:
            dict: Test results and metrics
        """
        if not self.baseline_established:
            raise ValueError("Baseline must be established before running tests")
        
        results = {
            'normal': [],
            'problem': [],
            'metrics': {}
        }
        
        # Process normal clips
        for clip_path in normal_clips:
            uncertainties = self.process_clip(clip_path)
            results['normal'].append({
                'clip': clip_path,
                'uncertainties': uncertainties,
                'expected_issue': False
            })
        
        # Process problem clips
        for i, clip_path in enumerate(problem_clips):
            uncertainties = self.process_clip(clip_path)
            problem_type = problem_labels[i] if problem_labels and i < len(problem_labels) else None
            results['problem'].append({
                'clip': clip_path,
                'uncertainties': uncertainties,
                'expected_issue': True,
                'expected_type': problem_type
            })
        
        # Calculate metrics
        normal_correct = sum(1 for r in results['normal'] 
                            if not r['uncertainties']['issues']['has_issue'])
        problem_correct = sum(1 for r in results['problem'] 
                             if r['uncertainties']['issues']['has_issue'])
        
        total = len(normal_clips) + len(problem_clips)
        accuracy = (normal_correct + problem_correct) / total if total > 0 else 0
        
        # Type accuracy
        type_correct = 0
        for r in results['problem']:
            if (r['uncertainties']['issues']['has_issue'] and 
                r['expected_type'] == r['uncertainties']['issues']['issue_type']):
                type_correct += 1
        
        type_accuracy = type_correct / len(problem_clips) if problem_clips else 0
        
        results['metrics'] = {
            'normal_accuracy': normal_correct / len(normal_clips) if normal_clips else 0,
            'problem_detection': problem_correct / len(problem_clips) if problem_clips else 0,
            'overall_accuracy': accuracy,
            'type_accuracy': type_accuracy
        }
        
        return results
    
    def optimize_thresholds(self, normal_clips, problem_clips, problem_labels=None):
        """
        Find optimal thresholds for uncertainty monitoring.
        
        Args:
            normal_clips (list): Paths to normal operation clips
            problem_clips (list): Paths to problem clips
            problem_labels (list): Optional problem type labels
            
        Returns:
            dict: Optimal thresholds and performance metrics
        """
        if not self.baseline_established:
            raise ValueError("Baseline must be established before optimizing thresholds")
        
        # Process all clips first to avoid reprocessing
        normal_results = []
        for clip_path in normal_clips:
            uncertainties = self.process_clip(clip_path)
            normal_results.append(uncertainties)
        
        problem_results = []
        for clip_path in problem_clips:
            uncertainties = self.process_clip(clip_path)
            problem_results.append(uncertainties)
        
        # Try different threshold combinations
        best_accuracy = 0
        best_thresholds = None
        
        # Grid search for thresholds
        for aleatoric_z in [2.0, 2.5, 3.0, 3.5]:
            for epistemic_z in [2.0, 2.5, 3.0, 3.5]:
                for ratio_threshold in [1.5, 2.0, 2.5, 3.0]:
                    thresholds = {
                        'aleatoric_z_threshold': aleatoric_z,
                        'epistemic_z_threshold': epistemic_z,
                        'ratio_threshold': ratio_threshold
                    }
                    
                    # Save original thresholds
                    original_thresholds = self.alert_thresholds.copy()
                    self.alert_thresholds = thresholds
                    
                    # Evaluate with these thresholds
                    normal_correct = 0
                    for result in normal_results:
                        issues = self._detect_issues(result)
                        normal_correct += 0 if issues['has_issue'] else 1
                    
                    problem_correct = 0
                    for result in problem_results:
                        issues = self._detect_issues(result)
                        problem_correct += 1 if issues['has_issue'] else 0
                    
                    total = len(normal_results) + len(problem_results)
                    accuracy = (normal_correct + problem_correct) / total if total > 0 else 0
                    
                    # Check if better than current best
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_thresholds = thresholds.copy()
                    
                    # Restore original thresholds
                    self.alert_thresholds = original_thresholds
        
        # Set to best thresholds
        if best_thresholds:
            self.alert_thresholds = best_thresholds
            print(f"Optimized thresholds: {best_thresholds}")
            print(f"Best accuracy: {best_accuracy:.2%}")
        
        return {
            'optimal_thresholds': best_thresholds,
            'accuracy': best_accuracy
        }
    
    def save(self, filepath):
        """
        Save monitor state and baseline statistics to file.
        
        Args:
            filepath (str): Path to save file
        """
        state = {
            'baseline_stats': self.baseline_stats,
            'alert_thresholds': self.alert_thresholds,
            'baseline_established': self.baseline_established,
            'history': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Saved monitor state to {filepath}")
    
    def load(self, filepath):
        """
        Load monitor state and baseline statistics from file.
        
        Args:
            filepath (str): Path to load file
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.baseline_stats = state['baseline_stats']
        self.alert_thresholds = state['alert_thresholds']
        self.baseline_established = state['baseline_established']
        self.history = state['history']
        
        print(f"Loaded monitor state from {filepath}")
        
        if self.baseline_established:
            print("Baseline statistics:")
            print(f"  Aleatoric: {self.baseline_stats['aleatoric']['mean']:.6f} ± {self.baseline_stats['aleatoric']['std']:.6f}")
            print(f"  Epistemic: {self.baseline_stats['epistemic']['mean']:.6f} ± {self.baseline_stats['epistemic']['std']:.6f}")
            print(f"  Ratio: {self.baseline_stats['ratio']['mean']:.2f} ± {self.baseline_stats['ratio']['std']:.2f}")


# Example usage
if __name__ == "__main__":
    # Initialize monitor
    monitor = UncertaintyMonitor(
        model_path="yolov8n.pt",
        conf_threshold=0.5,
        iou_threshold=0.5,
        mc_passes=10
    )
    
    # Establish baseline from good clips
    baseline_clips = [
        "data/good_clip1.mp4",
        "data/good_clip2.mp4",
        "data/good_clip3.mp4",
        "data/good_clip4.mp4",
        "data/good_clip5.mp4"
    ]
    
    monitor.establish_baseline(baseline_clips)
    
    # Process a new clip
    result = monitor.process_clip("data/new_clip.mp4", save_results=True, output_dir="output")
    
    if result['issues']['has_issue']:
        issue_type = result['issues']['issue_type']
        print(f"Issue detected: {issue_type}")
        print(f"Aleatoric Z-score: {result['issues']['metrics']['aleatoric_z']:.2f}")
        print(f"Epistemic Z-score: {result['issues']['metrics']['epistemic_z']:.2f}")
    else:
        print("No issues detected.")
    
    # Save monitor state
    monitor.save("monitor_state.json")