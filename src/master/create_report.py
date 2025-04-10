import os
import json
import glob
from pathlib import Path
import sys

# Import the report generator class
from src.master.report_generator import SegmentationReportGenerator
from src.master.data import KFoldSegmentationManager

def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    with open(file_path, 'r') as f:
        return json.load(f)

def reconstruct_all_results(results_dir):
    """
    Reconstruct the all_results dictionary from individual result files.
    
    Args:
        results_dir: Path to the directory containing the result files
        
    Returns:
        Dictionary with reconstructed results
    """
    results_dir = Path(results_dir)
    
    # Initialize the structure
    all_results = {
        'per_dataset': {},
        'training_size': {},
        'best_configs': {}
    }
    
    # Load dataset-specific metrics files
    dataset_files = glob.glob(str(results_dir / "*_metrics.json"))
    for file_path in dataset_files:
        file_name = os.path.basename(file_path)
        
        # Parse file name to extract components
        parts = file_name.split('_')
        if len(parts) < 3:
            continue
            
        dataset_name = parts[0]
        embedding_name = parts[1]
        classifier_name = parts[2].split('.')[0]  # Remove file extension
        
        # Skip training_size metrics files
        if "training_size" in file_name:
            continue
        
        # Load the metrics
        metrics = load_json_file(file_path)
        
        # Add to per_dataset results
        if dataset_name not in all_results['per_dataset']:
            all_results['per_dataset'][dataset_name] = {}
            
        if embedding_name not in all_results['per_dataset'][dataset_name]:
            all_results['per_dataset'][dataset_name][embedding_name] = {}
            
        all_results['per_dataset'][dataset_name][embedding_name][classifier_name] = metrics
    
    # Load training size metrics files
    training_size_files = glob.glob(str(results_dir / "*_training_size_metrics.json"))
    for file_path in training_size_files:
        file_name = os.path.basename(file_path)
        
        # Parse file name to extract components
        parts = file_name.split('_')
        if len(parts) < 4:
            continue
            
        dataset_name = parts[0]
        embedding_name = parts[1]
        classifier_name = parts[2]
        
        # Load the metrics
        metrics = load_json_file(file_path)
        
        # Add to training_size results
        if dataset_name not in all_results['training_size']:
            all_results['training_size'][dataset_name] = {}
            
        if embedding_name not in all_results['training_size'][dataset_name]:
            all_results['training_size'][dataset_name][embedding_name] = {}
            
        all_results['training_size'][dataset_name][embedding_name][classifier_name] = metrics
    
    # Load best config files
    best_config_files = glob.glob(str(results_dir / "*_best_config.json"))
    for file_path in best_config_files:
        file_name = os.path.basename(file_path)
        
        # Parse file name to extract components
        parts = file_name.split('_')
        if len(parts) < 4:
            continue
            
        dataset_name = parts[0]
        embedding_name = parts[1]
        classifier_name = parts[2]
        
        # Load the best config
        best_config = load_json_file(file_path)
        
        # Add to best_configs results
        if dataset_name not in all_results['best_configs']:
            all_results['best_configs'][dataset_name] = {}
            
        if embedding_name not in all_results['best_configs'][dataset_name]:
            all_results['best_configs'][dataset_name][embedding_name] = {}
            
        all_results['best_configs'][dataset_name][embedding_name][classifier_name] = best_config
    
    # Load consistency analysis if it exists
    consistency_file = results_dir / "model_consistency_analysis.json"
    if consistency_file.exists():
        all_results['consistency'] = load_json_file(consistency_file)
    
    # Load pipeline comparison if it exists
    pipeline_file = results_dir / "pipeline_comparison_results.json"
    if pipeline_file.exists():
        all_results['pipeline_comparison'] = load_json_file(pipeline_file)
    
    return all_results

def initialize_dataset_managers(dataset_paths):
    """
    Initialize dataset managers for the report generator
    
    Args:
        dataset_paths: Dictionary mapping dataset names to paths
        
    Returns:
        Dictionary of dataset managers
    """
    dataset_managers = {}
    
    for name, path in dataset_paths.items():
        try:
            # Initialize dataset manager with default class ID of 1
            dataset_manager = KFoldSegmentationManager(path, class_id=1)
            dataset_managers[name] = dataset_manager
            print(f"Loaded dataset: {name} from {path}")
        except Exception as e:
            print(f"Error loading dataset {name}: {e}")
    
    return dataset_managers

def main():
    """Main function to regenerate the report"""
    # Set paths
    results_dir = "ablation_results"
    
    # Load and reconstruct all results
    print("Loading and reconstructing results...")
    all_results = reconstruct_all_results(results_dir)
    
    # Define dataset paths - adjust this to match your actual paths
    project_root = Path.cwd()
    processed_data_dir = project_root / "data" / "processed"
    
    dataset_paths = {
        'meatballs': str(processed_data_dir / 'meatballs'),
        # Uncomment if you have these datasets
        # 'cans': str(processed_data_dir / 'cans'),
        # 'doughs': str(processed_data_dir / 'doughs'),
        # 'bottles': str(processed_data_dir / 'bottles'),
    }
    
    # Initialize dataset managers
    print("Initializing dataset managers...")
    dataset_managers = initialize_dataset_managers(dataset_paths)
    
    if not dataset_managers:
        print("Warning: No dataset managers could be initialized. Report may be incomplete.")
    
    # Generate report
    print("Generating report...")
    report_generator = SegmentationReportGenerator(results_dir)
    report_generator.generate_full_report(all_results, dataset_managers)
    
    print(f"Report generation complete. Check {results_dir} for the report.")

if __name__ == "__main__":
    main()