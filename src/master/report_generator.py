import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class SegmentationReportGenerator:
    def __init__(self, output_dir):
        """
        Initialize the report generator
        
        Args:
            output_dir: Directory where report files will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_full_report(self, all_results, dataset_managers):
        """
        Generate a comprehensive HTML report
        
        Args:
            all_results: Dictionary with all the ablation results
            dataset_managers: Dictionary of dataset manager objects
        """
        # Extract per_dataset results - handle both direct and nested structure
        per_dataset_results = {}
        if 'per_dataset' in all_results:
            per_dataset_results = all_results['per_dataset']
        elif 'ablation' in all_results and 'per_dataset' in all_results['ablation']:
            per_dataset_results = all_results['ablation']['per_dataset']
        
        # Create a summary DataFrame for per-dataset results
        df = self._create_summary_dataframe(per_dataset_results)
        
        # Save to CSV
        df.to_csv(os.path.join(self.output_dir, 'ablation_summary.csv'), index=False)
        
        # Create visualizations
        self._plot_f1_by_embedding_classifier(df)
        self._plot_f1_by_dataset(df)
        self._plot_timing_metrics(df)
        
        # Generate HTML report base
        self._generate_html_report(all_results, df, dataset_managers)
        
        # Prepare data structure for additional sections
        report_data = {}
        
        # Extract training size data
        if 'training_size' in all_results:
            report_data['training_size'] = all_results['training_size']
        elif 'ablation' in all_results and 'training_size' in all_results['ablation']:
            report_data['training_size'] = all_results['ablation']['training_size']
        
        # Extract per_dataset for reference in training size section
        if 'per_dataset' in all_results:
            report_data['per_dataset'] = all_results['per_dataset']
        elif 'ablation' in all_results and 'per_dataset' in all_results['ablation']:
            report_data['per_dataset'] = all_results['ablation']['per_dataset']
        
        # Add training size analysis if available
        if 'training_size' in report_data:
            with open(os.path.join(self.output_dir, 'ablation_report.html'), 'a') as f:
                self._add_training_size_section(f, report_data)
        
        # Extract consistency results
        consistency_results = None
        if 'consistency' in all_results:
            consistency_results = all_results['consistency']
        elif 'ablation' in all_results and 'consistency' in all_results['ablation']:
            consistency_results = all_results['ablation']['consistency']
        
        # Add consistency section if available
        if consistency_results:
            self.plot_consistency_metrics(consistency_results)
            self.add_consistency_section(consistency_results)
        
        # Extract pipeline comparison results
        pipeline_comparison = None
        if 'pipeline_comparison' in all_results:
            pipeline_comparison = all_results['pipeline_comparison']
        elif 'comparison' in all_results:
            pipeline_comparison = all_results['comparison']
        
        # Add pipeline comparison if available
        if pipeline_comparison:
            # Extract ideal and realistic results
            ideal_results = pipeline_comparison.get('ideal', {})
            realistic_results = pipeline_comparison.get('realistic', {})
            
            if ideal_results and realistic_results:
                self.add_pipeline_comparison(ideal_results, realistic_results)
        
        return df
    
    def _create_summary_dataframe(self, per_dataset_results):
        """Create a summary DataFrame from the results"""
        rows = []
        
        for dataset_name, dataset_results in per_dataset_results.items():
            for embedding_name, embedding_results in dataset_results.items():
                for classifier_name, metrics in embedding_results.items():
                    row = {
                        'Dataset': dataset_name,
                        'Embedding': embedding_name,
                        'Classifier': classifier_name,
                        'Precision': metrics.get('mask_precision', 0),
                        'Recall': metrics.get('mask_recall', 0),
                        'F1 Score': metrics.get('mask_f1', 0),
                        'Avg IoU': metrics.get('avg_iou_detected', 0),
                        'Detected/GT': f"{metrics.get('detected_masks', 0)}/{metrics.get('total_gt_masks', 0)}",
                        'Training Time (s)': metrics.get('avg_training_time', 0),
                        'Inference Time (s)': metrics.get('avg_inference_time', 0)
                    }
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _plot_f1_by_embedding_classifier(self, df):
        """Plot F1 scores by embedding and classifier with improved layout"""
        # Create one large average plot
        plt.figure(figsize=(15, 10))
        pivot_df = df.pivot_table(
            index='Classifier', 
            columns='Embedding', 
            values='F1 Score',
            aggfunc='mean'
        )
        ax = pivot_df.plot(kind='bar', rot=0, figsize=(15, 10))
        ax.set_title('Average F1 Score by Embedding and Classifier', fontsize=16)
        ax.set_ylabel('F1 Score', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(title='Embedding', fontsize=12, title_fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_by_embedding_classifier_avg.png'), dpi=300)
        plt.close()
        
        # Create small plots for each dataset
        datasets = df['Dataset'].unique()
        for dataset in datasets:
            dataset_df = df[df['Dataset'] == dataset]
            plt.figure(figsize=(8, 6))
            pivot_df = dataset_df.pivot_table(
                index='Classifier', 
                columns='Embedding', 
                values='F1 Score'
            )
            ax = pivot_df.plot(kind='bar', rot=0)
            ax.set_title(f'F1 Score for {dataset}', fontsize=12)
            ax.set_ylabel('F1 Score')
            plt.legend(title='Embedding')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'f1_by_embedding_classifier_{dataset}.png'), dpi=300)
            plt.close()
    
    def _plot_f1_by_dataset(self, df):
        """Plot F1 scores by dataset"""
        plt.figure(figsize=(12, 8))
        pivot_df = df.pivot_table(
            index='Dataset', 
            columns=['Embedding', 'Classifier'], 
            values='F1 Score'
        )
        ax = pivot_df.plot(kind='bar', rot=45)
        ax.set_title('F1 Score by Dataset, Embedding, and Classifier')
        ax.set_ylabel('F1 Score')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_by_dataset.png'), dpi=300)
        plt.close()

    def _plot_timing_metrics(self, df):
        """Plot timing metrics by embedding and classifier"""
        # Training time plot
        plt.figure(figsize=(12, 8))
        pivot_df = df.pivot_table(
            index='Classifier', 
            columns='Embedding', 
            values='Training Time (s)',
            aggfunc='mean'
        )
        ax = pivot_df.plot(kind='bar', rot=0)
        ax.set_title('Average Training Time by Embedding and Classifier')
        ax.set_ylabel('Time (seconds)')
        plt.legend(title='Embedding')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_time.png'), dpi=300)
        plt.close()
        
        # Inference time plot
        plt.figure(figsize=(12, 8))
        pivot_df = df.pivot_table(
            index='Classifier', 
            columns='Embedding', 
            values='Inference Time (s)',
            aggfunc='mean'
        )
        ax = pivot_df.plot(kind='bar', rot=0)
        ax.set_title('Average Inference Time by Embedding and Classifier')
        ax.set_ylabel('Time (seconds)')
        plt.legend(title='Embedding')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'inference_time.png'), dpi=300)
        plt.close()
    
    def _generate_html_report(self, all_results, df, dataset_managers):
        """Generate a comprehensive HTML report with modified F1 plot layout"""
        with open(os.path.join(self.output_dir, 'ablation_report.html'), 'w') as f:
            f.write('<html><head><title>Ablation Study Report</title>')
            f.write('<style>body{font-family:Arial;max-width:1200px;margin:0 auto;padding:20px}')
            f.write('table{border-collapse:collapse;width:100%;margin:20px 0}')
            f.write('th,td{border:1px solid #ddd;padding:8px;text-align:left}')
            f.write('th{background-color:#f2f2f2}')
            f.write('img{max-width:100%;height:auto}')
            f.write('h1,h2,h3{color:#333}')
            f.write('.timing-grid{display:grid;grid-template-columns:repeat(auto-fit, minmax(500px, 1fr));gap:20px}')
            f.write('.dataset-grid{display:grid;grid-template-columns:repeat(2, 1fr);gap:15px}')
            f.write('</style></head><body>')
            
            # Title
            f.write('<h1>Segmentation Model Ablation Study Report</h1>')
            
            # Summary
            f.write('<h2>Summary</h2>')
            f.write(f'<p>This report presents the results of an ablation study for segmentation models using different embeddings and classifiers.</p>')
            f.write(f'<p>Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
            
            # Dataset information
            f.write('<h2>Datasets</h2>')
            f.write('<table><tr><th>Dataset</th><th>Samples</th><th>Class ID</th></tr>')
            for name, manager in dataset_managers.items():
                info = manager.get_dataset_info()
                f.write(f'<tr><td>{name}</td><td>{info["total_samples"]}</td><td>{info["class_id"]}</td></tr>')
            f.write('</table>')
            
            # Per-dataset results
            f.write('<h2>Per-Dataset Results</h2>')
            f.write(df.to_html(index=False, float_format='%.4f'))
            
            # F1 score plots - MODIFIED LAYOUT
            f.write('<h3>F1 Score Visualizations</h3>')
            
            # Large overall average plot first
            f.write('<div style="text-align:center;margin:20px">')
            f.write('<h4>Average F1 Score Across All Datasets</h4>')
            f.write('<img src="f1_by_embedding_classifier_avg.png" alt="Average F1 by Embedding and Classifier" style="max-width:90%"/>')
            f.write('</div>')
            
            # Per-dataset F1 visualizations - two plots per row
            f.write('<h4>F1 Scores by Individual Dataset</h4>')
            f.write('<div class="dataset-grid">')
            
            # Get list of datasets
            datasets = df['Dataset'].unique()
            
            for dataset in datasets:
                f.write(f'<div style="text-align:center;margin:10px">')
                f.write(f'<img src="f1_by_embedding_classifier_{dataset}.png" alt="F1 for {dataset}"/>')
                f.write(f'<p>{dataset}</p>')
                f.write('</div>')
            
            # If odd number of datasets, add an empty div to maintain grid layout
            if len(datasets) % 2 == 1:
                f.write('<div></div>')
                
            f.write('</div>')  # Close the dataset-grid

            # Add training size analysis if available
            if 'training_size' in all_results:
                self._add_training_size_section(f, all_results)

            # Add timing section
            f.write('<h2>Computational Performance</h2>')
            f.write('<p>This section analyzes the computational efficiency of different model combinations.</p>')

            # Add timing visualizations
            f.write('<h3>Training and Inference Time</h3>')
            f.write('<div style="display:flex;flex-wrap:wrap;justify-content:space-between">')
            f.write('<div style="flex:1;min-width:45%;margin:10px"><img src="training_time.png" alt="Training Time"/></div>')
            f.write('<div style="flex:1;min-width:45%;margin:10px"><img src="inference_time.png" alt="Inference Time"/></div>')
            f.write('</div>')

            # Add timing table
            f.write('<h3>Timing Metrics by Model</h3>')
            timing_df = df[['Dataset', 'Embedding', 'Classifier', 'F1 Score', 'Training Time (s)', 'Inference Time (s)']]
            timing_df = timing_df.sort_values(by=['Embedding', 'Classifier', 'Dataset'])
            f.write(timing_df.to_html(index=False, float_format='%.4f'))
            f.write('</body></html>')
    
    def _add_training_size_section(self, f, all_results):
        """Add training size analysis section to the report, with classifier-focused plots"""
        f.write('<h2>Training Size Analysis</h2>')
        f.write('<p>This section shows how model performance scales with the number of training images.</p>')
        
        # Create a table summarizing the best models for few-shot learning
        f.write('<h3>Best Models for Few-Shot Learning</h3>')
        f.write('<table><tr><th>Dataset</th><th>Training Size</th><th>Best Embedding</th><th>Best Classifier</th><th>F1 Score</th></tr>')
        
        embeddings_set = set()
        classifiers_set = set()
        datasets_set = set()
        
        # Collect all unique embeddings, classifiers, and datasets
        for dataset_name, dataset_results in all_results['training_size'].items():
            datasets_set.add(dataset_name)
            for embedding_name in dataset_results.keys():
                embeddings_set.add(embedding_name)
                for classifier_name in dataset_results[embedding_name].keys():
                    classifiers_set.add(classifier_name)
        
        for dataset_name, dataset_results in all_results['training_size'].items():
            # Get all training sizes from the first available model
            first_emb = next(iter(dataset_results.values()))
            first_cls = next(iter(first_emb.values()))
            train_sizes = sorted([int(ts) for ts in first_cls.keys() if ts.isdigit()])
            
            for train_size in train_sizes:
                # Find best model for this dataset and training size
                best_f1 = -1
                best_model = {'embedding': '', 'classifier': '', 'f1': 0}
                
                for embedding_name, embedding_results in dataset_results.items():
                    for classifier_name, size_metrics in embedding_results.items():
                        if str(train_size) in size_metrics:
                            f1 = size_metrics[str(train_size)].get('mask_f1', 0)
                            if f1 > best_f1:
                                best_f1 = f1
                                best_model = {
                                    'embedding': embedding_name,
                                    'classifier': classifier_name,
                                    'f1': f1
                                }
                
                f.write(f'<tr><td>{dataset_name}</td><td>{train_size}</td><td>{best_model["embedding"]}</td>')
                f.write(f'<td>{best_model["classifier"]}</td><td>{best_model["f1"]:.4f}</td></tr>')
        
        f.write('</table>')
        
        # Create one plot per classifier, showing average performance across datasets for each embedding
        f.write('<h3>Learning Curves by Classifier</h3>')
        f.write('<p>These plots show how average F1 score improves with more training data for each embedding type, grouped by classifier.</p>')
        
        # Create a directory for learning curve plots
        learning_curves_dir = os.path.join(self.output_dir, 'learning_curves')
        os.makedirs(learning_curves_dir, exist_ok=True)
        
        # For each classifier, create one plot showing all embeddings
        for classifier_name in classifiers_set:
            plt.figure(figsize=(10, 6))
            
            # Dict to store aggregated data for each embedding
            embedding_data = {emb: {'sizes': [], 'f1s': []} for emb in embeddings_set}
            
            # Collect data points for each embedding across all datasets
            for dataset_name in datasets_set:
                if dataset_name not in all_results['training_size']:
                    continue
                
                dataset_results = all_results['training_size'][dataset_name]
                
                for embedding_name in embeddings_set:
                    if embedding_name not in dataset_results or classifier_name not in dataset_results[embedding_name]:
                        continue
                    
                    size_metrics = dataset_results[embedding_name][classifier_name]
                    
                    for size_str, metrics in size_metrics.items():
                        if size_str.isdigit():  # Skip non-numeric keys
                            size = int(size_str)
                            f1 = metrics.get('mask_f1', 0)
                            embedding_data[embedding_name]['sizes'].append(size)
                            embedding_data[embedding_name]['f1s'].append(f1)
            
            # Plot a line for each embedding, showing average performance by training size
            for embedding_name, data in embedding_data.items():
                if not data['sizes']:
                    continue  # Skip if no data
                    
                # Group by size and calculate average
                df = pd.DataFrame({'size': data['sizes'], 'f1': data['f1s']})
                
                if df.empty:
                    continue
                    
                avg_by_size = df.groupby('size')['f1'].mean().reset_index()
                
                # If we have at least 2 points, we can plot a line
                if len(avg_by_size) >= 2:
                    # Sort by size
                    avg_by_size = avg_by_size.sort_values('size')
                    
                    # Plot line for this embedding
                    plt.plot(avg_by_size['size'], avg_by_size['f1'], 'o-', linewidth=2, 
                            label=embedding_name, markersize=6)
            
            # Add reference line for maximum performance if available
            full_training_results = all_results.get('per_dataset', {})
            if full_training_results:
                max_f1s = []
                
                for dataset_name in datasets_set:
                    if dataset_name in full_training_results:
                        for embedding_name in embeddings_set:
                            if embedding_name in full_training_results[dataset_name]:
                                if classifier_name in full_training_results[dataset_name][embedding_name]:
                                    max_f1 = full_training_results[dataset_name][embedding_name][classifier_name].get('mask_f1', 0)
                                    max_f1s.append(max_f1)
                
                if max_f1s:
                    avg_max_f1 = np.mean(max_f1s)
                    plt.axhline(y=avg_max_f1, linestyle='--', color='black', alpha=0.5,
                            label=f'Avg. Full Training ({avg_max_f1:.3f})')
            
            plt.xlabel('Number of Training Images')
            plt.ylabel('Average F1 Score')
            plt.title(f'Learning Curve for {classifier_name} Classifier')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Use a logarithmic x-axis if we have a wide range of training sizes
            sizes = []
            for data in embedding_data.values():
                sizes.extend(data['sizes'])
            
            if sizes and max(sizes) / min(sizes) > 10:
                plt.xscale('log')
                # Add custom grid lines for log scale
                import matplotlib.ticker as ticker
                plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
                plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())
            
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f'{classifier_name}_learning_curve.png'
            plt.savefig(os.path.join(learning_curves_dir, plot_filename), dpi=300)
            plt.close()
        
        # Add the plots to the HTML report
        f.write('<div style="display:flex;flex-wrap:wrap;justify-content:space-around">')
        for classifier_name in classifiers_set:
            plot_filename = f'{classifier_name}_learning_curve.png'
            plot_path = os.path.join('learning_curves', plot_filename)
            
            if os.path.exists(os.path.join(self.output_dir, plot_path)):
                f.write(f'<div style="flex:1;min-width:45%;max-width:45%;margin:15px;text-align:center">')
                f.write(f'<img src="{plot_path}" alt="Learning Curve: {classifier_name}" style="max-width:100%"/>')
                f.write(f'<p>{classifier_name} Classifier</p>')
                f.write('</div>')
        f.write('</div>')
        
        # Add few-shot insights section
        f.write('<h3>Few-Shot Learning Insights</h3>')
        f.write('<p>This section highlights which models perform best with limited training data.</p>')
        
        # Find best model for minimal training data
        min_train_size = float('inf')
        for dataset_results in all_results['training_size'].values():
            for embedding_results in dataset_results.values():
                for size_metrics in embedding_results.values():
                    sizes = [int(s) for s in size_metrics.keys() if s.isdigit()]
                    if sizes:
                        min_train_size = min(min_train_size, min(sizes))
        
        # Collect results for the minimum training size
        few_shot_results = {}
        for dataset_name, dataset_results in all_results['training_size'].items():
            for embedding_name, embedding_results in dataset_results.items():
                for classifier_name, size_metrics in embedding_results.items():
                    if str(min_train_size) in size_metrics:
                        key = f"{embedding_name}_{classifier_name}"
                        if key not in few_shot_results:
                            few_shot_results[key] = {
                                'embedding': embedding_name,
                                'classifier': classifier_name,
                                'f1_scores': [],
                                'datasets': []
                            }
                        few_shot_results[key]['f1_scores'].append(size_metrics[str(min_train_size)].get('mask_f1', 0))
                        few_shot_results[key]['datasets'].append(dataset_name)
        
        # Calculate average performance with minimal training
        for key, results in few_shot_results.items():
            results['avg_f1'] = np.mean(results['f1_scores']) if results['f1_scores'] else 0
        
        # Sort by average F1 score
        sorted_few_shot = sorted(few_shot_results.items(), key=lambda x: x[1]['avg_f1'], reverse=True)
        
        # Show the top 3 models for few-shot learning
        f.write('<p><strong>Best models for minimal training data (' + str(min_train_size) + ' images):</strong></p>')
        f.write('<ol>')
        for key, results in sorted_few_shot[:3]:
            f.write(f'<li><strong>{results["embedding"]} + {results["classifier"]}</strong>: ')
            f.write(f'Average F1 Score: {results["avg_f1"]:.4f} across {len(results["datasets"])} datasets</li>')
        f.write('</ol>')
        
    
    def add_consistency_section(self, consistency_results):
        """
        Add model consistency analysis to the report
        
        Args:
            consistency_results: Dictionary with consistency metrics
        """
        report_path = os.path.join(self.output_dir, 'ablation_report.html')
        
        # Check if report exists
        if not os.path.exists(report_path):
            print("Error: Report not found. Generate full report first.")
            return
        
        # Read existing report
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        # Create consistency section
        consistency_section = '<h2>Model Consistency Analysis</h2>'
        consistency_section += '<p>This section analyzes how consistently different model combinations perform across datasets.</p>'
        
        # CV table
        consistency_section += '<h3>Performance Variability (Coefficient of Variation)</h3>'
        consistency_section += '<p>Lower values indicate more consistent performance across datasets.</p>'
        consistency_section += '<table><tr><th>Embedding</th><th>Classifier</th><th>Mean F1</th><th>CV</th><th>Min F1</th><th>Max F1</th><th>Range</th></tr>'
        
        for _, metrics in sorted(consistency_results['model_consistency'].items(), 
                               key=lambda x: x[1]['cv']):
            consistency_section += f'''<tr>
                <td>{metrics['embedding']}</td>
                <td>{metrics['classifier']}</td>
                <td>{metrics['mean_f1']:.4f}</td>
                <td>{metrics['cv']:.4f}</td>
                <td>{metrics['min_f1']:.4f}</td>
                <td>{metrics['max_f1']:.4f}</td>
                <td>{metrics['range']:.4f}</td>
            </tr>'''
        
        consistency_section += '</table>'
        
        # Images
        consistency_section += '<h3>Visualizations</h3>'
        consistency_section += '<div style="text-align:center;margin:10px">'
        consistency_section += '<img src="model_consistency_cv.png" alt="Model Consistency" style="max-width:90%"/>'
        consistency_section += '</div>'
        
        # Insert at end of document (replacing conclusion)
        end_body_tag = '</body></html>'
        new_report = report_content.replace(end_body_tag, consistency_section + end_body_tag)
        
        # Write updated report
        with open(report_path, 'w') as f:
            f.write(new_report)
    
    def plot_consistency_metrics(self, consistency_results):
        """
        Create visualizations for consistency metrics
        
        Args:
            consistency_results: Dictionary with consistency metrics
        """
        # 1. Coefficient of Variation plot
        plt.figure(figsize=(14, 10))
        model_metrics = consistency_results['model_consistency']
        
        # Extract data for plotting
        models = []
        cvs = []
        mean_f1s = []
        
        for combo, metrics in model_metrics.items():
            models.append(f"{metrics['embedding']}\n{metrics['classifier']}")
            cvs.append(metrics['cv'])
            mean_f1s.append(metrics['mean_f1'])
        
        # Sort by CV (lower is better)
        sorted_indices = np.argsort(cvs)
        models = [models[i] for i in sorted_indices]
        cvs = [cvs[i] for i in sorted_indices]
        mean_f1s = [mean_f1s[i] for i in sorted_indices]
        
        # Create plot
        fig, ax1 = plt.subplots(figsize=(14, 10))
        
        # Plot CV bars
        bars = ax1.bar(models, cvs, alpha=0.7, color='skyblue')
        ax1.set_ylabel('Coefficient of Variation (lower is better)', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title('Model Consistency Across Datasets', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        # Add mean F1 as a line on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(models, mean_f1s, 'ro-', linewidth=2)
        ax2.set_ylabel('Mean F1 Score', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add a grid for better readability
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, color='skyblue', alpha=0.7),
            Line2D([0], [0], color='r', marker='o', linewidth=2)
        ]
        ax1.legend(legend_elements, ['Coefficient of Variation', 'Mean F1 Score'], 
                  loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_consistency_cv.png'), dpi=300)
        plt.close()

    def analyze_model_rank_stability(self, df):
        """
        Analyze how consistently each model combination ranks across datasets
        
        Args:
            df: DataFrame with ablation results
            
        Returns:
            Dictionary with rank stability metrics for each model
        """
        # Create a composite key for model combinations
        df['Model'] = df['Embedding'] + '_' + df['Classifier']
        
        # Calculate ranks within each dataset
        rank_df = df.copy()
        rank_df['Rank'] = rank_df.groupby('Dataset')['F1 Score'].rank(ascending=False, method='min')
        
        # Group by model combination to analyze rank stability
        model_stability = {}
        
        for model, group in rank_df.groupby('Model'):
            embedding, classifier = model.split('_')
            
            # Calculate metrics on ranks
            ranks = group['Rank'].values
            datasets = group['Dataset'].values
            f1_scores = group['F1 Score'].values
            
            model_stability[model] = {
                'embedding': embedding,
                'classifier': classifier,
                'datasets': list(datasets),
                'ranks': list(ranks.astype(int)),
                'f1_scores': list(f1_scores),
                'mean_rank': np.mean(ranks),
                'median_rank': np.median(ranks),
                'std_rank': np.std(ranks),
                'min_rank': np.min(ranks),
                'max_rank': np.max(ranks),
                'rank_range': np.max(ranks) - np.min(ranks),
                'rank_cv': np.std(ranks) / np.mean(ranks) if np.mean(ranks) > 0 else 0,
                'consistent_top_3': np.all(ranks <= 3),
                'datasets_count': len(datasets)
            }
        
        return model_stability

    def plot_rank_stability(self, model_stability):
        """
        Create visualizations for rank stability metrics
        
        Args:
            model_stability: Dictionary with rank stability metrics
        """
        # 1. Rank variability chart
        plt.figure(figsize=(14, 8))
        
        # Extract data for plotting
        models = []
        mean_ranks = []
        rank_ranges = []
        
        for model_name, metrics in sorted(model_stability.items(), 
                                        key=lambda x: x[1]['mean_rank']):
            models.append(f"{metrics['embedding']}\n{metrics['classifier']}")
            mean_ranks.append(metrics['mean_rank'])
            rank_ranges.append(metrics['rank_range'])
        
        # Plot mean ranks
        x = np.arange(len(models))
        plt.bar(x, mean_ranks, width=0.6, yerr=rank_ranges, 
                capsize=5, color='lightblue', label='Mean Rank')
        
        # Add labels and formatting
        plt.axhline(y=3.5, color='r', linestyle='--', 
                label='Top 3 Threshold')
        plt.xlabel('Model')
        plt.ylabel('Rank (lower is better)')
        plt.title('Model Rank Stability Across Datasets')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'model_rank_stability.png'), dpi=300)
        plt.close()
        
        # 2. Rank distribution plot (box plot)
        plt.figure(figsize=(14, 8))
        
        # Prepare data for box plot
        boxplot_data = []
        boxplot_labels = []
        
        for model_name, metrics in sorted(model_stability.items(), 
                                        key=lambda x: x[1]['mean_rank']):
            if len(metrics['ranks']) >= 2:  # Need at least 2 points for a box
                boxplot_data.append(metrics['ranks'])
                boxplot_labels.append(f"{metrics['embedding']}\n{metrics['classifier']}")
        
        # Create box plot
        plt.boxplot(boxplot_data, labels=boxplot_labels, vert=True)
        plt.axhline(y=3.5, color='r', linestyle='--', 
                label='Top 3 Threshold')
        plt.ylabel('Rank (lower is better)')
        plt.title('Rank Distribution Across Datasets')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'rank_distribution.png'), dpi=300)
        plt.close()

    def add_rank_stability_section(self, model_rank_stability):
        """
        Add model rank stability analysis to the report
        
        Args:
            model_rank_stability: Dictionary with rank stability metrics from analyze_model_rank_stability method
        """
        report_path = os.path.join(self.output_dir, 'ablation_report.html')
        
        # Check if report exists
        if not os.path.exists(report_path):
            print("Error: Report not found. Generate full report first.")
            return
        
        # Read existing report
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        # Create rank stability section
        stability_section = '<h2>Model Rank Stability Analysis</h2>'
        stability_section += '<p>This section analyzes how consistently each model combination ranks across different datasets.</p>'
        
        # Rank stability table
        stability_section += '<h3>Rank Stability Metrics</h3>'
        stability_section += '<p>Lower mean rank is better. Lower rank range indicates more consistent ranking across datasets.</p>'
        stability_section += '<table><tr><th>Embedding</th><th>Classifier</th><th>Mean Rank</th><th>Median Rank</th>'
        stability_section += '<th>Rank Range</th><th>Std Dev</th><th>Consistently in Top 3</th></tr>'
        
        # Sort by mean rank (best to worst)
        for model_name, metrics in sorted(model_rank_stability.items(), 
                                        key=lambda x: x[1]['mean_rank']):
            top_3 = "Yes" if metrics['consistent_top_3'] else "No"
            stability_section += f'''<tr>
                <td>{metrics['embedding']}</td>
                <td>{metrics['classifier']}</td>
                <td>{metrics['mean_rank']:.2f}</td>
                <td>{metrics['median_rank']:.1f}</td>
                <td>{metrics['rank_range']}</td>
                <td>{metrics['std_rank']:.2f}</td>
                <td>{top_3}</td>
            </tr>'''
        
        stability_section += '</table>'
        
        # Add detailed rank table showing exact ranks per dataset
        stability_section += '<h3>Per-Dataset Rankings</h3>'
        stability_section += '<p>This table shows the exact rank of each model combination on each dataset (1 is best).</p>'
        stability_section += '<div style="overflow-x:auto;"><table><tr><th>Embedding</th><th>Classifier</th>'
        
        # Find all unique datasets
        all_datasets = set()
        for metrics in model_rank_stability.values():
            all_datasets.update(metrics['datasets'])
        
        # Sort datasets for consistent column order
        all_datasets = sorted(list(all_datasets))
        
        # Add dataset columns to header
        for dataset in all_datasets:
            stability_section += f'<th>{dataset}</th>'
        stability_section += '</tr>'
        
        # Add rows for each model
        for model_name, metrics in sorted(model_rank_stability.items(), 
                                        key=lambda x: x[1]['mean_rank']):
            stability_section += f'<tr><td>{metrics["embedding"]}</td><td>{metrics["classifier"]}</td>'
            
            # Create a map of dataset to rank for this model
            dataset_to_rank = {metrics['datasets'][i]: int(metrics['ranks'][i]) 
                            for i in range(len(metrics['datasets']))}
            
            # Add rank for each dataset
            for dataset in all_datasets:
                if dataset in dataset_to_rank:
                    rank = dataset_to_rank[dataset]
                    # Highlight top 3 ranks with color
                    if rank <= 3:
                        cell_style = f' style="background-color:rgba(0,200,0,{0.9 - 0.2*rank});"'
                    else:
                        cell_style = ''
                    stability_section += f'<td{cell_style}>{rank}</td>'
                else:
                    stability_section += '<td>-</td>'  # Missing data
            
            stability_section += '</tr>'
        
        stability_section += '</table></div>'
        
        # Images
        stability_section += '<h3>Visualizations</h3>'
        stability_section += '<div style="display:flex;flex-wrap:wrap;justify-content:space-between">'
        stability_section += '<div style="flex:1;min-width:45%;margin:10px"><img src="model_rank_stability.png" alt="Model Rank Stability"/></div>'
        stability_section += '<div style="flex:1;min-width:45%;margin:10px"><img src="rank_distribution.png" alt="Rank Distribution"/></div>'
        stability_section += '</div>'
        
        # Insights section
        stability_section += '<h3>Key Insights</h3>'
        
        # Find the most stable models (lowest rank range or std)
        most_stable_models = sorted(model_rank_stability.items(), 
                                key=lambda x: (x[1]['rank_range'], x[1]['std_rank']))[:3]
        
        # Find the best performing models (lowest mean rank)
        best_models = sorted(model_rank_stability.items(), 
                        key=lambda x: x[1]['mean_rank'])[:3]
        
        # Find consistently top-performing models
        top_consistent = [model for model, metrics in model_rank_stability.items() 
                        if metrics['consistent_top_3']]
        
        # Generate insights
        stability_section += '<ul>'
        
        if top_consistent:
            consistency_text = ', '.join([f"{model_rank_stability[model]['embedding']} + {model_rank_stability[model]['classifier']}" 
                                        for model in top_consistent[:3]])
            stability_section += f'<li><strong>Consistently Top Performing:</strong> {consistency_text} ranked in the top 3 across all datasets.</li>'
        
        # Most stable models
        if most_stable_models:
            most_stable = most_stable_models[0][1]
            stability_section += f'<li><strong>Most Stable Model:</strong> {most_stable["embedding"]} + {most_stable["classifier"]} '
            stability_section += f'showed the most consistent ranking across datasets (range: {most_stable["rank_range"]}, std: {most_stable["std_rank"]:.2f}).</li>'
        
        # Best overall models
        if best_models:
            best_model = best_models[0][1]
            stability_section += f'<li><strong>Best Overall Model:</strong> {best_model["embedding"]} + {best_model["classifier"]} '
            stability_section += f'had the lowest average rank of {best_model["mean_rank"]:.2f} across datasets.</li>'
        
        stability_section += '</ul>'
        
        # Insert at end of document before </body> tag
        end_body_tag = '</body></html>'
        new_report = report_content.replace(end_body_tag, stability_section + end_body_tag)
        
        # Write updated report
        with open(report_path, 'w') as f:
            f.write(new_report)

    def add_pipeline_comparison(self, ideal_results, realistic_results):
        """
        Add a comparison between ideal (full recall) and realistic (SAM2-based) pipeline results
        
        Args:
            ideal_results: Dictionary with results from full recall prediction track
            realistic_results: Dictionary with results from realistic pipeline with SAM2 masks
        """
        report_path = os.path.join(self.output_dir, 'ablation_report.html')
        
        # Check if report exists
        if not os.path.exists(report_path):
            print("Error: Report not found. Generate full report first.")
            return
        
        # Read existing report
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        # Create pipeline comparison section
        comparison_section = '<h2>Pipeline Comparison: Ideal vs. Realistic</h2>'
        comparison_section += '<p>This section compares the performance between two tracks:</p>'
        comparison_section += '<ul>'
        comparison_section += '<li><strong>Ideal Track:</strong> Predictions on data with full recall (perfect mask proposals)</li>'
        comparison_section += '<li><strong>Realistic Track:</strong> Predictions using the full pipeline with SAM2-generated masks</li>'
        comparison_section += '</ul>'
        comparison_section += '<p>This comparison helps quantify how much performance is lost due to imperfect mask proposals from SAM2.</p>'
        
        # Generate comparison visualizations
        self._generate_pipeline_comparison_plots(ideal_results, realistic_results)
        
        # Create comparison table
        comparison_section += '<h3>Performance Gap Analysis</h3>'
        
        # Create a table showing the difference between ideal and realistic performance
        comparison_section += '<table>'
        comparison_section += '<tr><th>Dataset</th><th>Model</th><th>Ideal F1</th><th>Realistic F1</th>'
        comparison_section += '<th>Absolute Gap</th><th>Relative Gap (%)</th><th>SAM2 Error Attribution (%)</th></tr>'
        
        # Process comparison data
        for dataset_name, ideal_dataset_results in ideal_results.items():
            if dataset_name not in realistic_results:
                continue
                
            realistic_dataset_results = realistic_results[dataset_name]
            
            for embedding_name, ideal_embedding_results in ideal_dataset_results.items():
                if embedding_name not in realistic_dataset_results:
                    continue
                    
                realistic_embedding_results = realistic_dataset_results[embedding_name]
                
                for classifier_name, ideal_metrics in ideal_embedding_results.items():
                    if classifier_name not in realistic_embedding_results:
                        continue
                        
                    realistic_metrics = realistic_embedding_results[classifier_name]
                    
                    # Get F1 scores
                    ideal_f1 = ideal_metrics.get('mask_f1', 0)
                    realistic_f1 = realistic_metrics.get('mask_f1', 0)
                    
                    # Calculate gaps
                    absolute_gap = ideal_f1 - realistic_f1
                    relative_gap = (absolute_gap / ideal_f1 * 100) if ideal_f1 > 0 else 0
                    
                    # Calculate theoretical maximum (oracle performance would be 1.0)
                    theoretical_gap = 1.0 - ideal_f1
                    
                    # Calculate how much of the gap between oracle (1.0) and realistic is due to SAM2
                    # The remaining gap would be due to the classification method
                    sam2_attribution = (absolute_gap / (theoretical_gap + absolute_gap) * 100) if (theoretical_gap + absolute_gap) > 0 else 0
                    
                    model_name = f"{embedding_name} + {classifier_name}"
                    
                    # Add row to table
                    comparison_section += f'<tr>'
                    comparison_section += f'<td>{dataset_name}</td>'
                    comparison_section += f'<td>{model_name}</td>'
                    comparison_section += f'<td>{ideal_f1:.4f}</td>'
                    comparison_section += f'<td>{realistic_f1:.4f}</td>'
                    comparison_section += f'<td>{absolute_gap:.4f}</td>'
                    comparison_section += f'<td>{relative_gap:.2f}%</td>'
                    comparison_section += f'<td>{sam2_attribution:.2f}%</td>'
                    comparison_section += f'</tr>'
        
        comparison_section += '</table>'
        
        # Add visualizations - one per line
        comparison_section += '<h3>Visualizations</h3>'
        
        # Overall comparison plot
        comparison_section += '<div style="margin:20px 0;text-align:center">'
        comparison_section += '<h4>Overall Performance Comparison</h4>'
        comparison_section += '<img src="pipeline_comparison_overall.png" alt="Pipeline Comparison" style="max-width:90%;"/>'
        comparison_section += '</div>'
        
        # Dataset-specific comparison
        comparison_section += '<div style="margin:20px 0;text-align:center">'
        comparison_section += '<h4>Dataset-Specific Comparison</h4>'
        comparison_section += '<img src="pipeline_comparison_by_dataset.png" alt="Pipeline Comparison by Dataset" style="max-width:90%;"/>'
        comparison_section += '</div>'
        
        # SAM2 attribution heatmap
        comparison_section += '<div style="margin:20px 0;text-align:center">'
        comparison_section += '<h4>SAM2 Error Attribution by Dataset and Embedding</h4>'
        comparison_section += '<img src="sam2_attribution_heatmap.png" alt="SAM2 Attribution Heatmap" style="max-width:90%;"/>'
        comparison_section += '</div>'
        
        # Add insights section
        comparison_section += '<h3>Key Insights</h3>'
        comparison_section += '<ul>'
        comparison_section += '<li><strong>SAM2 Impact:</strong> The difference between ideal and realistic performance shows the impact of using SAM2-generated masks instead of perfect mask proposals.</li>'
        comparison_section += '<li><strong>Error Attribution:</strong> The "SAM2 Error Attribution" shows what percentage of the gap between perfect performance (1.0) and actual performance is due to SAM2\'s mask generation.</li>'
        comparison_section += '<li><strong>Optimization Opportunities:</strong> Models with high SAM2 error attribution would benefit most from improvements to the mask proposal method.</li>'
        comparison_section += '</ul>'
        
        # Insert at end of document before </body> tag
        end_body_tag = '</body></html>'
        new_report = report_content.replace(end_body_tag, comparison_section + end_body_tag)
        
        # Write updated report
        with open(report_path, 'w') as f:
            f.write(new_report)

    def _generate_pipeline_comparison_plots(self, ideal_results, realistic_results):
        """
        Generate visualizations comparing ideal and realistic pipeline performance
        
        Args:
            ideal_results: Dictionary with results from full recall prediction track
            realistic_results: Dictionary with results from realistic pipeline with SAM2 masks
        """
        # Prepare data for plotting
        datasets = []
        models = []
        ideal_f1s = []
        realistic_f1s = []
        
        for dataset_name, ideal_dataset_results in ideal_results.items():
            if dataset_name not in realistic_results:
                continue
                
            realistic_dataset_results = realistic_results[dataset_name]
            
            for embedding_name, ideal_embedding_results in ideal_dataset_results.items():
                if embedding_name not in realistic_dataset_results:
                    continue
                    
                realistic_embedding_results = realistic_dataset_results[embedding_name]
                
                for classifier_name, ideal_metrics in ideal_embedding_results.items():
                    if classifier_name not in realistic_embedding_results:
                        continue
                        
                    realistic_metrics = realistic_embedding_results[classifier_name]
                    
                    # Get F1 scores
                    ideal_f1 = ideal_metrics.get('mask_f1', 0)
                    realistic_f1 = realistic_metrics.get('mask_f1', 0)
                    
                    # Add to data lists
                    datasets.append(dataset_name)
                    models.append(f"{embedding_name}_{classifier_name}")
                    ideal_f1s.append(ideal_f1)
                    realistic_f1s.append(realistic_f1)
        
        # Create DataFrame for easier manipulation
        comparison_df = pd.DataFrame({
            'Dataset': datasets,
            'Model': models,
            'Ideal_F1': ideal_f1s,
            'Realistic_F1': realistic_f1s
        })
        
        # Extract embedding and classifier from model
        comparison_df[['Embedding', 'Classifier']] = comparison_df['Model'].str.split('_', expand=True)
        
        # Calculate gaps
        comparison_df['Absolute_Gap'] = comparison_df['Ideal_F1'] - comparison_df['Realistic_F1']
        comparison_df['Relative_Gap'] = comparison_df['Absolute_Gap'] / comparison_df['Ideal_F1'] * 100
        comparison_df['Relative_Gap'] = comparison_df['Relative_Gap'].fillna(0)
        
        # Calculate SAM2 attribution
        theoretical_gap = 1.0 - comparison_df['Ideal_F1']
        comparison_df['SAM2_Attribution'] = comparison_df['Absolute_Gap'] / (theoretical_gap + comparison_df['Absolute_Gap']) * 100
        comparison_df['SAM2_Attribution'] = comparison_df['SAM2_Attribution'].fillna(0)
        
        # 1. Overall comparison plot
        plt.figure(figsize=(12, 8))
        
        # Group by model and calculate mean
        model_summary = comparison_df.groupby('Model')[['Ideal_F1', 'Realistic_F1', 'Absolute_Gap', 'SAM2_Attribution']].mean().reset_index()
        
        # Sort by ideal F1 score
        model_summary = model_summary.sort_values('Ideal_F1', ascending=False)
        
        # Extract embeddings and classifiers for better labels
        model_summary[['Embedding', 'Classifier']] = model_summary['Model'].str.split('_', expand=True)
        
        # Create bar plot for each model
        x = np.arange(len(model_summary))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(15, 8))
        
        # Plot bars for F1 scores
        bars1 = ax1.bar(x - width/2, model_summary['Ideal_F1'], width, label='Ideal F1', color='#2C7BB6', alpha=0.8)
        bars2 = ax1.bar(x + width/2, model_summary['Realistic_F1'], width, label='Realistic F1', color='#D7191C', alpha=0.8)
        
        # Add labels and grid
        ax1.set_ylabel('F1 Score', fontsize=12)
        ax1.set_title('Comparison of Ideal vs. Realistic Pipeline Performance', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{row['Embedding']}\n{row['Classifier']}" for _, row in model_summary.iterrows()], rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Add a second y-axis for SAM2 attribution
        ax2 = ax1.twinx()
        ax2.plot(x, model_summary['SAM2_Attribution'], 'go-', linewidth=2, markersize=8, label='SAM2 Error Attribution')
        ax2.set_ylabel('SAM2 Error Attribution (%)', color='green', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim([0, 100])
        ax2.legend(loc='upper right')
        
        # Add data labels
        for i, (ideal, realistic) in enumerate(zip(model_summary['Ideal_F1'], model_summary['Realistic_F1'])):
            ax1.annotate(f"{ideal:.2f}", xy=(i - width/2, ideal), xytext=(0, 3), 
                    textcoords="offset points", ha='center', va='bottom', fontsize=8)
            ax1.annotate(f"{realistic:.2f}", xy=(i + width/2, realistic), xytext=(0, 3), 
                    textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pipeline_comparison_overall.png'), dpi=300)
        plt.close()
        
        # 2. Dataset-specific comparison
        # Group by dataset and model to get average performance across datasets
        dataset_model_summary = comparison_df.groupby(['Dataset', 'Embedding'])[['Ideal_F1', 'Realistic_F1', 'Absolute_Gap']].mean().reset_index()
        
        # Create a plot for each dataset side by side
        datasets = dataset_model_summary['Dataset'].unique()
        embeddings = dataset_model_summary['Embedding'].unique()
        
        # Set up plot grid
        fig, axes = plt.subplots(1, len(datasets), figsize=(15, 6), sharey=True)
        if len(datasets) == 1:
            axes = [axes]  # Make sure axes is a list even with one dataset
        
        # Color map for embeddings
        colors = plt.cm.tab10.colors[:len(embeddings)]
        embedding_colors = {embedding: color for embedding, color in zip(embeddings, colors)}
        
        for i, dataset in enumerate(datasets):
            ax = axes[i]
            dataset_data = dataset_model_summary[dataset_model_summary['Dataset'] == dataset]
            
            # Sort by ideal F1
            dataset_data = dataset_data.sort_values('Ideal_F1', ascending=False)
            
            # Width of bars
            width = 0.35
            x = np.arange(len(dataset_data))
            
            # Plot bars
            for j, (_, row) in enumerate(dataset_data.iterrows()):
                embedding = row['Embedding']
                color = embedding_colors[embedding]
                
                # Plot ideal and realistic F1 scores
                ax.bar(j - width/2, row['Ideal_F1'], width, color=color, alpha=0.8, hatch='')
                ax.bar(j + width/2, row['Realistic_F1'], width, color=color, alpha=0.4, hatch='//') 
                
                # Add text for the gap
                gap = row['Absolute_Gap']
                ax.annotate(f"Δ: {gap:.2f}", 
                        xy=(j, (row['Ideal_F1'] + row['Realistic_F1'])/2),
                        xytext=(0, 0), 
                        textcoords="offset points",
                        ha='center', va='center',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Set labels and title
            ax.set_title(dataset)
            ax.set_xticks(x)
            ax.set_xticklabels([row['Embedding'] for _, row in dataset_data.iterrows()], rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            if i == 0:
                ax.set_ylabel('F1 Score')
        
        # Create a custom legend for all axes
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, color='gray', alpha=0.8, label='Ideal (Perfect Masks)'),
            plt.Rectangle((0,0), 1, 1, color='gray', alpha=0.4, hatch='//', label='Realistic (SAM2 Masks)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Add space at the bottom for the legend
        plt.savefig(os.path.join(self.output_dir, 'pipeline_comparison_by_dataset.png'), dpi=300)
        plt.close()
        
        # 3. Create a heatmap of SAM2 attribution by dataset and model
        heatmap_data = comparison_df.pivot_table(
            index='Dataset', 
            columns='Embedding',
            values='SAM2_Attribution',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # Create heatmap
        im = ax.imshow(heatmap_data.values, cmap='YlOrRd')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('SAM2 Error Attribution (%)')
        
        # Add labels
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticklabels(heatmap_data.index)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                text = ax.text(j, i, f"{value:.1f}%", 
                        ha="center", va="center", 
                        color="white" if value > 50 else "black")
        
        ax.set_title("SAM2 Error Attribution by Dataset and Embedding")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sam2_attribution_heatmap.png'), dpi=300)
        plt.close()