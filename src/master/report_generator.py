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
        """
        # Create a summary DataFrame for per-dataset results
        df = self._create_summary_dataframe(all_results['per_dataset'])
        
        # Save to CSV
        df.to_csv(os.path.join(self.output_dir, 'ablation_summary.csv'), index=False)
        
        # Create visualizations
        self._plot_f1_by_embedding_classifier(df)
        self._plot_f1_by_dataset(df)
        self._plot_timing_metrics(df)  # New timing plots
        
        # Generate HTML report
        self._generate_html_report(all_results, df, dataset_managers)
        
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
        """Plot F1 scores by embedding and classifier"""
        plt.figure(figsize=(12, 8))
        pivot_df = df.pivot_table(
            index='Classifier', 
            columns='Embedding', 
            values='F1 Score',
            aggfunc='mean'
        )
        ax = pivot_df.plot(kind='bar', rot=0)
        ax.set_title('Average F1 Score by Embedding and Classifier')
        ax.set_ylabel('F1 Score')
        plt.legend(title='Embedding')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_by_embedding_classifier.png'), dpi=300)
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
        
        # Performance vs timing scatter plot
        plt.figure(figsize=(12, 8))
        for embedding in df['Embedding'].unique():
            subset = df[df['Embedding'] == embedding]
            plt.scatter(subset['Inference Time (s)'], subset['F1 Score'], 
                    label=embedding, alpha=0.7, s=100)
        
        plt.xlabel('Inference Time (seconds)')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs. Inference Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_vs_time.png'), dpi=300)
        plt.close()
    
    def _generate_html_report(self, all_results, df, dataset_managers):
        """Generate a comprehensive HTML report"""
        with open(os.path.join(self.output_dir, 'ablation_report.html'), 'w') as f:
            f.write('<html><head><title>Ablation Study Report</title>')
            f.write('<style>body{font-family:Arial;max-width:1200px;margin:0 auto;padding:20px}')
            f.write('table{border-collapse:collapse;width:100%;margin:20px 0}')
            f.write('th,td{border:1px solid #ddd;padding:8px;text-align:left}')
            f.write('th{background-color:#f2f2f2}')
            f.write('img{max-width:100%;height:auto}')
            f.write('h1,h2,h3{color:#333}</style></head><body>')
            
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
            
            # F1 score plots
            f.write('<h3>F1 Score Visualizations</h3>')
            f.write('<div style="display:flex;flex-wrap:wrap;justify-content:space-between">')
            f.write('<div style="flex:1;min-width:45%;margin:10px"><img src="f1_by_embedding_classifier.png" alt="F1 by Embedding and Classifier"/></div>')
            f.write('<div style="flex:1;min-width:45%;margin:10px"><img src="f1_by_dataset.png" alt="F1 by Dataset"/></div>')
            f.write('</div>')

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

            # Add performance vs time tradeoff
            f.write('<h3>Performance vs. Time Trade-off</h3>')
            f.write('<div style="text-align:center;margin:10px"><img src="f1_vs_time.png" alt="F1 vs Time"/></div>')

            # Add timing table
            f.write('<h3>Timing Metrics by Model</h3>')
            timing_df = df[['Dataset', 'Embedding', 'Classifier', 'F1 Score', 'Training Time (s)', 'Inference Time (s)']]
            timing_df = timing_df.sort_values(by=['Embedding', 'Classifier', 'Dataset'])
            f.write(timing_df.to_html(index=False, float_format='%.4f'))
            
            # Conclusion
            f.write('<h2>Conclusion and Recommendations</h2>')
            # Find best combination overall
            if not df.empty:
                best_row = df.loc[df['F1 Score'].idxmax()]
                f.write(f'<p>The best overall performance was achieved using <strong>{best_row["Embedding"]}</strong> embedding with <strong>{best_row["Classifier"]}</strong> classifier on the <strong>{best_row["Dataset"]}</strong> dataset, with an F1 score of <strong>{best_row["F1 Score"]:.4f}</strong>.</p>')
            
            # Dataset-specific recommendations
            f.write('<h3>Dataset-Specific Recommendations</h3>')
            f.write('<table><tr><th>Dataset</th><th>Recommended Embedding</th><th>Recommended Classifier</th><th>F1 Score</th></tr>')
            for dataset in df['Dataset'].unique():
                dataset_df = df[df['Dataset'] == dataset]
                if not dataset_df.empty:
                    best_row = dataset_df.loc[dataset_df['F1 Score'].idxmax()]
                    f.write(f'<tr><td>{dataset}</td><td>{best_row["Embedding"]}</td><td>{best_row["Classifier"]}</td><td>{best_row["F1 Score"]:.4f}</td></tr>')
            f.write('</table>')
            
            f.write('</body></html>')
    
    def _add_training_size_section(self, f, all_results):
        """Add training size analysis section to the report"""
        f.write('<h2>Training Size Analysis</h2>')
        f.write('<p>This section shows how model performance scales with the number of training images.</p>')
        
        # Create a table summarizing the best models for few-shot learning
        f.write('<h3>Best Models for Few-Shot Learning</h3>')
        f.write('<table><tr><th>Dataset</th><th>Training Size</th><th>Best Embedding</th><th>Best Classifier</th><th>F1 Score</th></tr>')
        
        embeddings_set = set()
        classifiers_set = set()
        
        # Collect all unique embeddings and classifiers
        for dataset_results in all_results['training_size'].values():
            for embedding_name in dataset_results.keys():
                embeddings_set.add(embedding_name)
                for classifier_name in dataset_results[embedding_name].keys():
                    classifiers_set.add(classifier_name)
        
        for dataset_name, dataset_results in all_results['training_size'].items():
            # Get all training sizes from the first available model
            first_emb = next(iter(dataset_results.values()))
            first_cls = next(iter(first_emb.values()))
            train_sizes = sorted([int(ts) for ts in first_cls.keys()])
            
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
        
        # Add learning curve plots for each dataset
        f.write('<h3>Learning Curves</h3>')
        f.write('<p>These plots show how F1 score improves with more training data.</p>')
        f.write('<div style="display:flex;flex-wrap:wrap;justify-content:space-between">')
        
        for dataset_name in all_results['training_size'].keys():
            for embedding_name in embeddings_set:
                for classifier_name in classifiers_set:
                    curve_path = f"{dataset_name}_{embedding_name}_{classifier_name}_learning_curve.png"
                    if os.path.exists(os.path.join(self.output_dir, curve_path)):
                        f.write(f'<div style="flex:1;min-width:30%;margin:10px">')
                        f.write(f'<img src="{curve_path}" alt="Learning Curve: {dataset_name} - {embedding_name} - {classifier_name}"/>')
                        f.write(f'<p class="caption">{dataset_name} - {embedding_name} - {classifier_name}</p>')
                        f.write('</div>')
        
        f.write('</div>')
    
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
        
        # Ranking correlation
        consistency_section += '<h3>Ranking Correlation Between Datasets</h3>'
        consistency_section += '<p>Higher values indicate that the relative ranking of models is preserved between datasets.</p>'
        consistency_section += '<table><tr><th>Dataset Pair</th><th>Spearman Rank Correlation</th></tr>'
        
        for pair, corr_data in consistency_results['ranking_correlations'].items():
            consistency_section += f'''<tr>
                <td>{pair}</td>
                <td>{corr_data['correlation']:.4f}</td>
            </tr>'''
        
        consistency_section += '</table>'
        
        # Images
        consistency_section += '<h3>Visualizations</h3>'
        consistency_section += '<div style="display:flex;flex-wrap:wrap;justify-content:space-between">'
        consistency_section += '<div style="flex:1;min-width:45%;margin:10px"><img src="model_consistency_cv.png" alt="Model Consistency"/></div>'
        if 'ranking_correlations' in consistency_results and consistency_results['ranking_correlations']:
            consistency_section += '<div style="flex:1;min-width:45%;margin:10px"><img src="dataset_ranking_correlation.png" alt="Dataset Ranking Correlation"/></div>'
        consistency_section += '</div>'
        
        # Insert before conclusion
        new_report = report_content.replace('<h2>Conclusion and Recommendations</h2>', 
                                           consistency_section + '<h2>Conclusion and Recommendations</h2>')
        
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
        plt.figure(figsize=(12, 8))
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
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot CV bars
        bars = ax1.bar(models, cvs, alpha=0.7, color='skyblue')
        ax1.set_ylabel('Coefficient of Variation (lower is better)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title('Model Consistency Across Datasets')
        plt.xticks(rotation=45, ha='right')
        
        # Add mean F1 as a line on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(models, mean_f1s, 'ro-', linewidth=2)
        ax2.set_ylabel('Mean F1 Score', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_consistency_cv.png'), dpi=300)
        plt.close()
        
        # 2. Ranking correlation chart
        rank_corrs = consistency_results['ranking_correlations']
        if rank_corrs:
            dataset_pairs = list(rank_corrs.keys())
            corr_values = [rc['correlation'] for rc in rank_corrs.values()]
            
            plt.figure(figsize=(10, 8))
            plt.bar(dataset_pairs, corr_values, color='green')
            plt.ylabel("Spearman's Rank Correlation")
            plt.title("Ranking Consistency Between Datasets")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'dataset_ranking_correlation.png'), dpi=300)
            plt.close()