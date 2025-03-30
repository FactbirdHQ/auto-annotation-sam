# Auto-Labeling Experiment Guide

## 1. Embeddings and Classifiers Comparison Study

### Experiment Design
Compare 3 different embeddings and 4 different classifiers (all with the option of PCA) on multiple datasets.

**Embeddings to evaluate:**
- Histogram of Oriented Gradients (HoG)
- CLIP embeddings
- ResNet18 layer features

**Classifiers to evaluate:**
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest

**Datasets:**
- Meatballs
- Doughs
- Cans
- Bottles

**Note:** Class imbalance exists across datasets.

### Evaluation Metrics

#### Core Classification Metrics
- Precision
- Recall
- F1-score
- Balanced Accuracy
- Area Under ROC Curve (AUC-ROC)
- Area Under Precision-Recall Curve (AUC-PR)

#### Additional Metrics
- **Cohen's Kappa:** Measures agreement between predicted and actual classifications while accounting for chance agreement
- **Matthews Correlation Coefficient (MCC):** Useful for imbalanced classes as it considers all four confusion matrix values
- **Computation metrics:**
  - Training time
  - Inference time
  - Memory usage
- **Cross-dataset performance:**
  - Relative Performance Consistency (ranking correlation across datasets)
  - Performance Variability (coefficient of variation across datasets)


### Analysis Framework
1. **Per-class performance:** Examine metrics for each class separately given class imbalance
2. **Cross-validation:** Use stratified k-fold to maintain class distributions
3. **Statistical significance testing:** Use McNemar's test or 5×2cv paired t-test to determine if differences between models are statistically significant
4. **Confusion matrices visualization:** To understand specific error patterns for each combination
5. **Learning curves:** To assess how each embedding+classifier combination scales with training data size

### Cross-Dataset Analysis Details

#### Relative Performance Consistency
This measures if the same embedding/classifier combinations maintain similar rankings across all datasets.

- Calculate performance metrics for each method on each dataset
- Rank the methods within each dataset based on performance
- Calculate rank correlation (Spearman or Kendall's Tau) between datasets
- High correlation values indicate that methods perform in similar relative order across different datasets

#### Performance Variability (Coefficient of Variation)
This measures how consistently a method performs across different datasets in absolute terms.

- For each method, collect its performance scores across all datasets
- Calculate the mean and standard deviation of these scores
- Compute the coefficient of variation (CV = standard deviation / mean)
- Lower CV values indicate methods with more consistent absolute performance across datasets
- Methods with low CV are more reliable regardless of the dataset

## 2. SAM2 Segmentation Tuning Study

### Background
The classifier comparison study assumes SAM2 segmentation as input. If SAM2 fails to detect segments (incomplete recall), downstream classifiers cannot recover those missed objects.

### Two-Stage Evaluation Approach

#### First Stage: SAM2 Segmentation Evaluation
Evaluate SAM2's performance independently with:
- IoU (Intersection over Union)
- Boundary F1-score
- Detection recall (percentage of ground truth objects that SAM detected at all)
- Over-segmentation and under-segmentation rates

#### Second Stage: Classification with Two Parallel Tracks

##### Track 1: Realistic Pipeline Performance
- Use the actual SAM2 segments as they would appear in a real-world pipeline
- Report this as the primary result set since it reflects real-world performance
- Shows true end-to-end performance metrics

##### Track 2: Classifier-Only Performance (Adjusted Metrics)
- Use the combination of SAM2 segments with ground truth to ensure full recall
- Label these as "adjusted metrics" or "classifier-only performance"
- Isolates the classifier performance from segmentation errors
- Shows the theoretical upper bound of your system

### Error Attribution Analysis
Calculate breakdown of error sources:
- Percentage of errors from missed segments (SAM2 failures)
- Percentage of errors from misclassifications (classifier failures)

This analysis helps determine whether to focus improvement efforts on better segmentation or better classification.

## 3. Experimental Protocol

1. **Data Preparation:**
   - Split datasets into training/validation/test sets
   - Ensure stratification to maintain class distributions
   - Document any preprocessing steps applied

2. **Hyperparameter Tuning:**
   - Use grid search or random search with cross-validation
   - Tune hyperparameters separately for each classifier and embedding combination
   - Document optimal hyperparameters for each configuration

3. **Training Procedure:**
   - Document batch sizes, epochs, optimization algorithms
   - Save model checkpoints for reproducibility
   - Log training curves

4. **Evaluation:**
   - Apply consistent evaluation protocol across all combinations
   - Report confidence intervals for key metrics
   - Visualize results with appropriate plots (confusion matrices, ROC curves, etc.)

5. **Ablation Studies:**
   - Test variations of best-performing configurations
   - Assess impact of key parameters or features

## 4. Documentation and Reproducibility

- Document all software dependencies and versions
- Save random seeds for reproducibility
- Archive trained models and embeddings
- Maintain detailed logs of experiments
- Document any data augmentation strategies used
