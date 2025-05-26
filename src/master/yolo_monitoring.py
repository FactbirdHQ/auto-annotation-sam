import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

# Define model for NumPyro with constraints to match PyMC version
def gamma_model(data):
    
    # Define priors matching original code
    alpha = numpyro.sample("alpha", dist.Gamma(0.5, 0.5))
    beta = numpyro.sample("beta", dist.Gamma(0.5, 0.5))
    
    # Likelihood
    numpyro.sample("obs", dist.Gamma(alpha, beta), obs=data)

class BayesianYOLOMonitorJAX:
    def __init__(self, min_deviation_threshold=0.2,
                 credible_interval=0.95, mcmc_samples=1000):
        """
        Initialize Bayesian YOLO Performance Monitor with JAX backend
        
        Parameters:
        - min_deviation_threshold: Minimum relative deviation to be considered practically significant
        - credible_interval: Credible interval for anomaly detection (e.g., 0.95 for 95%)
        - mcmc_samples: Number of posterior samples to generate
        """
        self.min_deviation_threshold = min_deviation_threshold 
        
        # Parameters for posterior predictive distribution
        self.credible_interval = credible_interval
        self.mcmc_samples = mcmc_samples
        
        # Initialize storage for results
        self.baseline_clips_raw = []
        self.baseline_mean_variance = None
        self.baseline_std_variance = None
        
        # Posterior samples
        self.alpha_posterior = None
        self.beta_posterior = None
        self.min_log_prob_posterior = None
        self.threshold_posterior = None  # Will be set directly to min_log_prob_posterior
        self.baseline_credible_interval = None
        self.baseline_ci_lower = None
        self.baseline_ci_upper = None
    
    def establish_baseline(self, baseline_clips, visualize=False):
        """
        Establish baseline using frame-level data with JAX-based Bayesian estimation
        
        Parameters:
        - baseline_clips: List of frame variance arrays from good clips
        - visualize: Whether to visualize the baseline distributions
        
        Returns:
        - baseline_info: Dictionary with baseline statistics
        """
        # Store raw clips
        self.baseline_clips_raw = baseline_clips
        
        # Combine all frame variances
        all_frame_variances = []
        for clip in baseline_clips:
            all_frame_variances.extend(clip)
        
        # Convert to JAX array
        data = jnp.array(all_frame_variances)
        
        # Calculate baseline mean variance and std variance
        self.baseline_mean_variance = float(jnp.mean(data))
        self.baseline_std_variance = float(jnp.std(data))
        
        # Start timing
        import time
        start_time = time.time()
        
        print("Starting JAX MCMC sampling...")
        
        # Run MCMC
        kernel = NUTS(gamma_model, init_strategy=init_to_median)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=self.mcmc_samples)
        mcmc.run(jax.random.PRNGKey(0), data)
        
        # Print summary
        mcmc.print_summary()
        
        # Get posterior samples
        samples = mcmc.get_samples()
        
        # Convert to numpy for compatibility
        self.alpha_posterior = np.array(samples["alpha"])
        self.beta_posterior = np.array(samples["beta"])
        
        # Calculate log probs for each clip - more similar to original approach
        min_log_probs = []
        for i, clip in enumerate(baseline_clips):
            clip_data = jnp.array(clip)
            # Calculate log probs for each sample
            frame_log_probs_by_sample = jax.vmap(
                lambda a, b: dist.Gamma(a, b).log_prob(clip_data)
            )(self.alpha_posterior, self.beta_posterior)
            
            # Calculate mean log prob for each sample (like avg_log_prob in original)
            avg_log_probs = jnp.mean(frame_log_probs_by_sample, axis=1)
            min_log_probs.append(avg_log_probs)
        
        # Stack as in original PyMC code and find min along axis 0 (clip dimension)
        stacked = np.stack(min_log_probs)
        self.min_log_prob_posterior = np.min(stacked, axis=0)
        
        # Set threshold directly to min_log_prob_posterior (no buffer adjustment)
        self.threshold_posterior = self.min_log_prob_posterior
        
        # Generate posterior predictive distribution samples for p-value calculation
        n_predictive = 50000  # Increase for more stable intervals
        self.predictive_samples = np.array([
            np.random.gamma(alpha, 1/beta) 
            for alpha, beta in zip(np.random.choice(self.alpha_posterior, n_predictive), 
                                  np.random.choice(self.beta_posterior, n_predictive))
        ])
        
        # Calculate credible interval (still kept for backward compatibility)
        self.baseline_ci_lower = np.percentile(self.predictive_samples, (1 - self.credible_interval) * 100 / 2)
        self.baseline_ci_upper = np.percentile(self.predictive_samples, 100 - (1 - self.credible_interval) * 100 / 2)
        self.baseline_credible_interval = (self.baseline_ci_lower, self.baseline_ci_upper)
        
        # Report timing
        elapsed = time.time() - start_time
        print(f"JAX sampling completed in {elapsed:.2f} seconds ({self.mcmc_samples/elapsed:.2f} samples/sec)")
        
        if visualize:
            self.visualize_baseline()
        
        # Return info
        return {
            'alpha_mean': float(np.mean(self.alpha_posterior)),
            'beta_mean': float(np.mean(self.beta_posterior)),
            'baseline_mean_variance': self.baseline_mean_variance,
            'min_log_prob_mean': float(np.mean(self.min_log_prob_posterior)),
            'threshold_mean': float(np.mean(self.threshold_posterior)),
            'threshold_95ci': np.percentile(self.threshold_posterior, [2.5, 97.5]).tolist(),
            'baseline_credible_interval': self.baseline_credible_interval,
        }
    
    def check_performance_probabilistic(self, clip_variances):
        """
        Method 1: Probabilistic approach using buffer probabilities
        Returns decision based on P(below threshold) > 0.5
        """
        mean_variance = np.mean(clip_variances)
        
        # Calculate relative deviation
        relative_deviation = (mean_variance - self.baseline_mean_variance) / self.baseline_mean_variance
        
        # Practical significance check
        is_practically_significant = relative_deviation > self.min_deviation_threshold
        is_worse_direction = relative_deviation > 0
        
        if not (is_practically_significant and is_worse_direction):
            return {
                'method': 'probabilistic',
                'has_issue': False,
                'p_below_threshold': 0.0,
                'mean_variance': mean_variance,
                'relative_deviation': relative_deviation
            }
        
        # Calculate log-likelihood of current clip with posterior samples
        clip_data = jnp.array(clip_variances)
        frame_log_probs_by_sample = jax.vmap(
            lambda a, b: dist.Gamma(a, b).log_prob(clip_data)
        )(self.alpha_posterior, self.beta_posterior)
        
        # Calculate mean log prob for each sample
        avg_log_probs = jnp.mean(frame_log_probs_by_sample, axis=1)
        
        # Compare to threshold posterior
        p_below_threshold = np.mean(avg_log_probs < self.threshold_posterior)
        
        return {
            'method': 'probabilistic',
            'has_issue': p_below_threshold > 0.5,  # Decision rule
            'p_below_threshold': p_below_threshold,
            'mean_variance': mean_variance,
            'relative_deviation': relative_deviation
        }

    def check_performance_credible_interval(self, clip_variances):
        """
        Method 2: Credible interval approach
        Returns decision based on whether mean falls outside credible interval
        """
        mean_variance = np.mean(clip_variances)
        
        # Calculate relative deviation
        relative_deviation = (mean_variance - self.baseline_mean_variance) / self.baseline_mean_variance
        
        # Practical significance check
        is_practically_significant = relative_deviation > self.min_deviation_threshold
        is_worse_direction = relative_deviation > 0
        
        if not (is_practically_significant and is_worse_direction):
            return {
                'method': 'credible_interval',
                'has_issue': False,
                'outside_credible_interval': False,
                'credible_interval_bounds': (self.baseline_ci_lower, self.baseline_ci_upper),
                'mean_variance': mean_variance,
                'relative_deviation': relative_deviation
            }
        
        # Check if outside credible interval
        outside_credible_interval = (mean_variance < self.baseline_ci_lower or 
                                mean_variance > self.baseline_ci_upper)
        
        return {
            'method': 'credible_interval',
            'has_issue': outside_credible_interval,  # Decision rule
            'outside_credible_interval': outside_credible_interval,
            'credible_interval_bounds': (self.baseline_ci_lower, self.baseline_ci_upper),
            'mean_variance': mean_variance,
            'relative_deviation': relative_deviation
        }
        
    def check_performance_pp_pvalue(self, clip_variances, alpha=0.05):
        """
        Method 2: Posterior Predictive p-value approach
        Calculate posterior predictive p-value for the mean variance
        
        Parameters:
        - clip_variances: Frame variances from clip to check
        - alpha: Significance level for p-value threshold (default: 0.05)
        
        Returns:
        - result: Dictionary with check results
        """
        mean_variance = np.mean(clip_variances)
        
        # Calculate relative deviation
        relative_deviation = (mean_variance - self.baseline_mean_variance) / self.baseline_mean_variance
        
        # Practical significance check
        is_practically_significant = relative_deviation > self.min_deviation_threshold
        is_worse_direction = relative_deviation > 0
        
        if not (is_practically_significant and is_worse_direction):
            return {
                'method': 'pp_pvalue',
                'has_issue': False,
                'pp_pvalue': 1.0,
                'mean_variance': mean_variance,
                'relative_deviation': relative_deviation
            }
        
        # Calculate p-value (proportion of predictive samples more extreme than observed)
        # For one-sided test looking for large values (assuming higher variance is worse):
        pp_pvalue = np.mean(self.predictive_samples >= mean_variance)
        
        # If higher variance is worse, we want a small p-value to indicate anomaly
        # So we actually want to find how UNLIKELY it is to see a value this high or higher
        # This is 1 - p(X >= x) which is equivalent to p(X < x)
        pp_pvalue = 1 - pp_pvalue
        
        return {
            'method': 'pp_pvalue',
            'has_issue': pp_pvalue < alpha,  # Decision rule
            'pp_pvalue': pp_pvalue,
            'significance_level': alpha,
            'mean_variance': mean_variance,
            'relative_deviation': relative_deviation
        }

    def check_performance_both(self, clip_variances):
        """
        Run both methods and return results from both approaches
        """
        prob_result = self.check_performance_probabilistic(clip_variances)
        pp_result = self.check_performance_pp_pvalue(clip_variances)
        
        return {
            'probabilistic': prob_result,
            'pp_pvalue': pp_result,
            'mean_variance': prob_result['mean_variance'],
            'relative_deviation': prob_result['relative_deviation']
        }
    
    def check_performance(self, clip_variances):
        """
        Main method to check performance of a clip
        Combines probabilistic and posterior predictive p-value approaches
        
        Parameters:
        - clip_variances: Frame variances from clip to check
        
        Returns:
        - result: Dictionary with check results
        """
        # Get results from both methods
        prob_result = self.check_performance_probabilistic(clip_variances)
        pp_result = self.check_performance_pp_pvalue(clip_variances)
        
        # Calculate combined decision
        has_issue = prob_result['has_issue'] or pp_result['has_issue']
        
        # Calculate likelihood percentile
        clip_data = jnp.array(clip_variances)
        frame_log_probs_by_sample = jax.vmap(
            lambda a, b: dist.Gamma(a, b).log_prob(clip_data)
        )(self.alpha_posterior, self.beta_posterior)
        avg_log_probs = jnp.mean(frame_log_probs_by_sample, axis=1)
        
        # Calculate where this ranks in the posterior predictive distribution
        n_predictive = 10000
        predictive_log_probs = []
        for _ in range(n_predictive):
            # Get random posterior sample
            idx = np.random.randint(0, len(self.alpha_posterior))
            alpha = self.alpha_posterior[idx]
            beta = self.beta_posterior[idx]
            
            # Generate sample from this gamma
            sample = np.random.gamma(alpha, 1/beta, size=len(clip_variances))
            
            # Calculate log prob
            log_prob = np.mean(dist.Gamma(alpha, beta).log_prob(jnp.array(sample)))
            predictive_log_probs.append(log_prob)
        
        # Calculate percentile
        mean_log_prob = np.mean(avg_log_probs)
        likelihood_percentile = np.percentile(np.array([mean_log_prob]), 
                                           np.arange(0, 101, 1), 
                                           method='linear')[0]
        
        # Calculate probability of having an issue using a mixture of both methods
        p_has_issue = 0.7 * prob_result['p_below_threshold'] + 0.3 * float(pp_result['pp_pvalue'] < 0.05)
        
        return {
            'has_issue': has_issue,
            'probabilistic': prob_result,
            'pp_pvalue': pp_result,
            'p_below_threshold': prob_result['p_below_threshold'],
            'pp_pvalue_result': pp_result['pp_pvalue'],
            'mean_variance': prob_result['mean_variance'],
            'relative_deviation': prob_result['relative_deviation'],
            'likelihood_percentile': likelihood_percentile,
            'p_has_issue': p_has_issue
        }
    
    def update_baseline(self, new_clip_variances):
        """
        Update baseline distribution with new data
        
        Parameters:
        - new_clip_variances: Frame variances from new clip to add to baseline
        
        Returns:
        - updated_params: Dictionary with updated parameters
        """
        # Add new clip to raw baseline clips
        self.baseline_clips_raw.append(new_clip_variances)
        
        # Re-establish baseline with all clips
        return self.establish_baseline(self.baseline_clips_raw, visualize=False)
    
    def visualize_baseline(self):
        """
        Visualize baseline distributions, posterior, and predictive distributions
        """
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Plot all frame variances from baseline clips with posterior fit
        ax = axs[0, 0]
        
        # Combine all variances for histogram
        all_variances = []
        for clip in self.baseline_clips_raw:
            all_variances.extend(clip)
        
        # Plot histogram
        ax.hist(all_variances, bins=50, alpha=0.7, density=True, color='green', 
                label='Baseline frame variances')
        
        # Plot the posterior expected gamma distribution
        x = np.linspace(min(all_variances), max(all_variances), 1000)
        
        # Plot multiple posterior samples
        n_curves = 30
        indices = np.random.choice(len(self.alpha_posterior), n_curves)
        for i in indices:
            alpha = self.alpha_posterior[i]
            beta = self.beta_posterior[i]
            pdf = gamma.pdf(x, alpha, scale=1/beta)
            ax.plot(x, pdf, 'r-', lw=0.1, alpha=0.1)
        
        # Plot the posterior mean gamma distribution
        alpha_mean = np.mean(self.alpha_posterior)
        beta_mean = np.mean(self.beta_posterior)
        pdf_mean = gamma.pdf(x, alpha_mean, scale=1/beta_mean)
        ax.plot(x, pdf_mean, 'r-', lw=2, 
                label=f'Posterior mean gamma (α≈{alpha_mean:.2f}, β≈{beta_mean:.2f})')
        
        # Mark the credible interval
        ax.axvline(x=self.baseline_credible_interval[0], color='purple', linestyle='--')
        ax.axvline(x=self.baseline_credible_interval[1], color='purple', linestyle='--',
                   label=f'{self.credible_interval:.0%} credible interval')
        
        # Mark the mean variance
        ax.axvline(x=self.baseline_mean_variance, color='blue', linestyle='-', 
                   label=f'Mean variance: {self.baseline_mean_variance:.6f}')
        
        ax.set_title('Baseline Frame Variance Distribution with Posterior Fits')
        ax.set_xlabel('Variance')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Plot joint posterior distribution of alpha and beta (contour plot)
        ax = axs[0, 1]
        
        # Create 2D histogram/density plot
        from scipy.stats import gaussian_kde
        
        # Estimate the 2D density of alpha and beta posterior samples
        xy = np.vstack([self.alpha_posterior, self.beta_posterior])
        kernel = gaussian_kde(xy)
        
        # Create a grid for contour plot
        alpha_min, alpha_max = np.percentile(self.alpha_posterior, [0.5, 99.5])
        beta_min, beta_max = np.percentile(self.beta_posterior, [0.5, 99.5])
        
        # Add some padding
        alpha_range = alpha_max - alpha_min
        beta_range = beta_max - beta_min
        alpha_min -= alpha_range * 0.05
        alpha_max += alpha_range * 0.05
        beta_min -= beta_range * 0.05
        beta_max += beta_range * 0.05
        
        alpha_grid, beta_grid = np.mgrid[alpha_min:alpha_max:100j, beta_min:beta_max:100j]
        positions = np.vstack([alpha_grid.ravel(), beta_grid.ravel()])
        density = kernel(positions).reshape(alpha_grid.shape)
        
        # Plot contour
        contour = ax.contourf(alpha_grid, beta_grid, density, cmap='viridis', levels=20)
        plt.colorbar(contour, ax=ax, label='Density')
        
        # Plot the mean
        ax.scatter(alpha_mean, beta_mean, color='red', s=100, marker='x',
                  label=f'Mean (α={alpha_mean:.2f}, β={beta_mean:.2f})')
        
        # Plot posterior samples as small points
        ax.scatter(self.alpha_posterior, self.beta_posterior, color='white', 
                  s=5, alpha=0.1, marker='.')
        
        # Calculate 95% HPD contour (approximate by selecting the 95% highest density points)
        sorted_idx = np.argsort(density.ravel())[::-1]  # Sort from highest to lowest
        cumsum = np.cumsum(density.ravel()[sorted_idx])
        cumsum = cumsum / cumsum[-1]  # Normalize
        idx_95 = np.searchsorted(cumsum, 0.95)
        threshold = density.ravel()[sorted_idx[idx_95]]
        
        # Add 95% HPD contour
        ax.contour(alpha_grid, beta_grid, density, levels=[threshold], colors='red',
                  linestyles='dashed', linewidths=2, 
                  label='95% HPD region')
        
        ax.set_title('Joint Posterior Distribution of Shape (α) and Rate (β)')
        ax.set_xlabel('Shape Parameter (α)')
        ax.set_ylabel('Rate Parameter (β)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Plot posterior distribution for log-likelihood threshold
        ax = axs[1, 0]
        
        # Plot histogram of posterior threshold samples
        ax.hist(self.threshold_posterior, bins=30, alpha=0.7, color='green',
                density=True, label='Raw threshold values')
        
        # Mark posterior mean
        threshold_mean = np.mean(self.threshold_posterior)
        ax.axvline(x=threshold_mean, color='red', linestyle='-',
                   label=f'Mean threshold: {threshold_mean:.2f}')
        
        # Calculate 95% credible interval
        threshold_lower = np.percentile(self.threshold_posterior, 2.5)
        threshold_upper = np.percentile(self.threshold_posterior, 97.5)
        
        # Mark credible interval
        ax.axvline(x=threshold_lower, color='red', linestyle='--')
        ax.axvline(x=threshold_upper, color='red', linestyle='--',
                   label=f'95% CI: [{threshold_lower:.2f}, {threshold_upper:.2f}]')
        
        ax.set_title('Posterior Distribution for Log-Likelihood Threshold')
        ax.set_xlabel('Threshold Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Plot posterior predictive distribution with p-value regions
        ax = axs[1, 1]
        
        # Plot histogram of posterior predictive samples
        counts, bins, _ = ax.hist(self.predictive_samples, bins=50, alpha=0.7, color='purple',
                density=True, label='Posterior predictive samples')
        
        # Calculate some potential critical values for p-values
        p05 = np.percentile(self.predictive_samples, 95)  # Critical value for p=0.05
        p01 = np.percentile(self.predictive_samples, 99)  # Critical value for p=0.01
        
        # Shade the rejection regions
        bin_width = bins[1] - bins[0]
        max_height = max(counts)
        
        # Add rejection regions
        ax.fill_between(bins, 0, max_height, where=bins >= p05, 
                      alpha=0.3, color='red', 
                      label='Rejection region (p < 0.05)')
        ax.fill_between(bins, 0, max_height, where=bins >= p01, 
                      alpha=0.5, color='red', 
                      label='Rejection region (p < 0.01)')
        
        # Mark critical values
        ax.axvline(x=p05, color='red', linestyle='--',
                  label=f'Critical value (p=0.05): {p05:.6f}')
        ax.axvline(x=p01, color='red', linestyle=':',
                  label=f'Critical value (p=0.01): {p01:.6f}')
        
        # Mark the mean variance
        ax.axvline(x=self.baseline_mean_variance, color='blue', linestyle='-',
                  label=f'Mean variance: {self.baseline_mean_variance:.6f}')
        
        ax.set_title('Posterior Predictive Distribution with Rejection Regions')
        ax.set_xlabel('Variance')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Create a QQ plot to assess gamma fit
        plt.figure(figsize=(10, 6))
        
        # Get all variances
        all_variances = []
        for clip in self.baseline_clips_raw:
            all_variances.extend(clip)
            
        # Sort the data
        all_variances = np.sort(all_variances)
        n = len(all_variances)
        
        # Calculate empirical CDF positions
        p = np.arange(1, n+1) / (n+1)  # Using (i)/(n+1) formula
        
        # Calculate theoretical quantiles using posterior mean parameters
        alpha_mean = np.mean(self.alpha_posterior)
        beta_mean = np.mean(self.beta_posterior)
        theoretical_quantiles = gamma.ppf(p, alpha_mean, scale=1/beta_mean)
        
        # Create QQ plot
        plt.scatter(theoretical_quantiles, all_variances, alpha=0.5)
        
        # Add reference line
        max_val = max(np.max(theoretical_quantiles), np.max(all_variances))
        min_val = min(np.min(theoretical_quantiles), np.min(all_variances))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Calculate correlation for PPCC
        correlation = np.corrcoef(theoretical_quantiles, all_variances)[0, 1]
        
        plt.title(f'Q-Q Plot (PPCC: {correlation:.4f})')
        plt.xlabel('Theoretical Quantiles (Gamma)')
        plt.ylabel('Sample Quantiles')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_results(self, test_clips, labels):
        """
        Visualize test results
        
        Parameters:
        - test_clips: List of clips to visualize
        - labels: Binary labels (1=issue, 0=good)
        """
        # Calculate metrics for each clip
        results = []
        for clip in test_clips:
            results.append(self.check_performance(clip))
        
        # Extract metrics
        mean_variances = [r['mean_variance'] for r in results]
        percentiles = [r['likelihood_percentile'] for r in results]
        p_below_thresholds = [r['p_below_threshold'] for r in results]
        pp_pvalues = [r['pp_pvalue_result'] for r in results]
        deviations = [r['relative_deviation'] for r in results]
        p_has_issues = [r['p_has_issue'] for r in results]
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # 1. Mean variances with credible interval
        plt.subplot(2, 1, 1)
        
        # Define colors and markers
        colors = ['green' if label == 0 else 'red' for label in labels]
        shapes = ['o' if label == 0 else 'X' for label in labels]
        
        # Plot each clip
        for i, (mv, c, m) in enumerate(zip(mean_variances, colors, shapes)):
            plt.scatter(i, mv, color=c, marker=m, s=100)
        
        # Add clip labels
        for i, mv in enumerate(mean_variances):
            plt.text(i, mv, f"{i}", fontsize=9, ha='center', va='bottom')
        
        # Plot baseline mean and p-value critical values
        p05 = np.percentile(self.predictive_samples, 95)  # Critical value for p=0.05
        p01 = np.percentile(self.predictive_samples, 99)  # Critical value for p=0.01
        
        plt.axhline(y=self.baseline_mean_variance, color='blue', linestyle='-',
                   label=f'Baseline mean: {self.baseline_mean_variance:.6f}')
        plt.axhline(y=p05, color='red', linestyle='--',
                   label=f'Critical value (p=0.05): {p05:.6f}')
        plt.axhline(y=p01, color='red', linestyle=':',
                   label=f'Critical value (p=0.01): {p01:.6f}')
        plt.axhline(y=self.baseline_mean_variance * (1 + self.min_deviation_threshold),
                   color='orange', linestyle='-.',
                   label=f'Practical threshold (+{self.min_deviation_threshold*100:.1f}%)')
        
        plt.title('Mean Variance by Clip with Critical Values')
        plt.xlabel('Clip Index')
        plt.ylabel('Mean Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Probability of issue
        plt.subplot(2, 1, 2)
        
        # Sort by probability for better visualization
        sorted_indices = np.argsort(p_has_issues)
        sorted_probs = [p_has_issues[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        
        # Plot bars
        for i, (p, c) in enumerate(zip(sorted_probs, sorted_colors)):
            plt.bar(i, p, color=c, alpha=0.7)
        
        # Add clip indices as labels
        for i, idx in enumerate(sorted_indices):
            plt.text(i, sorted_probs[i] + 0.05, f"{idx}", fontsize=9, ha='center')
        
        # Add threshold line
        plt.axhline(y=0.5, color='red', linestyle='--',
                   label='Decision threshold (p=0.5)')
        
        plt.title('Probability of Issue by Clip (Sorted)')
        plt.xlabel('Sorted Clip Index')
        plt.ylabel('Probability of Issue')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show

    def run_test(self, test_clips, labels):
        """
        Run test on labeled clips to evaluate monitor performance
        
        Parameters:
        - test_clips: List of clips to test
        - labels: Binary labels (1=issue, 0=good)
        
        Returns:
        - results: Test results with metrics
        """
        if self.alpha_posterior is None:
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
        
        total = len(results)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Visualize results
        self.visualize_results(test_clips, labels)
        
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