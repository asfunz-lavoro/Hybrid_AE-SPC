"""
Statistical Process Control (SPC) Monitoring using Hybrid Autoencoder

This module implements real-time SPC monitoring using:
- T2 Hotelling statistics on the latent space (encoder output)
- SPE (Squared Prediction Error) charts for reconstruction error
- Monte Carlo Dropout uncertainty quantification
- Control charts with UCL/LCL limits
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import mahalanobis
import torch
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class SPCMonitor:
    """
    Statistical Process Control Monitor using trained autoencoder.
    Monitors both latent space (T2 Hotelling) and reconstruction error (SPE).
    """
    
    def __init__(self, model, confidence_level=0.95):
        """
        Initialize SPC Monitor.
        
        Parameters:
            model: Trained MonteCarloDropoutAutoencoder instance
            confidence_level: Confidence level for control limits (default: 0.95)
        """
        self.model = model
        self.confidence_level = confidence_level
        self.reference_embeddings = None
        self.reference_errors = None
        self.mu = None  # Mean of reference embeddings
        self.sigma = None  # Covariance of reference embeddings
        self.spe_mean = None
        self.spe_std = None
        self.spe_threshold = None
        self.t2_threshold = None
        self.monitoring_history = []
        
    def fit_control_limits(self, X_reference):
        """
        Fit control limits using reference (in-control) data.
        
        Parameters:
            X_reference: Reference data for establishing control limits (DataFrame or array)
        """
        print("Fitting control limits from reference data...")
        
        # Get embeddings and errors from reference data
        self.reference_embeddings = self.model.embed(X_reference)
        self.reference_errors = self.model.reconstruction_error(X_reference)
        
        # Fit T2 Hotelling parameters
        self.mu = np.mean(self.reference_embeddings, axis=0)
        self.sigma = np.cov(self.reference_embeddings.T)
        
        # Add small regularization to avoid singular matrix
        self.sigma += np.eye(self.sigma.shape[0]) * 1e-6
        
        # Calculate T2 statistics for reference data
        n_samples = len(self.reference_embeddings)
        n_features = self.reference_embeddings.shape[1]
        
        # T2 threshold using F-distribution
        alpha = 1 - self.confidence_level
        f_critical = stats.f.ppf(1 - alpha, n_features, n_samples - n_features)
        self.t2_threshold = ((n_samples - 1) * n_features / (n_samples - n_features)) * f_critical
        
        # Fit SPE (reconstruction error) parameters
        self.spe_mean = np.mean(self.reference_errors)
        self.spe_std = np.std(self.reference_errors)
        
        # SPE threshold using chi-square distribution (3-sigma)
        z_critical = stats.norm.ppf(self.confidence_level)
        self.spe_threshold = self.spe_mean + z_critical * self.spe_std
        
        print(f"✅ Control limits fitted successfully")
        print(f"  T2 Hotelling threshold: {self.t2_threshold:.4f}")
        print(f"  SPE threshold: {self.spe_threshold:.4f}")
        print(f"  Reference data: {n_samples} samples, {n_features} features")
    
    def calculate_t2_statistic(self, embeddings):
        """
        Calculate T2 Hotelling statistics for embeddings.
        
        Parameters:
            embeddings: Embeddings from model (n_samples, n_features)
        
        Returns:
            t2_stats: T2 statistics for each sample
        """
        if self.mu is None or self.sigma is None:
            raise ValueError("Control limits not fitted. Call fit_control_limits() first.")
        
        t2_stats = np.zeros(len(embeddings))
        sigma_inv = np.linalg.inv(self.sigma)
        
        for i, emb in enumerate(embeddings):
            diff = emb - self.mu
            t2_stats[i] = diff @ sigma_inv @ diff.T
        
        return t2_stats
    
    def monitor(self, X_new, num_stochastic_passes=50):
        """
        Monitor new data and generate SPC statistics.
        
        Parameters:
            X_new: New data to monitor (DataFrame or array)
            num_stochastic_passes: Number of MC Dropout passes
        
        Returns:
            monitoring_results: Dictionary with SPC metrics
        """
        if self.mu is None:
            raise ValueError("Control limits not fitted. Call fit_control_limits() first.")
        
        # Get predictions with uncertainty
        embeddings = self.model.embed(X_new)
        spe_errors = self.model.reconstruction_error(X_new)
        mean_recon, std_recon, unc_error, mse_error = self.model.monte_carlo_dropout_inference(
            X_new, num_stochastic_passes=num_stochastic_passes
        )
        
        # Calculate T2 Hotelling statistics
        t2_stats = self.calculate_t2_statistic(embeddings)
        
        # Determine control status
        t2_out_of_control = t2_stats > self.t2_threshold
        spe_out_of_control = spe_errors > self.spe_threshold
        out_of_control = t2_out_of_control | spe_out_of_control
        
        # Create results dictionary
        results = {
            'timestamp': datetime.now(),
            'n_samples': len(X_new),
            'embeddings': embeddings,
            'spe_errors': spe_errors,
            'mse_errors': mse_error,
            'uncertainty_errors': unc_error,
            't2_statistics': t2_stats,
            't2_threshold': self.t2_threshold,
            'spe_threshold': self.spe_threshold,
            't2_out_of_control': t2_out_of_control,
            'spe_out_of_control': spe_out_of_control,
            'out_of_control': out_of_control,
            'n_out_of_control': np.sum(out_of_control),
            'ooc_percentage': 100 * np.sum(out_of_control) / len(X_new)
        }
        
        # Store in history
        self.monitoring_history.append(results)
        
        return results
    
    def get_anomaly_indices(self, monitoring_results, threshold_type='both'):
        """
        Get indices of anomalous samples.
        
        Parameters:
            monitoring_results: Results from monitor()
            threshold_type: 'both' (either T2 or SPE), 't2', or 'spe'
        
        Returns:
            anomaly_indices: Array of anomalous sample indices
        """
        if threshold_type == 't2':
            anomalies = monitoring_results['t2_out_of_control']
        elif threshold_type == 'spe':
            anomalies = monitoring_results['spe_out_of_control']
        else:  # 'both'
            anomalies = monitoring_results['out_of_control']
        
        return np.where(anomalies)[0]
    
    def plot_t2_control_chart(self, figsize=(14, 6), max_samples=None):
        """
        Plot T2 Hotelling control chart.
        
        Parameters:
            figsize: Figure size
            max_samples: Maximum number of samples to plot (for clarity)
        """
        if not self.monitoring_history:
            print("No monitoring results to plot.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        all_t2 = []
        for results in self.monitoring_history:
            all_t2.extend(results['t2_statistics'])
        
        all_t2 = np.array(all_t2)
        if max_samples:
            all_t2 = all_t2[-max_samples:]
        
        sample_indices = np.arange(len(all_t2))
        
        # Plot T2 statistics
        ax.plot(sample_indices, all_t2, 'b-', linewidth=1.5, label='T2 Statistic', alpha=0.8)
        
        # Plot control limit
        ax.axhline(y=self.t2_threshold, color='r', linestyle='--', linewidth=2, label='UCL')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Highlight out-of-control points
        ooc_mask = all_t2 > self.t2_threshold
        if np.any(ooc_mask):
            ax.scatter(sample_indices[ooc_mask], all_t2[ooc_mask], 
                      color='red', s=100, marker='x', linewidths=2, label='Out of Control')
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('T2 Hotelling Statistic', fontsize=12)
        ax.set_title('T2 Hotelling Control Chart (Latent Space)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_spe_control_chart(self, figsize=(14, 6), max_samples=None):
        """
        Plot SPE (reconstruction error) control chart.
        
        Parameters:
            figsize: Figure size
            max_samples: Maximum number of samples to plot (for clarity)
        """
        if not self.monitoring_history:
            print("No monitoring results to plot.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        all_spe = []
        for results in self.monitoring_history:
            all_spe.extend(results['spe_errors'])
        
        all_spe = np.array(all_spe)
        if max_samples:
            all_spe = all_spe[-max_samples:]
        
        sample_indices = np.arange(len(all_spe))
        
        # Plot SPE statistics
        ax.plot(sample_indices, all_spe, 'g-', linewidth=1.5, label='SPE (Reconstruction Error)', alpha=0.8)
        
        # Plot control limit
        ax.axhline(y=self.spe_threshold, color='r', linestyle='--', linewidth=2, label='UCL')
        ax.axhline(y=self.spe_mean, color='orange', linestyle=':', linewidth=2, label='Mean')
        
        # Highlight out-of-control points
        ooc_mask = all_spe > self.spe_threshold
        if np.any(ooc_mask):
            ax.scatter(sample_indices[ooc_mask], all_spe[ooc_mask], 
                      color='red', s=100, marker='x', linewidths=2, label='Out of Control')
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('SPE (Reconstruction Error)', fontsize=12)
        ax.set_title('SPE Control Chart (Model Reconstruction Error)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_combined_control_chart(self, figsize=(14, 10), max_samples=None):
        """
        Plot both T2 and SPE control charts side-by-side.
        
        Parameters:
            figsize: Figure size
            max_samples: Maximum number of samples to plot
        """
        if not self.monitoring_history:
            print("No monitoring results to plot.")
            return None
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        all_t2 = []
        all_spe = []
        for results in self.monitoring_history:
            all_t2.extend(results['t2_statistics'])
            all_spe.extend(results['spe_errors'])
        
        all_t2 = np.array(all_t2)
        all_spe = np.array(all_spe)
        
        if max_samples:
            all_t2 = all_t2[-max_samples:]
            all_spe = all_spe[-max_samples:]
        
        sample_indices = np.arange(len(all_t2))
        
        # T2 Chart
        axes[0].plot(sample_indices, all_t2, 'b-', linewidth=1.5, label='T2 Statistic', alpha=0.8)
        axes[0].axhline(y=self.t2_threshold, color='r', linestyle='--', linewidth=2, label='UCL')
        ooc_t2 = all_t2 > self.t2_threshold
        if np.any(ooc_t2):
            axes[0].scatter(sample_indices[ooc_t2], all_t2[ooc_t2], 
                           color='red', s=100, marker='x', linewidths=2, label='Out of Control')
        axes[0].set_ylabel('T2 Hotelling', fontsize=11)
        axes[0].set_title('T2 Hotelling Control Chart', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # SPE Chart
        axes[1].plot(sample_indices, all_spe, 'g-', linewidth=1.5, label='SPE', alpha=0.8)
        axes[1].axhline(y=self.spe_threshold, color='r', linestyle='--', linewidth=2, label='UCL')
        axes[1].axhline(y=self.spe_mean, color='orange', linestyle=':', linewidth=2, label='Mean')
        ooc_spe = all_spe > self.spe_threshold
        if np.any(ooc_spe):
            axes[1].scatter(sample_indices[ooc_spe], all_spe[ooc_spe], 
                           color='red', s=100, marker='x', linewidths=2, label='Out of Control')
        axes[1].set_xlabel('Sample Index', fontsize=11)
        axes[1].set_ylabel('SPE (Reconstruction Error)', fontsize=11)
        axes[1].set_title('SPE Control Chart', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_uncertainty_analysis(self, monitoring_results, figsize=(14, 6)):
        """
        Plot uncertainty estimates from MC Dropout.
        
        Parameters:
            monitoring_results: Results from monitor()
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        mse_errors = monitoring_results['mse_errors']
        unc_errors = monitoring_results['uncertainty_errors']
        
        # Scatter plot of MSE vs Uncertainty
        axes[0].scatter(mse_errors, unc_errors, alpha=0.6, s=50)
        axes[0].set_xlabel('MSE Error', fontsize=11)
        axes[0].set_ylabel('Uncertainty Estimate', fontsize=11)
        axes[0].set_title('Reconstruction Error vs Uncertainty', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of uncertainty
        axes[1].hist(unc_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1].set_xlabel('Uncertainty', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Distribution of Uncertainty Estimates', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, monitoring_results):
        """
        Generate a text report of monitoring results.
        
        Parameters:
            monitoring_results: Results from monitor()
        
        Returns:
            report_text: Formatted report string
        """
        report = []
        report.append("="*70)
        report.append("STATISTICAL PROCESS CONTROL MONITORING REPORT")
        report.append("="*70)
        report.append(f"\nTimestamp: {monitoring_results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of samples: {monitoring_results['n_samples']}")
        report.append(f"\n--- Control Limits ---")
        report.append(f"T2 Hotelling UCL: {self.t2_threshold:.4f}")
        report.append(f"SPE UCL: {self.spe_threshold:.4f}")
        report.append(f"SPE Mean: {self.spe_mean:.4f}")
        report.append(f"\n--- Monitoring Results ---")
        report.append(f"Out-of-Control Samples: {monitoring_results['n_out_of_control']} ({monitoring_results['ooc_percentage']:.2f}%)")
        report.append(f"T2 violations: {np.sum(monitoring_results['t2_out_of_control'])}")
        report.append(f"SPE violations: {np.sum(monitoring_results['spe_out_of_control'])}")
        report.append(f"\n--- Statistical Summary ---")
        report.append(f"T2 Statistic - Min: {np.min(monitoring_results['t2_statistics']):.4f}, "
                     f"Mean: {np.mean(monitoring_results['t2_statistics']):.4f}, "
                     f"Max: {np.max(monitoring_results['t2_statistics']):.4f}")
        report.append(f"SPE - Min: {np.min(monitoring_results['spe_errors']):.4f}, "
                     f"Mean: {np.mean(monitoring_results['spe_errors']):.4f}, "
                     f"Max: {np.max(monitoring_results['spe_errors']):.4f}")
        report.append(f"Uncertainty - Mean: {np.mean(monitoring_results['uncertainty_errors']):.4f}, "
                     f"Std: {np.std(monitoring_results['uncertainty_errors']):.4f}")
        report.append("="*70)
        
        return "\n".join(report)
    
    def save_results(self, filepath):
        """
        Save monitoring history to file.
        
        Parameters:
            filepath: Path to save results
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'monitoring_history': self.monitoring_history,
                'control_limits': {
                    't2_threshold': self.t2_threshold,
                    'spe_threshold': self.spe_threshold,
                    'mu': self.mu,
                    'sigma': self.sigma,
                    'spe_mean': self.spe_mean,
                    'spe_std': self.spe_std
                }
            }, f)
        print(f"✅ Results saved to: {filepath}")


# Example usage
if __name__ == "__main__":
    print("SPC Monitoring Module")
    print("=" * 70)
    print("This module is designed to be imported and used with a trained autoencoder.")
    print("\nExample usage:")
    print("""
    from spc_monitoring import SPCMonitor
    
    # Create monitor with trained model
    monitor = SPCMonitor(trained_model, confidence_level=0.95)
    
    # Fit control limits using reference (in-control) data
    monitor.fit_control_limits(X_reference)
    
    # Monitor new data
    results = monitor.monitor(X_new, num_stochastic_passes=50)
    
    # Get report
    print(monitor.generate_report(results))
    
    # Plot control charts
    monitor.plot_combined_control_chart()
    monitor.plot_uncertainty_analysis(results)
    
    # Save results
    monitor.save_results('spc_monitoring_results.pkl')
    """)
