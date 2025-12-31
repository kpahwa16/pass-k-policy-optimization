"""
Variance Analysis: Replicating Figure 4 from PKPO Paper
========================================================

This script analyzes the variance of different gradient estimators:
1. s (no baseline)
2. s^(loo) (LOO baseline)
3. s^(loo-1) (k-1 baseline) - lowest variance

We also compare to naive partitioned methods to show the benefit
of averaging over ALL subsets.

Key finding: s^(loo-1) has the lowest variance, especially as n grows.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
from tqdm import tqdm

from pkpo_core import s, sloo, sloo_minus_one


def reward_function(x: np.ndarray) -> np.ndarray:
    """Same reward function as toy example."""
    g = np.zeros_like(x)
    mask = (x >= 0) & (x <= 1)
    g[mask] = x[mask] ** 2
    return g


def compute_gradient_estimate(
    x: np.ndarray,
    theta: float,
    sigma: float,
    k: int,
    method: str
) -> float:
    """
    Compute a single gradient estimate using specified method.
    
    Args:
        x: Samples from policy
        theta: Policy mean
        sigma: Policy std
        k: Target k for pass@k
        method: Estimation method
        
    Returns:
        Gradient estimate
    """
    g = reward_function(x)
    
    # Get transformed rewards
    if method == 's':
        transformed = s(g, k)
    elif method == 'sloo':
        transformed = sloo(g, k)
    elif method == 'sloo_minus_one':
        transformed = sloo_minus_one(g, k)
    elif method == 'naive_partitioned':
        # Naive method: partition into groups of k
        transformed = naive_partitioned_rewards(g, k)
    elif method == 'naive_partitioned_baselined':
        transformed = naive_partitioned_baselined(g, k)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Score function for Gaussian
    score = (x - theta) / (sigma ** 2)
    
    # Policy gradient
    return np.sum(transformed * score)


def naive_partitioned_rewards(g: np.ndarray, k: int) -> np.ndarray:
    """
    Naive partitioning: divide into groups of k, use max within each group.
    
    This is what you'd do without the PKPO formulation.
    """
    n = len(g)
    transformed = np.zeros(n)
    
    # How many complete groups of k
    n_groups = n // k
    
    for i in range(n_groups):
        start = i * k
        end = start + k
        group_max = np.max(g[start:end])
        transformed[start:end] = group_max
    
    # Handle remainder (if any)
    if n % k > 0:
        start = n_groups * k
        group_max = np.max(g[start:])
        transformed[start:] = group_max
    
    return transformed


def naive_partitioned_baselined(g: np.ndarray, k: int) -> np.ndarray:
    """
    Naive partitioning with leave-one-out baseline across groups.
    """
    transformed = naive_partitioned_rewards(g, k)
    
    # Subtract mean of other groups
    n = len(g)
    n_groups = n // k
    
    if n_groups > 1:
        baselined = np.zeros(n)
        for i in range(n_groups):
            start = i * k
            end = start + k
            # Mean of rewards from other groups
            other_mask = np.ones(n, dtype=bool)
            other_mask[start:end] = False
            baseline = np.mean(transformed[other_mask])
            baselined[start:end] = transformed[start:end] - baseline
        return baselined
    
    return transformed


def compute_estimator_variance(
    theta: float,
    sigma: float,
    k: int,
    n_samples: int,
    method: str,
    n_estimates: int = 10000
) -> Tuple[float, float]:
    """
    Compute the variance of a gradient estimator.
    
    Args:
        theta: Policy mean
        sigma: Policy std
        k: Target k
        n_samples: Number of samples per estimate
        method: Estimation method
        n_estimates: Number of gradient estimates to compute
        
    Returns:
        (mean, variance) of gradient estimates
    """
    estimates = []
    
    for _ in range(n_estimates):
        x = np.random.normal(theta, sigma, size=n_samples)
        grad = compute_gradient_estimate(x, theta, sigma, k, method)
        estimates.append(grad)
    
    estimates = np.array(estimates)
    return np.mean(estimates), np.var(estimates)


def plot_variance_comparison(
    save_path: str = 'figures/variance_comparison.png',
    k: int = 4,
    theta: float = 1.0,
    sigma: float = 0.1
):
    """
    Replicate Figure 4: Compare variance of different estimators.
    
    Shows variance as a function of n (number of samples).
    """
    n_values = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36]
    methods = ['s', 'sloo', 'sloo_minus_one', 
               'naive_partitioned', 'naive_partitioned_baselined']
    method_labels = {
        's': 'all subsets no baseline (s)',
        'sloo': 'all subsets baselined (s^loo)',
        'sloo_minus_one': 'loo minus one all subsets (s^loo-1)',
        'naive_partitioned': 'naive partitioned no baseline',
        'naive_partitioned_baselined': 'naive partitioned baselined'
    }
    
    results = {m: {'variance': [], 'mean': []} for m in methods}
    
    print(f"Computing variance for k={k}, θ={theta}, σ={sigma}")
    print("=" * 60)
    
    for n in tqdm(n_values, desc="Processing n values"):
        if n < k:
            for m in methods:
                results[m]['variance'].append(np.nan)
                results[m]['mean'].append(np.nan)
            continue
            
        for method in methods:
            try:
                mean, var = compute_estimator_variance(
                    theta, sigma, k, n, method, n_estimates=10000
                )
                results[method]['variance'].append(var)
                results[method]['mean'].append(mean)
            except Exception as e:
                print(f"  Error with {method}, n={n}: {e}")
                results[method]['variance'].append(np.nan)
                results[method]['mean'].append(np.nan)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        's': 'blue',
        'sloo': 'green', 
        'sloo_minus_one': 'red',
        'naive_partitioned': 'purple',
        'naive_partitioned_baselined': 'orange'
    }
    markers = {
        's': 'o',
        'sloo': 's',
        'sloo_minus_one': '^',
        'naive_partitioned': 'D',
        'naive_partitioned_baselined': 'v'
    }
    
    for method in methods:
        variances = results[method]['variance']
        valid_mask = ~np.isnan(variances)
        valid_n = np.array(n_values)[valid_mask]
        valid_var = np.array(variances)[valid_mask]
        
        ax.semilogy(valid_n, valid_var, 
                   marker=markers[method], 
                   color=colors[method],
                   label=method_labels[method],
                   linewidth=2,
                   markersize=8)
    
    ax.set_xlabel('Number of model samples n', fontsize=12)
    ax.set_ylabel('Empirical Variance of the Gradient Estimator', fontsize=12)
    ax.set_title(f'Variance of Estimated Gradient of max_g@k for k = {k}', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(n_values) - 1, max(n_values) + 1)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    
    plt.show()
    
    return results


def analyze_variance_vs_k(
    save_path: str = 'figures/variance_vs_k.png',
    n: int = 16,
    theta: float = 0.8,
    sigma: float = 0.1
):
    """
    Analyze how variance changes with k for fixed n.
    """
    k_values = [2, 4, 6, 8, 10, 12, 14, 16]
    methods = ['s', 'sloo', 'sloo_minus_one']
    
    results = {m: [] for m in methods}
    
    print(f"\nAnalyzing variance vs k for n={n}")
    print("=" * 60)
    
    for k in tqdm(k_values, desc="Processing k values"):
        for method in methods:
            try:
                _, var = compute_estimator_variance(
                    theta, sigma, k, n, method, n_estimates=10000
                )
                results[method].append(var)
            except Exception as e:
                print(f"  Error with {method}, k={k}: {e}")
                results[method].append(np.nan)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'s': 'blue', 'sloo': 'green', 'sloo_minus_one': 'red'}
    labels = {'s': 's (no baseline)', 'sloo': 's^loo', 'sloo_minus_one': 's^loo-1'}
    
    for method in methods:
        ax.semilogy(k_values, results[method], 'o-', 
                   color=colors[method], label=labels[method],
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('k value', fontsize=12)
    ax.set_ylabel('Variance of Gradient Estimator', fontsize=12)
    ax.set_title(f'Variance vs k (n = {n}, θ = {theta})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    
    plt.show()
    
    return results


def analyze_bias_variance_tradeoff(
    save_path: str = 'figures/bias_variance.png',
    n: int = 16,
    k: int = 4,
    sigma: float = 0.1
):
    """
    Verify that all estimators are unbiased while showing variance differences.
    """
    theta_values = np.linspace(0.2, 1.0, 9)
    methods = ['s', 'sloo', 'sloo_minus_one']
    
    results = {m: {'means': [], 'stds': []} for m in methods}
    
    print(f"\nAnalyzing bias-variance tradeoff")
    print("=" * 60)
    
    for theta in tqdm(theta_values, desc="Processing θ values"):
        for method in methods:
            mean, var = compute_estimator_variance(
                theta, sigma, k, n, method, n_estimates=5000
            )
            results[method]['means'].append(mean)
            results[method]['stds'].append(np.sqrt(var))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'s': 'blue', 'sloo': 'green', 'sloo_minus_one': 'red'}
    labels = {'s': 's (no baseline)', 'sloo': 's^loo', 'sloo_minus_one': 's^loo-1'}
    
    # Mean (should be similar for all - unbiasedness)
    ax1 = axes[0]
    for method in methods:
        ax1.plot(theta_values, results[method]['means'], 'o-',
                color=colors[method], label=labels[method],
                linewidth=2, markersize=8)
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('θ', fontsize=12)
    ax1.set_ylabel('Mean Gradient Estimate', fontsize=12)
    ax1.set_title('Mean (all should be similar - unbiased)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Std (variance reduction comparison)
    ax2 = axes[1]
    for method in methods:
        ax2.plot(theta_values, results[method]['stds'], 'o-',
                color=colors[method], label=labels[method],
                linewidth=2, markersize=8)
    
    ax2.set_xlabel('θ', fontsize=12)
    ax2.set_ylabel('Std of Gradient Estimate', fontsize=12)
    ax2.set_title('Standard Deviation (lower is better)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    
    plt.show()
    
    return results


def plot_effective_rewards(
    save_path: str = 'figures/effective_rewards.png',
    n: int = 32
):
    """
    Replicate Figure 5: Show raw vs effective rewards.
    
    Demonstrates how the baseline centers the rewards.
    """
    k_values = [2, 4, 8, 16]
    
    # Sample random rewards
    np.random.seed(42)
    raw_rewards = np.random.uniform(-0.5, 0.5, size=n)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(k_values)))
    
    # Sort for visualization
    sort_idx = np.argsort(raw_rewards)
    raw_sorted = raw_rewards[sort_idx]
    
    methods = ['s', 'sloo', 'sloo_minus_one']
    titles = ['s (no baseline)', 's^loo (LOO baseline)', 's^loo-1 (k-1 baseline)']
    
    for ax, method, title in zip(axes, methods, titles):
        for k, color in zip(k_values, colors):
            if method == 's':
                transformed = s(raw_rewards, k)
            elif method == 'sloo':
                transformed = sloo(raw_rewards, k)
            else:
                transformed = sloo_minus_one(raw_rewards, k)
            
            # Sort by raw reward for visualization
            transformed_sorted = transformed[sort_idx]
            
            ax.scatter(raw_sorted, transformed_sorted, 
                      c=[color], label=f'k = {k}', alpha=0.7, s=30)
        
        ax.set_xlabel('raw reward g(x_i)', fontsize=11)
        ax.set_ylabel(f'Effective Reward', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PKPO Variance Analysis')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick version')
    args = parser.parse_args()
    
    np.random.seed(42)
    
    print("PKPO Variance Analysis")
    print("=" * 60)
    
    # 1. Main variance comparison (Figure 4)
    print("\n1. Variance comparison across n values (Figure 4)...")
    results_n = plot_variance_comparison(k=4)
    
    # 2. Variance vs k
    print("\n2. Variance vs k...")
    results_k = analyze_variance_vs_k()
    
    # 3. Bias-variance analysis
    print("\n3. Bias-variance tradeoff...")
    results_bv = analyze_bias_variance_tradeoff()
    
    # 4. Effective rewards visualization (Figure 5)
    print("\n4. Effective rewards visualization (Figure 5)...")
    plot_effective_rewards()
    
    print("\n✓ Variance analysis complete!")
