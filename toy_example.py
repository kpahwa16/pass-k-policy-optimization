"""
Toy Example: Replicating Figure 1 from PKPO Paper
==================================================

This script demonstrates how different values of k affect:
1. The optimal policy (optimal θ)
2. The max_g@k objective landscape
3. The gradient of max_g@k

Setup:
- Policy: x ~ N(θ, σ) with fixed σ = 0.1
- Reward: g(x) = x² if 0 ≤ x ≤ 1, else 0
- Goal: Find θ that maximizes max_g@k

Key insight: Higher k leads to more risk-tolerant policies
(optimal θ closer to 1) because we only need ONE good sample.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List
import os

# Import our PKPO implementation
from pkpo_core import s, sloo, sloo_minus_one, rho_continuous


def reward_function(x: np.ndarray) -> np.ndarray:
    """
    Reward function from the paper (Equation 35):
    g(x) = x² if 0 ≤ x ≤ 1, else 0
    
    This creates a scenario where:
    - Being in [0,1] is good (reward increases towards 1)
    - Going past 1 is catastrophic (zero reward)
    """
    g = np.zeros_like(x)
    mask = (x >= 0) & (x <= 1)
    g[mask] = x[mask] ** 2
    return g


def compute_max_g_at_k_monte_carlo(
    theta: float, 
    sigma: float, 
    k: int, 
    n_simulations: int = 100000
) -> float:
    """
    Compute E[max{g(x_i)}_{i=1}^k] via Monte Carlo.
    
    Args:
        theta: Mean of Gaussian policy
        sigma: Std of Gaussian policy
        k: Number of samples
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        Estimated max_g@k
    """
    # Sample k values for each simulation
    samples = np.random.normal(theta, sigma, size=(n_simulations, k))
    rewards = reward_function(samples)
    max_rewards = np.max(rewards, axis=1)
    return np.mean(max_rewards)


def compute_gradient_monte_carlo(
    theta: float,
    sigma: float,
    k: int,
    n_samples: int = 16,
    n_simulations: int = 10000,
    method: str = 'sloo_minus_one'
) -> float:
    """
    Compute gradient of max_g@k using PKPO estimator.
    
    Uses the policy gradient theorem:
    ∇_θ max_g@k = E[Σ_i s_i * ∇_θ log p(x_i|θ)]
    
    For Gaussian: ∇_θ log p(x|θ) = (x - θ) / σ²
    """
    gradients = []
    
    for _ in range(n_simulations):
        # Sample n responses
        x = np.random.normal(theta, sigma, size=n_samples)
        
        # Compute rewards
        g = reward_function(x)
        
        # Transform rewards using PKPO
        if method == 's':
            transformed = s(g, k)
        elif method == 'sloo':
            transformed = sloo(g, k)
        elif method == 'sloo_minus_one':
            transformed = sloo_minus_one(g, k)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute score function ∇_θ log p(x|θ) = (x - θ) / σ²
        score = (x - theta) / (sigma ** 2)
        
        # Policy gradient: Σ_i s_i * score_i
        grad = np.sum(transformed * score)
        gradients.append(grad)
    
    return np.mean(gradients), np.std(gradients) / np.sqrt(n_simulations)


def plot_figure_1(save_path: str = 'figures/figure1_replication.png'):
    """
    Replicate Figure 1 from the paper.
    
    Shows how different k values change:
    - Left: The max_g@k objective as a function of θ
    - Right: The gradient of max_g@k as a function of θ
    """
    sigma = 0.1
    theta_range = np.linspace(-0.2, 1.4, 50)
    k_values = [1, 2, 4, 8, 16]
    
    # Pre-compute for plotting
    print("Computing max_g@k for different θ and k values...")
    
    # Store results
    max_g_at_k = {k: [] for k in k_values}
    
    for theta in theta_range:
        for k in k_values:
            val = compute_max_g_at_k_monte_carlo(theta, sigma, k, n_simulations=50000)
            max_g_at_k[k].append(val)
        print(f"  θ = {theta:.2f} done")
    
    # Also compute the raw reward function for reference
    x_range = np.linspace(-0.2, 1.4, 200)
    raw_reward = reward_function(x_range)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: max_g@k vs θ
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(k_values)))
    
    for k, color in zip(k_values, colors):
        ax1.plot(theta_range, max_g_at_k[k], label=f'k = {k}', color=color, linewidth=2)
    
    # Add raw reward on secondary axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_range, raw_reward, 'k--', alpha=0.5, label='g(x)')
    ax1_twin.set_ylabel('raw reward g (function of x)', color='gray')
    ax1_twin.tick_params(axis='y', labelcolor='gray')
    ax1_twin.set_ylim(0, 1.1)
    
    ax1.set_xlabel('x and θ')
    ax1.set_ylabel('max_g@k (function of θ)')
    ax1.set_title(f'max_g@k for x ~ N(θ, {sigma}) and various k')
    ax1.legend(loc='upper left')
    ax1.set_xlim(-0.2, 1.4)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Gradient vs θ
    ax2 = axes[1]
    print("\nComputing gradients...")
    
    theta_range_grad = np.linspace(-0.2, 1.4, 30)
    gradients = {k: [] for k in k_values[1:]}  # Skip k=1 as in paper
    
    for theta in theta_range_grad:
        for k in k_values[1:]:
            grad, _ = compute_gradient_monte_carlo(
                theta, sigma, k, 
                n_samples=16, 
                n_simulations=5000,
                method='sloo_minus_one'
            )
            gradients[k].append(grad)
        print(f"  θ = {theta:.2f} done")
    
    for k, color in zip(k_values[1:], colors[1:]):
        ax2.plot(theta_range_grad, gradients[k], label=f'k = {k}', color=color, linewidth=2)
    
    # Add raw reward on secondary axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_range, raw_reward, 'k--', alpha=0.5, label='g(x)')
    ax2_twin.set_ylabel('raw reward g (function of x)', color='gray')
    ax2_twin.tick_params(axis='y', labelcolor='gray')
    ax2_twin.set_ylim(0, 1.1)
    
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('x and θ')
    ax2.set_ylabel('∇_θ max_g@k (function of θ)')
    ax2.set_title(f'∇_θ max_g@k for x ~ N(θ, {sigma}) and various k')
    ax2.legend(loc='upper left')
    ax2.set_xlim(-0.2, 1.4)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    
    # Also show
    plt.show()
    
    return max_g_at_k, gradients


def analyze_optimal_theta():
    """
    Find and analyze the optimal θ for different k values.
    
    This demonstrates the key insight: higher k → more risk-tolerant policy
    """
    sigma = 0.1
    k_values = [1, 2, 4, 8, 16]
    
    print("Finding optimal θ for different k values...")
    print("=" * 60)
    
    results = {}
    
    for k in k_values:
        # Grid search for optimal theta
        theta_range = np.linspace(0.3, 1.0, 50)
        max_g_values = []
        
        for theta in theta_range:
            val = compute_max_g_at_k_monte_carlo(theta, sigma, k, n_simulations=20000)
            max_g_values.append(val)
        
        optimal_idx = np.argmax(max_g_values)
        optimal_theta = theta_range[optimal_idx]
        optimal_value = max_g_values[optimal_idx]
        
        results[k] = {
            'optimal_theta': optimal_theta,
            'optimal_value': optimal_value,
            'prob_exceed_1': 1 - stats.norm.cdf(1, optimal_theta, sigma)
        }
        
        print(f"k = {k:2d}: optimal θ = {optimal_theta:.3f}, "
              f"max_g@k = {optimal_value:.4f}, "
              f"P(x > 1) = {results[k]['prob_exceed_1']:.2%}")
    
    print("\n" + "=" * 60)
    print("Key insight: As k increases, optimal θ moves towards 1,")
    print("accepting higher probability of some samples exceeding 1")
    print("(and getting zero reward) to maximize chance of hitting")
    print("the 'sweet spot' just below 1.")
    
    return results


def plot_sample_weights(save_path: str = 'figures/sample_weights.png'):
    """
    Replicate Figure 2: Show how k affects sample weights.
    
    For sorted samples, higher k gives more weight to larger samples.
    """
    n = 8
    k_values = [1, 2, 4, 8]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create dummy rewards (actual values don't matter for weight visualization)
    g = np.arange(n).astype(float)
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(k_values)))
    
    for k, color in zip(k_values, colors):
        # Get weights (s values)
        weights = s(g, k)
        # Normalize for visualization
        weights_normalized = weights / weights.max() if weights.max() > 0 else weights
        
        ax.plot(range(n), weights_normalized, 'o-', 
                label=f'k = {k}', color=color, markersize=8, linewidth=2)
    
    ax.set_xlabel('ascending sort index i')
    ax.set_ylabel('μ_i (normalized)')
    ax.set_title(f'Element Weight for Various k (n = {n})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(n))
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PKPO Toy Example')
    parser.add_argument('--quick', action='store_true', 
                        help='Run quick version with fewer samples')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("PKPO Toy Example: Replicating Figure 1")
    print("=" * 60)
    
    # First, show sample weights (Figure 2)
    print("\n1. Plotting sample weights (Figure 2)...")
    plot_sample_weights()
    
    # Find optimal theta for different k
    print("\n2. Analyzing optimal θ for different k values...")
    results = analyze_optimal_theta()
    
    # Replicate Figure 1
    print("\n3. Replicating Figure 1 (this may take a few minutes)...")
    if args.quick:
        print("   (Running quick version)")
    plot_figure_1()
    
    print("\n✓ Done!")
