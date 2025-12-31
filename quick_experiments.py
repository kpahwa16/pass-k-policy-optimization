"""
Quick PKPO Experiments
======================
A streamlined version that generates key figures quickly.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Set up matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Import PKPO functions
from pkpo_core import s, sloo, sloo_minus_one, rho_continuous, rho_binary


def reward_function(x):
    """g(x) = x² if 0 ≤ x ≤ 1, else 0"""
    g = np.zeros_like(x, dtype=float)
    mask = (x >= 0) & (x <= 1)
    g[mask] = x[mask] ** 2
    return g


def experiment_1_optimal_theta():
    """Find optimal theta for different k values."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Optimal θ for Different k")
    print("=" * 60)
    
    sigma = 0.1
    k_values = [1, 2, 4, 8, 16]
    theta_range = np.linspace(0.2, 1.1, 40)
    n_sim = 20000
    
    results = {}
    
    for k in tqdm(k_values, desc="Computing max_g@k"):
        max_g_at_k = []
        for theta in theta_range:
            samples = np.random.normal(theta, sigma, size=(n_sim, k))
            rewards = reward_function(samples)
            max_rewards = np.max(rewards, axis=1)
            max_g_at_k.append(np.mean(max_rewards))
        
        results[k] = {
            'theta': theta_range,
            'max_g_at_k': np.array(max_g_at_k),
            'optimal_theta': theta_range[np.argmax(max_g_at_k)],
            'optimal_value': np.max(max_g_at_k)
        }
    
    # Print results
    print("\nOptimal θ for different k:")
    print("-" * 40)
    for k in k_values:
        print(f"k = {k:2d}: θ* = {results[k]['optimal_theta']:.3f}, "
              f"max_g@k = {results[k]['optimal_value']:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_values)))
    
    for k, color in zip(k_values, colors):
        ax.plot(results[k]['theta'], results[k]['max_g_at_k'], 
               label=f'k = {k}', color=color, linewidth=2)
        # Mark optimal
        ax.axvline(x=results[k]['optimal_theta'], color=color, 
                  linestyle='--', alpha=0.5)
    
    # Add reward function reference
    x = np.linspace(0, 1.1, 100)
    ax.plot(x, reward_function(x), 'k--', alpha=0.3, label='g(x)')
    
    ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.5, label='x=1 boundary')
    ax.set_xlabel('θ (policy mean)', fontsize=14)
    ax.set_ylabel('max_g@k', fontsize=14)
    ax.set_title(f'max_g@k vs θ for x ~ N(θ, {sigma})', fontsize=16)
    ax.legend()
    ax.set_xlim(0.2, 1.1)
    
    plt.tight_layout()
    plt.savefig('figures/exp1_optimal_theta.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Figure saved: figures/exp1_optimal_theta.png")
    return results


def experiment_2_variance_comparison():
    """Compare variance of different gradient estimators."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Gradient Estimator Variance")
    print("=" * 60)
    
    sigma = 0.1
    theta = 0.8  # Near the interesting region
    k = 4
    n_values = [4, 8, 12, 16, 20, 24, 32]
    n_estimates = 5000
    
    methods = {
        's': lambda g, k: s(g, k),
        'sloo': lambda g, k: sloo(g, k),
        'sloo_minus_one': lambda g, k: sloo_minus_one(g, k),
    }
    
    results = {m: [] for m in methods}
    
    for n in tqdm(n_values, desc="Computing variance"):
        for method_name, method_fn in methods.items():
            gradients = []
            for _ in range(n_estimates):
                x = np.random.normal(theta, sigma, size=n)
                g = reward_function(x)
                transformed = method_fn(g, k)
                score = (x - theta) / (sigma ** 2)
                grad = np.sum(transformed * score)
                gradients.append(grad)
            
            results[method_name].append(np.var(gradients))
    
    # Print results
    print(f"\nVariance comparison for k={k}, θ={theta}:")
    print("-" * 50)
    print(f"{'n':>4} {'s':>12} {'sloo':>12} {'sloo-1':>12}")
    print("-" * 50)
    for i, n in enumerate(n_values):
        print(f"{n:4d} {results['s'][i]:12.2f} {results['sloo'][i]:12.2f} "
              f"{results['sloo_minus_one'][i]:12.2f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'s': 'blue', 'sloo': 'green', 'sloo_minus_one': 'red'}
    labels = {'s': 's (no baseline)', 'sloo': 's^(loo)', 'sloo_minus_one': 's^(loo-1)'}
    
    for method_name in methods:
        ax.semilogy(n_values, results[method_name], 'o-',
                   color=colors[method_name], label=labels[method_name],
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of samples n', fontsize=14)
    ax.set_ylabel('Variance of gradient estimator (log scale)', fontsize=14)
    ax.set_title(f'Gradient Estimator Variance (k={k}, θ={theta})', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/exp2_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Figure saved: figures/exp2_variance.png")
    return results


def experiment_3_sample_weights():
    """Visualize how k affects sample weights."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Sample Weight Distribution")
    print("=" * 60)
    
    n = 8
    k_values = [1, 2, 4, 8]
    
    # Dummy rewards for weight visualization
    g = np.arange(n).astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_values)))
    
    for k, color in zip(k_values, colors):
        weights = s(g, k)
        # Normalize for visualization
        if weights.max() > 0:
            weights_norm = weights / weights.max()
        else:
            weights_norm = weights
        
        ax.plot(range(n), weights_norm, 'o-', 
               label=f'k = {k}', color=color, linewidth=2, markersize=10)
    
    ax.set_xlabel('Ascending sort index i', fontsize=14)
    ax.set_ylabel('μ_i (normalized weight)', fontsize=14)
    ax.set_title(f'Sample Weights for n={n} and Various k', fontsize=16)
    ax.set_xticks(range(n))
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/exp3_weights.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Key insight: Higher k → more weight on high-reward samples")
    print(f"\n✓ Figure saved: figures/exp3_weights.png")


def experiment_4_training():
    """Train policies with different k values."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Policy Training with Different k")
    print("=" * 60)
    
    sigma = 0.1
    n_samples = 16
    n_steps = 1500
    lr = 0.02
    k_values = [1, 4, 8]
    
    results = {}
    
    for k in k_values:
        theta = 0.5  # Initial
        theta_history = [theta]
        
        print(f"\nTraining with k={k}...")
        
        for step in tqdm(range(n_steps), desc=f"k={k}"):
            # Sample
            x = np.random.normal(theta, sigma, size=n_samples)
            g = reward_function(x)
            
            # Transform rewards
            if k == 1:
                # For k=1, just use mean-centered rewards
                transformed = g - np.mean(g)
            else:
                transformed = sloo_minus_one(g, k)
            
            # Compute gradient
            score = (x - theta) / (sigma ** 2)
            gradient = np.sum(transformed * score)
            
            # Update
            theta += lr * gradient
            theta_history.append(theta)
        
        results[k] = {
            'theta_history': theta_history,
            'final_theta': theta
        }
        print(f"  Final θ = {theta:.4f}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(k_values)))
    
    # Left: theta trajectory
    ax1 = axes[0]
    for k, color in zip(k_values, colors):
        ax1.plot(results[k]['theta_history'], label=f'k = {k}', 
                color=color, linewidth=2)
    
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Training step', fontsize=14)
    ax1.set_ylabel('θ', fontsize=14)
    ax1.set_title('Policy Mean θ During Training', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Right: Cross-evaluation
    ax2 = axes[1]
    k_eval_values = [1, 2, 4, 8, 16]
    
    for k_opt, color in zip(k_values, colors):
        theta_final = results[k_opt]['final_theta']
        cross_eval = []
        
        for k_eval in k_eval_values:
            # Evaluate
            n_eval = 5000
            samples = np.random.normal(theta_final, sigma, size=(n_eval, k_eval))
            rewards = reward_function(samples)
            max_rewards = np.max(rewards, axis=1)
            cross_eval.append(np.mean(max_rewards))
        
        ax2.plot(k_eval_values, cross_eval, 'o-', label=f'k_opt = {k_opt}',
                color=color, linewidth=2, markersize=8)
    
    ax2.set_xlabel('k_eval', fontsize=14)
    ax2.set_ylabel('max_g@k_eval', fontsize=14)
    ax2.set_title('Cross-Evaluation of Trained Policies', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/exp4_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Figure saved: figures/exp4_training.png")
    return results


def experiment_5_k_annealing():
    """Demonstrate k annealing strategy."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: K Annealing")
    print("=" * 60)
    
    sigma = 0.1
    n_samples = 16
    n_steps = 2000
    switch_step = 1000
    lr = 0.02
    
    # Train with annealing
    theta = 0.5
    theta_history = [theta]
    k_history = []
    
    for step in tqdm(range(n_steps), desc="K annealing"):
        # Annealing schedule
        k = 8 if step < switch_step else 1
        k_history.append(k)
        
        # Sample
        x = np.random.normal(theta, sigma, size=n_samples)
        g = reward_function(x)
        
        # Transform
        if k == 1:
            transformed = g - np.mean(g)
        else:
            transformed = sloo_minus_one(g, k)
        
        # Gradient
        score = (x - theta) / (sigma ** 2)
        gradient = np.sum(transformed * score)
        
        # Update
        theta += lr * gradient
        theta_history.append(theta)
    
    # Also train fixed k=1 and k=8 for comparison
    fixed_results = {}
    for k_fixed in [1, 8]:
        theta = 0.5
        history = [theta]
        for step in range(n_steps):
            x = np.random.normal(theta, sigma, size=n_samples)
            g = reward_function(x)
            if k_fixed == 1:
                transformed = g - np.mean(g)
            else:
                transformed = sloo_minus_one(g, k_fixed)
            score = (x - theta) / (sigma ** 2)
            gradient = np.sum(transformed * score)
            theta += lr * gradient
            history.append(theta)
        fixed_results[k_fixed] = history
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(fixed_results[1], label='k=1 (fixed)', color='blue', linewidth=2)
    ax.plot(fixed_results[8], label='k=8 (fixed)', color='red', linewidth=2)
    ax.plot(theta_history, label='k annealed (8→1)', color='green', linewidth=2)
    
    ax.axvline(x=switch_step, color='gray', linestyle='--', alpha=0.5, 
              label=f'Switch at step {switch_step}')
    ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Training step', fontsize=14)
    ax.set_ylabel('θ', fontsize=14)
    ax.set_title('K Annealing: Combining Exploration and Exploitation', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/exp5_annealing.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFinal θ values:")
    print(f"  k=1 (fixed):    {fixed_results[1][-1]:.4f}")
    print(f"  k=8 (fixed):    {fixed_results[8][-1]:.4f}")
    print(f"  k annealed:     {theta_history[-1]:.4f}")
    
    print(f"\n✓ Figure saved: figures/exp5_annealing.png")


def main():
    """Run all experiments."""
    np.random.seed(42)
    os.makedirs('figures', exist_ok=True)
    
    print("=" * 60)
    print("PKPO EXPERIMENTS")
    print("=" * 60)
    
    # Run experiments
    exp1_results = experiment_1_optimal_theta()
    exp2_results = experiment_2_variance_comparison()
    experiment_3_sample_weights()
    exp4_results = experiment_4_training()
    experiment_5_k_annealing()
    
    print("\n" + "=" * 60)
    print("✓ All experiments completed!")


if __name__ == "__main__":
    main()
