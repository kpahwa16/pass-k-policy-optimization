"""
PKPO Training on Toy Example
============================

This script demonstrates actual policy optimization using PKPO
on the toy Gaussian policy problem.

We train a policy π(x|θ) = N(θ, σ) to maximize max_g@k
for different values of k and observe:
1. Learning curves
2. Final optimal θ values
3. Effect of k on exploration and convergence
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass
from tqdm import tqdm

from pkpo_core import s, sloo, sloo_minus_one, rho_continuous


def reward_function(x: np.ndarray) -> np.ndarray:
    """g(x) = x² if 0 ≤ x ≤ 1, else 0"""
    g = np.zeros_like(x)
    mask = (x >= 0) & (x <= 1)
    g[mask] = x[mask] ** 2
    return g


@dataclass
class TrainingConfig:
    """Configuration for PKPO training."""
    sigma: float = 0.1          # Policy std (fixed)
    n_samples: int = 16         # Samples per update
    learning_rate: float = 0.01
    n_steps: int = 1000
    k: int = 4                  # Target k for pass@k
    method: str = 'sloo_minus_one'  # Gradient estimation method
    theta_init: float = 0.5     # Initial policy mean
    log_interval: int = 50


class GaussianPolicy:
    """
    Simple Gaussian policy with learnable mean.
    π(x|θ) = N(θ, σ²)
    """
    def __init__(self, theta_init: float, sigma: float):
        self.theta = theta_init
        self.sigma = sigma
    
    def sample(self, n: int) -> np.ndarray:
        """Sample n actions from the policy."""
        return np.random.normal(self.theta, self.sigma, size=n)
    
    def log_prob_grad(self, x: np.ndarray) -> np.ndarray:
        """
        Compute ∇_θ log π(x|θ) = (x - θ) / σ²
        """
        return (x - self.theta) / (self.sigma ** 2)
    
    def update(self, gradient: float, lr: float):
        """Update policy parameters."""
        self.theta += lr * gradient


def compute_pkpo_gradient(
    policy: GaussianPolicy,
    rewards: np.ndarray,
    samples: np.ndarray,
    k: int,
    method: str
) -> float:
    """
    Compute PKPO gradient estimate.
    
    ∇_θ J ≈ Σ_i s_i * ∇_θ log π(x_i|θ)
    """
    # Transform rewards
    if method == 's':
        transformed = s(rewards, k)
    elif method == 'sloo':
        transformed = sloo(rewards, k)
    elif method == 'sloo_minus_one':
        transformed = sloo_minus_one(rewards, k)
    elif method == 'baseline_only':
        # Standard RL baseline (mean subtraction)
        transformed = rewards - np.mean(rewards)
    elif method == 'raw':
        # No transformation at all
        transformed = rewards
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute score function
    scores = policy.log_prob_grad(samples)
    
    # Policy gradient
    gradient = np.sum(transformed * scores)
    
    return gradient


def evaluate_policy(
    policy: GaussianPolicy,
    k_eval: int,
    n_episodes: int = 1000
) -> Dict[str, float]:
    """
    Evaluate policy performance.
    
    Returns max_g@k and other metrics.
    """
    max_rewards = []
    mean_rewards = []
    
    for _ in range(n_episodes):
        samples = policy.sample(k_eval)
        rewards = reward_function(samples)
        max_rewards.append(np.max(rewards))
        mean_rewards.append(np.mean(rewards))
    
    return {
        'max_g_at_k': np.mean(max_rewards),
        'mean_reward': np.mean(mean_rewards),
        'theta': policy.theta
    }


def train_pkpo(config: TrainingConfig, verbose: bool = True) -> Dict:
    """
    Train a Gaussian policy using PKPO.
    
    Returns:
        Dictionary with training history and final policy
    """
    # Initialize policy
    policy = GaussianPolicy(config.theta_init, config.sigma)
    
    # History
    history = {
        'theta': [policy.theta],
        'max_g_at_k': [],
        'mean_reward': [],
        'gradients': []
    }
    
    # Initial evaluation
    eval_result = evaluate_policy(policy, config.k)
    history['max_g_at_k'].append(eval_result['max_g_at_k'])
    history['mean_reward'].append(eval_result['mean_reward'])
    
    if verbose:
        print(f"Training with k={config.k}, method={config.method}")
        print(f"Initial θ = {policy.theta:.4f}")
        print("-" * 50)
    
    # Training loop
    iterator = range(config.n_steps)
    if verbose:
        iterator = tqdm(iterator, desc="Training")
    
    for step in iterator:
        # Sample from policy
        samples = policy.sample(config.n_samples)
        rewards = reward_function(samples)
        
        # Compute gradient
        gradient = compute_pkpo_gradient(
            policy, rewards, samples, config.k, config.method
        )
        
        # Update policy
        policy.update(gradient, config.learning_rate)
        
        # Log
        history['theta'].append(policy.theta)
        history['gradients'].append(gradient)
        
        if (step + 1) % config.log_interval == 0:
            eval_result = evaluate_policy(policy, config.k)
            history['max_g_at_k'].append(eval_result['max_g_at_k'])
            history['mean_reward'].append(eval_result['mean_reward'])
            
            if verbose and (step + 1) % (config.log_interval * 4) == 0:
                tqdm.write(f"Step {step+1}: θ = {policy.theta:.4f}, "
                          f"max_g@{config.k} = {eval_result['max_g_at_k']:.4f}")
    
    return {
        'history': history,
        'policy': policy,
        'config': config
    }


def compare_k_values(
    k_values: List[int] = [1, 2, 4, 8],
    save_path: str = 'figures/training_comparison.png'
):
    """
    Compare training dynamics for different k values.
    """
    results = {}
    
    print("Comparing training with different k values")
    print("=" * 60)
    
    for k in k_values:
        config = TrainingConfig(
            k=k,
            n_steps=1500,
            learning_rate=0.02,
            n_samples=16,
            theta_init=0.5,
            method='sloo_minus_one'
        )
        
        print(f"\nTraining with k = {k}")
        results[k] = train_pkpo(config, verbose=True)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(k_values)))
    
    # Plot 1: θ trajectory
    ax1 = axes[0, 0]
    for k, color in zip(k_values, colors):
        ax1.plot(results[k]['history']['theta'], 
                label=f'k = {k}', color=color, linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('θ')
    ax1.set_title('Policy Mean θ During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: max_g@k
    ax2 = axes[0, 1]
    for k, color in zip(k_values, colors):
        steps = np.arange(0, len(results[k]['history']['max_g_at_k'])) * 50
        ax2.plot(steps, results[k]['history']['max_g_at_k'],
                label=f'k = {k}', color=color, linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('max_g@k')
    ax2.set_title('max_g@k During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final θ vs k
    ax3 = axes[1, 0]
    final_thetas = [results[k]['history']['theta'][-1] for k in k_values]
    ax3.bar(range(len(k_values)), final_thetas, color=colors)
    ax3.set_xticks(range(len(k_values)))
    ax3.set_xticklabels([f'k={k}' for k in k_values])
    ax3.set_ylabel('Final θ')
    ax3.set_title('Final Policy Mean for Different k')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='x=1 boundary')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cross-evaluation
    ax4 = axes[1, 1]
    k_eval_values = [1, 2, 4, 8, 16]
    
    for k_opt, color in zip(k_values, colors):
        cross_eval = []
        policy = results[k_opt]['policy']
        for k_eval in k_eval_values:
            eval_result = evaluate_policy(policy, k_eval, n_episodes=2000)
            cross_eval.append(eval_result['max_g_at_k'])
        ax4.plot(k_eval_values, cross_eval, 'o-',
                label=f'k_opt = {k_opt}', color=color, linewidth=2, markersize=8)
    
    ax4.set_xlabel('k_eval')
    ax4.set_ylabel('max_g@k_eval')
    ax4.set_title('Cross-Evaluation: Policies Trained with Different k')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    
    plt.show()
    
    return results


def compare_methods(
    methods: List[str] = ['sloo_minus_one', 'sloo', 's', 'baseline_only'],
    k: int = 4,
    save_path: str = 'figures/method_comparison.png'
):
    """
    Compare different gradient estimation methods.
    """
    results = {}
    
    print(f"Comparing methods for k = {k}")
    print("=" * 60)
    
    for method in methods:
        config = TrainingConfig(
            k=k,
            n_steps=1000,
            learning_rate=0.02,
            n_samples=16,
            theta_init=0.5,
            method=method
        )
        
        print(f"\nTraining with method = {method}")
        results[method] = train_pkpo(config, verbose=True)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(methods)))
    
    # θ trajectory
    ax1 = axes[0]
    for method, color in zip(methods, colors):
        ax1.plot(results[method]['history']['theta'],
                label=method, color=color, linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('θ')
    ax1.set_title('Policy Mean θ During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gradient variance (smoothed)
    ax2 = axes[1]
    window = 50
    for method, color in zip(methods, colors):
        grads = np.array(results[method]['history']['gradients'])
        # Compute rolling variance
        variances = []
        for i in range(len(grads) - window):
            variances.append(np.var(grads[i:i+window]))
        ax2.semilogy(range(len(variances)), variances,
                    label=method, color=color, linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Rolling Gradient Variance')
    ax2.set_title('Gradient Variance During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    
    plt.show()
    
    return results


def k_annealing_experiment(
    save_path: str = 'figures/k_annealing.png'
):
    """
    Demonstrate k annealing: start with high k, reduce to k=1.
    
    This gets the best of both worlds:
    - High k early: better exploration
    - Low k later: optimize for single-sample performance
    """
    n_steps = 2000
    switch_step = 1000
    
    # Fixed k policies
    k_values = [1, 8]
    fixed_results = {}
    
    print("K Annealing Experiment")
    print("=" * 60)
    
    for k in k_values:
        config = TrainingConfig(
            k=k,
            n_steps=n_steps,
            learning_rate=0.02,
            n_samples=16,
            theta_init=0.5,
            method='sloo_minus_one'
        )
        print(f"\nTraining with fixed k = {k}")
        fixed_results[k] = train_pkpo(config, verbose=True)
    
    # Annealed k
    print(f"\nTraining with annealed k (8 → 1 at step {switch_step})")
    
    policy = GaussianPolicy(0.5, 0.1)
    history = {
        'theta': [policy.theta],
        'max_g_at_1': [],
        'max_g_at_8': [],
        'k': []
    }
    
    for step in tqdm(range(n_steps)):
        # Annealing schedule
        k = 8 if step < switch_step else 1
        history['k'].append(k)
        
        samples = policy.sample(16)
        rewards = reward_function(samples)
        
        gradient = compute_pkpo_gradient(
            policy, rewards, samples, k, 'sloo_minus_one'
        )
        policy.update(gradient, 0.02)
        
        history['theta'].append(policy.theta)
        
        if step % 50 == 0:
            eval_1 = evaluate_policy(policy, 1)
            eval_8 = evaluate_policy(policy, 8)
            history['max_g_at_1'].append(eval_1['max_g_at_k'])
            history['max_g_at_8'].append(eval_8['max_g_at_k'])
    
    annealed_result = {'history': history, 'policy': policy}
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # θ trajectory
    ax1 = axes[0, 0]
    ax1.plot(fixed_results[1]['history']['theta'], 
             label='k=1 (fixed)', color='blue', linewidth=2)
    ax1.plot(fixed_results[8]['history']['theta'],
             label='k=8 (fixed)', color='red', linewidth=2)
    ax1.plot(annealed_result['history']['theta'],
             label='k annealed (8→1)', color='green', linewidth=2)
    ax1.axvline(x=switch_step, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('θ')
    ax1.set_title('Policy Mean θ During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # max_g@1
    ax2 = axes[0, 1]
    steps = np.arange(0, len(fixed_results[1]['history']['max_g_at_k'])) * 50
    ax2.plot(steps, fixed_results[1]['history']['max_g_at_k'],
             label='k=1 (fixed)', color='blue', linewidth=2)
    
    steps_8 = np.arange(0, len(fixed_results[8]['history']['max_g_at_k'])) * 50
    # For k=8, we evaluate at k=1 too
    policy_8 = fixed_results[8]['policy']
    # We don't have the history, so skip this for now
    
    steps_ann = np.arange(0, len(annealed_result['history']['max_g_at_1'])) * 50
    ax2.plot(steps_ann, annealed_result['history']['max_g_at_1'],
             label='k annealed', color='green', linewidth=2)
    ax2.axvline(x=switch_step, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('max_g@1')
    ax2.set_title('max_g@1 During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # max_g@8
    ax3 = axes[1, 0]
    ax3.plot(steps_ann, annealed_result['history']['max_g_at_8'],
             label='k annealed', color='green', linewidth=2)
    ax3.axvline(x=switch_step, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('max_g@8')
    ax3.set_title('max_g@8 During Training (Annealed)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final cross-evaluation
    ax4 = axes[1, 1]
    k_eval_values = [1, 2, 4, 8, 16]
    
    for label, result, color in [
        ('k=1 fixed', fixed_results[1], 'blue'),
        ('k=8 fixed', fixed_results[8], 'red'),
        ('k annealed', annealed_result, 'green')
    ]:
        cross_eval = []
        policy = result['policy']
        for k_eval in k_eval_values:
            eval_result = evaluate_policy(policy, k_eval, n_episodes=2000)
            cross_eval.append(eval_result['max_g_at_k'])
        ax4.plot(k_eval_values, cross_eval, 'o-',
                label=label, color=color, linewidth=2, markersize=8)
    
    ax4.set_xlabel('k_eval')
    ax4.set_ylabel('max_g@k_eval')
    ax4.set_title('Final Cross-Evaluation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    
    plt.show()
    
    return fixed_results, annealed_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PKPO Training Experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick version')
    args = parser.parse_args()
    
    np.random.seed(42)
    
    print("PKPO Training Experiments")
    print("=" * 60)
    
    # 1. Compare different k values
    print("\n1. Comparing different k values...")
    results_k = compare_k_values()
    
    # 2. Compare different methods
    print("\n2. Comparing different gradient estimation methods...")
    results_methods = compare_methods()
    
    # 3. K annealing experiment
    print("\n3. K annealing experiment...")
    results_annealing = k_annealing_experiment()
    
    print("\n✓ Training experiments complete!")
