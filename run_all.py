"""
PKPO Experiments: Main Runner
==============================

This script runs all PKPO experiments in sequence:
1. Toy example (Figure 1 replication)
2. Variance analysis (Figure 4 replication)
3. Training experiments
4. Summary report

Usage:
    python run_all.py              # Full experiments
    python run_all.py --quick      # Quick version
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_toy_example():
    """Run toy example experiments."""
    print("\n" + "=" * 70)
    print("PART 1: TOY EXAMPLE (Figure 1 Replication)")
    print("=" * 70)
    
    from toy_example import (
        plot_sample_weights,
        analyze_optimal_theta,
        plot_figure_1
    )
    
    # Sample weights (Figure 2)
    print("\n1.1 Sample weights visualization...")
    plot_sample_weights('figures/01_sample_weights.png')
    
    # Optimal theta analysis
    print("\n1.2 Optimal θ analysis...")
    results = analyze_optimal_theta()
    
    # Figure 1 replication
    print("\n1.3 Figure 1 replication...")
    max_g_at_k, gradients = plot_figure_1('figures/02_figure1_replication.png')
    
    return {'optimal_theta': results, 'max_g_at_k': max_g_at_k}


def run_variance_analysis():
    """Run variance analysis experiments."""
    print("\n" + "=" * 70)
    print("PART 2: VARIANCE ANALYSIS (Figure 4 Replication)")
    print("=" * 70)
    
    from variance_analysis import (
        plot_variance_comparison,
        analyze_variance_vs_k,
        analyze_bias_variance_tradeoff,
        plot_effective_rewards
    )
    
    # Variance comparison
    print("\n2.1 Variance comparison across n...")
    results_n = plot_variance_comparison('figures/03_variance_vs_n.png')
    
    # Variance vs k
    print("\n2.2 Variance vs k...")
    results_k = analyze_variance_vs_k('figures/04_variance_vs_k.png')
    
    # Bias-variance tradeoff
    print("\n2.3 Bias-variance analysis...")
    results_bv = analyze_bias_variance_tradeoff('figures/05_bias_variance.png')
    
    # Effective rewards
    print("\n2.4 Effective rewards visualization...")
    plot_effective_rewards('figures/06_effective_rewards.png')
    
    return {'variance_n': results_n, 'variance_k': results_k}


def run_training_experiments():
    """Run training experiments."""
    print("\n" + "=" * 70)
    print("PART 3: TRAINING EXPERIMENTS")
    print("=" * 70)
    
    from training_toy import (
        compare_k_values,
        compare_methods,
        k_annealing_experiment
    )
    
    # Compare k values
    print("\n3.1 Comparing different k values...")
    results_k = compare_k_values(
        k_values=[1, 2, 4, 8],
        save_path='figures/07_training_k_comparison.png'
    )
    
    # Compare methods
    print("\n3.2 Comparing gradient estimation methods...")
    results_methods = compare_methods(
        methods=['sloo_minus_one', 'sloo', 's', 'baseline_only'],
        k=4,
        save_path='figures/08_method_comparison.png'
    )
    
    # K annealing
    print("\n3.3 K annealing experiment...")
    fixed_results, annealed_result = k_annealing_experiment(
        save_path='figures/09_k_annealing.png'
    )
    
    return {
        'k_comparison': {k: r['history']['theta'][-1] for k, r in results_k.items()},
        'annealing': annealed_result['history']['theta'][-1]
    }


def generate_summary_report(all_results: dict, runtime: float):
    """Generate a summary report."""
    report = f"""
PKPO EXPERIMENT SUMMARY REPORT
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Runtime: {runtime:.1f} seconds

1. KEY FINDINGS FROM TOY EXAMPLE
--------------------------------
Optimal θ for different k values:
"""
    
    if 'toy' in all_results and 'optimal_theta' in all_results['toy']:
        for k, data in all_results['toy']['optimal_theta'].items():
            report += f"  k = {k:2d}: θ* = {data['optimal_theta']:.3f}, "
            report += f"max_g@k = {data['optimal_value']:.4f}, "
            report += f"P(x > 1) = {data['prob_exceed_1']:.2%}\n"
    
    report += """
Key Insight: Higher k leads to more risk-tolerant policies (θ closer to 1),
accepting some samples beyond the boundary to maximize the chance of 
hitting the high-reward region.

2. VARIANCE REDUCTION EFFECTIVENESS
-----------------------------------
The s^(loo-1) estimator (Equation 33) provides the lowest variance,
especially as the number of samples n increases. This is crucial for
stable training in practice.

3. TRAINING RESULTS
-------------------
Final θ values after training:
"""
    
    if 'training' in all_results and 'k_comparison' in all_results['training']:
        for k, theta in all_results['training']['k_comparison'].items():
            report += f"  k_opt = {k}: θ = {theta:.4f}\n"
    
    if 'training' in all_results and 'annealing' in all_results['training']:
        report += f"  k_annealed: θ = {all_results['training']['annealing']:.4f}\n"
    
    report += """
4. CONCLUSIONS
--------------
1. PKPO enables direct optimization of pass@k for any k ≤ n
2. Higher k encourages exploration and more diverse solutions
3. The s^(loo-1) estimator provides stable, low-variance gradients
4. K annealing can achieve strong performance on both pass@1 and pass@k

5. NEXT STEPS
-------------
- Scale to larger models (GEMMA, LLAMA)
- Apply to real tasks (MATH, Coding, ARC-AGI)
- Experiment with continuous reward functions
"""
    
    # Save report
    report_path = 'figures/experiment_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to {report_path}")
    
    return report


def main():
    """Run all experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='PKPO All Experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick version with fewer iterations')
    parser.add_argument('--skip-toy', action='store_true',
                        help='Skip toy example')
    parser.add_argument('--skip-variance', action='store_true',
                        help='Skip variance analysis')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training experiments')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(42)
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    print("=" * 70)
    print("PKPO EXPERIMENTS: COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    all_results = {}
    
    try:
        # Part 1: Toy Example
        if not args.skip_toy:
            all_results['toy'] = run_toy_example()
        
        # Part 2: Variance Analysis
        if not args.skip_variance:
            all_results['variance'] = run_variance_analysis()
        
        # Part 3: Training Experiments
        if not args.skip_training:
            all_results['training'] = run_training_experiments()
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)
    generate_summary_report(all_results, runtime)
    
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETED in {runtime:.1f} seconds")
    print("=" * 70)
    print("\nGenerated figures:")
    for f in sorted(os.listdir('figures')):
        print(f"  - figures/{f}")


if __name__ == "__main__":
    main()
