
# PKPO Experiments: Pass@K Policy Optimization

This repository contains experiments to understand and replicate the **Pass@K Policy Optimization (PKPO)** paper by Walder & Karkhanis from Google DeepMind.

## 📄 Paper Summary

**Problem**: Standard RL optimizes for pass@1 (single sample performance), but in practice we often generate multiple samples and pick the best one. This under-utilizes sampling capacity and limits exploration.

**Solution**: PKPO transforms rewards to directly optimize pass@k (probability that at least one of k samples is correct) using:
1. **Unbiased estimators** for pass@k and its gradient
2. **Efficient computation** in O(k + n log n)

## 📁 Repository Structure

```
pkpo_experiments/
├── pkpo_core.py          # Core PKPO implementation
├── toy_example.py        # Figure 1 replication
├── variance_analysis.py  # Figure 4 replication (variance comparison)
├── training_toy.py       # Training experiments
├── run_all.py           # Main runner script
├── requirements.txt     # Dependencies
└── figures/             # Generated figures
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments (takes ~10-15 minutes)
python run_all.py

# Or run individual components
python toy_example.py
python variance_analysis.py
python training_toy.py
```

## 📊 Key Experiments

### 1. Toy Example (Figure 1)
- Policy: x ~ N(θ, 0.1)
- Reward: g(x) = x² if 0 ≤ x ≤ 1, else 0
- Shows how optimal θ changes with k

**Key Finding**: Higher k → more risk-tolerant policy (θ closer to 1)

### 2. Variance Analysis (Figure 4)
Compares gradient estimator variance for:
- `s`: No baseline
- `s^(loo)`: Leave-one-out baseline
- `s^(loo-1)`: k-1 baseline (lowest variance!)

**Key Finding**: s^(loo-1) has ~10x lower variance than alternatives

### 3. Training Experiments
- Compare training with different k values
- Compare gradient estimation methods
- Demonstrate k annealing (k=8 → k=1)

## 🔧 Core Functions

```python
from pkpo_core import s, sloo, sloo_minus_one

# Raw rewards from your model
rewards = [0.1, 0.3, 0.5, 0.7, 0.9]

# Transform for pass@k optimization (k=4)
transformed = sloo_minus_one(rewards, k=4)

# Use in policy gradient:
# gradient = sum(transformed * score_function)
```

## 🧮 Mathematical Background

### Pass@k Definition
```
pass@k = P(at least one of k samples is correct)
       = 1 - (1-p)^k  (for binary rewards)
```

### PKPO Gradient Estimator
For continuous rewards g, the transformed reward s_i is:
```
s_i = (1/(n choose k)) * Σ_{j=i}^{n} m_ij * g_j
```
where m_ij counts subsets with j as max and containing i.

### Variance Reduction
The s^(loo-1) estimator subtracts for each sample i:
- The maximum of the subset WITH i
- The maximum of the subset WITHOUT i

This measures each sample's "contribution" to achieving the maximum.


## 📚 References

- [PKPO Paper](https://arxiv.org/abs/2505.15201)
- [Policy Gradient Methods](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [Original pass@k estimator](https://arxiv.org/abs/2107.03374) (Chen et al., 2021)


## License

MIT License - see the paper for attribution requirements.
