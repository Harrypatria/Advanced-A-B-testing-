# py-ab-testing: A/B Testing and Multivariate Testing Library

![A/B Testing](https://img.shields.io/badge/A%2FB%20Testing-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-95%25-brightgreen)

A comprehensive Python library for designing, analyzing, and visualizing A/B tests and multivariate tests with built-in support for statistical analysis and multi-armed bandit algorithms.

<p align="center">
  <img src="https://www.abtasty.com/wp-content/uploads/2022/11/Equal-traffic-allocation.png" alt="A/B Testing Diagram" width="600"/>
</p>

<p align="center">
  <img src="https://www.convertize.com/wp-content/uploads/2020/02/larger-samples-ab-testing-statistics.jpg" alt="A/B Testing Diagram" width="600"/>
</p>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [A/B Testing](#ab-testing)
  - [Multivariate Testing](#multivariate-testing)
  - [Multi-Armed Bandits](#multi-armed-bandits)
  - [Bayesian A/B Testing](#bayesian-ab-testing)
  - [Sequential Testing](#sequential-testing)
- [Mathematical Background](#mathematical-background)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [FAQ](#faq)

## 🔍 Overview

`py-ab-testing` is a Python toolkit that provides a robust framework for implementing, analyzing, and visualizing A/B tests and multivariate tests. Whether you're optimizing a website, improving a product, or refining marketing strategies, this library provides the statistical tools required to make data-driven decisions with confidence.

## ✨ Features

- **A/B Testing**: Compare two variants with proper statistical analysis
- **Multivariate Testing**: Test multiple variations simultaneously
- **Multi-Armed Bandits**: Implement adaptive allocation algorithms (Thompson Sampling, Epsilon-Greedy, UCB)
- **Bayesian Methods**: Optional Bayesian approach to test evaluation
- **Sequential Testing**: Methods for continuous monitoring with controlled error rates
- **Statistical Rigor**: Proper hypothesis testing and confidence intervals
- **Sample Size Calculation**: Determine required sample size before running tests
- **Data Visualization**: Generate publication-quality visualizations of test results
- **Segmentation Analysis**: Analyze test results across different user segments

## 📦 Installation

```bash
pip install py-ab-testing
```

For development installation:

```bash
git clone https://github.com/user/py-ab-testing.git
cd py-ab-testing
pip install -e .
```

## 🚀 Quick Start

```python
from py_ab_testing import ABTest
import numpy as np
import matplotlib.pyplot as plt

# Initialize test
ab_test = ABTest(baseline_conv_rate=0.12)

# Calculate required sample size
sample_size = ab_test.calculate_sample_size(mde=0.20)  # 20% minimum detectable effect
print(f"Required sample size per variant: {sample_size}")

# Analyze test results
results = ab_test.analyze_results(
    control_visitors=1000, 
    control_conversions=120,  # 12% conversion rate
    treatment_visitors=1000, 
    treatment_conversions=150  # 15% conversion rate
)

# Print key metrics
print(f"Relative uplift: {results['relative_uplift']:.2%}")
print(f"Statistical significance: {'Yes' if results['is_significant'] else 'No'}")

# Visualize results
ab_test.visualize_results(results)
plt.show()
```

## 📊 Usage Examples

### A/B Testing

```python
from py_ab_testing import ABTest

# Initialize test with baseline conversion rate
ab_test = ABTest(baseline_conv_rate=0.12)

# Calculate required sample size (20% minimum detectable effect)
sample_size = ab_test.calculate_sample_size(mde=0.20)
print(f"Required sample size per variant: {sample_size}")

# Analyze test results
results = ab_test.analyze_results(
    control_visitors=1000, 
    control_conversions=120,  # 12% conversion rate
    treatment_visitors=1000, 
    treatment_conversions=150  # 15% conversion rate
)

# Print key results
print(f"Control Conversion Rate: {results['control_cr']:.2%}")
print(f"Treatment Conversion Rate: {results['treatment_cr']:.2%}")
print(f"Relative Uplift: {results['relative_uplift']:.2%}")
print(f"P-value: {results['p_value']:.6f}")
print(f"Statistical Significance: {'Yes' if results['is_significant'] else 'No'}")

# Visualize results
ab_test.visualize_results(results)
```

### Multivariate Testing

```python
import pandas as pd
from py_ab_testing import MultivariateTest

# Initialize multivariate test
mv_test = MultivariateTest()

# Prepare test data
mv_data = pd.DataFrame({
    'variant': ['Control', 'Variant B', 'Variant C', 'Variant D'],
    'visitors': [1000, 1000, 1000, 1000],
    'conversions': [120, 150, 100, 140]
})

# Analyze results
mv_results = mv_test.analyze_results(mv_data)

# Print summary
print(mv_results[['variant', 'conversion_rate', 'relative_uplift', 
                'p_value', 'is_significant', 'is_winner']])

# Visualize results
mv_test.visualize_results(mv_results)
```

### Multi-Armed Bandits

```python
from py_ab_testing import MultiArmedBandit
import numpy as np

# Initialize bandit with Thompson Sampling
bandit = MultiArmedBandit(n_arms=4, algorithm='thompson_sampling')

# Simulate user interactions
true_conversion_rates = [0.05, 0.08, 0.12, 0.06]
total_reward = 0

for _ in range(1000):
    # Select arm to show next user
    arm = bandit.select_arm()
    
    # Simulate conversion based on true rates
    conversion = np.random.binomial(1, true_conversion_rates[arm])
    total_reward += conversion
    
    # Update bandit with result
    bandit.update(arm, conversion)

# Get final probability estimates
probabilities = bandit.get_arm_probabilities()
print("Final probability estimates:")
for i, prob in enumerate(probabilities):
    print(f"Arm {i} (true rate: {true_conversion_rates[i]:.2%}): {prob:.4f}")

print(f"Total reward: {total_reward}")

# Plot arm selection distribution over time
bandit.plot_arm_distribution_over_time()
```

### Bayesian A/B Testing

```python
from py_ab_testing.bayesian import BayesianABTest
import matplotlib.pyplot as plt

# Initialize Bayesian test with prior
bayesian_test = BayesianABTest(
    prior_alpha=1, 
    prior_beta=1
)

# Add observations
bayesian_test.add_observations(
    control_conversions=120,
    control_visitors=1000,
    treatment_conversions=150,
    treatment_visitors=1000
)

# Get probability that treatment is better
prob_better = bayesian_test.probability_b_better_than_a()
print(f"Probability treatment is better: {prob_better:.2%}")

# Expected loss (opportunity cost) of choosing treatment
exp_loss = bayesian_test.expected_loss()
print(f"Expected loss of choosing treatment: {exp_loss:.4f}")

# Risk of choosing treatment
risk = bayesian_test.expected_loss_relative()
print(f"Relative expected loss: {risk:.2%}")

# Visualize posterior distributions
bayesian_test.plot_posteriors()
plt.show()
```

### Sequential Testing

```python
from py_ab_testing.sequential import SequentialTest
import numpy as np
import matplotlib.pyplot as plt

# Initialize sequential test
seq_test = SequentialTest(
    baseline_cr=0.12,
    mde=0.20,
    alpha=0.05,
    beta=0.20
)

# Simulate observations in batches
control_cr = 0.12
treatment_cr = 0.15
batch_size = 100
max_batches = 20

for batch in range(max_batches):
    # Generate batch data
    control_conv = np.random.binomial(batch_size, control_cr)
    treatment_conv = np.random.binomial(batch_size, treatment_cr)
    
    # Add observations
    seq_test.add_observations(
        control_visitors=batch_size,
        control_conversions=control_conv,
        treatment_visitors=batch_size,
        treatment_conversions=treatment_conv
    )
    
    # Check if we can stop
    decision = seq_test.get_decision()
    
    if decision != "continue":
        print(f"Decision after {batch+1} batches: {decision}")
        break
    else:
        print(f"Batch {batch+1}: Continue testing...")

# If we reached max batches without decision
if batch == max_batches - 1 and decision == "continue":
    print("Reached maximum batches without conclusive result")

# Plot sequential boundaries and test trajectory
seq_test.plot_boundaries()
plt.show()
```

## 📐 Mathematical Background

### Z-Test for Proportions

The statistical significance of conversion rate differences is calculated using the Z-test:

$$Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_A} + \frac{1}{n_B})}}$$

Where:
- $\hat{p}_A$ and $\hat{p}_B$ are the observed conversion rates
- $\hat{p} = \frac{x_A + x_B}{n_A + n_B}$ is the pooled proportion
- $n_A$ and $n_B$ are the sample sizes
- $x_A$ and $x_B$ are the number of conversions

### Sample Size Calculation

Required sample size per variant is calculated as:

$$n = \frac{2 \times (z_{\alpha/2} + z_{\beta})^2 \times p(1-p)}{(\delta)^2}$$

Where:
- $z_{\alpha/2}$ is the critical value for confidence level $1-\alpha$ (typically 1.96 for 95% confidence)
- $z_{\beta}$ is the critical value for power $1-\beta$ (typically 0.84 for 80% power)
- $p$ is the baseline conversion rate
- $\delta$ is the minimum detectable effect (the smallest meaningful difference)

### Confidence Intervals

Confidence intervals for the difference in proportions:

$$CI = \hat{p}_B - \hat{p}_A \pm z_{\alpha/2} \times \sqrt{\frac{\hat{p}_A(1-\hat{p}_A)}{n_A} + \frac{\hat{p}_B(1-\hat{p}_B)}{n_B}}$$

### Bayesian Method

Bayesian A/B testing uses Beta distributions to model conversion rates:

$$P(p|data) \propto \text{Beta}(\alpha + x, \beta + n - x)$$

Where:
- $\alpha$ and $\beta$ are prior parameters
- $x$ is the number of conversions
- $n$ is the number of visitors

The probability that variant B is better than A is:

$$P(p_B > p_A) = \int_0^1 \int_{p_A}^1 P(p_A|data_A) \times P(p_B|data_B) \, dp_B \, dp_A$$

### Multi-Armed Bandit Algorithms

#### Thompson Sampling

1. Model each arm's reward probability as a Beta distribution
2. Sample from each distribution
3. Select the arm with the highest sample

#### Upper Confidence Bound (UCB)

Select arm $a$ that maximizes:

$$UCB_a = \hat{\mu}_a + \sqrt{\frac{2\ln{N}}{n_a}}$$

Where:
- $\hat{\mu}_a$ is the mean reward for arm $a$
- $N$ is the total number of pulls across all arms
- $n_a$ is the number of times arm $a$ has been pulled

## 📚 Documentation

Full documentation with API reference is available at [https://py-ab-testing.readthedocs.io/](https://py-ab-testing.readthedocs.io/)

### Key Modules

- `py_ab_testing.ABTest`: Core A/B testing functionality
- `py_ab_testing.MultivariateTest`: Multivariate testing tools
- `py_ab_testing.MultiArmedBandit`: Bandit algorithm implementations
- `py_ab_testing.bayesian`: Bayesian testing methods
- `py_ab_testing.sequential`: Sequential testing procedures
- `py_ab_testing.utils`: Utility functions and helpers

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and follow the code style.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📝 Citation

If you use this library in your research, please cite:

```
@software{py_ab_testing,
  author = {Harry Patria},
  title = {A/B Testing Experimentation},
  year = {2025},
  url = {https://github.com/Harrypatria/Advanced-A-B-testing-},
}
```

## ❓ FAQ

### When should I use A/B testing vs. multivariate testing?

- **A/B testing** is ideal when you want to test a single change or hypothesis, such as a new headline or button color. It provides clear causality and requires less traffic than multivariate testing.
- **Multivariate testing** is better when you want to test multiple elements simultaneously and understand their interactions. It requires more traffic but can be more efficient for testing multiple changes at once.

### How long should I run my test?

Tests should run until:
1. You reach the pre-calculated sample size
2. You complete at least one full business cycle (typically 1-2 weeks)
3. Results stabilize (showing consistent patterns)

The library's `calculate_test_duration()` method can help estimate appropriate duration based on your traffic and expected conversion rates.

### What is statistical power, and why does it matter?

Statistical power (typically set at 80%) is the probability of detecting a real effect when it exists. Higher power decreases false negatives but requires larger sample sizes. The library uses power in sample size calculations to ensure your tests are adequately powered to detect meaningful effects.

### Can I stop a test early if I see significant results?

It's generally not recommended to stop tests early based on peeking at results, as this increases the risk of false positives. If you need flexibility in test duration, consider using the sequential testing module, which is designed for continuous monitoring with controlled error rates.

### How do multi-armed bandits compare to traditional A/B testing?

- **Traditional A/B testing** is focused on learning: equal traffic allocation to determine the best variant with statistical confidence.
- **Multi-armed bandits** balance learning and earning: they adaptively allocate more traffic to better-performing variants, minimizing opportunity cost during the test.

### How do I interpret a p-value?

The p-value represents the probability of observing a difference as extreme as the one in your sample, assuming no real difference exists. A small p-value (typically <0.05) suggests the observed difference is unlikely to be due to random chance. However, p-values should be interpreted alongside effect sizes and confidence intervals for a complete understanding.

### What's the difference between frequentist and Bayesian approaches?

- **Frequentist approach** (default): Provides p-values and confidence intervals. Answers "How likely is this data, given my hypothesis?"
- **Bayesian approach** (optional): Provides direct probability statements about which variant is better. Answers "How likely is my hypothesis, given this data?"

Both approaches are valid, with trade-offs in interpretation and flexibility.

---

For more information, examples, and tutorials, visit our [website]([https://py-ab-testing.readthedocs.io/](https://learning.patriaco.id/)).
