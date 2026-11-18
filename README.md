# Bayesian vs Perceptron Classification Study

**Project No. 15 · BCS-DS-738T Mini Project**  
**Course Outcomes:** CO2, CO3 | **Bloom's Taxonomy Level (BTL):** 5

## 📖 Description

This project systematically compares **generative (Bayesian)** and **discriminative (Perceptron)** classifiers under controlled synthetic datasets and real-world benchmarks. Implemented completely from scratch using NumPy, the notebook isolates specific statistical properties of data—such as covariance mismatches, feature correlation, class overlap, non-Gaussian skewness, and non-linear XOR geometry—to empirically evaluate how feature distributions govern decision boundaries and convergence.

---

## 🧠 Theoretical Overview

### Bayesian Classifiers (Generative)
Models each class using a Gaussian probability density function and assigns labels via Bayes' rule:

$$\hat{y} = \arg\max_c \Big[ \log P(c) + \log \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_c, \Sigma_c) \Big]$$

- **Full-Covariance Gaussian Bayes (QDA-style):** Estimates an individual covariance matrix $\Sigma_c$ per class, producing **quadratic (curved) decision boundaries**. When classes share identical covariance, the boundary collapses to a linear hyperplane (LDA).
- **Gaussian Naive Bayes:** Assumes conditional feature independence given the class, constraining $\Sigma_c$ to a diagonal matrix and yielding axis-aligned quadratic boundaries. Performance degrades when features exhibit strong correlation.

### Perceptron (Discriminative)
Learns a linear separating hyperplane $\mathbf{w}^\top \mathbf{x} + b = 0$ directly without distributional assumptions:
- Updates weights iteratively on misclassified samples: $\mathbf{w} \leftarrow \mathbf{w} + \eta y \mathbf{x}$.
- Restricted strictly to **linear decision boundaries** and guaranteed to converge **only on linearly separable data** (Novikoff's theorem), oscillating indefinitely when classes overlap[cite: 1].

---

## 🎯 Objectives

- Implement **Gaussian Naive Bayes**, **Full-Covariance Gaussian Bayes**, and **Perceptron** algorithms from scratch using only NumPy[cite: 1].
- Validate custom Bayesian implementations against `scikit-learn` baselines[cite: 1].
- Analyze how feature skewness, kurtosis, correlation, covariance mismatch, and class overlap impact classification accuracy and F1-score[cite: 1].
- Visualize decision boundaries, confusion matrices, parameter sensitivity sweeps, and Perceptron epoch-by-epoch convergence[cite: 1].
- Evaluate model generalization on standard real-world datasets (Iris and Breast Cancer Wisconsin)[cite: 1].

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **NumPy** (Core linear algebra and scratch classifier implementations)[cite: 1]
- **Pandas** (Experiment logging and tabular summaries)[cite: 1]
- **SciPy** (Statistical distribution tests: skewness, kurtosis, normality)[cite: 1]
- **Scikit-learn** (Reference validation, train/test splitting, evaluation metrics, benchmark datasets)[cite: 1]
- **Matplotlib & Seaborn** (Decision boundary contour plots, confusion matrices, convergence curves)[cite: 1]

---

## 📂 Notebook Structure

The Jupyter/Colab notebook is self-contained with zero external file dependencies and executes in ~1–3 minutes[cite: 1]:

| Section | Title | Description |
| :--- | :--- | :--- |
| **0** | Theory in Brief | Mathematical formulation of Bayesian and Perceptron decision rules[cite: 1]. |
| **1** | Setup | Global configuration, random seed initialization (`SEED = 42`), and visualization styling[cite: 1]. |
| **2** | Models from Scratch | NumPy implementations of `GaussianBayes` (diagonal & full covariance) and `PerceptronScratch`[cite: 1]. |
| **3** | Controlled Datasets | Generation of six 2-D synthetic scenarios isolating distinct geometric/statistical properties[cite: 1]. |
| **4** | Validation | Numerical verification of custom Bayes models against `GaussianNB` and `QuadraticDiscriminantAnalysis`[cite: 1]. |
| **5** | Feature Distribution Analysis | Quantitative profiling of skewness, kurtosis, normality, correlation, and covariance ratios[cite: 1]. |
| **6** | Main Experiment | Multi-seed evaluation (`10` random seeds per scenario) reporting mean Accuracy and F1-score[cite: 1]. |
| **7** | Decision Boundaries | 2-D decision surface plots comparing linear vs. quadratic boundaries across all scenarios[cite: 1]. |
| **8** | Confusion Matrices | Class-wise error breakdowns per scenario and classifier[cite: 1]. |
| **9** | Controlled Sweeps | Sensitivity analysis across class separation, correlation strength, covariance mismatch, and sample size[cite: 1]. |
| **10** | Perceptron Convergence | Epoch-wise mistake tracking comparing linearly separable vs. overlapping distributions[cite: 1]. |
| **11** | Real Datasets | Benchmark evaluation on **Iris** and **Breast Cancer Wisconsin** datasets[cite: 1]. |
| **12** | Interactive Experiment | User-prompted (`input()`) interactive scenario selector for live testing[cite: 1]. |
| **13** | Findings & Conclusion | Automated summary of empirical results and theoretical takeaways[cite: 1]. |

---

## 🧪 Controlled Synthetic Scenarios

Six 2-D datasets (`N = 600` samples each) isolate a single data characteristic per experiment[cite: 1]:

| Code | Scenario | Isolated Property | Expected Behavior |
| :---: | :--- | :--- | :--- |
| **A** | Equal Covariance, Linearly Separable | Ideal baseline conditions[cite: 1] | All models achieve ~100% accuracy; Perceptron converges rapidly[cite: 1]. |
| **B** | Unequal Covariance | Requires a curved boundary[cite: 1] | Full-covariance Bayes outperforms Perceptron's linear constraint[cite: 1]. |
| **C** | Strongly Correlated Features | Violates feature independence[cite: 1] | Full-covariance Bayes surpasses Gaussian Naive Bayes[cite: 1]. |
| **D** | Heavy Class Overlap | Non-separable distributions[cite: 1] | Bayesian posteriors remain stable; Perceptron fails to converge[cite: 1]. |
| **E** | Non-Gaussian Features | Uniform vs. skewed exponential[cite: 1] | Evaluates robustness of Gaussian assumptions under heavy skewness[cite: 1]. |
| **F** | XOR Layout | Non-linearly separable clusters[cite: 1] | Perceptron and Naive Bayes perform near chance; full-covariance Bayes succeeds via tilted class ellipses[cite: 1]. |

---

## ⚙️ Installation & Usage

### Running Locally
```bash
# Clone the repository
git clone <repository-url>
cd <repository-folder>

# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter

# Launch Jupyter Notebook
jupyter notebook
