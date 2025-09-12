# Manifold Learning Methods Documentation

This document provides detailed explanations of the dimensionality reduction and manifold learning methods implemented in MEA-Flow's feature space analysis module.

## Overview

All methods operate on a standardized feature matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ where:
- $n$ = number of samples (experimental observations)
- $p$ = number of features (MEA metrics)
- Data is standardized: $\mathbf{X}_{scaled} = \frac{\mathbf{X} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$

The goal is to find a lower-dimensional representation $\mathbf{Y} \in \mathbb{R}^{n \times d}$ where $d \ll p$ (typically $d = 2$ for visualization).

---

## 1. Principal Component Analysis (PCA)

### Mathematical Formulation

PCA finds orthogonal directions of maximum variance in the data through eigendecomposition of the covariance matrix.

**Covariance Matrix:**
$$\mathbf{C} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$$

**Eigendecomposition:**
$$\mathbf{C}\mathbf{v}_i = \lambda_i\mathbf{v}_i$$

where $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_p$ are eigenvalues and $\mathbf{v}_i$ are corresponding eigenvectors.

**Projection:**
$$\mathbf{Y} = \mathbf{X}\mathbf{W}$$

where $\mathbf{W} = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_d]$ contains the first $d$ principal components.

**Explained Variance Ratio:**
$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j}$$

### Properties
- **Linear transformation**: Preserves linear relationships
- **Orthogonal components**: Each PC is uncorrelated with others
- **Variance maximization**: First PC captures maximum variance
- **Interpretable**: Component loadings show feature contributions

### Use Cases
- Initial exploratory analysis
- Feature importance through component loadings
- Dimensionality reduction for downstream analysis
- Noise reduction (keeping top components)

---

## 2. Multidimensional Scaling (MDS)

### Mathematical Formulation

MDS preserves pairwise distances between samples in the lower-dimensional space.

**Distance Matrix:**
$$\mathbf{D}_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|_2$$

**Objective Function (Stress):**
$$\text{Stress} = \sqrt{\frac{\sum_{i<j}(d_{ij} - \hat{d}_{ij})^2}{\sum_{i<j}d_{ij}^2}}$$

where:
- $d_{ij}$ = original distance between samples $i$ and $j$
- $\hat{d}_{ij}$ = distance in embedded space

**Optimization:**
MDS minimizes stress through iterative optimization (SMACOF algorithm):
$$\mathbf{Y}^{(t+1)} = \frac{1}{n}\mathbf{B}(\mathbf{Y}^{(t)})\mathbf{Y}^{(t)}$$

### Properties
- **Distance preservation**: Maintains global distance relationships
- **Non-linear**: Can capture non-linear data structures
- **Metric**: Assumes distances have meaningful interpretation
- **Global optimization**: Considers all pairwise relationships

### Use Cases
- When distance relationships are important
- Comparing similarity between experimental conditions
- Visualizing global data structure
- Quality assessment via stress values (< 0.1 = good fit)

---

## 3. t-Distributed Stochastic Neighbor Embedding (t-SNE)

### Mathematical Formulation

t-SNE preserves local neighborhood structure by modeling similarities as probability distributions.

**High-Dimensional Similarities:**
$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**Low-Dimensional Similarities:**
$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

**Objective Function (KL Divergence):**
$$C = \text{KL}(P||Q) = \sum_{i}\sum_{j}p_{ij}\log\frac{p_{ij}}{q_{ij}}$$

**Gradient:**
$$\frac{\delta C}{\delta \mathbf{y}_i} = 4\sum_{j}(p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$$

### Properties
- **Local structure preservation**: Excellent for clustering visualization
- **Non-linear**: Captures complex manifold structures
- **Perplexity parameter**: Controls local vs. global structure balance
- **Stochastic**: Different runs may yield different results

### Use Cases
- Visualizing clusters and local neighborhoods
- Identifying condition-specific groupings
- Exploratory data analysis
- Non-linear pattern detection

---

## 4. Uniform Manifold Approximation and Projection (UMAP)

### Mathematical Formulation

UMAP constructs a fuzzy topological representation and optimizes a low-dimensional embedding.

**Fuzzy Simplicial Set Construction:**
$$\rho_i = \min_{j \in N_k(i), j \neq i} d(\mathbf{x}_i, \mathbf{x}_j)$$

$$\sigma_i = \text{solution to } \sum_{j \in N_k(i)} \exp\left(-\frac{\max(0, d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i)}{\sigma_i}\right) = \log_2(k)$$

**High-Dimensional Weights:**
$$w_{ij} = \exp\left(-\frac{\max(0, d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i)}{\sigma_i}\right)$$

**Symmetric Weights:**
$$A_{ij} = w_{ij} + w_{ji} - w_{ij}w_{ji}$$

**Low-Dimensional Optimization:**
$$\Phi(\mathbf{a}, \mathbf{b}) = \frac{1}{1 + a\|\mathbf{y}_i - \mathbf{y}_j\|_2^{2b}}$$

**Objective Function:**
$$C = \sum_{(i,j) \in E} A_{ij}\log\Phi(\mathbf{a}, \mathbf{b}) + \gamma\sum_{(i,j) \notin E}(1-A_{ij})\log(1-\Phi(\mathbf{a}, \mathbf{b}))$$

### Properties
- **Topological foundation**: Based on Riemannian geometry
- **Local and global structure**: Balances both aspects
- **Fast computation**: More efficient than t-SNE
- **Deterministic**: More reproducible results

### Use Cases
- Large dataset visualization
- Preserving both local and global structure
- Faster alternative to t-SNE
- Continuous data exploration

---

## 5. Spectral Embedding

### Mathematical Formulation

Spectral embedding uses eigendecomposition of graph Laplacians to find low-dimensional representations.

**Similarity Graph Construction:**
$$W_{ij} = \begin{cases}
\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma^2) & \text{if } j \in N_k(i) \\
0 & \text{otherwise}
\end{cases}$$

**Graph Laplacian:**
$$\mathbf{L} = \mathbf{D} - \mathbf{W}$$

where $\mathbf{D}$ is the degree matrix: $D_{ii} = \sum_j W_{ij}$

**Normalized Laplacian:**
$$\mathbf{L}_{norm} = \mathbf{D}^{-1/2}\mathbf{L}\mathbf{D}^{-1/2}$$

**Eigendecomposition:**
$$\mathbf{L}_{norm}\mathbf{v}_i = \lambda_i\mathbf{v}_i$$

**Embedding:**
$$\mathbf{Y} = [\mathbf{v}_2, \mathbf{v}_3, \ldots, \mathbf{v}_{d+1}]$$

(Skip the first eigenvector $\mathbf{v}_1$ corresponding to $\lambda_1 = 0$)

### Properties
- **Graph-based**: Captures manifold structure through graphs
- **Spectral theory**: Based on eigenanalysis of Laplacians
- **Non-linear**: Can reveal non-linear structures
- **Parameter sensitive**: Requires careful tuning of $k$ and $\sigma$

### Use Cases
- Manifold learning on graph-structured data
- Clustering visualization
- Non-linear dimensionality reduction
- Community detection in feature space

---

## Implementation Details

### Data Preprocessing
1. **Standardization**: All methods use `StandardScaler` to normalize features
2. **Missing values**: Handled via mean imputation using `SimpleImputer`
3. **Parameter adaptation**: Sample size-dependent parameter adjustment

### Parameter Guidelines

| Method | Key Parameters | Recommendations |
|--------|---------------|-----------------|
| PCA | `n_components` | Start with 2-3 for visualization |
| MDS | `max_iter` | 300-1000 depending on convergence |
| t-SNE | `perplexity` | 5-50, adjust based on sample size |
| UMAP | `n_neighbors`, `min_dist` | 5-50 neighbors, 0.1-0.5 min_dist |
| Spectral | `n_neighbors`, `affinity` | 5-15 neighbors, 'nearest_neighbors' |

### Quality Metrics

- **PCA**: Explained variance ratio (higher = better)
- **MDS**: Stress value (lower = better, < 0.1 good)
- **t-SNE**: KL divergence (lower = better)
- **UMAP**: No direct quality metric
- **Spectral**: Eigenvalue gaps (larger gaps = better separation)

### Computational Complexity

| Method | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| PCA | $O(np^2 + p^3)$ | $O(p^2)$ |
| MDS | $O(n^3)$ | $O(n^2)$ |
| t-SNE | $O(n^2)$ per iteration | $O(n^2)$ |
| UMAP | $O(n^{1.14})$ | $O(n)$ |
| Spectral | $O(n^3)$ | $O(n^2)$ |

---

## Choosing the Right Method

### Decision Framework

1. **Linear relationships**: Use **PCA** first
2. **Global distance preservation**: Use **MDS**
3. **Local clustering structure**: Use **t-SNE**
4. **Balanced local/global + speed**: Use **UMAP**
5. **Graph-based manifolds**: Use **Spectral Embedding**

### Sample Size Considerations

- **Small samples (n < 50)**: PCA, MDS
- **Medium samples (50 < n < 1000)**: All methods applicable
- **Large samples (n > 1000)**: UMAP, PCA (avoid t-SNE)

### Feature Space Considerations

- **High-dimensional (p > 100)**: PCA preprocessing recommended
- **Mixed feature types**: Standardization crucial
- **Sparse features**: Consider feature selection first

---

## Interpretation Guidelines

### PCA
- **Component loadings**: Show feature contributions to each PC
- **Explained variance**: Indicates information retention
- **Biplot**: Combine samples and feature vectors

### MDS
- **Stress values**: < 0.05 excellent, < 0.1 good, < 0.2 fair
- **Distance preservation**: Check correlation between original and embedded distances

### t-SNE
- **Cluster separation**: Well-separated clusters indicate distinct conditions
- **Perplexity effects**: Lower values emphasize local structure
- **Multiple runs**: Compare results across different random seeds

### UMAP
- **Continuous structure**: Better preservation of global topology than t-SNE
- **Parameter sensitivity**: Test different `n_neighbors` values
- **Density preservation**: Maintains relative density information

### Spectral Embedding
- **Eigenvalue gaps**: Large gaps indicate clear cluster structure
- **Graph connectivity**: Ensure graph is connected for meaningful results
- **Parameter tuning**: Critical for good performance
