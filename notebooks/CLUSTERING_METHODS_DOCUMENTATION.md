# Clustering Methods Documentation

This document provides detailed explanations of the clustering algorithms implemented in MEA-Flow's clustering analysis module.

## Overview

All clustering methods operate on a standardized feature matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ where:
- $n$ = number of samples (experimental observations)
- $p$ = number of features (MEA metrics)
- Data is standardized: $\mathbf{X}_{scaled} = \frac{\mathbf{X} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$

The goal is to partition samples into $k$ clusters $\mathcal{C} = \{C_1, C_2, \ldots, C_k\}$ such that samples within clusters are similar and samples between clusters are dissimilar.

---

## 1. K-means Clustering

### Mathematical Formulation

K-means partitions data into $k$ clusters by minimizing within-cluster sum of squared distances.

**Objective Function:**
$$J = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$

where $\boldsymbol{\mu}_i$ is the centroid of cluster $C_i$:
$$\boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{\mathbf{x} \in C_i} \mathbf{x}$$

**Lloyd's Algorithm:**
1. **Initialize** centroids $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k$ randomly
2. **Assignment step**: Assign each point to nearest centroid
   $$C_i^{(t)} = \{\mathbf{x}_j : \|\mathbf{x}_j - \boldsymbol{\mu}_i^{(t)}\| \leq \|\mathbf{x}_j - \boldsymbol{\mu}_l^{(t)}\| \text{ for all } l\}$$
3. **Update step**: Recompute centroids
   $$\boldsymbol{\mu}_i^{(t+1)} = \frac{1}{|C_i^{(t)}|} \sum_{\mathbf{x} \in C_i^{(t)}} \mathbf{x}$$
4. **Repeat** until convergence

**Convergence Criterion:**
$$\sum_{i=1}^{k} \|\boldsymbol{\mu}_i^{(t+1)} - \boldsymbol{\mu}_i^{(t)}\| < \epsilon$$

### Properties
- **Centroid-based**: Each cluster represented by its mean
- **Spherical clusters**: Assumes clusters are roughly spherical and equal-sized
- **Hard assignment**: Each point belongs to exactly one cluster
- **Sensitive to initialization**: Multiple runs with different initializations recommended

### Use Cases
- Well-separated, compact clusters
- Known number of clusters
- Fast baseline clustering method
- Preprocessing for other algorithms

---

## 2. Gaussian Mixture Models (GMM)

### Mathematical Formulation

GMM models data as a mixture of $k$ multivariate Gaussian distributions.

**Mixture Model:**
$$p(\mathbf{x}) = \sum_{i=1}^{k} \pi_i \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$$

where:
- $\pi_i$ = mixing coefficient (prior probability of cluster $i$)
- $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$ = multivariate Gaussian with mean $\boldsymbol{\mu}_i$ and covariance $\boldsymbol{\Sigma}_i$

**Multivariate Gaussian:**
$$\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

**Log-Likelihood:**
$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{j=1}^{n} \log\left(\sum_{i=1}^{k} \pi_i \mathcal{N}(\mathbf{x}_j; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)\right)$$

**Expectation-Maximization Algorithm:**

**E-step**: Compute posterior probabilities (responsibilities)
$$\gamma_{ji} = \frac{\pi_i \mathcal{N}(\mathbf{x}_j; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)}{\sum_{l=1}^{k} \pi_l \mathcal{N}(\mathbf{x}_j; \boldsymbol{\mu}_l, \boldsymbol{\Sigma}_l)}$$

**M-step**: Update parameters
$$\pi_i^{new} = \frac{1}{n}\sum_{j=1}^{n} \gamma_{ji}$$

$$\boldsymbol{\mu}_i^{new} = \frac{\sum_{j=1}^{n} \gamma_{ji} \mathbf{x}_j}{\sum_{j=1}^{n} \gamma_{ji}}$$

$$\boldsymbol{\Sigma}_i^{new} = \frac{\sum_{j=1}^{n} \gamma_{ji} (\mathbf{x}_j - \boldsymbol{\mu}_i^{new})(\mathbf{x}_j - \boldsymbol{\mu}_i^{new})^T}{\sum_{j=1}^{n} \gamma_{ji}}$$

**Model Selection Criteria:**
- **AIC**: $AIC = -2\mathcal{L} + 2d$
- **BIC**: $BIC = -2\mathcal{L} + d \log n$

where $d$ is the number of parameters.

### Properties
- **Probabilistic**: Provides soft cluster assignments
- **Flexible shapes**: Can model elliptical clusters with different orientations
- **Covariance types**: Full, tied, diagonal, or spherical covariance matrices
- **Model selection**: AIC/BIC for choosing optimal number of components

### Use Cases
- Overlapping clusters
- Uncertainty quantification in cluster assignments
- Non-spherical cluster shapes
- Density estimation

---

## 3. DBSCAN (Density-Based Spatial Clustering)

### Mathematical Formulation

DBSCAN groups together points in high-density regions and marks points in low-density regions as outliers.

**Key Definitions:**
- **$\epsilon$-neighborhood**: $N_\epsilon(\mathbf{x}) = \{\mathbf{y} \in \mathbf{X} : d(\mathbf{x}, \mathbf{y}) \leq \epsilon\}$
- **Core point**: $|N_\epsilon(\mathbf{x})| \geq \text{MinPts}$
- **Border point**: Not core but in $\epsilon$-neighborhood of a core point
- **Noise point**: Neither core nor border

**Density Reachability:**
Point $\mathbf{q}$ is **directly density-reachable** from $\mathbf{p}$ if:
1. $\mathbf{q} \in N_\epsilon(\mathbf{p})$
2. $\mathbf{p}$ is a core point

Point $\mathbf{q}$ is **density-reachable** from $\mathbf{p}$ if there exists a chain:
$$\mathbf{p} = \mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_n = \mathbf{q}$$
where each $\mathbf{p}_{i+1}$ is directly density-reachable from $\mathbf{p}_i$.

**Density Connectivity:**
Points $\mathbf{p}$ and $\mathbf{q}$ are **density-connected** if there exists a core point $\mathbf{o}$ such that both $\mathbf{p}$ and $\mathbf{q}$ are density-reachable from $\mathbf{o}$.

**Cluster Definition:**
A cluster $C$ satisfies:
1. **Maximality**: If $\mathbf{p} \in C$ and $\mathbf{q}$ is density-reachable from $\mathbf{p}$, then $\mathbf{q} \in C$
2. **Connectivity**: All points in $C$ are density-connected

**Algorithm:**
1. For each unvisited point $\mathbf{p}$:
   - Find $N_\epsilon(\mathbf{p})$
   - If $|N_\epsilon(\mathbf{p})| < \text{MinPts}$, mark as noise
   - Otherwise, start new cluster and expand
2. For cluster expansion:
   - Add all density-reachable points to cluster
   - Recursively check neighbors of core points

### Properties
- **No assumption on cluster number**: Automatically determines clusters
- **Arbitrary shapes**: Can find non-convex clusters
- **Noise detection**: Identifies outliers explicitly
- **Parameter sensitive**: Performance depends on $\epsilon$ and MinPts

### Use Cases
- Unknown number of clusters
- Non-convex cluster shapes
- Outlier detection
- Varying cluster densities

---

## 4. Agglomerative Hierarchical Clustering

### Mathematical Formulation

Hierarchical clustering builds a tree of clusters by iteratively merging the closest pairs.

**Distance Matrix:**
$$\mathbf{D} = [d_{ij}]_{n \times n}$$
where $d_{ij} = d(\mathbf{x}_i, \mathbf{x}_j)$ is the distance between samples $i$ and $j$.

**Linkage Criteria:**
Define distance between clusters $C_i$ and $C_j$:

**Single Linkage (Minimum):**
$$d_{single}(C_i, C_j) = \min_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$$

**Complete Linkage (Maximum):**
$$d_{complete}(C_i, C_j) = \max_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$$

**Average Linkage:**
$$d_{average}(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{\mathbf{x} \in C_i} \sum_{\mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$$

**Ward Linkage:**
$$d_{ward}(C_i, C_j) = \sqrt{\frac{|C_i||C_j|}{|C_i| + |C_j|}} \|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|^2$$

where $\boldsymbol{\mu}_i$ and $\boldsymbol{\mu}_j$ are cluster centroids.

**Algorithm:**
1. **Initialize**: Each point as its own cluster
2. **Repeat** until one cluster remains:
   - Find closest pair of clusters $(C_i, C_j)$
   - Merge into new cluster $C_{new} = C_i \cup C_j$
   - Update distance matrix
3. **Cut** dendrogram at desired level to get $k$ clusters

**Dendrogram:**
Tree structure showing merge hierarchy with heights representing merge distances.

### Properties
- **Hierarchical structure**: Shows relationships at multiple scales
- **Deterministic**: Same result every run (given same linkage)
- **No cluster number assumption**: Can cut at any level
- **Linkage sensitivity**: Different linkage criteria give different results

### Use Cases
- Understanding data hierarchy
- Exploratory analysis of cluster structure
- When cluster number is unknown
- Phylogenetic analysis

---

## Implementation Details

### Data Preprocessing
1. **Standardization**: All methods use `StandardScaler` to normalize features
2. **Missing values**: Handled via mean imputation using `SimpleImputer`
3. **Parameter adaptation**: Sample size-dependent parameter adjustment

### Parameter Guidelines

| Method | Key Parameters | Recommendations |
|--------|---------------|-----------------|
| K-means | `n_clusters`, `n_init` | Try 2-10 clusters, 10+ initializations |
| GMM | `n_components`, `covariance_type` | Use BIC for model selection |
| DBSCAN | `eps`, `min_samples` | Use k-distance plot for eps |
| Hierarchical | `n_clusters`, `linkage` | Ward for compact clusters |

### Cluster Validation Metrics

**Internal Metrics** (no ground truth needed):
- **Silhouette Score**: $s = \frac{b - a}{\max(a, b)}$ where $a$ = intra-cluster distance, $b$ = nearest-cluster distance
- **Calinski-Harabasz Index**: $\frac{SS_B/(k-1)}{SS_W/(n-k)}$ (between/within cluster variance ratio)
- **Davies-Bouldin Index**: $\frac{1}{k}\sum_{i=1}^{k}\max_{j \neq i}\frac{\sigma_i + \sigma_j}{d_{ij}}$ (lower is better)

**External Metrics** (require ground truth):
- **Adjusted Rand Index**: $ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}$
- **Normalized Mutual Information**: $NMI = \frac{MI(C, T)}{\sqrt{H(C)H(T)}}$

### Computational Complexity

| Method | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| K-means | $O(nkdi)$ | $O(nk)$ |
| GMM | $O(nk^2di)$ | $O(nk + k^2d)$ |
| DBSCAN | $O(n^2)$ | $O(n)$ |
| Hierarchical | $O(n^3)$ | $O(n^2)$ |

where $n$ = samples, $k$ = clusters, $d$ = features, $i$ = iterations.

---

## Choosing the Right Method

### Decision Framework

1. **Known cluster number + spherical clusters**: **K-means**
2. **Probabilistic assignments + model selection**: **GMM**
3. **Unknown clusters + noise detection**: **DBSCAN**
4. **Hierarchical relationships**: **Agglomerative**

### Data Characteristics

| Characteristic | Best Method |
|---------------|-------------|
| Spherical clusters | K-means, GMM |
| Arbitrary shapes | DBSCAN, Hierarchical |
| Overlapping clusters | GMM |
| Noise/outliers | DBSCAN |
| Small datasets | Hierarchical |
| Large datasets | K-means, DBSCAN |

### Sample Size Considerations

- **Small samples (n < 100)**: All methods applicable
- **Medium samples (100 < n < 10,000)**: All methods applicable
- **Large samples (n > 10,000)**: K-means, DBSCAN (avoid hierarchical)

---

## Interpretation Guidelines

### K-means
- **Inertia**: Lower values indicate tighter clusters
- **Elbow method**: Plot inertia vs. k to find optimal number
- **Centroids**: Interpret cluster centers in feature space

### GMM
- **BIC/AIC**: Use for model selection (lower is better)
- **Probabilities**: Soft assignments show uncertainty
- **Covariance**: Elliptical cluster shapes and orientations

### DBSCAN
- **Noise ratio**: High ratios may indicate poor parameters
- **Cluster sizes**: Can vary significantly
- **Parameter tuning**: Use k-distance plot for eps selection

### Hierarchical
- **Dendrogram**: Shows merge order and distances
- **Cophenetic correlation**: Measures dendrogram quality
- **Cut height**: Determines final number of clusters

### Validation Strategy

1. **Visual inspection**: Plot clusters in reduced dimensions
2. **Multiple metrics**: Don't rely on single validation measure
3. **Domain knowledge**: Consider biological/experimental meaning
4. **Stability**: Test robustness across parameter ranges
5. **Comparison**: Try multiple methods and compare results

