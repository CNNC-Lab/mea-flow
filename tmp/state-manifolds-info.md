---
{}
---

# Neural Manifold Learning

A notion that has been gaining traction in theoretical and computational neuroscience is that of *intrinsic manifolds* or *neural manifolds*, which refers to the notion that distributed, high-dimensional population-dynamics is effectively constrained to a lower-dimensional subspace (a *latent manifold*), i.e., the neural population, as a high-dimensional nonlinear dynamical system, tends to operate along a smaller, restricted subspace.

This idea has led to several methods that are increasingly common to use when investigating task-related dynamics and, for example, separate the responses to different stimuli or task conditions. Here, we will use it to compare the population recordings in the 3 different experimental conditions.

Several common **dimensionality reduction** and **manifold learning** methods (as well as some more refined derivations) can be employed for this analysis. Given that the dataset refers to recordings of *spontaneous activity* in different experimental conditions and there are no task-related variables / labels, we will make some modifications to the usual
analysis.
<img width="1374" alt="image" src="https://user-images.githubusercontent.com/38789733/161796468-ca38b653-ed4a-43e7-b8f4-bc25a9548539.png">**Fig. 1**: Schematic depiction of how a manifold learning algorithm may reduce the dimensionality of a high dimensional neural population time series to produce a more interpretable low dimensional representation. A high-dimensional neural population activity matrix, X, with N neurons and T time points, is projected into a lower dimensional manifold space and the trajectory visualised in the space formed by the first 3 dimensions, c1, c2 and c3.

There are multiple dimensionality reduction and manifold learning algorithms that can be applied. Given that the main purpose of this analysis is to obtain a better intuition for feature / condition separation and internal representations, the criterion that determines which algorithm to choose is usually the one that yields the most informative projection. In the following, we will start by comparing the main algorithms and the embeddings they generate for the dataset in question. Beyond visual inspection, we quantify how accurately we can reconstruct the original activity from the embedding.

**Notes:**
1. The analysis relies on continuous state variables so a common approach is to apply a low-pass filter to the spiking activity. This is the first step in our analyses (see examples in point 1 below)
2. The recordings do not reflect the activity of a single network, but rather several dissociated networks (different wells). So, a more careful analysis may be needed to look into the activity in individual wells
3. The recordings are quite long and, for some analyses, the memory required to compute is prohibitive (e.g. distance matrices). Also, for visualization purposes, it is often simpler to look at smaller time windows. So, in several instances, we analyse population activity in 5-10s windows
4. Given that the analysis we aim for consists of comparing 3 experimental conditions, we will merge the recordings and label the conditions before performing the analysis (see point 2 below)

## A) Analyzing a single experimental condition
(see [[state-manifolds-single.ipynb]])
##### 1. From spikes to continuous state matrices
The analysis relies on continuous state variables so a common approach is to apply a low-pass filter to the spiking activity. We employ an exponential filter kernel $g(t)=\mathrm{exp}(-t/\tau)$ with $\tau=20 \mathrm{ms}$. This is common practice to smoothen the spiking signal and as a first approximation to postsynaptic integration via fast AMPARs. Some people use Gaussian kernels, but this breaks causality, so I always prefer to use an exponential. Nevertheless, it is important to keep in mind that the choice of $\tau$ has a significant impact on the result.
![[sample_traces1.png]]
![[sample-states1.png]]
(note that the mean activity $\bar{X}$ can be negative because the matrix is standardized to zero mean and unit variance, this can be changed, it's used here for convenience)

##### 2. Pair-wise state distances (complexity of network dynamics)
We can quantify how complex are the network's states by evaluating pairwise distances between states as, for example, using Euclidean (L2) distances:
$$
d = <|| X_{i} - X_{j} ||^{2}>_{T}
$$
![[Euc_dist_n1.png]]
or using cosine dissimilarity, which yields values in $[0,1]$
![[cosine_dist_n1.png]]
It is clear that there are 4 independent wells, with a strong inter-well similarity, so the recordings consist of 4 disjoint populations
##### 3. Sample trajectories and effective sub-space dimensionality
Let's first look at a sample trajectory, embedded in a low-dimensional space, using PCA (the simplest and most straightforward dimensionality reduction method). We can also quantify the effective dimensionality as:
$$
\lambda_{\mathrm{eff}} = \left(\sum_{a=1}^N \tilde{\lambda}_a^2\right)^{-1}
$$
which quantifies how explained variance is distributed among the principal components (PC). If all PCs capture equal amounts of variance in the data, λeff will be high. If some PCs explain a large amount of variance relative to other PCs, λeff will be low.
![[sample_trajectory.png]]- sample for the first 5 sec
![[full_trajectory.png]]- entire recording
Note that the trajectories have an odd shape (like emerging from a central point) because of the separated wells, which yield essentially 4 different populations. A thorough analysis would need to separate these elements.

We can also quickly check if, within the trajectory, there are regions of state-space where the network stays for longer periods of time. We use 2D kernel-density estimation (with a Gaussian kernel) to obtain estimates of state density:
![[state_density_small.png]]

##### 4. Manifold extraction and evaluation (for a single condition)
(see [[manifold-comparisons.ipynb]])

Because there are no task conditions (to look for representations), we will use time as the label, i.e., we get a monotonically increasing numerical label for the state vectors. So, the representations are not really meaningful but the application of these methods requires labels. The goal of this initial analysis is to choose the method that is best suited for the data at hand. We will test and compare the reconstruction accuracy for the different methods, for the different task conditions, to decide which method we will apply in the comparisons.  
  
Also, these methods are typically used to identify latent variables and encoding sub-spaces. Our aim for this analysis is not to identify latent dynamics, but rather to identify and compare the overall manifold geometry across the different experimental conditions.

>[!INFO] Algorithms tested:
>1. Principal Component Analysis (PCA)
>2. Multi-Dimensional Scaling (MDS)
>3. Isomap Embedding
>4. Locally Linear Embedding (LLE)
>5. Laplacian Eigenmaps (LEM) or Spectral Embedding
>6. t-distributed Stochastic Neighbour Embedding (t-SNE)
>7. Uniform Manifold Approximation and Projection (UMAP)

![[Research/MEA-data/code/PyDataAnalysis/plots/pca1.png]]
![[pca2-dim.png]]
![[reconstruction-pca.png]]


![[mds1.png]]
![[mds2.png]]

(...) repeat for all algorithms (...)
![[Isomap1.png]]
![[LLE1.png]]
![[Spectral1.png]]
![[tSNE1.png]]
![[UMAP1.png]]


##### 5. Reliability and comparison of different methods

>[!info] Evaluation Metrics: 
>1. Optimal linear estimator
>2. Reconstruction error / score
>3. Intrinsic dimensionality

![[dimensionality_real.png]]

![[decode_rmseErr_real.png]]
![[totalRMSE_real.png]]


![[rec_corr_real.png]]
![[totalRecAcc_real.png]]
![[totalRecAcc_real.png]]

![[decode_corrErr_real.png]]
![[totalDecAcc_real.png]]

**Notes:**
- There are severe constraints because of how these results were acquired (single dataset, time as labels, multiple sub-populations, etc.)
- The reliability and validity of these results is questionable, but they mean to illustrate the kind of analysis that can be done to systematically assess which methods are more suitable
- Based on these (sketchy) results, PCA stands out as one with highest reconstruction accuracy and UMAP as the one with highest correlation between original and reconstructed data
## B) Comparison across conditions
(see [[global-manifolds.ipynb]])

Now, we repeat the analysis on the concatenated responses. To simplify, we start by using a random sample of the population responses and mixing the conditions (with appropriate labels). We first get a quick idea of the data structure using PCA:

![[Pasted image 20240403160549.png]]
- the 3 experimental conditions are clearly segregated, with condition C3 leading to more compact population activity

Let's see the outcome using UMAP
![[comparison-umap.png]]