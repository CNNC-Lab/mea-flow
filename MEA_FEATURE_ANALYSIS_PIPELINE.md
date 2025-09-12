# MEA Feature Analysis Pipeline Design

## Overview

This document outlines a comprehensive, robust feature analysis pipeline specifically designed for MEA neural metrics data. The pipeline is structured in phases to systematically identify critical, redundant, and irrelevant features for differentiating experimental conditions.

## Pipeline Architecture

### Phase 1: Data Preparation and Initial Assessment
**Objective**: Understand data structure and prepare for analysis

#### 1.1 Data Loading and Validation
- Load MEA metrics dataset (CSV format)
- Validate data integrity (missing values, outliers, data types)
- Separate features from target variable (experimental conditions)
- Basic descriptive statistics

#### 1.2 Exploratory Data Analysis
- Feature distribution analysis
- Target class distribution and balance
- Initial correlation heatmap visualization
- Identify potential data quality issues

### Phase 2: Redundancy Detection and Preprocessing
**Objective**: Remove highly redundant and uninformative features

#### 2.1 Basic Filtering
```python
# Remove constant and near-constant features
variance_threshold = VarianceThreshold(threshold=0.01)
```

#### 2.2 Multicollinearity Detection
```python
# Calculate VIF for all features
# Remove features with VIF > 10 (severe multicollinearity)
# Iterative process: remove highest VIF, recalculate
```

#### 2.3 High Correlation Removal
```python
# Calculate correlation matrix
# Remove one feature from pairs with |correlation| > 0.9
# Prefer to keep features with higher variance or domain relevance
```

**Expected Outcome**: Reduced feature set with minimal redundancy

### Phase 3: Core Feature Selection Methods
**Objective**: Apply multiple complementary feature selection approaches

#### 3.1 Filter Methods (Statistical Significance)
```python
# Method 1: Mutual Information
mutual_info_scores = mutual_info_classif(X, y, random_state=42)

# Method 2: ANOVA F-test
f_scores, p_values = f_classif(X, y)

# Method 3: Kruskal-Wallis (non-parametric alternative)
kruskal_scores = [kruskal(*[X[y==class_i, feature_idx] 
                           for class_i in np.unique(y)])[0] 
                  for feature_idx in range(X.shape[1])]
```

#### 3.2 Embedded Methods (Model-Based)
```python
# Method 4: Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda_weights = np.abs(lda.fit(X, y).coef_[0])

# Method 5: Random Forest with Permutation Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

# Method 6: LASSO Regression (L1 Regularization)
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y_encoded)  # For regression formulation
lasso_weights = np.abs(lasso.coef_)
```

#### 3.3 Wrapper Methods (Performance-Based)
```python
# Method 7: Recursive Feature Elimination with Cross-Validation
rfecv = RFECV(estimator=RandomForestClassifier(random_state=42), 
              cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X, y)

# Method 8: Boruta Algorithm (if computational resources allow)
boruta = BorutaPy(RandomForestClassifier(random_state=42), 
                  n_estimators=100, verbose=2, random_state=42)
boruta.fit(X, y)
```

### Phase 4: Advanced Methods (Selected Based on Data Characteristics)

#### 4.1 Minimum Redundancy Maximum Relevance (mRMR)
```python
# Specifically designed for handling redundant features
selected_features = mrmr_classif(X=X_df, y=y, K=20)  # Select top 20
```

#### 4.2 Dimensionality Reduction (if needed)
```python
# PCA for linear dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# UMAP for non-linear visualization
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)
```

### Phase 5: Validation and Statistical Testing
**Objective**: Ensure robustness and statistical significance

#### 5.1 Cross-Validation Stability
```python
# Measure stability of feature selection across CV folds
def calculate_stability_score(feature_selections):
    # Jaccard similarity across folds
    # Kuncheva stability index
    return stability_metrics

stability_scores = cross_val_stability(X, y, feature_selectors, cv=5)
```

#### 5.2 Permutation Testing
```python
# Test statistical significance of feature importance scores
def permutation_test(X, y, importance_func, n_permutations=1000):
    # Permute target labels and recalculate importance
    # Compare original scores to null distribution
    return p_values

p_values = permutation_test(X, y, mutual_info_classif, n_permutations=1000)
```

#### 5.3 Bootstrap Confidence Intervals
```python
# Estimate confidence intervals for feature importance
def bootstrap_importance(X, y, importance_func, n_bootstrap=1000):
    # Bootstrap sampling with replacement
    # Calculate importance for each bootstrap sample
    return confidence_intervals

ci_lower, ci_upper = bootstrap_importance(X, y, mutual_info_classif)
```

### Phase 6: Consensus Ranking and Feature Categorization
**Objective**: Combine results from multiple methods for robust conclusions

#### 6.1 Consensus Ranking
```python
# Combine rankings from all methods using rank aggregation
def consensus_ranking(method_rankings, weights=None):
    # Borda count or weighted rank aggregation
    # Handle missing rankings (methods that don't rank all features)
    return consensus_scores

final_ranking = consensus_ranking({
    'mutual_info': mi_ranking,
    'anova_f': f_ranking,
    'lda_weights': lda_ranking,
    'rf_importance': rf_ranking,
    'permutation_imp': perm_ranking,
    'lasso_weights': lasso_ranking,
    'rfecv': rfecv_ranking,
    'mrmr': mrmr_ranking
})
```

#### 6.2 Feature Categorization
```python
# Categorize features based on consensus results and statistical significance
def categorize_features(consensus_scores, stability_scores, p_values, thresholds):
    categories = {
        'critical': [],      # High consensus score, stable, significant
        'important': [],     # Moderate consensus score, stable
        'method_specific': [], # High in some methods, low stability
        'redundant': [],     # Removed in redundancy detection
        'irrelevant': []     # Low consensus score, not significant
    }
    return categories

feature_categories = categorize_features(
    consensus_scores=final_ranking,
    stability_scores=stability_scores,
    p_values=p_values,
    thresholds={'critical': 0.8, 'important': 0.6, 'significant': 0.05}
)
```

### Phase 7: Results Interpretation and Visualization
**Objective**: Generate interpretable outputs and actionable insights

#### 7.1 Feature Importance Visualization
- Consensus ranking heatmap across methods
- Feature importance distributions with confidence intervals
- Stability scores visualization
- Method agreement matrix

#### 7.2 Biological/Experimental Interpretation
- Map selected features to neural activity domains:
  - Firing rate metrics
  - Spike timing metrics  
  - Synchrony/connectivity metrics
  - Burst activity metrics
- Identify which experimental conditions are best differentiated
- Suggest biological mechanisms underlying feature importance

#### 7.3 Performance Validation
```python
# Test classification performance with selected feature subsets
def validate_feature_subsets(X, y, feature_subsets):
    results = {}
    for subset_name, features in feature_subsets.items():
        # Cross-validation with multiple classifiers
        cv_scores = cross_validate_multiple_classifiers(
            X[:, features], y, 
            classifiers=['LDA', 'RandomForest', 'SVM', 'LogisticRegression']
        )
        results[subset_name] = cv_scores
    return results

performance_results = validate_feature_subsets(X, y, {
    'critical_features': feature_categories['critical'],
    'critical_plus_important': feature_categories['critical'] + feature_categories['important'],
    'top_10_consensus': final_ranking[:10],
    'mrmr_top_10': mrmr_features[:10],
    'all_features': list(range(X.shape[1]))
})
```

## Implementation Strategy

### Core Methods to Implement (Phase 1)
1. **VIF and Correlation Analysis** - Essential preprocessing
2. **Mutual Information** - Captures non-linear relationships
3. **LDA with Feature Weights** - Excellent performance in your data
4. **Random Forest + Permutation Importance** - Robust, handles interactions
5. **Cross-Validation Stability** - Ensures reliability
6. **Consensus Ranking** - Combines multiple methods

### Extended Methods (Phase 2)
7. **mRMR** - Specifically designed for redundant features
8. **ANOVA F-test** - Statistical significance
9. **LASSO/Elastic Net** - Automatic feature selection
10. **Boruta Algorithm** - Statistical validation
11. **Permutation Testing** - Statistical significance validation

### Advanced Methods (Phase 3 - Optional)
12. **RFECV** - Wrapper method validation
13. **PCA/UMAP** - Dimensionality reduction and visualization
14. **Bootstrap Confidence Intervals** - Uncertainty quantification
15. **Nested Cross-Validation** - Unbiased performance estimation

## Expected Outputs

### 1. Feature Rankings and Scores
- Consensus feature importance ranking
- Method-specific rankings
- Statistical significance (p-values)
- Stability scores across CV folds

### 2. Feature Categories
- **Critical Features** (5-15): Essential for condition discrimination
- **Important Features** (10-25): Contribute meaningfully but not essential
- **Method-Specific Features** (5-20): Important for specific methods only
- **Redundant Features** (10-50): Removed due to multicollinearity
- **Irrelevant Features** (remaining): No significant contribution

### 3. Performance Metrics
- Classification accuracy with different feature subsets
- Comparison across multiple classifiers
- Cross-validation stability measures
- Statistical significance tests

### 4. Interpretable Insights
- Which neural activity domains are most discriminative
- Biological interpretation of selected features
- Recommendations for experimental design
- Feature importance confidence intervals

## Computational Considerations

### For Your MEA Dataset
- **Dataset Size**: ~hundreds of samples, ~50-200 features
- **Recommended Methods**: All core methods feasible
- **Computational Time**: ~10-30 minutes for full pipeline
- **Memory Requirements**: Standard (< 4GB RAM)

### Scalability Options
- **Parallel Processing**: Most methods support joblib parallelization
- **Incremental Processing**: For larger datasets, implement batch processing
- **Method Selection**: Can skip computationally expensive methods if needed

## Quality Assurance

### Statistical Rigor
- Multiple hypothesis correction (Bonferroni, FDR)
- Cross-validation for all performance estimates
- Bootstrap confidence intervals for uncertainty quantification
- Permutation tests for significance validation

### Robustness Checks
- Stability across different random seeds
- Sensitivity to hyperparameter choices
- Consistency across different train/test splits
- Agreement between different methods

This pipeline provides a comprehensive, statistically rigorous approach to feature analysis that will identify the most important neural metrics for differentiating your experimental conditions while handling the challenges of correlated, high-dimensional MEA data.

---

Perfect! Let me design a comprehensive feature analysis pipeline specifically tailored for your MEA neural data. I'll create a structured approach that builds on the methods from our reference document.

I've created a comprehensive MEA Feature Analysis Pipeline that's specifically designed for your neural data. Here's the structured approach:

## Pipeline Overview

The pipeline is organized into **7 phases** that systematically progress from data preparation to actionable insights:

### **Phase 1: Data Preparation** 
- Load and validate MEA metrics dataset
- Exploratory data analysis
- Basic quality checks

### **Phase 2: Redundancy Detection**
- Variance threshold filtering (remove constant features)
- VIF analysis (remove features with VIF > 10)
- High correlation removal (|r| > 0.9)

### **Phase 3: Core Feature Selection Methods**
- **Filter Methods**: Mutual Information, ANOVA F-test, Kruskal-Wallis
- **Embedded Methods**: LDA weights, Random Forest + Permutation Importance, LASSO
- **Wrapper Methods**: RFECV, Boruta (optional)

### **Phase 4: Advanced Methods**
- mRMR (specifically for redundant neural features)
- PCA/UMAP for dimensionality reduction

### **Phase 5: Validation & Statistical Testing**
- Cross-validation stability analysis
- Permutation testing for significance
- Bootstrap confidence intervals

### **Phase 6: Consensus Ranking**
- Combine results from all methods using rank aggregation
- Categorize features into: **Critical**, **Important**, **Method-Specific**, **Redundant**, **Irrelevant**

### **Phase 7: Results & Interpretation**
- Performance validation with different feature subsets
- Biological interpretation of selected features
- Visualization and reporting

## Implementation Strategy

I've organized the methods into **3 implementation phases**:

**Core Methods (Implement First - 6 methods)**:
1. VIF and correlation analysis
2. Mutual Information  
3. LDA with feature weights
4. Random Forest + permutation importance
5. Cross-validation stability
6. Consensus ranking

**Extended Methods (Phase 2 - 5 methods)**:
7. mRMR, ANOVA F-test, LASSO, Boruta, Permutation testing

**Advanced Methods (Optional - 4 methods)**:
12. RFECV, PCA/UMAP, Bootstrap CI, Nested CV

## Expected Outcomes

The pipeline will produce:
- **Feature Categories**: Critical (5-15), Important (10-25), Method-specific (5-20), etc.
- **Performance Metrics**: Classification accuracy with different feature subsets
- **Statistical Validation**: P-values, confidence intervals, stability scores
- **Biological Insights**: Which neural activity domains are most discriminative

The pipeline is designed to be computationally feasible for your dataset size (~hundreds of samples, ~50-200 features) and should complete in 10-30 minutes.

Would you like me to start implementing the core methods, or would you prefer to select specific methods from the pipeline to focus on first?