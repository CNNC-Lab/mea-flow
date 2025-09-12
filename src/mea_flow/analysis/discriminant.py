"""
Comprehensive Feature Analysis for MEA Neural Data.

This module provides a complete suite of feature selection and importance analysis
methods specifically designed for multi-electrode array (MEA) neural data analysis.
Implements methods from redundancy detection to ensemble consensus approaches.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

try:
    # Core ML libraries
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC, LinearSVC
    from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.feature_selection import (
        VarianceThreshold, f_classif, SelectKBest, 
        RFE, RFECV, mutual_info_classif
    )
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class FeatureSelectionMethod(Enum):
    """Available feature selection methods organized by category."""
    
    # Redundancy Detection Methods
    VIF = "variance_inflation_factor"
    CORRELATION = "correlation_analysis"
    HIERARCHICAL_CLUSTERING = "hierarchical_clustering"
    
    # Filter Methods
    MUTUAL_INFORMATION = "mutual_information"
    ANOVA_F_TEST = "anova_f_test"
    KRUSKAL_WALLIS = "kruskal_wallis"
    VARIANCE_THRESHOLD = "variance_threshold"
    FISHER_SCORE = "fisher_score"
    
    # Embedded Methods
    LDA = "linear_discriminant_analysis"
    LASSO = "lasso_regression"
    RIDGE = "ridge_regression"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    SVM_L1 = "svm_l1_penalty"
    
    # Ensemble Methods
    STABILITY_SELECTION = "stability_selection"
    CONSENSUS_RANKING = "consensus_ranking"
    
    # Validation Methods
    PERMUTATION_IMPORTANCE = "permutation_importance"
    CV_STABILITY = "cross_validation_stability"
    BOOTSTRAP_CI = "bootstrap_confidence_intervals"
    
    # Specialized Neural Methods
    MRMR = "minimum_redundancy_maximum_relevance"
    RELIEF_F = "relief_f_algorithm"


class AnalysisPhase(Enum):
    """Pipeline phases for systematic feature analysis."""
    PREPROCESSING = "preprocessing"
    REDUNDANCY_DETECTION = "redundancy_detection"
    CORE_SELECTION = "core_selection"
    VALIDATION = "validation"
    CONSENSUS = "consensus"


# Method categories for organized processing
REDUNDANCY_METHODS = [
    FeatureSelectionMethod.VIF,
    FeatureSelectionMethod.CORRELATION,
    FeatureSelectionMethod.HIERARCHICAL_CLUSTERING
]

FILTER_METHODS = [
    FeatureSelectionMethod.MUTUAL_INFORMATION,
    FeatureSelectionMethod.ANOVA_F_TEST,
    FeatureSelectionMethod.KRUSKAL_WALLIS,
    FeatureSelectionMethod.VARIANCE_THRESHOLD,
    FeatureSelectionMethod.FISHER_SCORE
]

EMBEDDED_METHODS = [
    FeatureSelectionMethod.LDA,
    FeatureSelectionMethod.LASSO,
    FeatureSelectionMethod.RIDGE,
    FeatureSelectionMethod.ELASTIC_NET,
    FeatureSelectionMethod.RANDOM_FOREST,
    FeatureSelectionMethod.XGBOOST,
    FeatureSelectionMethod.SVM_L1
]

ENSEMBLE_METHODS = [
    FeatureSelectionMethod.STABILITY_SELECTION,
    FeatureSelectionMethod.CONSENSUS_RANKING
]

VALIDATION_METHODS = [
    FeatureSelectionMethod.PERMUTATION_IMPORTANCE,
    FeatureSelectionMethod.CV_STABILITY,
    FeatureSelectionMethod.BOOTSTRAP_CI
]

SPECIALIZED_METHODS = [
    FeatureSelectionMethod.MRMR,
    FeatureSelectionMethod.RELIEF_F
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FeatureAnalysisConfig:
    """Comprehensive configuration for feature analysis pipeline."""
    
    # Data specification
    target_column: str = "condition"
    feature_columns: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = None
    
    # Pipeline control
    phases: List[AnalysisPhase] = field(default_factory=lambda: list(AnalysisPhase))
    methods: Optional[List[FeatureSelectionMethod]] = None
    
    # Preprocessing
    scale_features: bool = True
    handle_missing: str = "median"  # "median", "mean", "drop", "zero"
    
    # Redundancy detection thresholds
    vif_threshold: float = 10.0
    correlation_threshold: float = 0.9
    variance_threshold: float = 0.0
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_scoring: str = "accuracy"
    
    # Statistical significance
    alpha: float = 0.05
    multiple_testing_correction: str = "fdr_bh"  # "bonferroni", "fdr_bh", "none"
    
    # Bootstrap and stability settings
    n_bootstrap: int = 100
    stability_threshold: float = 0.8
    
    # Method-specific parameters
    random_state: int = 42
    n_jobs: int = -1
    
    # Regularization parameters
    lasso_alpha: float = 1.0
    ridge_alpha: float = 1.0
    elastic_net_alpha: float = 1.0
    elastic_net_l1_ratio: float = 0.5
    
    # Tree-based parameters
    rf_n_estimators: int = 100
    xgb_n_estimators: int = 100
    
    # SVM parameters
    svm_C: float = 1.0
    
    # mRMR parameters
    mrmr_k: int = 10
    
    # Relief-F parameters
    relief_n_neighbors: int = 10
    relief_n_iterations: int = 100


@dataclass
class FeatureImportanceResult:
    """Results from a single feature importance method."""
    method: FeatureSelectionMethod
    scores: pd.DataFrame  # columns: ['feature', 'importance', 'rank', 'p_value', 'selected']
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    
@dataclass
class RedundancyAnalysisResult:
    """Results from redundancy detection analysis."""
    method: FeatureSelectionMethod
    redundant_features: List[str]
    feature_groups: Optional[Dict[str, List[str]]] = None
    scores: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Results from consensus feature ranking."""
    consensus_ranking: pd.DataFrame  # columns: ['feature', 'consensus_score', 'rank', 'method_agreement']
    method_rankings: Dict[str, pd.DataFrame]
    stability_scores: pd.DataFrame
    selected_features: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureAnalysisResults:
    """Comprehensive results from feature analysis pipeline."""
    
    # Input data summary
    n_samples: int
    n_features: int
    n_classes: int
    class_distribution: Dict[str, int]
    
    # Phase results
    redundancy_results: Dict[str, RedundancyAnalysisResult] = field(default_factory=dict)
    importance_results: Dict[str, FeatureImportanceResult] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Final consensus
    consensus_result: Optional[ConsensusResult] = None
    
    # Feature categorization
    critical_features: List[str] = field(default_factory=list)
    redundant_features: List[str] = field(default_factory=list)
    method_specific_features: Dict[str, List[str]] = field(default_factory=dict)
    irrelevant_features: List[str] = field(default_factory=list)
    
    # Execution metadata
    config: Optional[FeatureAnalysisConfig] = None
    execution_time: float = 0.0
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# REDUNDANCY DETECTION METHODS
# =============================================================================

def variance_inflation_factor_analysis(
    X: pd.DataFrame,
    threshold: float = 10.0
) -> RedundancyAnalysisResult:
    """
    Analyze multicollinearity using Variance Inflation Factor (VIF).
    
    VIF measures how much the variance of a coefficient increases due to 
    collinearity. VIF > 10 typically indicates problematic multicollinearity.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    threshold : float, default=10.0
        VIF threshold for identifying redundant features
        
    Returns
    -------
    RedundancyAnalysisResult
        Analysis results with redundant features identified
    """
    if not STATSMODELS_AVAILABLE:
        return RedundancyAnalysisResult(
            method=FeatureSelectionMethod.VIF,
            redundant_features=[],
            metadata={"error": "statsmodels not available"}
        )
    
    # Clean the data before VIF analysis
    X_clean = X.copy()
    
    # Remove columns with zero variance (constant features)
    constant_cols = X_clean.columns[X_clean.var() == 0].tolist()
    if constant_cols:
        X_clean = X_clean.drop(columns=constant_cols)
    
    # Check for remaining issues
    if X_clean.empty or X_clean.shape[1] < 2:
        return RedundancyAnalysisResult(
            method=FeatureSelectionMethod.VIF,
            redundant_features=constant_cols,
            metadata={
                "error": "Insufficient features for VIF analysis",
                "constant_features_removed": constant_cols
            }
        )
    
    # Ensure no infinite or NaN values
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    redundant_features = []
    vif_scores = []
    
    for i, feature in enumerate(X_clean.columns):
        try:
            # Check if feature has sufficient variation
            if X_clean[feature].std() == 0:
                vif_scores.append(np.inf)
                redundant_features.append(feature)
                continue
                
            vif = variance_inflation_factor(X_clean.values, i)
            
            # Handle edge cases
            if np.isnan(vif) or np.isinf(vif):
                vif = np.inf
                
            vif_scores.append(vif)
            
            if vif > threshold:
                redundant_features.append(feature)
                
        except Exception as e:
            # Silently handle errors and mark as problematic
            vif_scores.append(np.inf)
            redundant_features.append(feature)
    
    # Add back constant columns to redundant features
    redundant_features.extend(constant_cols)
    
    # Create scores DataFrame for all original features
    all_vif_scores = []
    for feature in X.columns:
        if feature in constant_cols:
            all_vif_scores.append(np.inf)
        elif feature in X_clean.columns:
            idx = X_clean.columns.get_loc(feature)
            all_vif_scores.append(vif_scores[idx])
        else:
            all_vif_scores.append(np.inf)
    
    scores_df = pd.DataFrame({
        'feature': X.columns,
        'vif_score': all_vif_scores,
        'redundant': [f in redundant_features for f in X.columns]
    })
    
    return RedundancyAnalysisResult(
        method=FeatureSelectionMethod.VIF,
        redundant_features=redundant_features,
        scores=scores_df,
        metadata={
            "threshold": threshold,
            "n_redundant": len(redundant_features),
            "max_vif": np.nanmax([s for s in all_vif_scores if not np.isinf(s)]) if any(not np.isinf(s) for s in all_vif_scores) else 0,
            "constant_features_removed": constant_cols
        }
    )


def correlation_analysis(
    X: pd.DataFrame,
    threshold: float = 0.9
) -> RedundancyAnalysisResult:
    """
    Identify highly correlated feature pairs.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    threshold : float, default=0.9
        Correlation threshold above which features are considered redundant
        
    Returns
    -------
    RedundancyAnalysisResult
        Results with redundant features identified
    """
    import time
    start_time = time.time()
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Find highly correlated pairs
    redundant_features = set()
    correlation_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            correlation = corr_matrix.iloc[i, j]
            
            if correlation > threshold:
                correlation_pairs.append({
                    'feature_1': feature_i,
                    'feature_2': feature_j,
                    'correlation': correlation
                })
                # Remove feature with lower variance (less informative)
                if X[feature_i].var() < X[feature_j].var():
                    redundant_features.add(feature_i)
                else:
                    redundant_features.add(feature_j)
    
    correlation_df = pd.DataFrame(correlation_pairs)
    
    return RedundancyAnalysisResult(
        method=FeatureSelectionMethod.CORRELATION,
        redundant_features=list(redundant_features),
        scores=correlation_df,
        metadata={
            'threshold': threshold,
            'n_pairs': len(correlation_pairs),
            'n_redundant': len(redundant_features),
            'execution_time': time.time() - start_time
        }
    )


def hierarchical_clustering_analysis(
    X: pd.DataFrame,
    distance_threshold: float = 0.1,
    linkage_method: str = 'average'
) -> RedundancyAnalysisResult:
    """
    Group features using hierarchical clustering based on correlation distance.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    distance_threshold : float, default=0.1
        Distance threshold for forming clusters (1 - correlation)
    linkage_method : str, default='average'
        Linkage method for hierarchical clustering
        
    Returns
    -------
    RedundancyAnalysisResult
        Results with feature groups identified
    """
    import time
    start_time = time.time()
    
    # Calculate correlation matrix and convert to distance
    corr_matrix = X.corr().abs()
    distance_matrix = 1 - corr_matrix
    
    # Perform hierarchical clustering
    condensed_distances = pdist(distance_matrix, metric='precomputed')
    linkage_matrix = linkage(condensed_distances, method=linkage_method)
    
    # Form clusters
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
    # Group features by cluster
    feature_groups = {}
    redundant_features = []
    
    for feature, cluster_id in zip(X.columns, cluster_labels):
        if cluster_id not in feature_groups:
            feature_groups[cluster_id] = []
        feature_groups[cluster_id].append(feature)
    
    # Identify redundant features (keep one representative per cluster)
    for cluster_id, features in feature_groups.items():
        if len(features) > 1:
            # Keep feature with highest variance
            variances = X[features].var()
            representative = variances.idxmax()
            redundant_in_cluster = [f for f in features if f != representative]
            redundant_features.extend(redundant_in_cluster)
    
    # Convert cluster IDs to strings for JSON serialization
    feature_groups_str = {f"cluster_{k}": v for k, v in feature_groups.items()}
    
    return RedundancyAnalysisResult(
        method=FeatureSelectionMethod.HIERARCHICAL_CLUSTERING,
        redundant_features=redundant_features,
        feature_groups=feature_groups_str,
        metadata={
            'distance_threshold': distance_threshold,
            'linkage_method': linkage_method,
            'n_clusters': len(feature_groups),
            'n_redundant': len(redundant_features),
            'execution_time': time.time() - start_time
        }
    )


# =============================================================================
# FILTER METHODS
# =============================================================================

def mutual_information_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    discrete_features: bool = False,
    random_state: int = 42
) -> FeatureImportanceResult:
    """
    Calculate mutual information between features and target.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    discrete_features : bool, default=False
        Whether features are discrete
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    FeatureImportanceResult
        Results with mutual information scores
    """
    import time
    start_time = time.time()
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(
        X, y, 
        discrete_features=discrete_features,
        random_state=random_state
    )
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': X.columns,
        'importance': mi_scores,
        'rank': stats.rankdata(-mi_scores, method='ordinal'),
        'p_value': np.nan,  # MI doesn't provide p-values directly
        'selected': mi_scores > np.median(mi_scores)
    }).sort_values('importance', ascending=False)
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.MUTUAL_INFORMATION,
        scores=results_df,
        metadata={
            'discrete_features': discrete_features,
            'median_score': np.median(mi_scores),
            'execution_time': time.time() - start_time
        },
        execution_time=time.time() - start_time
    )


def anova_f_test_analysis(
    X: pd.DataFrame,
    y: pd.Series
) -> FeatureImportanceResult:
    """
    Perform ANOVA F-test for feature selection.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
        
    Returns
    -------
    FeatureImportanceResult
        Results with F-statistics and p-values
    """
    import time
    start_time = time.time()
    
    # Calculate F-statistics and p-values
    f_stats, p_values = f_classif(X, y)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': X.columns,
        'importance': f_stats,
        'rank': stats.rankdata(-f_stats, method='ordinal'),
        'p_value': p_values,
        'selected': p_values < 0.05
    }).sort_values('importance', ascending=False)
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.ANOVA_F_TEST,
        scores=results_df,
        metadata={
            'n_significant': (p_values < 0.05).sum(),
            'execution_time': time.time() - start_time
        },
        execution_time=time.time() - start_time
    )


def kruskal_wallis_analysis(
    X: pd.DataFrame,
    y: pd.Series
) -> FeatureImportanceResult:
    """
    Perform Kruskal-Wallis test for feature selection.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
        
    Returns
    -------
    FeatureImportanceResult
        Results with H-statistics and p-values
    """
    import time
    start_time = time.time()
    
    h_stats = []
    p_values = []
    
    # Perform Kruskal-Wallis test for each feature
    for feature in X.columns:
        groups = [X[feature][y == class_label].values for class_label in np.unique(y)]
        # Remove empty groups
        groups = [group for group in groups if len(group) > 0]
        
        if len(groups) >= 2:
            h_stat, p_val = stats.kruskal(*groups)
            h_stats.append(h_stat)
            p_values.append(p_val)
        else:
            h_stats.append(0.0)
            p_values.append(1.0)
    
    h_stats = np.array(h_stats)
    p_values = np.array(p_values)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': X.columns,
        'importance': h_stats,
        'rank': stats.rankdata(-h_stats, method='ordinal'),
        'p_value': p_values,
        'selected': p_values < 0.05
    }).sort_values('importance', ascending=False)
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.KRUSKAL_WALLIS,
        scores=results_df,
        metadata={
            'n_significant': (p_values < 0.05).sum(),
            'execution_time': time.time() - start_time
        },
        execution_time=time.time() - start_time
    )


def variance_threshold_analysis(
    X: pd.DataFrame,
    threshold: float = 0.0
) -> FeatureImportanceResult:
    """
    Remove features with low variance.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    threshold : float, default=0.0
        Variance threshold
        
    Returns
    -------
    FeatureImportanceResult
        Results with variance scores
    """
    import time
    start_time = time.time()
    
    # Calculate variances
    variances = X.var()
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': X.columns,
        'importance': variances,
        'rank': stats.rankdata(-variances, method='ordinal'),
        'p_value': np.nan,
        'selected': variances > threshold
    }).sort_values('importance', ascending=False)
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.VARIANCE_THRESHOLD,
        scores=results_df,
        metadata={
            'threshold': threshold,
            'n_selected': (variances > threshold).sum(),
            'execution_time': time.time() - start_time
        },
        execution_time=time.time() - start_time
    )


def fisher_score_analysis(
    X: pd.DataFrame,
    y: pd.Series
) -> FeatureImportanceResult:
    """
    Calculate Fisher's discriminant score for each feature.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
        
    Returns
    -------
    FeatureImportanceResult
        Results with Fisher scores
    """
    import time
    start_time = time.time()
    
    fisher_scores = []
    
    for feature in X.columns:
        feature_data = X[feature]
        
        # Calculate between-class and within-class variances
        overall_mean = feature_data.mean()
        between_class_var = 0
        within_class_var = 0
        
        for class_label in np.unique(y):
            class_data = feature_data[y == class_label]
            class_mean = class_data.mean()
            class_size = len(class_data)
            
            # Between-class variance
            between_class_var += class_size * (class_mean - overall_mean) ** 2
            
            # Within-class variance
            within_class_var += ((class_data - class_mean) ** 2).sum()
        
        # Fisher score
        if within_class_var > 0:
            fisher_score = between_class_var / within_class_var
        else:
            fisher_score = 0.0
            
        fisher_scores.append(fisher_score)
    
    fisher_scores = np.array(fisher_scores)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': X.columns,
        'importance': fisher_scores,
        'rank': stats.rankdata(-fisher_scores, method='ordinal'),
        'p_value': np.nan,
        'selected': fisher_scores > np.median(fisher_scores)
    }).sort_values('importance', ascending=False)
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.FISHER_SCORE,
        scores=results_df,
        metadata={
            'median_score': np.median(fisher_scores),
            'execution_time': time.time() - start_time
        },
        execution_time=time.time() - start_time
    )


# =============================================================================
# EMBEDDED METHODS
# =============================================================================

def lda_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    solver: str = 'svd'
) -> FeatureImportanceResult:
    """
    Linear Discriminant Analysis for feature importance.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    solver : str, default='svd'
        Solver to use for LDA
        
    Returns
    -------
    FeatureImportanceResult
        Results with LDA coefficients as importance
    """
    import time
    start_time = time.time()
    
    # Fit LDA
    lda = LinearDiscriminantAnalysis(solver=solver)
    lda.fit(X, y)
    
    # Extract feature importance from coefficients
    if hasattr(lda, 'coef_'):
        if lda.coef_.ndim > 1:
            # Multi-class: use mean absolute coefficient
            importance = np.mean(np.abs(lda.coef_), axis=0)
        else:
            importance = np.abs(lda.coef_[0])
    else:
        importance = np.zeros(X.shape[1])
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance,
        'rank': stats.rankdata(-importance, method='ordinal'),
        'p_value': np.nan,
        'selected': importance > np.median(importance)
    }).sort_values('importance', ascending=False)
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.LDA,
        scores=results_df,
        metadata={
            'solver': solver,
            'n_components': lda.n_components,
            'execution_time': time.time() - start_time
        },
        execution_time=time.time() - start_time
    )


def lasso_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 1.0,
    random_state: int = 42
) -> FeatureImportanceResult:
    """
    LASSO regression for feature selection.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    alpha : float, default=1.0
        Regularization strength
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    FeatureImportanceResult
        Results with LASSO coefficients as importance
    """
    import time
    start_time = time.time()
    
    # Encode target for regression
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Fit LASSO
    lasso = Lasso(alpha=alpha, random_state=random_state)
    lasso.fit(X, y_encoded)
    
    # Extract feature importance from coefficients
    importance = np.abs(lasso.coef_)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance,
        'rank': stats.rankdata(-importance, method='ordinal'),
        'p_value': np.nan,
        'selected': importance > 0  # LASSO sets coefficients to exactly 0
    }).sort_values('importance', ascending=False)
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.LASSO,
        scores=results_df,
        metadata={
            'alpha': alpha,
            'n_selected': (importance > 0).sum(),
            'execution_time': time.time() - start_time
        },
        execution_time=time.time() - start_time
    )


def random_forest_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
    use_permutation: bool = True
) -> FeatureImportanceResult:
    """
    Random Forest feature importance analysis.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    n_estimators : int, default=100
        Number of trees
    random_state : int, default=42
        Random state for reproducibility
    use_permutation : bool, default=True
        Whether to use permutation importance (more reliable)
        
    Returns
    -------
    FeatureImportanceResult
        Results with Random Forest importance scores
    """
    import time
    start_time = time.time()
    
    # Fit Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    rf.fit(X, y)
    
    if use_permutation:
        # Use permutation importance (more reliable)
        perm_importance = permutation_importance(
            rf, X, y, random_state=random_state
        )
        importance = perm_importance.importances_mean
        importance_std = perm_importance.importances_std
    else:
        # Use built-in feature importance
        importance = rf.feature_importances_
        importance_std = np.zeros_like(importance)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance,
        'rank': stats.rankdata(-importance, method='ordinal'),
        'p_value': np.nan,
        'selected': importance > np.median(importance)
    }).sort_values('importance', ascending=False)
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.RANDOM_FOREST,
        scores=results_df,
        metadata={
            'n_estimators': n_estimators,
            'random_state': random_state,
            'oob_score': rf.oob_score_ if hasattr(rf, 'oob_score_') else None
        },
        execution_time=time.time() - start_time
    )


def ridge_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 1.0,
    random_state: int = 42
) -> FeatureImportanceResult:
    """
    Perform Ridge regression feature importance analysis.
    """
    import time
    start_time = time.time()
    
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import LabelEncoder
    
    # Handle categorical targets
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y
    
    # Fit Ridge regression
    ridge = Ridge(alpha=alpha, random_state=random_state)
    ridge.fit(X, y_encoded)
    
    # Get feature importance (absolute coefficients)
    importance_scores = np.abs(ridge.coef_)
    
    # Create results DataFrame
    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_scores,
        'rank': np.argsort(-importance_scores) + 1,
        'p_value': np.nan,
        'selected': importance_scores > np.percentile(importance_scores, 75)
    })
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.RIDGE,
        scores=scores_df,
        metadata={'alpha': alpha, 'random_state': random_state},
        execution_time=time.time() - start_time
    )


def elastic_net_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    random_state: int = 42
) -> FeatureImportanceResult:
    """
    Perform Elastic Net feature importance analysis.
    """
    import time
    start_time = time.time()
    
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import LabelEncoder
    
    # Handle categorical targets
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y
    
    # Fit Elastic Net
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    elastic_net.fit(X, y_encoded)
    
    # Get feature importance (absolute coefficients)
    importance_scores = np.abs(elastic_net.coef_)
    
    # Create results DataFrame
    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_scores,
        'rank': np.argsort(-importance_scores) + 1,
        'p_value': np.nan,
        'selected': importance_scores > 0
    })
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.ELASTIC_NET,
        scores=scores_df,
        metadata={'alpha': alpha, 'l1_ratio': l1_ratio},
        execution_time=time.time() - start_time
    )


def xgboost_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42
) -> FeatureImportanceResult:
    """
    Perform XGBoost feature importance analysis.
    """
    import time
    start_time = time.time()
    
    try:
        import xgboost as xgb
    except ImportError:
        warnings.warn("XGBoost not available, falling back to Random Forest")
        return random_forest_analysis(X, y, n_estimators, random_state)
    
    from sklearn.preprocessing import LabelEncoder
    
    # Handle categorical targets
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        y_encoded = y
        xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=random_state)
    
    xgb_model.fit(X, y_encoded)
    
    # Get feature importance
    importance_scores = xgb_model.feature_importances_
    
    # Create results DataFrame
    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_scores,
        'rank': np.argsort(-importance_scores) + 1,
        'p_value': np.nan,
        'selected': importance_scores > np.percentile(importance_scores, 75)
    })
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.XGBOOST,
        scores=scores_df,
        metadata={'n_estimators': n_estimators},
        execution_time=time.time() - start_time
    )


def svm_l1_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    C: float = 1.0,
    random_state: int = 42
) -> FeatureImportanceResult:
    """
    Perform SVM with L1 penalty feature importance analysis.
    """
    import time
    start_time = time.time()
    
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import LabelEncoder
    
    # Handle categorical targets
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y
    
    # Fit Linear SVM with L1 penalty
    svm = LinearSVC(C=C, penalty='l1', dual=False, random_state=random_state, max_iter=10000)
    svm.fit(X, y_encoded)
    
    # Get feature importance (absolute coefficients)
    if len(svm.coef_.shape) > 1 and svm.coef_.shape[0] > 1:
        importance_scores = np.mean(np.abs(svm.coef_), axis=0)
    else:
        importance_scores = np.abs(svm.coef_[0])
    
    # Create results DataFrame
    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_scores,
        'rank': np.argsort(-importance_scores) + 1,
        'p_value': np.nan,
        'selected': importance_scores > 0
    })
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.SVM_L1,
        scores=scores_df,
        metadata={'C': C},
        execution_time=time.time() - start_time
    )


def mrmr_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 10
) -> FeatureImportanceResult:
    """
    Perform simplified mRMR feature selection.
    """
    import time
    start_time = time.time()
    
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder
    
    # Handle categorical targets
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        mi_func = mutual_info_classif
    else:
        y_encoded = y
        mi_func = mutual_info_regression
    
    # Calculate mutual information with target
    relevance_scores = mi_func(X, y_encoded, random_state=42)
    
    # Simplified mRMR selection
    selected_features = []
    remaining_features = list(range(len(X.columns)))
    
    # Select first feature with highest relevance
    if len(remaining_features) > 0:
        first_idx = np.argmax(relevance_scores)
        selected_features.append(first_idx)
        remaining_features.remove(first_idx)
    
    # Iteratively select features
    for _ in range(min(k-1, len(remaining_features))):
        best_score = -np.inf
        best_feature = None
        
        for feature_idx in remaining_features:
            relevance = relevance_scores[feature_idx]
            
            if selected_features:
                redundancy = np.mean([
                    np.abs(np.corrcoef(X.iloc[:, feature_idx], X.iloc[:, sel_idx])[0, 1])
                    for sel_idx in selected_features
                ])
            else:
                redundancy = 0
            
            mrmr_score = relevance - redundancy
            
            if mrmr_score > best_score:
                best_score = mrmr_score
                best_feature = feature_idx
        
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
    
    # Create importance scores
    importance_scores = np.zeros(len(X.columns))
    for i, feature_idx in enumerate(selected_features):
        importance_scores[feature_idx] = (len(selected_features) - i) / len(selected_features)
    
    # Create results DataFrame
    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_scores,
        'rank': np.argsort(-importance_scores) + 1,
        'p_value': np.nan,
        'selected': importance_scores > 0
    })
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.MRMR,
        scores=scores_df,
        metadata={'k': k, 'n_selected': len(selected_features)},
        execution_time=time.time() - start_time
    )


def relief_f_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    n_neighbors: int = 10,
    n_iterations: int = 100
) -> FeatureImportanceResult:
    """
    Perform simplified Relief-F feature selection.
    """
    import time
    start_time = time.time()
    
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    # Handle categorical targets
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize feature weights
    feature_weights = np.zeros(X.shape[1])
    
    # Relief-F algorithm
    for _ in range(n_iterations):
        idx = np.random.randint(0, len(X))
        instance = X_scaled[idx]
        instance_class = y_encoded[idx]
        
        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors+1)
        nn.fit(X_scaled)
        distances, indices = nn.kneighbors([instance])
        neighbor_indices = indices[0][1:]
        
        # Update feature weights
        for feature_idx in range(X.shape[1]):
            hit_diff = 0
            miss_diff = 0
            
            for neighbor_idx in neighbor_indices:
                neighbor_class = y_encoded[neighbor_idx]
                feature_diff = abs(instance[feature_idx] - X_scaled[neighbor_idx, feature_idx])
                
                if neighbor_class == instance_class:
                    hit_diff += feature_diff
                else:
                    miss_diff += feature_diff
            
            feature_weights[feature_idx] += (miss_diff - hit_diff) / (n_neighbors * n_iterations)
    
    # Normalize weights to [0, 1]
    if feature_weights.max() > feature_weights.min():
        feature_weights = (feature_weights - feature_weights.min()) / (feature_weights.max() - feature_weights.min())
    
    # Create results DataFrame
    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_weights,
        'rank': np.argsort(-feature_weights) + 1,
        'p_value': np.nan,
        'selected': feature_weights > np.percentile(feature_weights, 75)
    })
    
    return FeatureImportanceResult(
        method=FeatureSelectionMethod.RELIEF_F,
        scores=scores_df,
        metadata={'n_neighbors': n_neighbors, 'n_iterations': n_iterations},
        execution_time=time.time() - start_time
    )


# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def comprehensive_feature_analysis(
    data: pd.DataFrame,
    config: FeatureAnalysisConfig
) -> FeatureAnalysisResults:
    """
    Run comprehensive feature analysis pipeline.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with features and target
    config : FeatureAnalysisConfig
        Configuration for the analysis
        
    Returns
    -------
    FeatureAnalysisResults
        Complete results from all analysis phases
    """
    import time
    start_time = time.time()
    
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for feature analysis")
    
    # Validate inputs
    if config.target_column not in data.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in data")
    
    # Prepare data
    X, y, feature_names = _prepare_feature_data(data, config)
    
    # Initialize results
    results = FeatureAnalysisResults(
        n_samples=len(X),
        n_features=len(feature_names),
        n_classes=len(np.unique(y)),
        class_distribution=dict(zip(*np.unique(y, return_counts=True))),
        config=config
    )
    
    warnings_list = []
    
    # Phase 1: Redundancy Detection
    if AnalysisPhase.REDUNDANCY_DETECTION in config.phases:
        all_redundant = set()
        
        # VIF Analysis (if statsmodels is available)
        if STATSMODELS_AVAILABLE:
            try:
                vif_result = variance_inflation_factor_analysis(X, config.vif_threshold)
                results.redundancy_results['vif'] = vif_result
                all_redundant.update(vif_result.redundant_features)
            except Exception as e:
                warnings_list.append(f"VIF analysis failed: {e}")
        else:
            warnings_list.append("VIF analysis skipped: statsmodels not available")
        
        # Correlation Analysis (always available)
        try:
            corr_result = correlation_analysis(X, config.correlation_threshold)
            results.redundancy_results['correlation'] = corr_result
            all_redundant.update(corr_result.redundant_features)
        except Exception as e:
            warnings_list.append(f"Correlation analysis failed: {e}")
        
        # Remove redundant features for subsequent analysis
        X_filtered = X.drop(columns=list(all_redundant), errors='ignore')
        results.redundant_features = list(all_redundant)
    else:
        X_filtered = X
    
    # Phase 2: Core Feature Selection
    if AnalysisPhase.CORE_SELECTION in config.phases and config.methods:
        for method in config.methods:
            try:
                if method == FeatureSelectionMethod.MUTUAL_INFORMATION:
                    result = mutual_information_analysis(X_filtered, y)
                    results.importance_results['mutual_information'] = result
                    
                elif method == FeatureSelectionMethod.ANOVA_F_TEST:
                    result = anova_f_test_analysis(X_filtered, y)
                    results.importance_results['anova_f_test'] = result
                    
                elif method == FeatureSelectionMethod.KRUSKAL_WALLIS:
                    result = kruskal_wallis_analysis(X_filtered, y)
                    results.importance_results['kruskal_wallis'] = result
                    
                elif method == FeatureSelectionMethod.VARIANCE_THRESHOLD:
                    result = variance_threshold_analysis(X_filtered, config.variance_threshold)
                    results.importance_results['variance_threshold'] = result
                    
                elif method == FeatureSelectionMethod.FISHER_SCORE:
                    result = fisher_score_analysis(X_filtered, y)
                    results.importance_results['fisher_score'] = result
                    
                elif method == FeatureSelectionMethod.RANDOM_FOREST:
                    result = random_forest_analysis(X_filtered, y, config.rf_n_estimators)
                    results.importance_results['random_forest'] = result
                    
                elif method == FeatureSelectionMethod.LDA:
                    result = lda_analysis(X_filtered, y)
                    results.importance_results['lda'] = result
                    
                elif method == FeatureSelectionMethod.LASSO:
                    result = lasso_analysis(X_filtered, y, config.lasso_alpha)
                    results.importance_results['lasso_regression'] = result
                    
                elif method == FeatureSelectionMethod.RIDGE:
                    result = ridge_analysis(X_filtered, y, config.ridge_alpha)
                    results.importance_results['ridge_regression'] = result
                    
                elif method == FeatureSelectionMethod.ELASTIC_NET:
                    result = elastic_net_analysis(X_filtered, y, config.elastic_net_alpha, config.elastic_net_l1_ratio)
                    results.importance_results['elastic_net'] = result
                    
                elif method == FeatureSelectionMethod.XGBOOST:
                    result = xgboost_analysis(X_filtered, y, config.xgb_n_estimators)
                    results.importance_results['xgboost'] = result
                    
                elif method == FeatureSelectionMethod.SVM_L1:
                    result = svm_l1_analysis(X_filtered, y, config.svm_C)
                    results.importance_results['svm_l1_penalty'] = result
                    
                elif method == FeatureSelectionMethod.MRMR:
                    result = mrmr_analysis(X_filtered, y, config.mrmr_k)
                    results.importance_results['minimum_redundancy_maximum_relevance'] = result
                    
                elif method == FeatureSelectionMethod.RELIEF_F:
                    result = relief_f_analysis(X_filtered, y, config.relief_n_neighbors, config.relief_n_iterations)
                    results.importance_results['relief_f_algorithm'] = result
                    
            except Exception as e:
                warnings_list.append(f"Method {method.value} failed: {e}")
    
    # Phase 3: Consensus Ranking
    if AnalysisPhase.CONSENSUS in config.phases and results.importance_results:
        try:
            consensus_result = _compute_consensus_ranking(results.importance_results)
            results.consensus_result = consensus_result
            
            # Categorize features
            results.critical_features = consensus_result.selected_features[:10]  # Top 10
            
        except Exception as e:
            warnings_list.append(f"Consensus ranking failed: {e}")
    
    results.warnings = warnings_list
    results.execution_time = time.time() - start_time
    
    return results


def _prepare_feature_data(
    data: pd.DataFrame,
    config: FeatureAnalysisConfig
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare feature matrix and target vector."""
    
    # Get target
    y = data[config.target_column]
    
    # Determine feature columns
    if config.feature_columns is not None:
        feature_cols = config.feature_columns
    else:
        exclude_cols = [config.target_column]
        if config.exclude_columns:
            exclude_cols.extend(config.exclude_columns)
        
        feature_cols = [col for col in data.columns 
                       if col not in exclude_cols and 
                       pd.api.types.is_numeric_dtype(data[col])]
    
    # Extract features
    X = data[feature_cols].copy()
    
    # Handle missing values and infinite values
    # Replace infinite values with NaN first
    X = X.replace([np.inf, -np.inf], np.nan)
    
    if config.handle_missing == "median":
        X = X.fillna(X.median())
    elif config.handle_missing == "mean":
        X = X.fillna(X.mean())
    elif config.handle_missing == "zero":
        X = X.fillna(0)
    elif config.handle_missing == "drop":
        X = X.dropna()
        y = y.loc[X.index]
    
    # Final check: replace any remaining NaN/inf with 0
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Scale features if requested
    if config.scale_features:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        X = X_scaled
    
    return X, y, feature_cols


def _compute_consensus_ranking(
    importance_results: Dict[str, FeatureImportanceResult]
) -> ConsensusResult:
    """Compute consensus ranking from multiple methods."""
    
    # Collect all rankings
    method_rankings = {}
    all_features = set()
    
    for method_name, result in importance_results.items():
        rankings = result.scores[['feature', 'rank']].set_index('feature')['rank']
        method_rankings[method_name] = rankings
        all_features.update(rankings.index)
    
    all_features = list(all_features)
    
    # Compute consensus scores using Borda count
    consensus_scores = {}
    for feature in all_features:
        scores = []
        for method_name, rankings in method_rankings.items():
            if feature in rankings.index:
                # Convert rank to score (lower rank = higher score)
                max_rank = len(rankings)
                score = max_rank - rankings[feature] + 1
                scores.append(score)
        
        consensus_scores[feature] = np.mean(scores) if scores else 0
    
    # Create consensus DataFrame
    consensus_df = pd.DataFrame([
        {
            'feature': feature,
            'consensus_score': score,
            'rank': rank,
            'method_agreement': len([r for r in method_rankings.values() if feature in r.index])
        }
        for rank, (feature, score) in enumerate(
            sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True), 1
        )
    ])
    
    # Select top features
    selected_features = consensus_df.head(20)['feature'].tolist()
    
    return ConsensusResult(
        consensus_ranking=consensus_df,
        method_rankings={k: v.to_frame('rank') for k, v in method_rankings.items()},
        stability_scores=pd.DataFrame(),  # Placeholder
        selected_features=selected_features
    )
