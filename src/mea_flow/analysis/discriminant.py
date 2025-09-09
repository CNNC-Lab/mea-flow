"""
Discriminant analysis for MEA data feature importance and classification.

This module provides tools for identifying which features best discriminate
between experimental conditions using various machine learning approaches.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DiscriminantMethod(Enum):
    """Available methods for discriminant analysis."""
    RANDOM_FOREST = "random_forest"
    LINEAR_DISCRIMINANT = "linear_discriminant"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"


# Method string mappings for user convenience
METHOD_MAPPINGS = {
    # Random Forest
    "RF": DiscriminantMethod.RANDOM_FOREST,
    "RANDOM_FOREST": DiscriminantMethod.RANDOM_FOREST,
    "random_forest": DiscriminantMethod.RANDOM_FOREST,
    
    # Linear Discriminant Analysis
    "LDA": DiscriminantMethod.LINEAR_DISCRIMINANT,
    "LINEAR_DISCRIMINANT": DiscriminantMethod.LINEAR_DISCRIMINANT,
    "linear_discriminant": DiscriminantMethod.LINEAR_DISCRIMINANT,
    
    # Support Vector Machine
    "SVM": DiscriminantMethod.SVM,
    "svm": DiscriminantMethod.SVM,
    
    # Logistic Regression
    "LR": DiscriminantMethod.LOGISTIC_REGRESSION,
    "LOGISTIC": DiscriminantMethod.LOGISTIC_REGRESSION,
    "LOGISTIC_REGRESSION": DiscriminantMethod.LOGISTIC_REGRESSION,
    "logistic_regression": DiscriminantMethod.LOGISTIC_REGRESSION,
}


def get_available_methods() -> List[str]:
    """
    Get list of available discriminant analysis methods.
    
    Returns
    -------
    List[str]
        List of method strings that can be used
    """
    return [
        "RF", "RANDOM_FOREST",           # Random Forest
        "LDA", "LINEAR_DISCRIMINANT",    # Linear Discriminant Analysis  
        "SVM",                           # Support Vector Machine
        "LR", "LOGISTIC", "LOGISTIC_REGRESSION"  # Logistic Regression
    ]


def _parse_method(method: Union[str, DiscriminantMethod]) -> DiscriminantMethod:
    """
    Parse method input to DiscriminantMethod enum.
    
    Parameters
    ----------
    method : str or DiscriminantMethod
        Method specification
        
    Returns
    -------
    DiscriminantMethod
        Parsed method enum
        
    Raises
    ------
    ValueError
        If method string is not recognized
    """
    if isinstance(method, DiscriminantMethod):
        return method
    
    if isinstance(method, str):
        if method in METHOD_MAPPINGS:
            return METHOD_MAPPINGS[method]
        else:
            available = get_available_methods()
            raise ValueError(f"Unknown method '{method}'. Available methods: {available}")
    
    raise TypeError(f"Method must be string or DiscriminantMethod, got {type(method)}")


@dataclass
class DiscriminantConfig:
    """Configuration for discriminant analysis."""
    method: Union[str, DiscriminantMethod] = "RF"
    target_column: str = "condition"
    feature_columns: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = None
    scale_features: bool = True
    cross_validation: bool = True
    cv_folds: int = 5
    random_state: int = 42
    
    # Method-specific parameters
    rf_n_estimators: int = 100
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    logistic_max_iter: int = 1000


@dataclass
class DiscriminantResults:
    """Results from discriminant analysis."""
    feature_importance: pd.DataFrame
    model_performance: Dict[str, float]
    classification_report: Optional[str] = None
    confusion_matrix: Optional[np.ndarray] = None
    model: Optional[Any] = None


def identify_discriminative_features(
    data: pd.DataFrame,
    config: DiscriminantConfig
) -> DiscriminantResults:
    """
    Identify features that best discriminate between conditions.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing features and target condition column
    config : DiscriminantConfig
        Configuration parameters for the analysis
        
    Returns
    -------
    DiscriminantResults
        Results containing feature importance scores and model performance
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for discriminant analysis")
    
    # Parse method string to enum
    method_enum = _parse_method(config.method)
    
    # Validate inputs
    if config.target_column not in data.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in data")
    
    # Prepare feature matrix and target vector
    X, y, feature_names = _prepare_data(data, config)
    
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 different conditions for discriminant analysis")
    
    # Scale features if requested
    scaler = None
    if config.scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Initialize model based on method
    model = _initialize_model(method_enum, config)
    
    # Fit model
    model.fit(X, y)
    
    # Extract feature importance
    importance_scores = _extract_feature_importance(model, method_enum, X, y)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    # Evaluate model performance
    performance = {}
    classification_rep = None
    conf_matrix = None
    
    if config.cross_validation:
        # Determine appropriate number of CV folds based on dataset size
        unique_labels, label_counts = np.unique(y, return_counts=True)
        min_class_size = min(label_counts)
        
        # Use at most min_class_size folds, but at least 2
        adaptive_cv_folds = min(config.cv_folds, min_class_size, len(y))
        adaptive_cv_folds = max(2, adaptive_cv_folds)
        
        if adaptive_cv_folds < config.cv_folds:
            warnings.warn(f"Reducing CV folds from {config.cv_folds} to {adaptive_cv_folds} due to small dataset size")
        
        try:
            cv_scores = cross_val_score(model, X, y, cv=adaptive_cv_folds)
            performance['cv_mean_accuracy'] = cv_scores.mean()
            performance['cv_std_accuracy'] = cv_scores.std()
            performance['cv_folds_used'] = adaptive_cv_folds
        except Exception as e:
            warnings.warn(f"Cross-validation failed: {e}. Skipping CV evaluation.")
            performance['cv_mean_accuracy'] = np.nan
            performance['cv_std_accuracy'] = np.nan
    
    # Get predictions for full dataset
    y_pred = model.predict(X)
    performance['train_accuracy'] = (y_pred == y).mean()
    
    # Generate classification report and confusion matrix
    try:
        classification_rep = classification_report(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
    except Exception as e:
        warnings.warn(f"Could not generate classification report: {e}")
    
    return DiscriminantResults(
        feature_importance=importance_df,
        model_performance=performance,
        classification_report=classification_rep,
        confusion_matrix=conf_matrix,
        model=model
    )


def _prepare_data(
    data: pd.DataFrame, 
    config: DiscriminantConfig
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare feature matrix and target vector from DataFrame."""
    
    # Get target vector
    y = data[config.target_column].values
    
    # Determine feature columns
    if config.feature_columns is not None:
        feature_cols = config.feature_columns
    else:
        # Use all numeric columns except target and excluded columns
        exclude_cols = [config.target_column]
        if config.exclude_columns:
            exclude_cols.extend(config.exclude_columns)
        
        feature_cols = [col for col in data.columns 
                       if col not in exclude_cols and 
                       pd.api.types.is_numeric_dtype(data[col])]
    
    # Validate feature columns exist
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Feature columns not found in data: {missing_cols}")
    
    # Extract feature matrix
    X = data[feature_cols].values
    
    # Handle missing values more robustly
    if np.any(np.isnan(X)):
        warnings.warn("NaN values found in features, using robust imputation")
        X_df = pd.DataFrame(X, columns=feature_cols)
        
        # For each column, fill NaNs with median (more robust than mean)
        for col in X_df.columns:
            if X_df[col].isna().any():
                # Use median for numeric columns, or drop columns that are all NaN
                if X_df[col].isna().all():
                    warnings.warn(f"Column {col} is entirely NaN, dropping it")
                    X_df = X_df.drop(col, axis=1)
                else:
                    median_val = X_df[col].median()
                    if np.isnan(median_val):
                        # If median is still NaN, use 0
                        X_df[col] = X_df[col].fillna(0)
                    else:
                        X_df[col] = X_df[col].fillna(median_val)
        
        X = X_df.values
        feature_cols = X_df.columns.tolist()
        
        # Final check for any remaining NaNs
        if np.any(np.isnan(X)):
            warnings.warn("Some NaN values remain, replacing with 0")
            X = np.nan_to_num(X, nan=0.0)
    
    return X, y, feature_cols


def _initialize_model(method: DiscriminantMethod, config: DiscriminantConfig):
    """Initialize the appropriate model based on configuration."""
    
    if method == DiscriminantMethod.RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            random_state=config.random_state
        )
    
    elif method == DiscriminantMethod.LINEAR_DISCRIMINANT:
        return LinearDiscriminantAnalysis()
    
    elif method == DiscriminantMethod.SVM:
        return SVC(
            kernel=config.svm_kernel,
            C=config.svm_C,
            random_state=config.random_state
        )
    
    elif method == DiscriminantMethod.LOGISTIC_REGRESSION:
        return LogisticRegression(
            max_iter=config.logistic_max_iter,
            random_state=config.random_state
        )
    
    else:
        raise ValueError(f"Unknown discriminant method: {method}")


def _extract_feature_importance(
    model, 
    method: DiscriminantMethod, 
    X: np.ndarray, 
    y: np.ndarray
) -> np.ndarray:
    """Extract and normalize feature importance scores from the fitted model."""
    
    if method == DiscriminantMethod.RANDOM_FOREST:
        importance = model.feature_importances_
    
    elif method == DiscriminantMethod.LINEAR_DISCRIMINANT:
        # Use absolute values of LDA coefficients as importance
        if hasattr(model, 'coef_'):
            if model.coef_.ndim > 1:
                # Multi-class case: use mean absolute coefficient
                importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                importance = np.abs(model.coef_[0])
        else:
            warnings.warn("Could not extract coefficients from LDA model")
            importance = np.zeros(X.shape[1])
    
    elif method == DiscriminantMethod.LOGISTIC_REGRESSION:
        # Use absolute values of logistic regression coefficients
        if hasattr(model, 'coef_'):
            if model.coef_.ndim > 1:
                # Multi-class case: use mean absolute coefficient
                importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                importance = np.abs(model.coef_[0])
        else:
            warnings.warn("Could not extract coefficients from logistic regression")
            importance = np.zeros(X.shape[1])
    
    elif method == DiscriminantMethod.SVM:
        # For SVM, use coefficient magnitude
        if hasattr(model, 'coef_') and model.coef_ is not None:
            if model.coef_.ndim > 1:
                importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                importance = np.abs(model.coef_[0])
        else:
            # For non-linear kernels, return uniform importance
            warnings.warn("SVM with non-linear kernel: using uniform feature importance")
            importance = np.ones(X.shape[1])
    
    else:
        raise ValueError(f"Feature importance extraction not implemented for {method}")
    
    # Normalize importance scores to [0, 1] for fair comparison across methods
    if np.sum(importance) > 0:
        importance = importance / np.sum(importance)
    else:
        # If all importance scores are 0, make them uniform
        importance = np.ones(len(importance)) / len(importance)
    
    return importance


def compare_discriminant_methods(
    data: pd.DataFrame,
    base_config: DiscriminantConfig,
    methods: Optional[List[DiscriminantMethod]] = None
) -> Dict[str, DiscriminantResults]:
    """
    Compare multiple discriminant analysis methods on the same dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    base_config : DiscriminantConfig
        Base configuration (method will be overridden)
    methods : List[DiscriminantMethod], optional
        Methods to compare. If None, uses all available methods.
        
    Returns
    -------
    Dict[str, DiscriminantResults]
        Results for each method
    """
    if methods is None:
        methods = list(DiscriminantMethod)
    
    results = {}
    
    for method in methods:
        try:
            config = DiscriminantConfig(**{
                **base_config.__dict__,
                'method': method.value if isinstance(method, DiscriminantMethod) else method
            })
            
            result = identify_discriminative_features(data, config)
            results[method.value] = result
            
        except Exception as e:
            warnings.warn(f"Method {method.value} failed: {e}")
            continue
    
    return results
