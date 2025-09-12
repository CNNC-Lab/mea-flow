# MEA-Flow Installation Guide

This guide provides comprehensive installation instructions for MEA-Flow with all advanced feature analysis dependencies.

## Prerequisites

- Python 3.10 or higher
- Git
- uv package manager (recommended) or pip

## Quick Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd mea-flow

# Install with all dependencies using uv
uv sync
```

### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd mea-flow

# Install in development mode with all dependencies
pip install -e ".[all]"
```

## Dependencies Overview

MEA-Flow includes several categories of dependencies for comprehensive neural data analysis:

### Core Dependencies
- **numpy** (≥1.24.0) - Numerical computing
- **scipy** (≥1.10.0) - Scientific computing
- **pandas** (≥2.0.0) - Data manipulation
- **scikit-learn** (≥1.3.0) - Machine learning
- **statsmodels** (≥0.14.0) - Statistical modeling
- **matplotlib** (≥3.7.0) - Plotting
- **seaborn** (≥0.12.0) - Statistical visualization
- **h5py** (≥3.8.0) - HDF5 file support

### Advanced ML Dependencies
- **xgboost** (≥3.0.5) - Gradient boosting
- **lightgbm** (≥4.6.0) - Gradient boosting
- **imbalanced-learn** (≥0.14.0) - Handling imbalanced datasets
- **mlxtend** (≥0.23.4) - Machine learning extensions
- **skrebate** (≥0.62) - Relief-based feature selection
- **boruta** (≥0.4.3) - All-relevant feature selection

### Optional Dependencies
- **pyspike** (≥0.7.0) - Spike train analysis
- **umap-learn** (≥0.5.9) - Manifold learning
- **jupyter** - Notebook support
- **plotly** - Interactive plotting

## Feature Analysis Pipeline Capabilities

The comprehensive feature analysis pipeline now supports:

### Redundancy Detection
- **Variance Inflation Factor (VIF)** - Multicollinearity detection
- **Correlation Analysis** - Feature correlation matrix
- **Hierarchical Clustering** - Feature grouping

### Filter Methods
- **Mutual Information** - Information-theoretic feature selection
- **ANOVA F-test** - Statistical significance testing
- **Kruskal-Wallis** - Non-parametric statistical testing
- **Variance Threshold** - Low-variance feature removal
- **Fisher Score** - Class separability scoring

### Embedded Methods
- **Linear Discriminant Analysis (LDA)** - Dimensionality reduction
- **LASSO Regression** - L1 regularization
- **Ridge Regression** - L2 regularization
- **Elastic Net** - Combined L1/L2 regularization
- **Random Forest** - Tree-based feature importance
- **XGBoost** - Gradient boosting feature importance
- **SVM with L1** - Support vector machine feature selection

### Advanced Methods
- **Relief-F** - Instance-based feature selection
- **Boruta** - All-relevant feature selection
- **Sequential Feature Selection** - Wrapper methods
- **SMOTE** - Synthetic minority oversampling

### Consensus and Validation
- **Borda Count Ranking** - Multi-method consensus
- **Cross-validation Stability** - Feature selection stability
- **Bootstrap Confidence Intervals** - Statistical validation

## Installation Verification

Run the comprehensive test suite to verify all dependencies:

```bash
# Test basic functionality
uv run python test_direct_import.py

# Test all advanced dependencies
uv run python test_advanced_dependencies.py
```

Expected output should show:
```
✓ All advanced dependency tests passed!

The comprehensive feature analysis pipeline is ready with:
  • XGBoost and LightGBM for gradient boosting
  • SMOTE for handling imbalanced data
  • Relief-F for instance-based feature selection
  • Boruta for all-relevant feature selection
  • Sequential Feature Selection for wrapper methods
  • Statsmodels for VIF analysis
  • Full integration with MEA-Flow discriminant module
```

## Usage Examples

### Basic Feature Analysis

```python
import pandas as pd
from mea_flow.analysis.discriminant import (
    comprehensive_feature_analysis,
    FeatureAnalysisConfig,
    AnalysisPhase,
    FeatureSelectionMethod
)

# Load your MEA data
data = pd.read_csv('your_mea_features.csv')

# Configure analysis
config = FeatureAnalysisConfig(
    target_column='condition',
    phases=[
        AnalysisPhase.PREPROCESSING,
        AnalysisPhase.REDUNDANCY_DETECTION,
        AnalysisPhase.CORE_SELECTION,
        AnalysisPhase.CONSENSUS
    ],
    methods=[
        FeatureSelectionMethod.MUTUAL_INFORMATION,
        FeatureSelectionMethod.ANOVA_F_TEST,
        FeatureSelectionMethod.RANDOM_FOREST,
        FeatureSelectionMethod.XGBOOST
    ],
    cv_folds=5,
    random_state=42
)

# Run analysis
results = comprehensive_feature_analysis(data, config)

# View results
print(f"Top consensus features:")
print(results.consensus_result.consensus_ranking.head(10))
```

### Advanced Configuration

```python
# Advanced configuration with all methods
config = FeatureAnalysisConfig(
    target_column='condition',
    methods=[
        # Filter methods
        FeatureSelectionMethod.MUTUAL_INFORMATION,
        FeatureSelectionMethod.ANOVA_F_TEST,
        FeatureSelectionMethod.KRUSKAL_WALLIS,
        FeatureSelectionMethod.FISHER_SCORE,
        
        # Embedded methods
        FeatureSelectionMethod.LDA,
        FeatureSelectionMethod.LASSO,
        FeatureSelectionMethod.RANDOM_FOREST,
        FeatureSelectionMethod.XGBOOST,
        
        # Advanced methods (when implemented)
        # FeatureSelectionMethod.RELIEF_F,
        # FeatureSelectionMethod.MRMR
    ],
    # Redundancy detection
    vif_threshold=10.0,
    correlation_threshold=0.9,
    
    # Cross-validation
    cv_folds=5,
    cv_scoring='accuracy',
    
    # Statistical significance
    alpha=0.05,
    multiple_testing_correction='fdr_bh',
    
    # Method-specific parameters
    lasso_alpha=1.0,
    rf_n_estimators=100,
    xgb_n_estimators=100,
    
    random_state=42
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   uv sync  # or pip install -e ".[all]"
   ```

2. **Missing h5py**: Required for HDF5 file support
   ```bash
   uv add h5py  # or pip install h5py
   ```

3. **Statsmodels Issues**: Required for VIF analysis
   ```bash
   uv add statsmodels  # or pip install statsmodels
   ```

4. **Memory Issues**: For large datasets, consider:
   - Reducing `cv_folds`
   - Using fewer methods simultaneously
   - Preprocessing to reduce feature dimensionality

### Performance Tips

1. **Parallel Processing**: Set `n_jobs=-1` in config for multi-core processing
2. **Method Selection**: Choose appropriate methods for your data size
3. **Cross-validation**: Reduce `cv_folds` for faster execution
4. **Feature Preprocessing**: Remove obviously irrelevant features first

## Development Installation

For development work:

```bash
# Clone and install in development mode
git clone <repository-url>
cd mea-flow
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Support

For issues and questions:
- Check the test scripts for usage examples
- Review the comprehensive documentation in `notebooks/`
- Examine the reference implementation in `src/mea_flow/analysis/discriminant.py`

## Version Compatibility

- **Python**: 3.10+
- **Core ML Libraries**: Latest stable versions
- **Advanced Libraries**: Tested with versions specified in `pyproject.toml`

The installation has been tested and verified to work with all specified dependency versions.
