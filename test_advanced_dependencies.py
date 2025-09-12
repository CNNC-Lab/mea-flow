#!/usr/bin/env python3
"""
Comprehensive test for all advanced dependencies in the feature analysis pipeline.
Tests XGBoost, LightGBM, imbalanced-learn, mlxtend, skrebate, and boruta.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_advanced_imports():
    """Test that all advanced ML libraries can be imported."""
    print("=== Advanced Dependencies Import Test ===\n")
    
    # Test XGBoost
    try:
        import xgboost as xgb
        print("✓ XGBoost imported successfully")
        print(f"  Version: {xgb.__version__}")
    except ImportError as e:
        print(f"✗ XGBoost import failed: {e}")
        return False
    
    # Test LightGBM
    try:
        import lightgbm as lgb
        print("✓ LightGBM imported successfully")
        print(f"  Version: {lgb.__version__}")
    except ImportError as e:
        print(f"✗ LightGBM import failed: {e}")
        return False
    
    # Test imbalanced-learn
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        print("✓ imbalanced-learn imported successfully")
    except ImportError as e:
        print(f"✗ imbalanced-learn import failed: {e}")
        return False
    
    # Test mlxtend
    try:
        from mlxtend.feature_selection import SequentialFeatureSelector
        from mlxtend.plotting import plot_sequential_feature_selection
        print("✓ mlxtend imported successfully")
    except ImportError as e:
        print(f"✗ mlxtend import failed: {e}")
        return False
    
    # Test skrebate (Relief-F)
    try:
        from skrebate import ReliefF, SURF, SURFstar, MultiSURF
        print("✓ skrebate imported successfully")
    except ImportError as e:
        print(f"✗ skrebate import failed: {e}")
        return False
    
    # Test boruta
    try:
        from boruta import BorutaPy
        print("✓ boruta imported successfully")
    except ImportError as e:
        print(f"✗ boruta import failed: {e}")
        return False
    
    # Test statsmodels
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import statsmodels.api as sm
        print("✓ statsmodels imported successfully")
        try:
            import statsmodels
            print(f"  Version: {statsmodels.__version__}")
        except:
            print("  Version: Available")
    except ImportError as e:
        print(f"✗ statsmodels import failed: {e}")
        return False
    
    return True

def test_advanced_functionality():
    """Test advanced ML functionality with synthetic data."""
    print("\n=== Advanced Functionality Test ===\n")
    
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Generate synthetic data
    print("1. Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='condition')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42, stratify=y_series
    )
    
    print(f"  Dataset shape: {X_df.shape}")
    print(f"  Classes: {np.unique(y_series)}")
    
    # Test XGBoost
    print("\n2. Testing XGBoost...")
    try:
        import xgboost as xgb
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        
        print(f"  ✓ XGBoost accuracy: {xgb_acc:.3f}")
        
        # Feature importance
        importance = xgb_model.feature_importances_
        top_features = np.argsort(importance)[-5:][::-1]
        print(f"  ✓ Top 5 features: {[feature_names[i] for i in top_features]}")
        
    except Exception as e:
        print(f"  ✗ XGBoost test failed: {e}")
        return False
    
    # Test LightGBM
    print("\n3. Testing LightGBM...")
    try:
        import lightgbm as lgb
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        
        print(f"  ✓ LightGBM accuracy: {lgb_acc:.3f}")
        
        # Feature importance
        importance = lgb_model.feature_importances_
        top_features = np.argsort(importance)[-5:][::-1]
        print(f"  ✓ Top 5 features: {[feature_names[i] for i in top_features]}")
        
    except Exception as e:
        print(f"  ✗ LightGBM test failed: {e}")
        return False
    
    # Test SMOTE (imbalanced-learn)
    print("\n4. Testing SMOTE (imbalanced-learn)...")
    try:
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"  ✓ Original shape: {X_train.shape}")
        print(f"  ✓ Resampled shape: {X_resampled.shape}")
        print(f"  ✓ Original class distribution: {np.bincount(y_train)}")
        print(f"  ✓ Resampled class distribution: {np.bincount(y_resampled)}")
        
    except Exception as e:
        print(f"  ✗ SMOTE test failed: {e}")
        return False
    
    # Test Relief-F (skrebate)
    print("\n5. Testing Relief-F (skrebate)...")
    try:
        from skrebate import ReliefF
        
        relief = ReliefF(n_features_to_select=10, n_neighbors=10)
        relief.fit(X_train.values, y_train.values)
        
        selected_features = relief.transform(X_train.values)
        print(f"  ✓ Relief-F selected {selected_features.shape[1]} features")
        
        # Get feature scores
        feature_scores = relief.feature_importances_
        top_indices = np.argsort(feature_scores)[-5:][::-1]
        print(f"  ✓ Top 5 Relief-F features: {[feature_names[i] for i in top_indices]}")
        
    except Exception as e:
        print(f"  ✗ Relief-F test failed: {e}")
        return False
    
    # Test Boruta
    print("\n6. Testing Boruta...")
    try:
        from boruta import BorutaPy
        
        rf = RandomForestClassifier(n_jobs=-1, random_state=42)
        boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=10)
        boruta.fit(X_train.values, y_train.values)
        
        selected_features = boruta.support_
        n_selected = np.sum(selected_features)
        print(f"  ✓ Boruta selected {n_selected} features")
        
        if n_selected > 0:
            selected_names = [feature_names[i] for i in range(len(selected_features)) if selected_features[i]]
            print(f"  ✓ Selected features: {selected_names[:5]}...")  # Show first 5
        
    except Exception as e:
        print(f"  ✗ Boruta test failed: {e}")
        return False
    
    # Test Sequential Feature Selection (mlxtend)
    print("\n7. Testing Sequential Feature Selection (mlxtend)...")
    try:
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        sfs = SFS(rf, k_features=5, forward=True, floating=False, 
                  scoring='accuracy', cv=3, n_jobs=1)
        sfs.fit(X_train, y_train)
        
        selected_indices = list(sfs.k_feature_idx_)
        selected_names = [feature_names[i] for i in selected_indices]
        print(f"  ✓ SFS selected {len(selected_indices)} features")
        print(f"  ✓ Selected features: {selected_names}")
        
    except Exception as e:
        print(f"  ✗ Sequential Feature Selection test failed: {e}")
        return False
    
    return True

def test_discriminant_with_advanced_methods():
    """Test the discriminant module with advanced methods."""
    print("\n=== Discriminant Module Advanced Methods Test ===\n")
    
    try:
        # Import our discriminant module
        sys.path.append('src')
        import mea_flow.analysis.discriminant as discriminant
        
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        
        # Generate test data
        X, y = make_classification(
            n_samples=500, 
            n_features=15, 
            n_informative=8,
            n_redundant=3,
            n_classes=2,
            random_state=42
        )
        
        feature_names = [f'neural_feature_{i}' for i in range(X.shape[1])]
        
        # Create DataFrame
        data = pd.DataFrame(X, columns=feature_names)
        data['condition'] = y
        
        print(f"1. Created test dataset: {data.shape}")
        
        # Test comprehensive analysis with all phases
        config = discriminant.FeatureAnalysisConfig(
            target_column='condition',
            phases=[
                discriminant.AnalysisPhase.PREPROCESSING,
                discriminant.AnalysisPhase.REDUNDANCY_DETECTION,
                discriminant.AnalysisPhase.CORE_SELECTION,
                discriminant.AnalysisPhase.CONSENSUS
            ],
            methods=[
                discriminant.FeatureSelectionMethod.MUTUAL_INFORMATION,
                discriminant.FeatureSelectionMethod.ANOVA_F_TEST,
                discriminant.FeatureSelectionMethod.KRUSKAL_WALLIS,
                discriminant.FeatureSelectionMethod.LDA,
                discriminant.FeatureSelectionMethod.LASSO,
                discriminant.FeatureSelectionMethod.RANDOM_FOREST
            ],
            cv_folds=3,
            random_state=42
        )
        
        print("2. Running comprehensive feature analysis...")
        results = discriminant.comprehensive_feature_analysis(data, config)
        
        print(f"  ✓ Analysis completed successfully")
        print(f"  ✓ Redundancy results: {len(results.redundancy_results)} methods")
        print(f"  ✓ Feature importance results: {len(results.importance_results)} methods")
        print(f"  ✓ Consensus ranking available: {results.consensus_result is not None}")
        
        if results.consensus_result:
            top_features = results.consensus_result.consensus_ranking.head(5)
            print(f"  ✓ Top 5 consensus features:")
            for idx, row in top_features.iterrows():
                print(f"    {row['feature']}: score={row['consensus_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Discriminant module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing Advanced Dependencies for MEA-Flow Feature Analysis Pipeline")
    print("=" * 70)
    
    success = True
    
    # Test imports
    if not test_advanced_imports():
        success = False
    
    # Test functionality
    if not test_advanced_functionality():
        success = False
    
    # Test discriminant module
    if not test_discriminant_with_advanced_methods():
        success = False
    
    print("\n" + "=" * 70)
    if success:
        print("✓ All advanced dependency tests passed!")
        print("\nThe comprehensive feature analysis pipeline is ready with:")
        print("  • XGBoost and LightGBM for gradient boosting")
        print("  • SMOTE for handling imbalanced data")
        print("  • Relief-F for instance-based feature selection")
        print("  • Boruta for all-relevant feature selection")
        print("  • Sequential Feature Selection for wrapper methods")
        print("  • Statsmodels for VIF analysis")
        print("  • Full integration with MEA-Flow discriminant module")
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
