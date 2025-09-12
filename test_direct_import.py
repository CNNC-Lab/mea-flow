#!/usr/bin/env python3
"""
Direct import test for discriminant analysis module.
"""

import sys
import os

# Add the source directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def test_direct_import():
    """Test direct import of discriminant module."""
    try:
        # Import the discriminant module directly
        import mea_flow.analysis.discriminant as discriminant
        
        print("✓ Direct import successful")
        print(f"  sklearn available: {discriminant.SKLEARN_AVAILABLE}")
        print(f"  statsmodels available: {discriminant.STATSMODELS_AVAILABLE}")
        
        # Test creating configuration
        config = discriminant.FeatureAnalysisConfig(
            target_column='condition',
            phases=[discriminant.AnalysisPhase.CORE_SELECTION],
            scale_features=True
        )
        print("✓ Configuration creation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Direct import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_sample_data():
    """Test with actual sample data."""
    try:
        import pandas as pd
        import numpy as np
        import mea_flow.analysis.discriminant as discriminant
        
        # Create simple test data
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'condition': ['A'] * 50 + ['B'] * 50
        })
        
        # Make feature1 discriminative
        data.loc[data['condition'] == 'B', 'feature1'] += 2
        
        print("✓ Test data created")
        
        # Test individual methods
        X = data[['feature1', 'feature2', 'feature3']]
        y = data['condition']
        
        # Test mutual information
        mi_result = discriminant.mutual_information_analysis(X, y)
        print(f"✓ Mutual Information: top feature = {mi_result.scores.iloc[0]['feature']}")
        
        # Test ANOVA
        anova_result = discriminant.anova_f_test_analysis(X, y)
        print(f"✓ ANOVA F-test: {anova_result.metadata['n_significant']} significant features")
        
        # Test comprehensive analysis
        config = discriminant.FeatureAnalysisConfig(
            target_column='condition',
            phases=[discriminant.AnalysisPhase.CORE_SELECTION, discriminant.AnalysisPhase.CONSENSUS],
            scale_features=True,
            rf_n_estimators=50
        )
        
        results = discriminant.comprehensive_feature_analysis(data, config)
        print(f"✓ Comprehensive analysis: {len(results.importance_results)} methods completed")
        
        if results.consensus_result:
            top_feature = results.consensus_result.consensus_ranking.iloc[0]['feature']
            print(f"✓ Top consensus feature: {top_feature}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Direct Import Test ===")
    
    print("\n1. Testing direct import...")
    success1 = test_direct_import()
    
    print("\n2. Testing with sample data...")
    success2 = test_with_sample_data()
    
    if success1 and success2:
        print("\n✓ All tests passed! The discriminant analysis module is working correctly.")
        print("\nTo use in your notebook:")
        print("import sys")
        print("sys.path.append('src')")
        print("import mea_flow.analysis.discriminant as discriminant")
    else:
        print("\n✗ Some tests failed")
