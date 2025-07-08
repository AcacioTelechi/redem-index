import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from equality_index_calculator import EqualityIndexCalculator

def demonstrate_normalization_methods():
    """
    Demonstrate different normalization methods for equality index calculation.
    """
    
    # Create sample data with different scenarios
    scenarios = {
        'perfect_equality': {
            'data': pd.DataFrame({
                'A': [1, 0, 1, 0],
                'B': [0, 1, 0, 1]
            }, index=['a', 'b', 'c', 'd']),
            'population_proportions': {'A': 0.5, 'B': 0.5},
            'description': 'Perfect equality scenario'
        },
        'moderate_inequality': {
            'data': pd.DataFrame({
                'A': [1, 1, 0, 0],
                'B': [0, 0, 1, 1]
            }, index=['a', 'b', 'c', 'd']),
            'population_proportions': {'A': 0.5, 'B': 0.5},
            'description': 'Moderate inequality scenario'
        },
        'high_inequality': {
            'data': pd.DataFrame({
                'A': [1, 1, 1, 0],
                'B': [0, 0, 0, 1]
            }, index=['a', 'b', 'c', 'd']),
            'population_proportions': {'A': 0.5, 'B': 0.5},
            'description': 'High inequality scenario'
        }
    }
    
    calculator = EqualityIndexCalculator()
    normalization_methods = ['theoretical_bounds', 'empirical_bounds', 'gini_based', 'entropy_based']
    
    results = []
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n=== {scenario['description']} ===")
        
        for method in normalization_methods:
            try:
                result_df = calculator.calculate(
                    scenario['data'], 
                    scenario['population_proportions'],
                    normalization_method=method
                )
                
                original_index = result_df['equality_index'].iloc[0]
                normalized_index = result_df['equality_index_normalized'].iloc[0]
                
                print(f"{method:20s}: Original={original_index:.4f}, Normalized={normalized_index:.4f}")
                
                results.append({
                    'scenario': scenario_name,
                    'method': method,
                    'original_index': original_index,
                    'normalized_index': normalized_index
                })
                
            except Exception as e:
                print(f"{method:20s}: Error - {e}")
    
    return pd.DataFrame(results)

def compare_across_time_periods():
    """
    Demonstrate how different normalization methods perform across time periods.
    """
    
    # Simulate data from different time periods with varying sample sizes
    time_periods = {
        'period_1': {
            'data': pd.DataFrame({
                'A': [1, 0, 1, 0, 1],
                'B': [0, 1, 0, 1, 0]
            }, index=['a', 'b', 'c', 'd', 'e']),
            'population_proportions': {'A': 0.5, 'B': 0.5}
        },
        'period_2': {
            'data': pd.DataFrame({
                'A': [1, 0, 1, 0, 1, 0, 1],
                'B': [0, 1, 0, 1, 0, 1, 0]
            }, index=['a', 'b', 'c', 'd', 'e', 'f', 'g']),
            'population_proportions': {'A': 0.5, 'B': 0.5}
        },
        'period_3': {
            'data': pd.DataFrame({
                'A': [1, 0, 1, 0],
                'B': [0, 1, 0, 1]
            }, index=['a', 'b', 'c', 'd']),
            'population_proportions': {'A': 0.5, 'B': 0.5}
        }
    }
    
    calculator = EqualityIndexCalculator()
    normalization_methods = ['theoretical_bounds', 'empirical_bounds', 'gini_based', 'entropy_based']
    
    print("\n=== Cross-Temporal Comparison ===")
    
    for method in normalization_methods:
        print(f"\n{method.upper()} METHOD:")
        for period, data in time_periods.items():
            try:
                result_df = calculator.calculate(
                    data['data'], 
                    data['population_proportions'],
                    normalization_method=method
                )
                
                normalized_index = result_df['equality_index_normalized'].iloc[0]
                sample_size = len(data['data'])
                
                print(f"  {period:10s} (n={sample_size}): {normalized_index:.4f}")
                
            except Exception as e:
                print(f"  {period:10s}: Error - {e}")

def demonstrate_weight_normalization():
    """
    Demonstrate different methods for normalizing individual equality weights.
    """
    
    # Create sample data with varying weights
    sample_data = pd.DataFrame({
        'A': [1, 0, 1, 0, 1],
        'B': [0, 1, 0, 1, 0]
    }, index=['a', 'b', 'c', 'd', 'e'])
    
    population_proportions = {'A': 0.5, 'B': 0.5}
    
    calculator = EqualityIndexCalculator()
    weight_methods = ['min_max', 'z_score', 'robust', 'rank']
    
    print("\n=== Individual Weight Normalization Methods ===")
    
    for method in weight_methods:
        try:
            result_df = calculator.calculate(
                sample_data, 
                population_proportions,
                normalization_method='theoretical_bounds',
                weight_normalization_method=method
            )
            
            print(f"\n{method.upper()} METHOD:")
            print("Original Weights vs Normalized Weights:")
            for idx in result_df.index:
                original = result_df.loc[idx, 'equality_weight']
                normalized = result_df.loc[idx, 'equality_weight_normalized']
                print(f"  {idx}: {original:.4f} -> {normalized:.4f}")
            
            # Show statistics
            original_weights = result_df['equality_weight']
            normalized_weights = result_df['equality_weight_normalized']
            
            print(f"  Original range: [{original_weights.min():.4f}, {original_weights.max():.4f}]")
            print(f"  Normalized range: [{normalized_weights.min():.4f}, {normalized_weights.max():.4f}]")
            
        except Exception as e:
            print(f"{method}: Error - {e}")

def compare_weight_normalization_across_periods():
    """
    Compare how different weight normalization methods perform across time periods.
    """
    
    # Simulate data from different time periods
    time_periods = {
        'period_1': {
            'data': pd.DataFrame({
                'A': [1, 0, 1, 0],
                'B': [0, 1, 0, 1]
            }, index=['a', 'b', 'c', 'd']),
            'population_proportions': {'A': 0.5, 'B': 0.5}
        },
        'period_2': {
            'data': pd.DataFrame({
                'A': [1, 1, 0, 0, 1],
                'B': [0, 0, 1, 1, 0]
            }, index=['a', 'b', 'c', 'd', 'e']),
            'population_proportions': {'A': 0.5, 'B': 0.5}
        }
    }
    
    calculator = EqualityIndexCalculator()
    weight_methods = ['min_max', 'z_score', 'robust', 'rank']
    
    print("\n=== Cross-Period Weight Normalization Comparison ===")
    
    for method in weight_methods:
        print(f"\n{method.upper()} METHOD:")
        for period, data in time_periods.items():
            try:
                result_df = calculator.calculate(
                    data['data'], 
                    data['population_proportions'],
                    normalization_method='theoretical_bounds',
                    weight_normalization_method=method
                )
                
                normalized_weights = result_df['equality_weight_normalized']
                print(f"  {period}: range [{normalized_weights.min():.4f}, {normalized_weights.max():.4f}], mean={normalized_weights.mean():.4f}")
                
            except Exception as e:
                print(f"  {period}: Error - {e}")

if __name__ == "__main__":
    print("Equality Index Normalization Methods Demonstration")
    print("=" * 50)
    
    # Demonstrate different scenarios for aggregated index
    results_df = demonstrate_normalization_methods()
    
    # Compare across time periods for aggregated index
    compare_across_time_periods()
    
    # Demonstrate individual weight normalization
    demonstrate_weight_normalization()
    
    # Compare weight normalization across periods
    compare_weight_normalization_across_periods()
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS:")
    print("\nFor Aggregated Index (equality_index_normalized):")
    print("1. 'theoretical_bounds': Best for cross-temporal comparison")
    print("2. 'gini_based': Good alternative, scale-invariant")
    print("3. 'entropy_based': Interpretable, good for comparison")
    print("4. 'empirical_bounds': Avoid for cross-temporal comparison")
    
    print("\nFor Individual Weights (equality_weight_normalized):")
    print("1. 'min_max': Simple linear scaling, good for most cases")
    print("2. 'rank': Preserves relative ordering, robust to outliers")
    print("3. 'robust': Good for data with outliers")
    print("4. 'z_score': Good for normally distributed weights") 