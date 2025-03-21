import pandas as pd
import numpy as np
from src.indices.base_index import BaseIndex

class EqualityIndexCalculator(BaseIndex):
    def __init__(self):
        super().__init__()
        self.weights = None
        self.adjustment_index = None
        
    def compute_individual_weights(self, df: pd.DataFrame, population_proportions: dict) -> tuple[dict, float]:
        """
        Computes individual weights using a weighted sum approach and calculates an adjustment index.
        
        Args:
            df: DataFrame with individuals as index and categories as columns (binary values)
            population_proportions: Dictionary with expected proportions for each category
            
        Returns:
            tuple: (weights_dict, adjustment_index)
        """
        # Compute the sum of occurrences for each category in the sample
        category_totals = df.sum(axis=0)  # Sum across individuals
        
        # Compute weights for each individual
        weights = df.apply(
            lambda row: sum(
                row[col] * (population_proportions[col] / category_totals[col]) 
                for col in df.columns 
                if category_totals[col] > 0
            ), 
            axis=1
        )

        # Normalize weights to keep total sum of weights = number of individuals
        weights *= len(df) / weights.sum()

        # Compute the adjustment index
        adjustment_index = 1 - np.mean(np.abs(weights - 1))

        return dict(zip(df.index, weights)), adjustment_index

    def calculate(self, df: pd.DataFrame, population_proportions: dict) -> pd.DataFrame:
        """
        Calculate equality index for the given DataFrame.
        
        Args:
            df: Input DataFrame with categorical columns
            population_proportions: Dictionary mapping column names to expected proportions
            
        Returns:
            DataFrame with added weights and equality index
        """
        self.df = df.copy()
        
        # Calculate weights and adjustment index
        self.weights, self.adjustment_index = self.compute_individual_weights(
            self.df,
            population_proportions
        )
        
        # Add weights to DataFrame
        self.df['equality_weight'] = self.df.index.map(self.weights)
        
        # Add adjustment index as a column (same value for all rows)
        self.df['equality_index'] = self.adjustment_index
        
        return self.df 