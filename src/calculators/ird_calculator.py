import pandas as pd
import numpy as np

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base_calculator import BaseCalculator
else:
    from src.calculators.base_calculator import BaseCalculator


class IRDCalculator(BaseCalculator):
    def __init__(self):
        super().__init__()
        self.weights = None
        self.adjustment_index = None
        self.normalized_index = None

        self.weight_normalization_method = "original"
        self.normalization_method = "original"

        self.population_proportions = None

    def compute_individual_weights(
        self, df: pd.DataFrame, population_proportions: dict
    ) -> tuple[dict, float]:
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

        total_population = len(df)
        category_proportions = {
            col: category_totals[col] / total_population for col in df.columns
        }
        # Compute weights for each individual
        weights = df.apply(
            lambda row: sum(
                (
                    row[col] * (population_proportions[col] / category_proportions[col])
                    if category_proportions[col] > 0
                    else 0
                )
                for col in df.columns
            ),
            axis=1,
        )

        # Normalize weights to keep total sum of weights = number of individuals
        # weights *= len(df) / weights.sum()

        # Compute the adjustment index
        # adjustment_index = 1 - np.mean(np.abs(weights - 1))

        return dict(zip(df.index, weights))

    def normalize_index(
        self, weights: pd.Series, method: str = "original"
    ) -> float:
        """
        Normalize the equality index to [0,1] interval using different methods.

        Args:
            weights: Series of individual weights
            method: Normalization method ('original' (default))

        Returns:
            float: Normalized equality index in [0,1]
        """
        self.normalization_method = method

        if method == "original":
            return self._normalize_index_original(weights)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def normalize_individual_weights(
        self, weights: pd.Series, method: str = "original"
    ) -> pd.Series:
        """
        Normalize individual equality weights to [0,1] interval.

        Args:
            weights: Series of individual wei'ghts
            method: Normalization method ('original' (default))

        Returns:
            pd.Series: Normalized weights in [0,1]
        """
        if method == "original":
            return self._normalize_weights_original(weights)
        else:
            raise ValueError(f"Unknown weight normalization method: {method}")

    def _normalize_index_original(self, weights: pd.Series) -> float:
        """
        Original normalization: 1 - sum(abs(weights - 1)) / (2 * (n - 1))
        """
        if int(len(weights)) != int(sum(weights)):
            raise ValueError(f"Weights do not sum to the number of individuals: {weights.sum()} != {len(weights)}")

        n = len(weights)
        normalizer_factor = 1 / (2 * (n - 1))
        i_ =  np.sum(np.abs(weights - 1))

        return 1 - normalizer_factor * i_

    def _normalize_weights_original(self, weights: pd.Series) -> pd.Series:
        """
        Original normalization: weights / q
        where q is the sum of the population proportionss
        """
        q = sum(self.population_proportions.values())
        weights /= q
        return weights
    
    def calculate(
        self,
        df: pd.DataFrame,
        population_proportions: dict,
        weight_normalization_method: str = "original",
        normalization_method: str = "original",
    ) -> pd.DataFrame:
        """
        Calculate equality index for the given DataFrame.

        Args:
            df: Input DataFrame with categorical columns
            population_proportions: Dictionary mapping column names to expected proportions
            weight_normalization_method: Method for normalizing individual weights to [0,1]
                'original' (default), 'robust', 'min_max', 'z_score', 'rank', 'none';
            normalization_method: Method for normalizing the equality index to [0,1]
                'original' (default), 'theoretical_bounds', 'empirical_bounds', 'gini_based', 'entropy_based'

        Returns:
            DataFrame with added weights and equality index
        """
        self.df = df.copy()
        self.population_proportions = population_proportions

        # Calculate weights and adjustment index
        self.weights = self.compute_individual_weights(self.df, population_proportions)

        # Add original weights to DataFrame
        self.df["equality_weight"] = self.df.index.map(self.weights)

        # Normalize individual weights
        weights_series = pd.Series(self.weights)
        normalized_weights = self.normalize_individual_weights(
            weights_series, weight_normalization_method
        )
        self.df["equality_weight_normalized"] = normalized_weights

        # Calculate normalized equality index
        self.normalized_index = self.normalize_index(
            normalized_weights, normalization_method
        )

        # Add both original and normalized indices
        self.df["equality_index_normalized"] = self.normalized_index

        return self.df


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    n_a = 10
    n_b = 2
    n_c = 3
    n_d = 9
    df = pd.DataFrame(
        {
            "A": [1 for _ in range(n_a)] + [0 for _ in range(n_b)],
            "B": [0 for _ in range(n_a)] + [1 for _ in range(n_b)],
            "C": [0 for _ in range(n_c)] + [1 for _ in range(n_d)],
            "D": [0 for _ in range(n_d)] + [1 for _ in range(n_c)],
        },
        index=list(range(n_a + n_b )),
    )
    calculator = IRDCalculator()
    print("Combination of normalization methods: original and original")
    print(
        calculator.calculate(
            df,
            {"A": 0.52, "B": 0.48, "C": 0.20, "D": 0.80},
            weight_normalization_method="original",
            normalization_method="original",
        )
    )
   