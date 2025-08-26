import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.calculators.base_index import BaseIndex


class IRDCalculator(BaseIndex):
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

    def normalize_equality_index(
        self, weights: pd.Series, method: str = "original"
    ) -> float:
        """
        Normalize the equality index to [0,1] interval using different methods.

        Args:
            weights: Series of individual weights
            method: Normalization method ('original' (default), 'theoretical_bounds', 'empirical_bounds', 'gini_based', 'entropy_based')

        Returns:
            float: Normalized equality index in [0,1]
        """
        self.normalization_method = method

        if method == "original":
            return self._normalize_original(weights)
        elif method == "theoretical_bounds":
            return self._normalize_theoretical_bounds(weights)
        elif method == "empirical_bounds":
            return self._normalize_empirical_bounds(weights)
        elif method == "gini_based":
            return self._normalize_gini_based(weights)
        elif method == "entropy_based":
            return self._normalize_entropy_based(weights)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _normalize_original(self, weights: pd.Series) -> float:
        """
        Original normalization: 1 - mean(abs(weights - 1))
        """
        n = len(weights)
        i_max = 2 * (1 - 1 / n)
        i_ =  np.mean(np.abs(weights - 1))
        return 1 - i_ / i_max

    def _normalize_theoretical_bounds(self, weights: pd.Series) -> float:
        """
        Normalize using theoretical bounds based on maximum possible inequality.
        This method is most suitable for cross-temporal comparison.

        The theoretical maximum occurs when all weight is concentrated on one individual.
        """
        n = len(weights)
        if n <= 1:
            return 1.0

        # Theoretical maximum inequality: all weight on one person
        max_inequality = 2 * (n - 1) / n  # Maximum mean absolute deviation from 1

        # Current inequality
        current_inequality = np.mean(np.abs(weights - 1))

        # Normalize to [0,1] where 1 = perfect equality
        normalized_index = max(0, 1 - current_inequality / max_inequality)

        return normalized_index

    def _normalize_empirical_bounds(self, weights: pd.Series) -> float:
        """
        Normalize using empirical bounds from the data.
        Less suitable for cross-temporal comparison as bounds may vary.
        """
        current_inequality = np.mean(np.abs(weights - 1))

        # Use the maximum observed inequality as the bound
        max_observed_inequality = np.max(np.abs(weights - 1))

        if max_observed_inequality == 0:
            return 1.0

        normalized_index = max(0, 1 - current_inequality / max_observed_inequality)

        return normalized_index

    def _normalize_gini_based(self, weights: pd.Series) -> float:
        """
        Normalize using Gini coefficient approach.
        Good for cross-temporal comparison as it's scale-invariant.
        """
        # Calculate Gini coefficient of weights
        sorted_weights = np.sort(weights)
        n = len(sorted_weights)

        if n <= 1 or np.sum(sorted_weights) == 0:
            return 1.0

        # Gini coefficient calculation
        cumsum = np.cumsum(sorted_weights)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

        # Convert to equality index (1 - Gini)
        equality_index = 1 - gini

        return max(0, min(1, equality_index))

    def _normalize_entropy_based(self, weights: pd.Series) -> float:
        """
        Normalize using entropy-based approach.
        Good for cross-temporal comparison and interpretable.
        """
        # Normalize weights to sum to 1 (probability distribution)
        normalized_weights = weights / weights.sum()

        # Calculate entropy
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        normalized_weights = normalized_weights + epsilon
        normalized_weights = normalized_weights / normalized_weights.sum()

        entropy = -np.sum(normalized_weights * np.log(normalized_weights))

        # Maximum entropy (uniform distribution)
        max_entropy = np.log(len(weights))

        if max_entropy == 0:
            return 1.0

        # Normalize entropy to [0,1]
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def normalize_individual_weights(
        self, weights: pd.Series, method: str = "original"
    ) -> pd.Series:
        """
        Normalize individual equality weights to [0,1] interval.

        Args:
            weights: Series of individual weights
            method: Normalization method ('original' (default), 'robust', 'min_max', 'z_score', 'rank', 'none')

        Returns:
            pd.Series: Normalized weights in [0,1]
        """
        if method == "original":
            return self._normalize_weights_original(weights)
        elif method == "min_max":
            return self._normalize_weights_min_max(weights)
        elif method == "z_score":
            return self._normalize_weights_z_score(weights)
        elif method == "robust":
            return self._normalize_weights_robust(weights)
        elif method == "rank":
            return self._normalize_weights_rank(weights)
        elif method == "none" or method is None:
            return weights
        else:
            raise ValueError(f"Unknown weight normalization method: {method}")

    def _normalize_weights_original(self, weights: pd.Series) -> pd.Series:
        """
        Original normalization: weights / q
        where q is the sum of the population proportionss
        """
        q = sum(self.population_proportions.values())
        weights /= q
        return weights

    def _normalize_weights_min_max(self, weights: pd.Series) -> pd.Series:
        """
        Min-max normalization: (w - min) / (max - min)
        Simple linear scaling to [0,1]
        """
        min_w = weights.min()
        max_w = weights.max()

        if max_w == min_w:
            return pd.Series(0.5, index=weights.index)

        normalized = (weights - min_w) / (max_w - min_w)
        return normalized

    def _normalize_weights_z_score(self, weights: pd.Series) -> pd.Series:
        """
        Z-score normalization with sigmoid transformation to [0,1]
        Good for handling outliers
        """
        mean_w = weights.mean()
        std_w = weights.std()

        if std_w == 0:
            return pd.Series(0.5, index=weights.index)

        z_scores = (weights - mean_w) / std_w
        # Use sigmoid function to map to [0,1]
        normalized = 1 / (1 + np.exp(-z_scores))
        return normalized

    def _normalize_weights_robust(self, weights: pd.Series) -> pd.Series:
        """
        Robust normalization using median and IQR with sigmoid transformation
        Less sensitive to outliers than min-max and z-score
        """
        median_w = weights.median()
        q75 = weights.quantile(0.75)
        q25 = weights.quantile(0.25)
        iqr = q75 - q25

        if iqr == 0:
            return pd.Series(0.5, index=weights.index)

        # Use robust scaling with sigmoid transformation
        robust_scores = (weights - median_w) / iqr
        normalized = 1 / (1 + np.exp(-robust_scores))
        return normalized

    def _normalize_weights_rank(self, weights: pd.Series) -> pd.Series:
        """
        Rank-based normalization
        Converts weights to ranks and normalizes to [0,1]
        """
        ranks = weights.rank(method="average")
        normalized = (ranks - 1) / (len(ranks) - 1)
        return normalized

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
        self.normalized_index = self.normalize_equality_index(
            weights_series, normalization_method
        )

        # Add both original and normalized indices
        self.df["equality_index_normalized"] = self.normalized_index

        return self.df


if __name__ == "__main__":
    n_a = 10
    n_b = 2
    df = pd.DataFrame(
        {
            "A": [1 for _ in range(n_a)] + [0 for _ in range(n_b)],
            "B": [0 for _ in range(n_a)] + [1 for _ in range(n_b)],
        },
        index=list(range(n_a + n_b)),
    )
    calculator = EqualityIndexCalculator()
    print("Combination of normalization methods: original and original")
    print(
        calculator.calculate(
            df,
            {"A": 0.52, "B": 0.48},
            weight_normalization_method="original",
            normalization_method="original",
        )
    )
   