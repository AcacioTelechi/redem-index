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
        if weights.sum() > 0:
            weights *= len(df) / weights.sum()

        # Compute the adjustment index
        # adjustment_index = 1 - np.mean(np.abs(weights - 1))

        q = sum(population_proportions.values())
        weights /= q

        return dict(zip(df.index, weights))

    def normalize_index(self, weights: pd.Series, method: str = "original") -> float:
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
        elif method == "square":
            return self._normalize_index_square(weights)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def normalize_individual_weights(
        self, weights: pd.Series
    ) -> pd.Series:
        """
        Normalize individual weights to [0,1] interval using softmax function
        """
        weights = np.exp(weights)
        weights /= weights.sum()
        return weights

    def _normalize_index_original(self, weights: pd.Series) -> float:
        """
        Original normalization: 1 - sum(abs(weights - 1)) / (2 * (n - 1))
        """
        if not np.isclose(weights.sum(), len(weights), rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"Weights do not sum to the number of individuals (within tolerance): {weights.sum()} != {len(weights)}"
            )

        n = len(weights)
        normalizer_factor = 1 / (2 * (n - 1))
        i_ = np.sum(np.abs(weights - 1))

        return 1 - normalizer_factor * i_

    def _normalize_index_square(self, weights: pd.Series) -> float:
        """
        Square normalization: 1 - sqrt(sum((w - 1)**2)) / sqrt(n * (n - 1))
        This is 1 minus the Euclidean distance of weights from a vector of 1s, normalized by the max possible distance.
        """
        # Use np.isclose to check if weights sum to the number of individuals due to floating point precision issues
        if not np.isclose(weights.sum(), len(weights), rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"Weights do not sum to the number of individuals (within tolerance): {weights.sum()} != {len(weights)}"
            )

        n = len(weights)

        if n <= 1:
            return 1.0

        i_ = np.sum((weights - 1) ** 2)
        # TODO validate!!
        max_i_ = n * (n - 1)

        normalized_deviation = np.sqrt(i_ / max_i_)

        return 1 - normalized_deviation

    def calculate(
        self,
        df: pd.DataFrame,
        population_proportions: dict,
        normalization_method: str = "original",
    ) -> pd.DataFrame:
        """
        Calculate equality index for the given DataFrame.

        Args:
            df: Input DataFrame with categorical columns
            population_proportions: Dictionary mapping column names to expected proportions
            normalization_method: Method for normalizing the equality index to [0,1]
                'original' (default), 'theoretical_bounds', 'empirical_bounds', 'gini_based', 'entropy_based'

        Returns:
            DataFrame with added weights and equality index
        """
        self.df = df.copy()
        self.population_proportions = population_proportions

        # Check for missing categories in the sample
        for col in self.population_proportions:
            if col not in self.df.columns or self.df[col].sum() == 0:
                self.df["equality_weight"] = np.nan
                self.df["equality_weight_normalized"] = np.nan
                self.df["equality_index_normalized"] = 0
                return self.df

        # Calculate weights and adjustment index
        self.weights = self.compute_individual_weights(self.df, population_proportions)

        # Add original weights to DataFrame
        self.df["equality_weight"] = self.df.index.map(self.weights)

        # Normalize individual weights
        weights_series = pd.Series(self.weights)
        
        normalized_weights = self.normalize_individual_weights(
            weights_series
        )
        self.df["equality_weight_normalized"] = normalized_weights

        # Calculate normalized equality index
        self.normalized_index = self.normalize_index(
            weights_series, normalization_method
        )

        # Add both original and normalized indices
        self.df["equality_index_normalized"] = self.normalized_index

        return self.df


if __name__ == "__main__":
    import sys
    import os
    import matplotlib.pyplot as plt

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    n_a = 1000
    n_b = 1000

    scenarios = []
    for alpha_ in range(0,101, 1):

        alpha = alpha_ / 100
        scenarios.append(
            {
                "desc": f"Alpha {alpha}",
                "A": [1] * int(n_a * alpha) + [0] * int(n_b * (1 - alpha)),
                "B": [0] * int(n_a * alpha) + [1] * int(n_b * (1 - alpha)),
            }
        )

    population_proportions = {"A": 0.5, "B": 0.5}

    calculator = IRDCalculator()
    results = []
    for scenario in scenarios:
        df = pd.DataFrame({"A": scenario["A"], "B": scenario["B"]})
        result = calculator.calculate(
            df=df,
            population_proportions=population_proportions,
        )
        results.append(
            {"desc": scenario["desc"], "ird": result["equality_index_normalized"][0]}
        )

    pd.DataFrame(results).plot(x="desc", y="ird", kind="line")
    plt.show()

    print(pd.DataFrame(results))
