from .base_calculator import BaseCalculator

from utils.df_operations import min_max_normalize, apply_weights


class IPPCalculator(BaseCalculator):
    def __init__(self):
        super().__init__()
        self.weights = {
            "tempo_atuacao_dias_cum_ln_norm": 1,
            "relatorias_ln_cum_norm": 2,
            # "pos_lider_cum_norm": 4,
            "pos_comiss_pr_cum_norm": 3,
            # "mesa_cum_norm": 5,
            # "mandatos_cum_norm": 1,
            "fid_gerais_cum_norm": 1,
        }

        self.dimensoes = {
            "dim_comprometimento": [
                # "mesa_cum_norm_w",
                # "pos_lider_cum_norm_w",
                "pos_comiss_pr_cum_norm_w",
                "relatorias_ln_cum_norm",
            ],
            "dim_carreira": [
                # "mandatos_cum_norm_w",
                "fid_gerais_cum_norm_w",
                "tempo_atuacao_dias_cum_ln_norm_w",
            ],
        }

    def calculate(self, df):
        """Calculate IPP index"""
        self.df = df.copy()

        # Normalize columns
        for col, _ in self.weights.items():
            base_col = col.replace("_norm", "")
            self.df[col] = min_max_normalize(self.df[base_col])

        # Apply weights
        self.df = apply_weights(self.df, self.weights)

        # Calculate dimensions
        self.calculate_dimensions()

        # Calculate final index
        self.df["ipp"] = (
            self.df["dim_comprometimento_norm"] + self.df["dim_carreira_norm"]
        ) / 2

        return self.df

    def calculate_dimensions(self):
        """Calculate and normalize dimensions"""
        # Calculate raw dimensions
        self.df["dim_comprometimento"] = sum(
            self.df[col] for col in self.dimensoes["dim_comprometimento"]
        )

        self.df["dim_carreira"] = sum(
            self.df[col] for col in self.dimensoes["dim_carreira"]
        )

        # Normalize dimensions
        for dim in ["dim_comprometimento", "dim_carreira"]:
            self.df[f"{dim}_norm"] = min_max_normalize(self.df[dim])
