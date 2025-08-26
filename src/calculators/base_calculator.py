import pandas as pd
from abc import ABC, abstractmethod


class BaseCalculator(ABC):
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
