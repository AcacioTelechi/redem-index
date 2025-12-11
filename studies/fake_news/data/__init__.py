"""
Data preparation and loading modules for fake news detection.
"""
from .dataset import FakeNewsDataset
from .prepare_data import (
    load_factcheck_dataset,
    load_annotated_dataset,
    prepare_train_data,
    create_data_splits,
)

__all__ = [
    "FakeNewsDataset",
    "load_factcheck_dataset",
    "load_annotated_dataset",
    "prepare_train_data",
    "create_data_splits",
]














