"""
Configuration file for fake news detection model training and inference.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent / "studies/fake_news"
DATA_DIR = Path(__file__).parent / "data/ifs"
MODEL_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

CHROMA_DB_PATH = DATA_DIR / ".chroma_db"

# Dataset paths
FACTCHECK_DATASET_PATH = DATA_DIR / "Analise manual Aos fatos_v2.xlsx"
ANNOTATED_DATASET_PATH = DATA_DIR / "df_train.xlsx"
POSTS_DATASET_PATH = DATA_DIR / "df_train.xlsx"

# ChromaDB collection names
FACTCHECK_COLLECTION = "checks_v2_rewrite_cosine_sentence_transformer"
POSTS_COLLECTION = "posts_cosine_sentence_transformer"

# Model configuration
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01
DROPOUT = 0.5
CLASSIFIER_HIDDEN_DIMS = [768, 384, 192]

# Training configuration
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Class imbalance handling
POSITIVE_CLASS_WEIGHT = 7  # Approximate ratio (620/86)
USE_FOCAL_LOSS = True
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# Claim retrieval
TOP_K_CLAIMS = 3
SIMILARITY_THRESHOLD = 0.6
AGGREGATION_METHOD = "max"  # "max", "mean", or "attention"

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC = "f1"  # "f1", "precision", "recall", "val_loss"

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logging
LOG_INTERVAL = 10
SAVE_BEST_MODEL = True

