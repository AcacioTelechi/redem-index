"""
Training script for fake news detection model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import config
from data.prepare_data import prepare_train_data, create_data_splits, get_class_weights
from data.dataset import FakeNewsDataset
from models.fake_news_model import FakeNewsDetector, FocalLoss, create_model


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving."""
    
    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        metric: str = config.EARLY_STOPPING_METRIC,
        mode: str = "max",
        min_delta: float = 0.0
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            metric: Metric to monitor
            mode: "max" or "min" (whether higher or lower is better)
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "max":
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == "min"
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            metrics["auc"] = auc
        except ValueError:
            metrics["auc"] = 0.0
    
    return metrics


def train_epoch(model, dataloader, optimizer, loss_fn, device, use_focal_loss=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        if use_focal_loss and loss_fn is not None:
            logits = outputs["logits"]
            loss = loss_fn(logits, labels)
        else:
            loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs["logits"], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return avg_loss, metrics


def validate(model, dataloader, loss_fn, device, use_focal_loss=False):
    """Validate on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            if use_focal_loss and loss_fn is not None:
                logits = outputs["logits"]
                loss = loss_fn(logits, labels)
            else:
                loss = outputs["loss"]
            
            total_loss += loss.item()
            preds = torch.argmax(outputs["logits"], dim=1)
            probas = torch.softmax(outputs["logits"], dim=1)[:, 1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds, all_probas)
    
    return avg_loss, metrics


def create_weighted_sampler(dataset):
    """Create weighted sampler to handle class imbalance."""
    labels = dataset.data["label"].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def train():
    """Main training function."""
    print("Starting...")
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Create checkpoint directory
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    df_train_data = prepare_train_data(use_retrieval=True, use_existing_matches=False)
    
    print(f"Total training pairs: {len(df_train_data)}")
    print(f"  - Fake news: {(df_train_data['label'] == 1).sum()}")
    print(f"  - Not fake: {(df_train_data['label'] == 0).sum()}")
    
    # Create splits
    df_train, df_val, df_test = create_data_splits(df_train_data)
    
    print(f"\nTrain: {len(df_train)}")
    print(f"Val: {len(df_val)}")
    print(f"Test: {len(df_test)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create datasets
    train_dataset = FakeNewsDataset(df_train, tokenizer)
    val_dataset = FakeNewsDataset(df_val, tokenizer)
    test_dataset = FakeNewsDataset(df_test, tokenizer)
    
    # Create weighted sampler for training
    train_sampler = create_weighted_sampler(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nInitializing model...")
    device = torch.device(config.DEVICE)
    model, loss_fn = create_model(
        model_name=config.MODEL_NAME,
        use_focal_loss=config.USE_FOCAL_LOSS,
        classifier_hidden_dims=config.CLASSIFIER_HIDDEN_DIMS
    )
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        metric=config.EARLY_STOPPING_METRIC,
        mode="max"
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_f1 = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": []
    }
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, config.USE_FOCAL_LOSS
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, loss_fn, device, config.USE_FOCAL_LOSS
        )
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)
        
        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            if config.SAVE_BEST_MODEL:
                checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_val_f1,
                    "val_metrics": val_metrics
                }, checkpoint_path)
                print(f"Saved best model with F1: {best_val_f1:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics[config.EARLY_STOPPING_METRIC]):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_metrics = validate(model, test_loader, loss_fn, device, config.USE_FOCAL_LOSS)
    print(f"Test Metrics: {test_metrics}")
    
    # Save final model and history
    final_checkpoint_path = config.CHECKPOINT_DIR / "final_model.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_metrics": test_metrics,
        "history": history
    }, final_checkpoint_path)
    
    # Save history as JSON
    history_path = config.CHECKPOINT_DIR / "training_history.json"
    # Convert numpy types to native Python types for JSON
    json_history = {}
    for key, value in history.items():
        if key.endswith("_metrics"):
            json_history[key] = [
                {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                 for k, v in m.items()}
                for m in value
            ]
        else:
            json_history[key] = [float(v) for v in value]
    
    with open(history_path, "w") as f:
        json.dump(json_history, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {config.CHECKPOINT_DIR}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    train()

