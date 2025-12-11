"""
PyTorch Dataset classes for fake news detection.
"""
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional
import pandas as pd
import config


class FakeNewsDataset(Dataset):
    """
    Dataset class for fake news detection.
    
    Each sample contains:
    - post: Social media post text
    - claim: Fake news claim text
    - label: Binary label (0 = not fake, 1 = fake)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = config.MAX_LENGTH,
        include_claim: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with columns 'post', 'claim', and 'label'
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            include_claim: If True, concatenate post and claim. If False, use only post.
        """
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_claim = include_claim
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'labels'
        """
        row = self.data.iloc[idx]
        post = str(row["post"])
        label = int(row["label"])
        
        if self.include_claim and "claim" in row and pd.notna(row["claim"]):
            claim = str(row["claim"])
            # Format: [CLS] post [SEP] claim [SEP]
            text = f"{post} [SEP] {claim}"
        else:
            # Use only post
            text = post
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class FakeNewsDatasetWithClaims(Dataset):
    """
    Dataset class that handles multiple claims per post.
    
    For each post, retrieves top-k claims and creates multiple samples.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = config.MAX_LENGTH,
        top_k_claims: int = config.TOP_K_CLAIMS
    ):
        """
        Initialize the dataset with multiple claims per post.
        
        Args:
            data: DataFrame with columns 'post', 'label', and optionally 'claims' (list of claim dicts)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            top_k_claims: Number of claims to use per post
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.top_k_claims = top_k_claims
        
        # Expand data to include multiple claims per post
        expanded_data = []
        for idx, row in data.iterrows():
            post = str(row["post"])
            label = int(row["label"])
            
            # Get claims (either from 'claims' column or from 'claim' column)
            if "claims" in row and isinstance(row["claims"], list):
                claims = row["claims"][:top_k_claims]
            elif "claim" in row and pd.notna(row["claim"]):
                claims = [{"claim": str(row["claim"])}]
            else:
                claims = [{"claim": ""}]  # Empty claim if none available
            
            # Create a sample for each claim
            for claim_info in claims:
                expanded_data.append({
                    "post": post,
                    "claim": claim_info.get("claim", ""),
                    "label": label,
                    "similarity": claim_info.get("similarity", None)
                })
        
        self.data = pd.DataFrame(expanded_data).reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        row = self.data.iloc[idx]
        post = str(row["post"])
        claim = str(row["claim"])
        label = int(row["label"])
        
        # Format: [CLS] post [SEP] claim [SEP]
        text = f"{post} [SEP] {claim}" if claim else post
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }














