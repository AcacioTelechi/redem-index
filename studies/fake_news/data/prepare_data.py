"""
Data preparation and loading utilities for fake news detection.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
import chromadb
from embeders import SentenceTransformerEmbeddingFunction
import config


def load_factcheck_dataset() -> pd.DataFrame:
    """
    Load the fact-check dataset from Excel file.
    
    Returns:
        DataFrame with fact-check reports and extracted fake news claims.
    """
    df = pd.read_excel(config.FACTCHECK_DATASET_PATH)
    df = df.dropna(subset=["resumo_2"])  # Keep only rows with rewritten summaries
    return df


def load_annotated_dataset() -> pd.DataFrame:
    """
    Load the annotated posts dataset.
    
    Returns:
        DataFrame with posts, labels (is_fn), and matched fake news claims (resumo_2).
    """
    df = pd.read_excel(config.ANNOTATED_DATASET_PATH)
    # Filter out uncertain labels (is_fn == 0.5) for binary classification
    df = df[df["is_fn"].isin([0.0, 1.0])].copy()
    return df


def get_factcheck_collection():
    """
    Get the ChromaDB collection containing fact-check reports.
    
    Returns:
        ChromaDB collection object.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
    ef = SentenceTransformerEmbeddingFunction()
    
    try:
        collection = client.get_collection(
            name=config.FACTCHECK_COLLECTION,
            embedding_function=ef
        )
        return collection
    except Exception as e:
        print(f"Warning: Could not load fact-check collection: {e}")
        return None


def retrieve_claims_for_post(
    post_text: str,
    collection,
    top_k: int = config.TOP_K_CLAIMS,
    threshold: float = config.SIMILARITY_THRESHOLD
) -> List[Dict]:
    """
    Retrieve top-k most similar fake news claims for a given post.
    
    Args:
        post_text: The social media post text
        collection: ChromaDB collection with fact-check reports
        top_k: Number of claims to retrieve
        threshold: Minimum similarity threshold
        
    Returns:
        List of dictionaries containing claim text, distance, and metadata
    """
    if collection is None:
        return []
    
    try:
        results = collection.query(
            query_texts=[post_text],
            n_results=top_k
        )
        
        claims = []
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            # Convert distance to similarity (assuming cosine distance)
            similarity = 1 - distance
            
            if similarity >= threshold:
                claims.append({
                    "claim": results["documents"][0][i],
                    "similarity": similarity,
                    "distance": distance,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                    "id": results["ids"][0][i]
                })
        
        return claims
    except Exception as e:
        print(f"Error retrieving claims: {e}")
        return []


def prepare_train_data(
    use_retrieval: bool = True,
    use_existing_matches: bool = True
) -> pd.DataFrame:
    """
    Prepare training data by loading annotated posts and matching with fake news claims.
    
    Args:
        use_retrieval: If True, retrieve claims using ChromaDB. If False, use existing resumo_2.
        use_existing_matches: If True and use_retrieval is False, use resumo_2 from dataset.
        
    Returns:
        DataFrame with columns: post, claim, label
    """
    # Load annotated dataset
    df_annotated = load_annotated_dataset()
    
    training_data = []
    
    if use_retrieval:
        # Retrieve claims using ChromaDB
        collection = get_factcheck_collection()
        
        for idx, row in df_annotated.iterrows():
            post = str(row["message"])
            label = int(row["is_fn"])
            
            # Retrieve top-k claims
            claims = retrieve_claims_for_post(post, collection)
            
            if claims:
                # Use the most similar claim (first one)
                training_data.append({
                    "post": post,
                    "claim": claims[0]["claim"],
                    "label": label,
                    "similarity": claims[0]["similarity"]
                })
            elif use_existing_matches and pd.notna(row.get("resumo_2")):
                # Fallback to existing match if retrieval fails
                training_data.append({
                    "post": post,
                    "claim": str(row["resumo_2"]),
                    "label": label,
                    "similarity": None
                })
    else:
        # Use existing matches from dataset
        for idx, row in df_annotated.iterrows():
            post = str(row["message"])
            label = int(row["is_fn"])
            
            if pd.notna(row.get("resumo_2")):
                training_data.append({
                    "post": post,
                    "claim": str(row["resumo_2"]),
                    "label": label,
                    "similarity": None
                })
    
    return pd.DataFrame(training_data)


def create_data_splits(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_SPLIT,
    val_ratio: float = config.VAL_SPLIT,
    test_ratio: float = config.TEST_SPLIT,
    random_state: int = config.RANDOM_SEED,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test splits with stratification to preserve class distribution.
    
    Args:
        df: DataFrame with training data
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by label
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    stratify_col = df["label"] if stratify else None
    df_train, df_temp = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Second split: val vs test
    val_size_relative = val_ratio / (val_ratio + test_ratio)
    stratify_col = df_temp["label"] if stratify else None
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1 - val_size_relative),
        random_state=random_state,
        stratify=stratify_col
    )
    
    return df_train, df_val, df_test


def get_class_weights(df: pd.DataFrame) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        df: DataFrame with 'label' column
        
    Returns:
        Dictionary mapping class indices to weights
    """
    label_counts = df["label"].value_counts().sort_index()
    total = len(df)
    n_classes = len(label_counts)
    
    weights = {}
    for label, count in label_counts.items():
        # Weight inversely proportional to frequency
        weights[int(label)] = total / (n_classes * count)
    
    return weights


if __name__ == "__main__":
    # Test data loading
    print("Loading datasets...")
    df_factcheck = load_factcheck_dataset()
    print(f"Fact-check dataset: {len(df_factcheck)} reports")
    
    df_annotated = load_annotated_dataset()
    print(f"Annotated dataset: {len(df_annotated)} posts")
    print(f"  - Fake news: {(df_annotated['is_fn'] == 1).sum()}")
    print(f"  - Not fake: {(df_annotated['is_fn'] == 0).sum()}")
    
    print("\nPreparing training data...")
    df_train_data = prepare_train_data(use_retrieval=False, use_existing_matches=True)
    print(f"Training pairs: {len(df_train_data)}")
    print(f"  - Fake news: {(df_train_data['label'] == 1).sum()}")
    print(f"  - Not fake: {(df_train_data['label'] == 0).sum()}")
    
    print("\nCreating splits...")
    df_train, df_val, df_test = create_data_splits(df_train_data)
    print(f"Train: {len(df_train)} ({len(df_train)/len(df_train_data)*100:.1f}%)")
    print(f"Val: {len(df_val)} ({len(df_val)/len(df_train_data)*100:.1f}%)")
    print(f"Test: {len(df_test)} ({len(df_test)/len(df_train_data)*100:.1f}%)")
    
    print("\nClass weights:")
    weights = get_class_weights(df_train)
    print(weights)

