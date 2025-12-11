"""
Inference utilities for fake news detection model.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
import config
from models.fake_news_model import FakeNewsDetector, ClaimRetriever


class FakeNewsPredictor:
    """
    Predictor class for fake news detection.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_name: str = config.MODEL_NAME,
        device: str = config.DEVICE,
        use_retrieval: bool = True,
        classifier_hidden_dims: Optional[List[int]] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            model_name: Pre-trained BERT model name
            device: Device to run inference on
            use_retrieval: Whether to retrieve claims from ChromaDB
        """
        self.device = torch.device(device)
        self.use_retrieval = use_retrieval
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        if classifier_hidden_dims is None:
            classifier_hidden_dims = getattr(config, "CLASSIFIER_HIDDEN_DIMS", None)

        self.model = FakeNewsDetector(
            model_name=model_name,
            classifier_hidden_dims=classifier_hidden_dims
        )
        
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize claim retriever if needed
        self.claim_retriever = None
        if use_retrieval:
            self.claim_retriever = ClaimRetriever()
    
    def predict(
        self,
        post: str,
        claim: Optional[str] = None,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predict if a post is fake news.
        
        Args:
            post: Social media post text
            claim: Optional fake news claim (if not provided and use_retrieval=True, will retrieve)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction and metadata
        """
        # Retrieve claim if not provided
        if claim is None and self.use_retrieval and self.claim_retriever is not None:
            claims = self.claim_retriever.retrieve(post)
            if claims:
                claim = claims[0]["claim"]
            else:
                claim = ""  # Empty claim if retrieval fails
        
        if claim is None:
            claim = ""
        
        # Format input
        text = f"{post} [SEP] {claim}" if claim else post
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LENGTH,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            fake_probability = probabilities[0, 1].item()
        
        result = {
            "prediction": predicted_class,
            "is_fake": bool(predicted_class == 1),
            "fake_probability": fake_probability,
            "not_fake_probability": probabilities[0, 0].item()
        }
        
        if return_probabilities:
            result["probabilities"] = {
                "not_fake": probabilities[0, 0].item(),
                "fake": probabilities[0, 1].item()
            }
        
        return result
    
    def predict_batch(
        self,
        posts: List[str],
        claims: Optional[List[Optional[str]]] = None,
        return_probabilities: bool = True,
        batch_size: int = config.BATCH_SIZE
    ) -> List[Dict]:
        """
        Predict on a batch of posts.
        
        Args:
            posts: List of post texts
            claims: Optional list of claims (None entries will trigger retrieval if enabled)
            return_probabilities: Whether to return class probabilities
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        if claims is None:
            claims = [None] * len(posts)
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(posts), batch_size):
            batch_posts = posts[i:i + batch_size]
            batch_claims = claims[i:i + batch_size]
            
            # Retrieve claims if needed
            if self.use_retrieval and self.claim_retriever is not None:
                for j, (post, claim) in enumerate(zip(batch_posts, batch_claims)):
                    if claim is None:
                        retrieved_claims = self.claim_retriever.retrieve(post)
                        if retrieved_claims:
                            batch_claims[j] = retrieved_claims[0]["claim"]
                        else:
                            batch_claims[j] = ""
            
            # Prepare texts
            texts = [
                f"{post} [SEP] {claim}" if claim else post
                for post, claim in zip(batch_posts, batch_claims)
            ]
            
            # Tokenize batch
            encoding = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=config.MAX_LENGTH,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                probabilities = F.softmax(logits, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                fake_probabilities = probabilities[:, 1]
            
            # Format results
            for j in range(len(batch_posts)):
                result = {
                    "prediction": predicted_classes[j].item(),
                    "is_fake": bool(predicted_classes[j].item() == 1),
                    "fake_probability": fake_probabilities[j].item(),
                    "not_fake_probability": probabilities[j, 0].item()
                }
                
                if return_probabilities:
                    result["probabilities"] = {
                        "not_fake": probabilities[j, 0].item(),
                        "fake": probabilities[j, 1].item()
                    }
                
                all_results.append(result)
        
        return all_results
    
    def predict_with_aggregation(
        self,
        post: str,
        top_k: int = config.TOP_K_CLAIMS,
        aggregation_method: str = config.AGGREGATION_METHOD
    ) -> Dict:
        """
        Predict using multiple claims and aggregate results.
        
        Args:
            post: Social media post text
            top_k: Number of claims to retrieve
            aggregation_method: "max" or "mean"
            
        Returns:
            Dictionary with aggregated prediction
        """
        if not self.use_retrieval or self.claim_retriever is None:
            return self.predict(post)
        
        # Retrieve multiple claims
        claims = self.claim_retriever.retrieve(post)
        if not claims:
            return self.predict(post)
        
        # Get predictions for each claim
        predictions = []
        for claim_info in claims[:top_k]:
            result = self.predict(post, claim=claim_info["claim"])
            predictions.append(result["fake_probability"])
        
        # Aggregate
        if aggregation_method == "max":
            aggregated_prob = max(predictions)
        elif aggregation_method == "mean":
            aggregated_prob = np.mean(predictions)
        else:
            aggregated_prob = max(predictions)  # Default to max
        
        return {
            "prediction": 1 if aggregated_prob >= 0.5 else 0,
            "is_fake": aggregated_prob >= 0.5,
            "fake_probability": aggregated_prob,
            "num_claims_used": len(predictions),
            "aggregation_method": aggregation_method
        }


def load_model(model_path: Union[str, Path]) -> FakeNewsPredictor:
    """
    Load a trained model for inference.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        FakeNewsPredictor instance
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return FakeNewsPredictor(model_path=model_path)


def predict_from_dataframe(
    df: pd.DataFrame,
    post_column: str = "post",
    claim_column: Optional[str] = None,
    model_path: Optional[Union[str, Path]] = None,
    use_retrieval: bool = True
) -> pd.DataFrame:
    """
    Predict fake news for posts in a DataFrame.
    
    Args:
        df: DataFrame with posts
        post_column: Name of column containing post text
        claim_column: Optional name of column containing claims
        model_path: Path to trained model
        use_retrieval: Whether to retrieve claims
        
    Returns:
        DataFrame with added prediction columns
    """
    predictor = FakeNewsPredictor(
        model_path=model_path,
        use_retrieval=use_retrieval
    )
    
    posts = df[post_column].astype(str).tolist()
    claims = None
    if claim_column and claim_column in df.columns:
        claims = df[claim_column].astype(str).tolist()
        claims = [None if pd.isna(c) or c == "nan" else c for c in claims]
    
    predictions = predictor.predict_batch(posts, claims)
    
    # Add predictions to DataFrame
    result_df = df.copy()
    result_df["is_fake"] = [p["is_fake"] for p in predictions]
    result_df["fake_probability"] = [p["fake_probability"] for p in predictions]
    result_df["prediction"] = [p["prediction"] for p in predictions]
    
    return result_df


if __name__ == "__main__":
    # Example usage
    predictor = FakeNewsPredictor(model_path="./checkpoints/final_model.pt", use_retrieval=True)

    print("Model loaded")
    print(f"Parameters: {sum(p.numel() for p in predictor.model.parameters())}")
    # print model architecture
    # print(predictor.model)
    
    # # Single prediction
    # test_post = "Compartilhem essa notícia urgente sobre política!"
    # result = predictor.predict(test_post)
    # print(f"Post: {test_post}")
    # print(f"Prediction: {'Fake' if result['is_fake'] else 'Not Fake'}")
    # print(f"Probability: {result['fake_probability']:.4f}")
    
    # # Batch prediction
    # test_posts = [
    #     "Notícia importante sobre saúde pública",
    #     "Mais uma fake news sendo desmentida",
    #     "Informação verificada pelos fatos"
    # ]
    # results = predictor.predict_batch(test_posts)
    # for post, result in zip(test_posts, results):
    #     print(f"\nPost: {post}")
    #     print(f"Prediction: {'Fake' if result['is_fake'] else 'Not Fake'}")
    #     print(f"Probability: {result['fake_probability']:.4f}")


    # # Predict with aggregation from dataframe
    # df = pd.read_excel("./df_train.xlsx")
    # df = df[df["analisado"] == 0]
    # result_df = predict_from_dataframe(df, model_path="./checkpoints/final_model.pt")
    # result_df.to_excel("./df_train_predictions.xlsx", index=False)
