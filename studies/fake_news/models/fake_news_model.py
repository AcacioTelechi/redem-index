"""
Model architectures for fake news detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Tuple
import chromadb
from embeders import SentenceTransformerEmbeddingFunction
import config


class ClaimRetriever:
    """
    Retrieves fake news claims from ChromaDB for a given post.
    """

    def __init__(
        self,
        collection_name: str = config.FACTCHECK_COLLECTION,
        chroma_db_path: str = str(config.CHROMA_DB_PATH),
        top_k: int = config.TOP_K_CLAIMS,
        threshold: float = config.SIMILARITY_THRESHOLD,
    ):
        """
        Initialize the claim retriever.

        Args:
            collection_name: Name of the ChromaDB collection
            chroma_db_path: Path to ChromaDB database
            top_k: Number of claims to retrieve
            threshold: Minimum similarity threshold
        """
        self.top_k = top_k
        self.threshold = threshold
        self.collection = None

        try:
            client = chromadb.PersistentClient(path=chroma_db_path)
            ef = SentenceTransformerEmbeddingFunction()
            self.collection = client.get_collection(
                name=collection_name, embedding_function=ef
            )
        except Exception as e:
            print(f"Warning: Could not load ChromaDB collection: {e}")

    def retrieve(self, post_text: str) -> List[Dict]:
        """
        Retrieve top-k claims for a given post.

        Args:
            post_text: Social media post text

        Returns:
            List of dictionaries with claim text, similarity, and metadata
        """
        if self.collection is None:
            return []

        try:
            results = self.collection.query(
                query_texts=[post_text], n_results=self.top_k
            )

            claims = []
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity

                if similarity >= self.threshold:
                    claims.append(
                        {
                            "claim": results["documents"][0][i],
                            "similarity": similarity,
                            "distance": distance,
                            "metadata": (
                                results["metadatas"][0][i]
                                if results["metadatas"]
                                else None
                            ),
                            "id": results["ids"][0][i],
                        }
                    )

            return claims
        except Exception as e:
            print(f"Error retrieving claims: {e}")
            return []


class ClassifierHead(nn.Module):
    """
    Multi-layer perceptron classifier head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        num_labels: int,
        dropout: float,
    ):
        super(ClassifierHead, self).__init__()
        self.classifier = self._build_classifier_head(
            input_dim, hidden_dims, num_labels, dropout
        )

    def _build_classifier_head(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        num_labels: int,
        dropout: float,
    ) -> nn.Sequential:
        """
        Construct a multi-layer perceptron classifier head.
        """
        if not hidden_dims:
            hidden_dims = [input_dim, max(input_dim // 2, num_labels * 2)]

        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_labels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.classifier(x)


class FakeNewsDetector(nn.Module):
    """
    Cross-encoder model for fake news detection.

    Takes a post and a fake news claim, outputs probability of being fake.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        dropout: float = config.DROPOUT,
        num_labels: int = 2,
        classifier_hidden_dims: Optional[List[int]] = None,
    ):
        """
        Initialize the fake news detector.

        Args:
            model_name: Pre-trained BERT model name
            dropout: Dropout rate
            num_labels: Number of output labels (2 for binary classification)
            classifier_hidden_dims: Hidden layer sizes for the classifier head MLP
        """
        super(FakeNewsDetector, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = ClassifierHead(
            input_dim=self.bert.config.hidden_size,
            hidden_dims=classifier_hidden_dims,
            num_labels=num_labels,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size] (optional)

        Returns:
            Dictionary with logits and optionally loss
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)


        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Classification head
        logits = self.classifier(pooled_output)

        output_dict = {"logits": logits}

        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            output_dict["loss"] = loss

        return output_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal loss focuses on hard examples and down-weights easy examples.
    """

    def __init__(
        self,
        alpha: float = config.FOCAL_LOSS_ALPHA,
        gamma: float = config.FOCAL_LOSS_GAMMA,
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.

        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Target labels [batch_size]

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_model(
    model_name: str = config.MODEL_NAME,
    # use_aggregation: bool = False,
    use_focal_loss: bool = config.USE_FOCAL_LOSS,
    classifier_hidden_dims: Optional[List[int]] = None,
) -> Tuple[nn.Module, Optional[nn.Module]]:
    """
    Factory function to create model and loss function.

    Args:
        model_name: Pre-trained model name
        # use_aggregation: Whether to use aggregation model
        use_focal_loss: Whether to use focal loss
        classifier_hidden_dims: Hidden layer sizes for the classifier MLP

    Returns:
        Tuple of (model, loss_fn)
    """

    model = FakeNewsDetector(
        model_name=model_name, classifier_hidden_dims=classifier_hidden_dims
    )

    loss_fn = FocalLoss() if use_focal_loss else None

    return model, loss_fn
