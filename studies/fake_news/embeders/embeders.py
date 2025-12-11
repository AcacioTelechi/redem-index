# from ollama import Client as OllamaClient
from typing import List, Sequence
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def __call__(self, inputs: Sequence[str]) -> List[List[float]]:
        if isinstance(inputs, str):
            inputs = [inputs]

        return self.model.encode(inputs)

    def embed_query(self, input: str) -> List[float]:
        return self.model.encode(input)
