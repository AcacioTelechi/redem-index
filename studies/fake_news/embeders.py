import re
import unidecode
from ollama import Client as OllamaClient
from typing import Callable, List, Sequence
from chromadb.api.types import EmbeddingFunction
from nltk.corpus import stopwords


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self, model: str = "mxbai-embed-large", host: str = "http://127.0.0.1:11434"
    ):
        self.model = model
        self.client = OllamaClient(host=host)

    def __call__(self, inputs: Sequence[str]) -> List[List[float]]:
        if isinstance(inputs, str):
            inputs = [inputs]

        inputs = [self.text_transform(input) for input in inputs]
        resp = self.client.embed(model=self.model, input=list(inputs))
        return resp["embeddings"]

    def text_transform(self, text: str) -> str:
        """
        - Normalizing whitespace and Unicode (to avoid different representations of the same token, e.g., café vs café).
        - Lowercasing (optional).
        - Remove noie (html tagas)
        """

        text = text.lower()
        text = unidecode.unidecode(text)

        #remove html tags
        text = re.sub(r'<[^>]*>', '', text)

        # remove all non-alphanumeric characters but accept accents
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

        return text
