import re
import unidecode
from nltk.corpus import stopwords



def text_transform(text: str) -> str:
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
