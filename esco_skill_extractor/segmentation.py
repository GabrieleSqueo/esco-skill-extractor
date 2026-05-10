import re
from typing import List


def texts_to_tokens(texts: List[str]) -> List[List[str]]:
    return [
        [s for s in re.split(r"\r|\n|\t|\.|\,|\;|and|or", text.strip()) if s]
        for text in texts
    ]
