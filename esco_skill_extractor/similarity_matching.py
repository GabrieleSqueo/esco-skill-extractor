import logging
from itertools import chain
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from .segmentation import texts_to_tokens

log = logging.getLogger(__name__)


def extract_entity_ids_per_text(
    texts: List[str],
    model: SentenceTransformer,
    device: str,
    entity_ids: np.ndarray,
    entity_embeddings: torch.Tensor,
    threshold: float,
) -> List[List[str]]:
    if all(not text for text in texts):
        return [[] for _ in texts]

    token_groups = texts_to_tokens(texts)
    tokens = list(chain.from_iterable(token_groups))

    if not tokens:
        return [[] for _ in texts]

    log.debug(
        "Similarity: encoding di %d segmenti di testo (soglia=%s)",
        len(tokens),
        threshold,
    )
    sentence_embeddings = model.encode(
        tokens,
        device=device,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )

    similarity_matrix = util.dot_score(sentence_embeddings, entity_embeddings)
    most_similar_entity_scores, most_similar_entity_indices = torch.max(
        similarity_matrix, dim=-1
    )

    entity_ids_per_text = []
    sentences = 0

    for text in token_groups:
        sentences_in_text = len(text)

        most_similar_entity_indices_text = most_similar_entity_indices[
            sentences : sentences + sentences_in_text
        ]
        most_similar_entity_scores_text = most_similar_entity_scores[
            sentences : sentences + sentences_in_text
        ]

        most_similar_entity_indices_text = (
            most_similar_entity_indices_text[
                torch.nonzero(most_similar_entity_scores_text > threshold)
            ]
            .squeeze(dim=-1)
            .unique()
            .tolist()
        )

        entity_ids_per_text.append(
            np.take(entity_ids, most_similar_entity_indices_text).tolist()
        )

        sentences += sentences_in_text

    n_match = sum(len(g) for g in entity_ids_per_text)
    log.debug(
        "Similarity: completata — %d entità matchate (soglia=%s)",
        n_match,
        threshold,
    )
    return entity_ids_per_text
