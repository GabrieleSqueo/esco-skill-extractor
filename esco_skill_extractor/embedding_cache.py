import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import List

import torch
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


def load_or_build_embeddings(
    cache_file: Path,
    texts: List[str],
    model: SentenceTransformer,
    device: str,
) -> torch.Tensor:
    if cache_file.exists():
        log.info(
            "Cache embedding trovata, caricamento da %s",
            cache_file.name,
        )
        with open(cache_file, "rb") as f:
            return pickle.load(f).to(device)
    if "skill" in cache_file.name and "occupation" not in cache_file.name:
        label = "Skill"
    else:
        label = "Occupation"
    log.info(
        "%s: cache assente (%s), calcolo embedding da zero…",
        label,
        cache_file.name,
    )
    vectors = model.encode(
        texts,
        device=device,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(cache_file, "wb") as f:
            pickle.dump(vectors, f)
    log.info("%s: embedding salvati in %s", label, cache_file.name)
    return vectors


def clear_embedding_cache_files(data_dir: Path) -> None:
    for name in ("skill_embeddings.bin", "occupation_embeddings.bin"):
        path = data_dir / name
        if path.exists():
            log.info("Rimossa cache embedding: %s", path.name)
            os.remove(path)
