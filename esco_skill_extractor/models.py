import logging
import warnings

from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


def load_sentence_transformer(model_name: str, device: str) -> SentenceTransformer:
    log.debug("Download/caricamento pesi SentenceTransformer se necessario…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SentenceTransformer(model_name, device=device)
    log.debug("SentenceTransformer pronto su device=%s", device)
    return model
