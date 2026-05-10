import logging
from typing import Union, List, Dict

import torch

from . import datasets as esco_data
from .embedding_cache import clear_embedding_cache_files, load_or_build_embeddings
from .esco_client import fetch_skill_labels_en
from .models import load_sentence_transformer
from .paths import package_data_dir
from .similarity_matching import extract_entity_ids_per_text

log = logging.getLogger(__name__)


class SkillExtractor:

    def __init__(
        self,
        model,
        skills_threshold,
        occupation_threshold,
        device: Union[str, None] = None,
    ):
        """
        Loads the models, skills and skill embeddings.

        Args:
            skills_threshold (float, optional): The similarity threshold for skill comparisons. Increase it to be more harsh. Defaults to 0.45. Range: [0, 1].
            occupation_threshold (float, optional): The similarity threshold for occupation comparisons. Increase it to be more harsh. Defaults to 0.55. Range: [0, 1].
            device (Union[str, None], optional): The device where the model will run. Defaults to "cuda" if available, otherwise "cpu".
        """

        self.model_name = model
        self.skills_threshold = skills_threshold
        self.occupation_threshold = occupation_threshold
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._data_dir = package_data_dir()
        log.info("Inizializzazione SkillExtractor (device=%s, data_dir=%s)", self.device, self._data_dir)

        log.info("Caricamento SentenceTransformer: %s", self.model_name)
        self._model = load_sentence_transformer(self.model_name, self.device)
        log.info("SentenceTransformer caricato")

        self._skills = esco_data.load_skills_table(self._data_dir)
        self._skill_ids = self._skills["id"].to_numpy()
        log.info("Tabella skill: %d righe", len(self._skills))

        self._occupations = esco_data.load_occupations_table(self._data_dir)
        self._occupation_ids = self._occupations["id"].to_numpy()
        log.info("Tabella occupazioni: %d righe", len(self._occupations))

        log.info("Embedding skill (cache o calcolo)…")
        self._skill_embeddings = load_or_build_embeddings(
            self._data_dir / "skill_embeddings.bin",
            self._skills["description"].to_list(),
            self._model,
            self.device,
        )
        log.info("Embedding skill pronti (shape=%s)", tuple(self._skill_embeddings.shape))

        log.info("Embedding occupazioni (cache o calcolo)…")
        self._occupation_embeddings = load_or_build_embeddings(
            self._data_dir / "occupation_embeddings.bin",
            self._occupations["description"].to_list(),
            self._model,
            self.device,
        )
        log.info("Embedding occupazioni pronti (shape=%s)", tuple(self._occupation_embeddings.shape))

    @staticmethod
    def remove_embeddings():
        """
        Removes the skill and occupation embeddings from the disk if the model changed.
        """
        clear_embedding_cache_files(package_data_dir())

    def get_skills(self, texts: List[str]) -> List[List[Dict[str, str]]]:
        """
        Extracts ESCO skills from the texts.

        Returns:
            List[List[dict]]: For each text, a list of {"uri", "label_en"} from the ESCO API.
        """

        log.debug("Similarity matching skill (soglia=%s)…", self.skills_threshold)
        uri_groups = extract_entity_ids_per_text(
            texts,
            self._model,
            self.device,
            self._skill_ids,
            self._skill_embeddings,
            self.skills_threshold,
        )
        all_uris = [u for group in uri_groups for u in group]
        log.info(
            "Similarity matching skill: trovati %d URI skill (unici: %d) su %d testo/i",
            len(all_uris),
            len(set(all_uris)),
            len(uri_groups),
        )

        unique_uris = set(all_uris)
        if unique_uris:
            log.info(
                "Recupero etichette EN da API ESCO per %d URI unici…",
                len(unique_uris),
            )
            labels = fetch_skill_labels_en(all_uris)
            missing = sum(1 for u in unique_uris if not labels.get(u))
            if missing:
                log.warning("API ESCO: %d URI senza etichetta in risposta", missing)
            else:
                log.info("API ESCO: etichette ricevute per tutti gli URI richiesti")
        else:
            log.info("Nessun URI skill estratto: salto la chiamata API ESCO")
            labels = {}

        return [
            [{"uri": u, "label_en": labels.get(u, "")} for u in group]
            for group in uri_groups
        ]

    def get_occupations(self, texts: List[str]) -> List[List[str]]:
        """
        Extracts ESCO occupations from the texts.

        Returns:
            List[List[str]]: IDs (URIs) of occupations for each text.
        """

        log.debug("Similarity matching occupazioni (soglia=%s)…", self.occupation_threshold)
        matched = extract_entity_ids_per_text(
            texts,
            self._model,
            self.device,
            self._occupation_ids,
            self._occupation_embeddings,
            self.occupation_threshold,
        )
        flat = [u for g in matched for u in g]
        log.info(
            "Similarity matching occupazioni: %d URI (unici: %d) su %d testo/i",
            len(flat),
            len(set(flat)),
            len(matched),
        )
        return matched
