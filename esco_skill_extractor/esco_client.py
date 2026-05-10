import json
import logging
import math
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

ESCO_API_SKILL = "https://ec.europa.eu/esco/api/resource/skill"
SKILL_URI_BATCH = 40


def fetch_skill_labels_en(uris: List[str]) -> Dict[str, str]:
    """Resolve English preferred labels (title) for skill URIs via the public ESCO API."""
    labels: Dict[str, str] = {}
    unique = list(dict.fromkeys(uris))
    if not unique:
        log.debug("API ESCO skill: nessun URI da risolvere")
        return labels

    n_batch = math.ceil(len(unique) / SKILL_URI_BATCH)
    log.info(
        "API ESCO skill: richiesta etichette EN per %d URI (%d batch da max %d URI)",
        len(unique),
        n_batch,
        SKILL_URI_BATCH,
    )

    for i in range(0, len(unique), SKILL_URI_BATCH):
        batch = unique[i : i + SKILL_URI_BATCH]
        batch_num = i // SKILL_URI_BATCH + 1
        log.debug(
            "API ESCO skill: batch %d/%d (%d URI)",
            batch_num,
            n_batch,
            len(batch),
        )
        query = urlencode([("uris", u) for u in batch] + [("language", "en")])
        url = f"{ESCO_API_SKILL}?{query}"
        req = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "esco-skill-extractor (Python urllib)",
            },
        )
        with urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
        embedded = payload.get("_embedded") or {}
        for skill in embedded.values():
            if not isinstance(skill, dict):
                continue
            uri = skill.get("uri")
            if not uri:
                continue
            title = skill.get("title")
            if title is not None:
                labels[uri] = title
        log.debug(
            "API ESCO skill: batch %d/%d completato (%d concetti in risposta)",
            batch_num,
            n_batch,
            len(embedded),
        )
    log.info("API ESCO skill: completata — %d etichette in mappa", len(labels))
    return labels
