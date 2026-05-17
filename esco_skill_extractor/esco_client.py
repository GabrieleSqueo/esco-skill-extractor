import json
import logging
import math
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

ESCO_API_BASE = "https://ec.europa.eu/esco/api/resource"
URI_BATCH = 40


def _fetch_labels_en(resource: str, uris: List[str]) -> Dict[str, str]:
    """Resolve English preferred labels (title) for concept URIs via the public ESCO API."""
    labels: Dict[str, str] = {}
    unique = list(dict.fromkeys(uris))
    if not unique:
        log.debug("API ESCO %s: nessun URI da risolvere", resource)
        return labels

    api_url = f"{ESCO_API_BASE}/{resource}"
    n_batch = math.ceil(len(unique) / URI_BATCH)
    log.info(
        "API ESCO %s: richiesta etichette EN per %d URI (%d batch da max %d URI)",
        resource,
        len(unique),
        n_batch,
        URI_BATCH,
    )

    for i in range(0, len(unique), URI_BATCH):
        batch = unique[i : i + URI_BATCH]
        batch_num = i // URI_BATCH + 1
        log.debug(
            "API ESCO %s: batch %d/%d (%d URI)",
            resource,
            batch_num,
            n_batch,
            len(batch),
        )
        query = urlencode([("uris", u) for u in batch] + [("language", "en")])
        url = f"{api_url}?{query}"
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
        for concept in embedded.values():
            if not isinstance(concept, dict):
                continue
            uri = concept.get("uri")
            if not uri:
                continue
            title = concept.get("title")
            if title is not None:
                labels[uri] = title
        log.debug(
            "API ESCO %s: batch %d/%d completato (%d concetti in risposta)",
            resource,
            batch_num,
            n_batch,
            len(embedded),
        )
    log.info("API ESCO %s: completata — %d etichette in mappa", resource, len(labels))
    return labels


def fetch_skill_labels_en(uris: List[str]) -> Dict[str, str]:
    return _fetch_labels_en("skill", uris)


def fetch_occupation_labels_en(uris: List[str]) -> Dict[str, str]:
    return _fetch_labels_en("occupation", uris)


def _normalize_link_list(value) -> List[dict]:
    if not value:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def fetch_skills_for_occupations(occupation_uris: List[str]) -> List[Dict[str, str]]:
    """
    Return essential and optional skills linked to the given occupation URIs
    (from ESCO occupation resource ``hasEssentialSkill`` / ``hasOptionalSkill``).
    """
    unique = list(dict.fromkeys(u for u in occupation_uris if u))
    if not unique:
        log.debug("API ESCO occupation→skill: nessuna occupazione in input")
        return []

    skills: Dict[str, str] = {}
    api_url = f"{ESCO_API_BASE}/occupation"
    n_batch = math.ceil(len(unique) / URI_BATCH)
    log.info(
        "API ESCO occupation→skill: recupero skill per %d occupazioni (%d batch)",
        len(unique),
        n_batch,
    )

    for i in range(0, len(unique), URI_BATCH):
        batch = unique[i : i + URI_BATCH]
        batch_num = i // URI_BATCH + 1
        query = urlencode([("uris", u) for u in batch] + [("language", "en")])
        url = f"{api_url}?{query}"
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
        batch_skills = 0
        for occupation in embedded.values():
            if not isinstance(occupation, dict):
                continue
            links = occupation.get("_links") or {}
            for rel in ("hasEssentialSkill", "hasOptionalSkill"):
                for link in _normalize_link_list(links.get(rel)):
                    uri = link.get("uri")
                    title = link.get("title")
                    if not uri:
                        continue
                    if uri not in skills:
                        batch_skills += 1
                    skills[uri] = title or skills.get(uri, "")
        log.debug(
            "API ESCO occupation→skill: batch %d/%d — %d skill nuove in questo batch",
            batch_num,
            n_batch,
            batch_skills,
        )

    result = [{"uri": u, "label_en": skills[u]} for u in skills]
    log.info(
        "API ESCO occupation→skill: completata — %d skill uniche da occupazioni",
        len(result),
    )
    return result
