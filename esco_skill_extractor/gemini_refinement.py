import json
import logging
from typing import Dict, List

log = logging.getLogger(__name__)


def refine_skills_with_gemini(
    source_text: str,
    candidates: List[Dict[str, str]],
    *,
    api_key: str,
    model_name: str,
) -> List[Dict[str, str]]:
    """
    Ask Gemini to keep only skills from ``candidates`` that are clearly supported by
    ``source_text``. URIs are whitelisted after the call so nothing outside
    ``candidates`` can appear. Labels are always taken from ``candidates``, not the model.
    """
    if not candidates:
        log.info("Gemini: nessun candidato da filtrare, salto la chiamata")
        return []

    log.info(
        "Gemini: chiamata modello=%s per filtrare %d skill candidate",
        model_name,
        len(candidates),
    )

    from google import genai as google_genai
    from google.genai import types

    client = google_genai.Client(api_key=api_key)

    by_uri: Dict[str, str] = {}
    for c in candidates:
        uri = c.get("uri")
        if not uri:
            continue
        by_uri[uri] = c.get("label_en", "") or ""

    candidates_min = [{"uri": u, "label_en": by_uri[u]} for u in by_uri]
    payload = json.dumps(candidates_min, ensure_ascii=False)

    prompt = f"""You validate ESCO skill suggestions against a source document.

SOURCE TEXT:
---
{source_text}
---

CANDIDATES (JSON). You may ONLY select skills from this list. Copy each "uri" exactly as given. Do not add skills, URIs, or labels that are not in CANDIDATES.
{payload}

Instructions:
- Pick only skills the source text clearly supports (explicit wording, clear paraphrase, or unambiguous professional context).
- Exclude skills that are vague, generic without support, or only loosely related.
- If none are justified, return an empty list.
- Do not invent or infer skills beyond the text and the candidate list.

Return ONLY valid JSON (no markdown fences) in this exact shape:
{{"selected":[{{"uri":"<exact uri from CANDIDATES>","reason":"<one short sentence in English explaining the match>"}}]}}
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        raw = response.text
        if not raw or not raw.strip():
            log.warning(
                "Gemini: risposta vuota, restituisco tutte le %d candidate senza filtro",
                len(candidates),
            )
            return _copy_candidates(candidates)

        data = json.loads(raw)
        selected = data.get("selected")
        if not isinstance(selected, list):
            log.warning(
                "Gemini: JSON senza lista 'selected' valida, restituisco tutte le candidate",
            )
            return _copy_candidates(candidates)

        out: List[Dict[str, str]] = []
        seen: set[str] = set()
        for item in selected:
            if not isinstance(item, dict):
                continue
            uri = item.get("uri")
            if not isinstance(uri, str) or uri not in by_uri or uri in seen:
                continue
            seen.add(uri)
            reason = item.get("reason")
            if reason is not None and not isinstance(reason, str):
                reason = str(reason)
            row: Dict[str, str] = {
                "uri": uri,
                "label_en": by_uri[uri],
            }
            if reason:
                row["reason"] = reason
            out.append(row)
        log.info(
            "Gemini: filtro applicato — %d skill confermate su %d candidate",
            len(out),
            len(candidates),
        )
        return out
    except Exception as e:
        log.warning(
            "Gemini: errore durante la chiamata (%s), restituisco tutte le candidate",
            e,
            exc_info=log.isEnabledFor(logging.DEBUG),
        )
        return _copy_candidates(candidates)


def _copy_candidates(candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "uri": c["uri"],
            "label_en": c.get("label_en", ""),
        }
        for c in candidates
        if c.get("uri")
    ]
