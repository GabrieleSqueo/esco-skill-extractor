import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from waitress import serve

from . import SkillExtractor
from .gemini_refinement import refine_skills_with_gemini

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ----------- Parse command line arguments -----------
parser = argparse.ArgumentParser(
    description="ESCO Skill Extractor: Extract ESCO skills and ISCO occupations from any text."
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="paraphrase-multilingual-mpnet-base-v2",
    help="Model to use for skill extraction. Default is 'all-MiniLM-L6-v2'.",
)
parser.add_argument(
    "--skill_threshold",
    "-s",
    type=float,
    default=0.55,
    help="Threshold for skill extraction. Default is 0.6",
)
parser.add_argument(
    "--occupation_threshold",
    "-o",
    type=float,
    default=0.55,
    help="Threshold for occupation extraction. Default is 0.55.",
)
parser.add_argument(
    "--device",
    "-d",
    type=str,
    default=None,
    help="Device to use for computations. Default is cuda if available, else CPU.",
)
parser.add_argument(
    "--host",
    "-c",
    type=str,
    default="localhost",
    help="Host to bind the server to. Default is localhost",
)
parser.add_argument(
    "--port",
    "-p",
    type=int,
    default=8001,
    help="Port to bind the server to. Default is 8000",
)
parser.add_argument(
    "--gemini-model",
    type=str,
    default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    help="Gemini model when GEMINI_API_KEY is set. Default: gemini-2.5-flash or GEMINI_MODEL env.",
)
parser.add_argument(
    "--log-level",
    type=str,
    default=os.environ.get("LOG_LEVEL", "INFO"),
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Verbosità dei log (default: INFO, oppure variabile LOG_LEVEL).",
)

args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log.info(
    "Avvio server (log=%s, modello embedding=%s, device da SkillExtractor)",
    args.log_level,
    args.model,
)

# ----------- Initialize the skill extractor -----------
extractor = SkillExtractor(
    model=args.model,
    skills_threshold=args.skill_threshold,
    occupation_threshold=args.occupation_threshold,
    device=args.device,
)

# ----------- Define the Flask app -----------
BASE_DIR = __file__.replace("__main__.py", "")
app = Flask(
    __name__,
    template_folder=BASE_DIR + "templates",
    static_folder=BASE_DIR + "static",
)


@app.after_request
def handle_options(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
    return response


@app.route("/")
def index():
    return render_template("index.html", host=args.host, port=args.port)


@app.route("/extract-skills", methods=["POST"])
def extract():
    texts = request.json
    if not isinstance(texts, list):
        return jsonify({"error": "Expected a JSON array of strings (one or more texts)."}), 400

    log.info("POST /extract-skills: ricevuti %d testo/i", len(texts))
    log.info("Step 1/3: estrazione skill (similarity + API ESCO etichette)…")
    raw = extractor.get_skills(texts)
    n_cand = sum(len(x) for x in raw)
    log.info(
        "Step 1/3: completato — %d skill candidate totali su %d testo/i",
        n_cand,
        len(raw),
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.info("Step 2/3: saltato — GEMINI_API_KEY non impostata, nessun filtro LLM")
        log.info("Step 3/3: risposta pronta (solo estrazione)")
        return jsonify(raw)

    log.info(
        "Step 2/3: filtro Gemini (modello=%s) sui candidati per testo…",
        args.gemini_model,
    )
    refined = []
    for idx, (text, skills) in enumerate(zip(texts, raw)):
        log.info(
            "  Testo %d/%d: invio a Gemini %d candidat%s",
            idx + 1,
            len(texts),
            len(skills),
            "o" if len(skills) == 1 else "i",
        )
        refined.append(
            refine_skills_with_gemini(
                text or "",
                skills,
                api_key=api_key,
                model_name=args.gemini_model,
            )
        )
    n_out = sum(len(x) for x in refined)
    log.info(
        "Step 2/3: completato — %d skill dopo filtro Gemini (prima: %d)",
        n_out,
        n_cand,
    )
    log.info("Step 3/3: risposta pronta")
    return jsonify(refined)


@app.route("/extract-occupations", methods=["POST"])
def extract_occupations():
    payload = request.json
    n = len(payload) if isinstance(payload, list) else 0
    log.info("POST /extract-occupations: %d testo/i", n)
    log.info("Estrazione occupazioni (similarity)…")
    out = extractor.get_occupations(payload)
    log.info(
        "Estrazione occupazioni completata — %d occupazioni totali",
        sum(len(x) for x in out),
    )
    return jsonify(out)


# 20 minutes timeout, our model might take a while to infer for really big loads
log.info("SkillExtractor pronto. Server in ascolto su http://%s:%s", args.host, args.port)
serve(
    app,
    host=args.host,
    port=args.port,
    channel_timeout=12000,
)
