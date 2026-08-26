from __future__ import annotations
from flask import Blueprint, current_app, jsonify, render_template, request
from .service import RagService
from .store import FaissStore

bp = Blueprint("api", __name__)


def service() -> RagService:
    if "RAG_SERVICE" not in current_app.extensions:
        current_app.extensions["RAG_SERVICE"] = RagService(current_app.config["SETTINGS"])
    return current_app.extensions["RAG_SERVICE"]


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/health")
def health():
    settings = current_app.config["SETTINGS"]
    return {"status": "ok", "index_ready": FaissStore(settings.index_dir).ready()}


@bp.post("/query")
def query():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "JSON body must include a non-empty 'question'"}), 400
    try:
        return jsonify(service().query(question, bool(body.get("force_fallback"))))
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
