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


@bp.post("/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400
    filename = file.filename
    if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".txt")):
        return jsonify({"error": "Only .pdf and .txt files are supported"}), 400
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_path = raw_dir / filename
    file.save(save_path)

    from .ingest import ingest
    try:
        stats = ingest(raw_dir)
        service().cache.clear()
        return jsonify({"message": f"Successfully uploaded and indexed '{filename}'!", "stats": stats})
    except Exception as error:
        return jsonify({"error": str(error)}), 500


