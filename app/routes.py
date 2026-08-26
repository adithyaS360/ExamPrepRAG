from __future__ import annotations

import os
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

from .service import RagService
from .store import FaissStore

bp = Blueprint("api", __name__)


def service() -> RagService:
    if "RAG_SERVICE" not in current_app.extensions:
        current_app.extensions["RAG_SERVICE"] = RagService(
            current_app.config["SETTINGS"]
        )
    return current_app.extensions["RAG_SERVICE"]


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/health")
def health():
    settings = current_app.config["SETTINGS"]
    return {
        "status": "ok",
        "index_ready": FaissStore(settings.index_dir).ready(),
    }


@bp.post("/query")
def query():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()

    if not question:
        return jsonify({
            "error": "JSON body must include a non-empty 'question'"
        }), 400

    try:
        return jsonify(
            service().query(
                question,
                bool(body.get("force_fallback"))
            )
        )
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    except Exception:
        current_app.logger.exception("Unhandled error in /query")
        return jsonify({"error": "Internal server error"}), 500


def _check_upload_auth() -> bool:
    """Require a shared-secret header for /upload.

    Set UPLOAD_API_KEY in the environment to enable uploads.
    If unset, uploads are disabled entirely.
    """
    expected = os.getenv("UPLOAD_API_KEY")

    if not expected:
        return False

    provided = request.headers.get("X-Upload-Key", "")
    return provided == expected


@bp.post("/upload")
def upload():
    if not _check_upload_auth():
        return jsonify({
            "error": "Uploads are disabled or unauthorized"
        }), 403

    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)

    if not filename or not (
        filename.lower().endswith(".pdf")
        or filename.lower().endswith(".txt")
    ):
        return jsonify({
            "error": "Only .pdf and .txt files are supported"
        }), 400

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    save_path = (raw_dir / filename).resolve()

    if raw_dir.resolve() not in save_path.parents:
        return jsonify({"error": "Invalid filename"}), 400

    file.save(save_path)

    from .ingest import ingest

    try:
        stats = ingest(raw_dir)
        service().cache.clear()

        return jsonify({
            "message": f"Successfully uploaded and indexed '{filename}'!",
            "stats": stats,
        })

    except Exception:
        current_app.logger.exception("Unhandled error in /upload")
        return jsonify({
            "error": "Failed to ingest uploaded file"
        }), 500