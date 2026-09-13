from io import BytesIO

import pytest

from app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # RagService._ensure_index() and the /upload route both read/write
    # "data/raw" and "data/fixtures" as paths relative to the current
    # working directory. Without this chdir, running the test suite writes
    # test fixtures (including test_upload_rejects_path_traversal_filename's
    # "evil.pdf") straight into the real project's data/raw/ directory,
    # which then breaks real ingestion the next time someone runs the app.
    monkeypatch.chdir(tmp_path)
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_query_requires_question(client):
    response = client.post("/query", json={})
    assert response.status_code == 400


def test_index_serves_research_interface(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Ask the" in response.data


def test_upload_disabled_without_key(client):
    response = client.post("/upload")
    assert response.status_code == 403


def test_upload_requires_file(client, monkeypatch):
    monkeypatch.setenv("UPLOAD_API_KEY", "test-key")

    response = client.post(
        "/upload",
        headers={"X-Upload-Key": "test-key"},
    )

    assert response.status_code == 400


def test_upload_rejects_path_traversal_filename(client, monkeypatch):
    monkeypatch.setenv("UPLOAD_API_KEY", "test-key")

    data = {
        "file": (BytesIO(b"not a real pdf"), "../../evil.pdf")
    }

    response = client.post(
        "/upload",
        data=data,
        headers={"X-Upload-Key": "test-key"},
        content_type="multipart/form-data",
    )

    assert response.status_code in (400, 500)