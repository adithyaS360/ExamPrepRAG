from io import BytesIO

import pytest

from app import create_app


@pytest.fixture()
def client():
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