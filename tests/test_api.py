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


def test_upload_requires_file(client):
    response = client.post("/upload")
    assert response.status_code == 400

