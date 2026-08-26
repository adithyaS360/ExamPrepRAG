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
