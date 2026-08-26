"""VTU RAG Flask application factory."""
from flask import Flask

from .config import Settings
from .routes import bp


def create_app() -> Flask:
    # Flask's app factory is analogous to a compact Spring Boot configuration class.
    app = Flask(__name__)
    settings = Settings.from_env()
    app.config["SETTINGS"] = settings
    app.register_blueprint(bp)
    return app


app = create_app()

