from __future__ import annotations

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    from .routes import bp as gui_bp

    app.register_blueprint(gui_bp)
    return app
