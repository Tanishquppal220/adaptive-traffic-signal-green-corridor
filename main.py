"""Adaptive Traffic Signal & Green Corridor — CLI entrypoint.

Launches the Flask-based GUI that serves both the live camera feed
and the image-upload test page.

Usage:
    python main.py              # starts the web server on 127.0.0.1:5000
"""

from config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT
from gui import create_app


def main() -> None:
    """Create and run the Flask application."""
    app = create_app()
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        use_reloader=FLASK_DEBUG,
    )


if __name__ == "__main__":
    main()
