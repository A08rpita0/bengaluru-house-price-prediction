"""WSGI entry point: gunicorn server.wsgi:app"""

from server.app import create_app

app = create_app()
