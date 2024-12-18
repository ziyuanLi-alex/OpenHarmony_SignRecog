# src/web/__init__.py
"""
Web server and API endpoints.
"""
from .server import create_app
from .routes import register_routes

__all__ = ['create_app', 'register_routes']