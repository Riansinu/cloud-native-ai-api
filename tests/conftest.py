"""Pytest fixtures for the Cloud-Native AI Prediction API tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """Create a TestClient with lifespan events so the model loads on startup."""
    with TestClient(app) as c:
        yield c
