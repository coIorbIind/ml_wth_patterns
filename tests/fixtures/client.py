import pytest

from fastapi.testclient import TestClient

from ds_model_with_fastapi.main import get_app
from ds_model_with_fastapi.model.model import get_model

from .model import override_get_model


@pytest.fixture
def client():
    app = get_app()
    app.dependency_overrides[get_model] = override_get_model
    return TestClient(app)
