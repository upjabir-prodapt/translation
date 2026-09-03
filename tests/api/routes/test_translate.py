import io
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from src.api.core.security import AuthenticatedUser
from src.api.core.security import get_current_user_context
from src.api.dependencies import get_translation_handler
from src.api.main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_current_user_context] = lambda: AuthenticatedUser(
        email="test@example.com", business_unit="bu1", organization="org1"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestTranslateRoutes:
    def test_submit_translation(self, client):
        mock_handler = AsyncMock()
        mock_handler.submit_translations.return_value = {
            "batch_id": "batch1",
            "jobs": [
                {
                    "job_id": "job1",
                    "target_language": "fr",
                    "status": "queued",
                    "status_url": "/api/v1/translate/job1",
                }
            ],
        }
        app.dependency_overrides[get_translation_handler] = lambda: mock_handler

        file_content = b"%PDF-1.4 test"
        files = {"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
        data = {
            "target_languages": ["French"],
            "source_language": "English",
            "domain": "legal",
        }

        response = client.post("/api/v1/translate", files=files, data=data)
        assert response.status_code == 202
        assert response.json()["batch_id"] == "batch1"
