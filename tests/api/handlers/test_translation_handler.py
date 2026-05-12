from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.api.handlers.translation_handler import TranslationHandler


@pytest.fixture
def mock_trans_service():
    return AsyncMock()


@pytest.fixture
def mock_job_service():
    return AsyncMock()


@pytest.fixture
def handler(mock_trans_service, mock_job_service):
    return TranslationHandler(
        translation_service=mock_trans_service, job_service=mock_job_service
    )


class TestTranslationHandler:
    async def test_submit_translation(self, handler, mock_trans_service):
        req = MagicMock()
        await handler.submit_translation(req)
        mock_trans_service.submit_translation.assert_called_once_with(req)

    async def test_get_translation_status(self, handler, mock_job_service):
        await handler.get_translation_status("job1")
        mock_job_service.get_translation_status.assert_called_once_with("job1")
