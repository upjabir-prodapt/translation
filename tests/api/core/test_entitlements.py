"""Tests for Firestore-backed entitlement lookups (src/api/core/entitlements.py)."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.api.core import entitlements


def _mock_client_returning(snapshot: MagicMock) -> MagicMock:
    mock_doc_ref = MagicMock()
    mock_doc_ref.get = AsyncMock(return_value=snapshot)
    mock_collection = MagicMock()
    mock_collection.document.return_value = mock_doc_ref
    mock_client = MagicMock()
    mock_client.collection.return_value = mock_collection
    return mock_client


@pytest.mark.asyncio
async def test_has_translation_access_true_when_field_true():
    snapshot = MagicMock(exists=True)
    snapshot.to_dict.return_value = {"translation_access": True}
    client = _mock_client_returning(snapshot)

    with patch.object(entitlements, "_get_client", return_value=client):
        assert await entitlements.has_translation_access("user@colt.net") is True


@pytest.mark.asyncio
async def test_has_translation_access_false_when_field_false():
    snapshot = MagicMock(exists=True)
    snapshot.to_dict.return_value = {"translation_access": False}
    client = _mock_client_returning(snapshot)

    with patch.object(entitlements, "_get_client", return_value=client):
        assert await entitlements.has_translation_access("user@colt.net") is False


@pytest.mark.asyncio
async def test_has_translation_access_false_when_field_missing():
    snapshot = MagicMock(exists=True)
    snapshot.to_dict.return_value = {
        "sales_agent_access": True
    }  # no translation_access key
    client = _mock_client_returning(snapshot)

    with patch.object(entitlements, "_get_client", return_value=client):
        assert await entitlements.has_translation_access("user@colt.net") is False


@pytest.mark.asyncio
async def test_has_translation_access_false_when_doc_missing():
    snapshot = MagicMock(exists=False)
    client = _mock_client_returning(snapshot)

    with patch.object(entitlements, "_get_client", return_value=client):
        assert await entitlements.has_translation_access("nouser@colt.net") is False


@pytest.mark.asyncio
async def test_has_translation_access_fails_closed_on_exception():
    mock_doc_ref = MagicMock()
    mock_doc_ref.get = AsyncMock(side_effect=Exception("firestore unavailable"))
    mock_collection = MagicMock()
    mock_collection.document.return_value = mock_doc_ref
    mock_client = MagicMock()
    mock_client.collection.return_value = mock_collection

    with patch.object(entitlements, "_get_client", return_value=mock_client):
        assert await entitlements.has_translation_access("user@colt.net") is False
