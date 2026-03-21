"""
Unit tests for FirestoreRepository.claim_job() — lines 137-171.

Uses a fully mocked Firestore async client with a transaction context manager.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from repository.firestore_repository import FirestoreRepository
from repository.repository_exception import FirestoreError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(collection_ref=None):
    mock_client = AsyncMock()
    # collection() is called synchronously in __init__, so must be MagicMock
    mock_client.collection = MagicMock(return_value=collection_ref or MagicMock())
    # transaction() is called synchronously in claim_job, so must be MagicMock
    mock_client.transaction = MagicMock()
    repo = FirestoreRepository(client=mock_client, collection="test_jobs")
    return repo, mock_client


def _make_transaction_mock():
    """Build a mock transaction that behaves as an async context manager."""
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=txn)
    txn.__aexit__ = AsyncMock(return_value=False)
    txn.update = MagicMock()
    return txn


def _make_snapshot(exists=True, status="queued"):
    snap = MagicMock()
    snap.exists = exists
    snap.get = lambda key: status if key == "status" else None
    return snap


# ---------------------------------------------------------------------------
# claim_job — success cases
# ---------------------------------------------------------------------------


class TestClaimJobSuccess:
    async def test_returns_true_for_queued_job(self):
        txn = _make_transaction_mock()
        doc_ref = AsyncMock()
        doc_ref.get.return_value = _make_snapshot(exists=True, status="queued")

        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref

        repo, mock_client = _make_repo(collection_ref=collection_ref)
        mock_client.transaction.return_value = txn

        result = await repo.claim_job("job-1")
        assert result is True

    async def test_calls_transaction_update_for_queued_job(self):
        txn = _make_transaction_mock()
        doc_ref = AsyncMock()
        doc_ref.get.return_value = _make_snapshot(exists=True, status="queued")

        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref

        repo, mock_client = _make_repo(collection_ref=collection_ref)
        mock_client.transaction.return_value = txn

        await repo.claim_job("job-1")
        txn.update.assert_called_once()

    async def test_update_sets_processing_status(self):
        txn = _make_transaction_mock()
        doc_ref = AsyncMock()
        doc_ref.get.return_value = _make_snapshot(exists=True, status="queued")

        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref

        repo, mock_client = _make_repo(collection_ref=collection_ref)
        mock_client.transaction.return_value = txn

        await repo.claim_job("job-1")
        # Check that update was called with a dict containing status=processing
        call_args = txn.update.call_args
        update_data = call_args[0][1] if call_args[0] else call_args[1]
        assert update_data.get("status") == "processing"


# ---------------------------------------------------------------------------
# claim_job — failure / skip cases
# ---------------------------------------------------------------------------


class TestClaimJobFailures:
    async def test_returns_false_when_job_not_found(self):
        txn = _make_transaction_mock()
        doc_ref = AsyncMock()
        doc_ref.get.return_value = _make_snapshot(exists=False)

        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref

        repo, mock_client = _make_repo(collection_ref=collection_ref)
        mock_client.transaction.return_value = txn

        result = await repo.claim_job("ghost-job")
        assert result is False

    async def test_returns_false_when_job_not_queued(self):
        txn = _make_transaction_mock()
        doc_ref = AsyncMock()
        doc_ref.get.return_value = _make_snapshot(exists=True, status="processing")

        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref

        repo, mock_client = _make_repo(collection_ref=collection_ref)
        mock_client.transaction.return_value = txn

        result = await repo.claim_job("busy-job")
        assert result is False

    async def test_returns_false_for_completed_job(self):
        txn = _make_transaction_mock()
        doc_ref = AsyncMock()
        doc_ref.get.return_value = _make_snapshot(exists=True, status="completed")

        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref

        repo, mock_client = _make_repo(collection_ref=collection_ref)
        mock_client.transaction.return_value = txn

        result = await repo.claim_job("done-job")
        assert result is False

    async def test_exception_raises_firestore_error(self):
        txn = _make_transaction_mock()
        # Make the transaction context manager itself raise
        txn.__aenter__ = AsyncMock(side_effect=Exception("Firestore down"))

        collection_ref = MagicMock()
        collection_ref.document.return_value = AsyncMock()

        repo, mock_client = _make_repo(collection_ref=collection_ref)
        mock_client.transaction.return_value = txn

        with pytest.raises(FirestoreError, match="Failed to claim job"):
            await repo.claim_job("job-1")

    async def test_no_update_called_when_job_not_found(self):
        txn = _make_transaction_mock()
        doc_ref = AsyncMock()
        doc_ref.get.return_value = _make_snapshot(exists=False)

        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref

        repo, mock_client = _make_repo(collection_ref=collection_ref)
        mock_client.transaction.return_value = txn

        await repo.claim_job("ghost-job")
        txn.update.assert_not_called()
