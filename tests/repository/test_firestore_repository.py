"""
Unit tests for repository/firestore_repository.py — FirestoreRepository.

All Firestore async client calls are mocked; no real GCP connection is made.

Async iteration (for list_jobs / get_jobs_by_status) is handled by a local
AsyncIterator helper that yields mock document snapshots.
"""

from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import NotFound

from repository.firestore_repository import FirestoreRepository
from repository.repository_exception import FirestoreError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AsyncIter:
    """Turn a list of items into an async iterator for mocking query.stream()."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None


def _make_mock_doc(data: dict) -> MagicMock:
    """Build a mock Firestore document snapshot."""
    doc = MagicMock()
    doc.exists = True
    doc.to_dict.return_value = data
    doc.get = lambda key: data.get(key)
    return doc


def _build_repo(client=None, collection_ref=None) -> FirestoreRepository:
    """Construct a FirestoreRepository with injected mock client.

    FirestoreRepository.__init__ calls self.client.collection() synchronously,
    so collection must be a plain MagicMock — not an AsyncMock attribute —
    otherwise the call returns a coroutine instead of the collection ref.
    """
    mock_client = client or AsyncMock()
    # Override collection with a synchronous MagicMock so __init__ receives
    # the collection ref immediately rather than a coroutine.
    resolved_ref = collection_ref if collection_ref is not None else MagicMock()
    mock_client.collection = MagicMock(return_value=resolved_ref)
    repo = FirestoreRepository(client=mock_client, collection="test_jobs")
    return repo


def _make_query_mock(docs: list):
    """Build a fully chained mock query that streams the given docs."""
    q = MagicMock()
    q.where.return_value = q
    q.order_by.return_value = q
    q.limit.return_value = q
    q.offset.return_value = q
    q.stream.return_value = _AsyncIter(docs)
    return q


# ---------------------------------------------------------------------------
# create_job
# ---------------------------------------------------------------------------


class TestCreateJob:
    async def test_creates_document(self):
        doc_ref = AsyncMock()
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        await repo.create_job("job-1", {"job_id": "job-1", "status": "queued"})
        doc_ref.set.assert_called_once()

    async def test_adds_timestamps(self):
        doc_ref = AsyncMock()
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        data = {"job_id": "job-1"}
        await repo.create_job("job-1", data)

        saved_data = doc_ref.set.call_args[0][0]
        assert "created_at" in saved_data
        assert "updated_at" in saved_data
        assert "expire_at" in saved_data

    async def test_ttl_is_in_future(self):
        doc_ref = AsyncMock()
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        data = {"job_id": "job-1"}
        await repo.create_job("job-1", data)

        saved = doc_ref.set.call_args[0][0]
        assert saved["expire_at"] > datetime.now(UTC)

    async def test_firestore_error_wrapped(self):
        doc_ref = AsyncMock()
        doc_ref.set.side_effect = Exception("Firestore down")
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        with pytest.raises(FirestoreError, match="Failed to create job"):
            await repo.create_job("job-1", {"job_id": "job-1"})


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------


class TestGetJob:
    async def test_returns_dict_for_existing_job(self):
        job_data = {"job_id": "job-1", "status": "queued"}
        snapshot = _make_mock_doc(job_data)
        doc_ref = AsyncMock()
        doc_ref.get.return_value = snapshot
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        result = await repo.get_job("job-1")
        assert result == job_data

    async def test_returns_none_for_missing_job(self):
        snapshot = MagicMock()
        snapshot.exists = False
        doc_ref = AsyncMock()
        doc_ref.get.return_value = snapshot
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        result = await repo.get_job("ghost-job")
        assert result is None

    async def test_firestore_error_wrapped(self):
        doc_ref = AsyncMock()
        doc_ref.get.side_effect = Exception("timeout")
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        with pytest.raises(FirestoreError):
            await repo.get_job("job-1")


# ---------------------------------------------------------------------------
# update_job
# ---------------------------------------------------------------------------


class TestUpdateJob:
    async def test_returns_true_on_success(self):
        doc_ref = AsyncMock()
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        result = await repo.update_job("job-1", {"status": "processing"})
        assert result is True

    async def test_update_called_with_correct_args(self):
        doc_ref = AsyncMock()
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        await repo.update_job("job-1", {"status": "completed", "progress": 1.0})
        call_args = doc_ref.update.call_args[0][0]
        assert call_args["status"] == "completed"

    async def test_updated_at_added(self):
        doc_ref = AsyncMock()
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        await repo.update_job("job-1", {"status": "processing"})
        call_args = doc_ref.update.call_args[0][0]
        assert "updated_at" in call_args

    async def test_returns_false_when_not_found(self):
        doc_ref = AsyncMock()
        doc_ref.update.side_effect = NotFound("doc")
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        result = await repo.update_job("ghost-job", {"status": "cancelled"})
        assert result is False

    async def test_other_error_wrapped(self):
        doc_ref = AsyncMock()
        doc_ref.update.side_effect = Exception("network error")
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        with pytest.raises(FirestoreError):
            await repo.update_job("job-1", {"status": "processing"})


# ---------------------------------------------------------------------------
# delete_job
# ---------------------------------------------------------------------------


class TestDeleteJob:
    async def test_returns_true_on_success(self):
        doc_ref = AsyncMock()
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        result = await repo.delete_job("job-1")
        assert result is True

    async def test_delete_called(self):
        doc_ref = AsyncMock()
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        await repo.delete_job("job-1")
        doc_ref.delete.assert_called_once()

    async def test_returns_false_when_not_found(self):
        doc_ref = AsyncMock()
        doc_ref.delete.side_effect = NotFound("doc")
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        result = await repo.delete_job("ghost-job")
        assert result is False

    async def test_other_error_wrapped(self):
        doc_ref = AsyncMock()
        doc_ref.delete.side_effect = Exception("timeout")
        collection_ref = MagicMock()
        collection_ref.document.return_value = doc_ref
        repo = _build_repo(collection_ref=collection_ref)

        with pytest.raises(FirestoreError):
            await repo.delete_job("job-1")


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    async def test_returns_list_of_dicts(self):
        doc1 = _make_mock_doc({"job_id": "j1", "status": "queued"})
        doc2 = _make_mock_doc({"job_id": "j2", "status": "completed"})
        query = _make_query_mock([doc1, doc2])
        collection_ref = MagicMock()
        collection_ref.order_by.return_value = query
        repo = _build_repo(collection_ref=collection_ref)

        result = await repo.list_jobs()
        assert len(result) == 2
        assert result[0]["job_id"] == "j1"

    async def test_empty_collection_returns_empty_list(self):
        query = _make_query_mock([])
        collection_ref = MagicMock()
        collection_ref.order_by.return_value = query
        repo = _build_repo(collection_ref=collection_ref)

        result = await repo.list_jobs()
        assert result == []

    async def test_status_filter_applied(self):
        doc = _make_mock_doc({"job_id": "j1", "status": "queued"})
        query = _make_query_mock([doc])
        collection_ref = MagicMock()
        collection_ref.order_by.return_value = query
        repo = _build_repo(collection_ref=collection_ref)

        # The where call should be chained on the query
        await repo.list_jobs(status="queued")
        query.where.assert_called_once_with("status", "==", "queued")

    async def test_no_status_filter_skips_where(self):
        query = _make_query_mock([])
        collection_ref = MagicMock()
        collection_ref.order_by.return_value = query
        repo = _build_repo(collection_ref=collection_ref)

        await repo.list_jobs(status=None)
        query.where.assert_not_called()

    async def test_limit_and_offset_applied(self):
        query = _make_query_mock([])
        collection_ref = MagicMock()
        collection_ref.order_by.return_value = query
        repo = _build_repo(collection_ref=collection_ref)

        await repo.list_jobs(limit=5, offset=10)
        query.limit.assert_called_with(5)
        query.offset.assert_called_with(10)

    async def test_firestore_error_wrapped(self):
        collection_ref = MagicMock()
        collection_ref.order_by.side_effect = Exception("boom")
        repo = _build_repo(collection_ref=collection_ref)

        with pytest.raises(FirestoreError):
            await repo.list_jobs()


# ---------------------------------------------------------------------------
# get_jobs_by_status
# ---------------------------------------------------------------------------


class TestGetJobsByStatus:
    async def test_returns_matching_jobs(self):
        doc = _make_mock_doc({"job_id": "j1", "status": "queued"})
        query = _make_query_mock([doc])
        collection_ref = MagicMock()
        collection_ref.where.return_value = query
        repo = _build_repo(collection_ref=collection_ref)

        result = await repo.get_jobs_by_status("queued")
        assert len(result) == 1

    async def test_empty_result(self):
        query = _make_query_mock([])
        collection_ref = MagicMock()
        collection_ref.where.return_value = query
        repo = _build_repo(collection_ref=collection_ref)

        result = await repo.get_jobs_by_status("processing")
        assert result == []

    async def test_firestore_error_wrapped(self):
        collection_ref = MagicMock()
        collection_ref.where.side_effect = Exception("error")
        repo = _build_repo(collection_ref=collection_ref)

        with pytest.raises(FirestoreError):
            await repo.get_jobs_by_status("queued")


# ---------------------------------------------------------------------------
# list_jobs_with_expired_outputs / list_expired_terminal_jobs
# ---------------------------------------------------------------------------


class TestCleanupQueries:
    async def test_expired_outputs_filters_state_and_expiry(self):
        doc = _make_mock_doc({"job_id": "j1", "status": "completed"})
        query = _make_query_mock([doc])
        collection_ref = MagicMock()
        collection_ref.where.return_value = query
        repo = _build_repo(collection_ref=collection_ref)

        now = datetime.now(UTC)
        result = await repo.list_jobs_with_expired_outputs(now, limit=50)

        assert len(result) == 1
        collection_ref.where.assert_called_once_with(
            "file_lifecycle.storage_state", "in", ["available", "delete_pending"]
        )
        query.where.assert_called_once_with("file_lifecycle.expires_at", "<=", now)
        query.limit.assert_called_once_with(50)

    async def test_expired_outputs_error_wrapped(self):
        collection_ref = MagicMock()
        collection_ref.where.side_effect = Exception("boom")
        repo = _build_repo(collection_ref=collection_ref)

        with pytest.raises(FirestoreError):
            await repo.list_jobs_with_expired_outputs(datetime.now(UTC))

    async def test_expired_terminal_jobs_filters_status_and_ttl(self):
        doc = _make_mock_doc({"job_id": "j2", "status": "failed"})
        query = _make_query_mock([doc])
        collection_ref = MagicMock()
        collection_ref.where.return_value = query
        repo = _build_repo(collection_ref=collection_ref)

        now = datetime.now(UTC)
        result = await repo.list_expired_terminal_jobs(now, limit=25)

        assert len(result) == 1
        collection_ref.where.assert_called_once_with(
            "status", "in", ["failed", "cancelled"]
        )
        query.where.assert_called_once_with("expire_at", "<=", now)
        query.limit.assert_called_once_with(25)

    async def test_expired_terminal_jobs_error_wrapped(self):
        collection_ref = MagicMock()
        collection_ref.where.side_effect = Exception("boom")
        repo = _build_repo(collection_ref=collection_ref)

        with pytest.raises(FirestoreError):
            await repo.list_expired_terminal_jobs(datetime.now(UTC))
