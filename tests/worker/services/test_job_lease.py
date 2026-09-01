"""Cross-instance job lease.

With `--concurrency=1` a Cloud Tasks retry can never be delivered to the
instance that is busy with the job, so `status == PROCESSING` on its own
could not distinguish a crashed owner from a live one -- and the handler's
crash-retry branch would rmtree the scratch directory of a translation that
was still running, on a second instance costing another ~6-7 GB.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

from src.worker.services import job_lease


class _FakeRedis:
    """In-memory stand-in implementing just enough of redis-py.

    `eval` is modelled faithfully as an atomic compare-and-swap because that
    atomicity is the point: a bare DEL/EXPIRE would let an instance whose
    lease had already expired -- and been taken over -- clobber the new
    owner's.
    """

    def __init__(self):
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        self.ttls[key] = ex
        return True

    def get(self, key):
        return self.store.get(key)

    def eval(self, script, numkeys, key, owner, *args):
        if self.store.get(key) != owner:
            return 0
        if "del" in script:
            del self.store[key]
            self.ttls.pop(key, None)
        else:
            self.ttls[key] = int(args[0])
        return 1

    def expire_now(self, key):
        self.store.pop(key, None)
        self.ttls.pop(key, None)


def _with_redis(client):
    return patch.object(job_lease, "get_redis_client", return_value=client)


class TestAcquire:
    def test_first_acquirer_wins(self):
        with _with_redis(_FakeRedis()):
            assert job_lease.acquire("job-1") is True

    def test_second_acquirer_is_refused(self):
        client = _FakeRedis()
        with _with_redis(client):
            assert job_lease.acquire("job-1") is True
            # A different process: a fresh owner ID against the same Redis.
            with patch.object(job_lease, "owner_id", return_value="other-instance"):
                assert job_lease.acquire("job-1") is False

    def test_ttl_is_applied(self):
        client = _FakeRedis()
        with _with_redis(client):
            job_lease.acquire("job-1", ttl_seconds=42)
        assert next(iter(client.ttls.values())) == 42

    def test_fails_open_when_redis_is_unavailable(self):
        """Redis is an optimisation and a safety net, never a dependency of
        correctness: a worker that cannot reach it must behave exactly as it
        did before this module existed."""
        with _with_redis(None):
            assert job_lease.acquire("job-1") is True

    def test_fails_open_when_redis_raises(self):
        client = MagicMock()
        client.set.side_effect = ConnectionError("no route to host")
        with _with_redis(client):
            assert job_lease.acquire("job-1") is True


class TestRefresh:
    def test_owner_can_refresh(self):
        client = _FakeRedis()
        with _with_redis(client):
            job_lease.acquire("job-1", ttl_seconds=10)
            assert job_lease.refresh("job-1", ttl_seconds=300) is True
        assert next(iter(client.ttls.values())) == 300

    def test_non_owner_cannot_refresh(self):
        client = _FakeRedis()
        with _with_redis(client):
            job_lease.acquire("job-1", ttl_seconds=10)
            with patch.object(job_lease, "owner_id", return_value="other-instance"):
                assert job_lease.refresh("job-1") is False
        assert next(iter(client.ttls.values())) == 10

    def test_refresh_of_an_expired_lease_fails(self):
        client = _FakeRedis()
        with _with_redis(client):
            job_lease.acquire("job-1")
            client.expire_now(next(iter(client.store)))
            assert job_lease.refresh("job-1") is False


class TestRelease:
    def test_owner_releases(self):
        client = _FakeRedis()
        with _with_redis(client):
            job_lease.acquire("job-1")
            assert job_lease.release("job-1") is True
        assert client.store == {}

    def test_release_by_a_non_owner_is_a_no_op(self):
        """Not a delete. A process whose lease already expired and was taken
        over must not be able to free the new owner's lease."""
        client = _FakeRedis()
        with _with_redis(client):
            job_lease.acquire("job-1")
            key = next(iter(client.store))
            with patch.object(job_lease, "owner_id", return_value="other-instance"):
                assert job_lease.release("job-1") is False
        assert key in client.store

    def test_expiry_allows_takeover(self):
        client = _FakeRedis()
        with _with_redis(client):
            job_lease.acquire("job-1")
            client.expire_now(next(iter(client.store)))
            with patch.object(job_lease, "owner_id", return_value="other-instance"):
                assert job_lease.acquire("job-1") is True


class TestHolder:
    def test_reports_the_current_owner(self):
        client = _FakeRedis()
        with _with_redis(client):
            job_lease.acquire("job-1")
            assert job_lease.holder("job-1") == job_lease.owner_id()

    def test_none_when_unleased(self):
        with _with_redis(_FakeRedis()):
            assert job_lease.holder("job-1") is None
