"""Shared Redis client.

Extracted from `translation_cache` so the translation cache and the job
lease share one pooled connection against the same Memorystore/PSC endpoint
instead of opening two and duplicating the TLS quirks documented in
docs/infra/redis-memorystore-psc-setup.md.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

from src.worker.services import redis_client


def _settings(**overrides):
    s = MagicMock()
    s.REDIS_HOST = "10.0.0.5"
    s.REDIS_PORT = 6379
    s.REDIS_DB = 0
    s.REDIS_PASSWORD = ""
    s.REDIS_TLS_ENABLED = True
    s.REDIS_SOCKET_TIMEOUT_SECONDS = 2.0
    s.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = 2.0
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


class TestGetRedisClient:
    def setup_method(self):
        redis_client.reset_for_tests()

    def teardown_method(self):
        redis_client.reset_for_tests()

    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    def test_client_is_a_singleton(self, mock_pool, mock_redis_cls):
        with patch.object(redis_client, "settings", _settings()):
            first = redis_client.get_redis_client()
            second = redis_client.get_redis_client()
        assert first is second
        assert mock_redis_cls.call_count == 1

    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    def test_tls_uses_ssl_connection_class(self, mock_pool, mock_redis_cls):
        import redis

        with patch.object(redis_client, "settings", _settings()):
            redis_client.get_redis_client()
        assert mock_pool.call_args.kwargs["connection_class"] is redis.SSLConnection
        assert mock_pool.call_args.kwargs["ssl_cert_reqs"] is None

    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    def test_plaintext_uses_plain_connection_class(self, mock_pool, mock_redis_cls):
        import redis

        with patch.object(redis_client, "settings", _settings(REDIS_TLS_ENABLED=False)):
            redis_client.get_redis_client()
        assert mock_pool.call_args.kwargs["connection_class"] is redis.Connection
        assert "ssl_cert_reqs" not in mock_pool.call_args.kwargs

    def test_no_host_disables_redis(self):
        with patch.object(redis_client, "settings", _settings(REDIS_HOST="")):
            assert redis_client.get_redis_client() is None
            assert redis_client.is_redis_enabled() is False

    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    def test_ping_failure_fails_closed_and_is_not_retried(
        self, mock_pool, mock_redis_cls
    ):
        """A dead endpoint disables Redis immediately rather than costing a
        failed round trip on every single operation."""
        client = MagicMock()
        client.ping.side_effect = ConnectionError("connection refused")
        mock_redis_cls.return_value = client

        with patch.object(redis_client, "settings", _settings()):
            assert redis_client.get_redis_client() is None
            assert redis_client.get_redis_client() is None
        assert mock_redis_cls.call_count == 1
