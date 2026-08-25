"""Redis-backed translation cache tests."""

from unittest.mock import MagicMock
from unittest.mock import patch

from src.worker.doctranslator.translator import translation_cache as tc_module
from src.worker.doctranslator.translator.translation_cache import TranslationCache
from src.worker.doctranslator.translator.translation_cache import build_cache_key


def _reset_module_singletons():
    tc_module._client = None
    tc_module._cache_enabled = None


class TestBuildCacheKey:
    def test_stable_and_deterministic(self):
        key1 = build_cache_key(
            provider="gemini_vertexai",
            model="gemini-2.5-flash",
            lang_in="en",
            lang_out="fr",
            text="hello world",
        )
        key2 = build_cache_key(
            provider="gemini_vertexai",
            model="gemini-2.5-flash",
            lang_in="en",
            lang_out="fr",
            text="hello world",
        )
        assert key1 == key2
        # sha256 hex digest (64 chars) namespaced under REDIS_KEY_PREFIX so
        # this service's keys cannot collide with other services sharing
        # the same Redis Cluster.
        from src.config.constants import settings

        assert key1.startswith(settings.REDIS_KEY_PREFIX)
        assert len(key1) == len(settings.REDIS_KEY_PREFIX) + 64

    def test_different_text_yields_different_key(self):
        key1 = build_cache_key(
            provider="gemini_vertexai",
            model="gemini-2.5-flash",
            lang_in="en",
            lang_out="fr",
            text="hello",
        )
        key2 = build_cache_key(
            provider="gemini_vertexai",
            model="gemini-2.5-flash",
            lang_in="en",
            lang_out="fr",
            text="goodbye",
        )
        assert key1 != key2

    def test_different_domain_yields_different_key(self):
        key_legal = build_cache_key(
            provider="gemini_vertexai",
            model="gemini-2.5-flash",
            lang_in="en",
            lang_out="fr",
            text="execute",
            domain="legal",
        )
        key_commercial = build_cache_key(
            provider="gemini_vertexai",
            model="gemini-2.5-flash",
            lang_in="en",
            lang_out="fr",
            text="execute",
            domain="commercial",
        )
        key_default = build_cache_key(
            provider="gemini_vertexai",
            model="gemini-2.5-flash",
            lang_in="en",
            lang_out="fr",
            text="execute",
        )
        assert key_legal != key_commercial
        assert key_legal != key_default
        assert key_commercial != key_default


class TestTranslationCacheRedisBackend:
    def setup_method(self):
        _reset_module_singletons()

    def teardown_method(self):
        _reset_module_singletons()

    @patch("src.worker.doctranslator.translator.translation_cache.settings")
    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    def test_get_returns_cached_value_on_hit(
        self, mock_pool, mock_redis_cls, mock_settings
    ):
        mock_settings.REDIS_HOST = "10.0.0.5"
        mock_settings.REDIS_PORT = 6379
        mock_settings.REDIS_DB = 0
        mock_settings.REDIS_PASSWORD = ""
        mock_settings.REDIS_TLS_ENABLED = True
        mock_settings.REDIS_SOCKET_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_CACHE_TTL_SECONDS = 604800

        mock_client = MagicMock()
        mock_client.get.return_value = "bonjour le monde"
        mock_redis_cls.return_value = mock_client

        cache = TranslationCache()
        result = cache.get("some-key")

        assert result == "bonjour le monde"
        mock_client.get.assert_called_once_with("some-key")

    @patch("src.worker.doctranslator.translator.translation_cache.settings")
    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    def test_get_returns_none_on_miss(self, mock_pool, mock_redis_cls, mock_settings):
        mock_settings.REDIS_HOST = "10.0.0.5"
        mock_settings.REDIS_PORT = 6379
        mock_settings.REDIS_DB = 0
        mock_settings.REDIS_PASSWORD = ""
        mock_settings.REDIS_TLS_ENABLED = True
        mock_settings.REDIS_SOCKET_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_CACHE_TTL_SECONDS = 604800

        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis_cls.return_value = mock_client

        cache = TranslationCache()
        assert cache.get("missing-key") is None

    @patch("src.worker.doctranslator.translator.translation_cache.settings")
    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    def test_set_writes_with_seven_day_ttl(
        self, mock_pool, mock_redis_cls, mock_settings
    ):
        mock_settings.REDIS_HOST = "10.0.0.5"
        mock_settings.REDIS_PORT = 6379
        mock_settings.REDIS_DB = 0
        mock_settings.REDIS_PASSWORD = ""
        mock_settings.REDIS_TLS_ENABLED = True
        mock_settings.REDIS_SOCKET_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_CACHE_TTL_SECONDS = 604800

        mock_client = MagicMock()
        mock_redis_cls.return_value = mock_client

        cache = TranslationCache()
        cache.set(
            "some-key",
            "bonjour le monde",
            provider="gemini_vertexai",
            model="gemini-2.5-flash",
            lang_in="en",
            lang_out="fr",
        )

        mock_client.set.assert_called_once_with(
            "some-key", "bonjour le monde", ex=604800
        )


class TestTranslationCacheFailOpen:
    def setup_method(self):
        _reset_module_singletons()

    def teardown_method(self):
        _reset_module_singletons()

    @patch("src.worker.doctranslator.translator.translation_cache.settings")
    def test_no_redis_host_disables_caching(self, mock_settings):
        mock_settings.REDIS_HOST = ""

        cache = TranslationCache()
        assert cache.get("any-key") is None
        # set() must be a silent no-op, never raise.
        cache.set(
            "any-key",
            "value",
            provider="p",
            model="m",
            lang_in="en",
            lang_out="fr",
        )

    @patch("src.worker.doctranslator.translator.translation_cache.settings")
    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    def test_ping_failure_disables_caching(
        self, mock_pool, mock_redis_cls, mock_settings
    ):
        mock_settings.REDIS_HOST = "10.0.0.5"
        mock_settings.REDIS_PORT = 6379
        mock_settings.REDIS_DB = 0
        mock_settings.REDIS_PASSWORD = ""
        mock_settings.REDIS_TLS_ENABLED = True
        mock_settings.REDIS_SOCKET_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_CACHE_TTL_SECONDS = 604800

        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("connection refused")
        mock_redis_cls.return_value = mock_client

        cache = TranslationCache()
        assert cache.get("any-key") is None

        # Subsequent calls should not attempt to reconnect (fails closed for
        # the lifetime of the process) and never raise.
        cache.set(
            "any-key",
            "value",
            provider="p",
            model="m",
            lang_in="en",
            lang_out="fr",
        )
        assert mock_redis_cls.call_count == 1

    @patch("src.worker.doctranslator.translator.translation_cache.settings")
    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    def test_get_error_is_swallowed(self, mock_pool, mock_redis_cls, mock_settings):
        mock_settings.REDIS_HOST = "10.0.0.5"
        mock_settings.REDIS_PORT = 6379
        mock_settings.REDIS_DB = 0
        mock_settings.REDIS_PASSWORD = ""
        mock_settings.REDIS_TLS_ENABLED = True
        mock_settings.REDIS_SOCKET_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_CACHE_TTL_SECONDS = 604800

        mock_client = MagicMock()
        mock_client.get.side_effect = TimeoutError("timed out")
        mock_redis_cls.return_value = mock_client

        cache = TranslationCache()
        assert cache.get("any-key") is None

    @patch("src.worker.doctranslator.translator.translation_cache.settings")
    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    def test_set_error_is_swallowed(self, mock_pool, mock_redis_cls, mock_settings):
        mock_settings.REDIS_HOST = "10.0.0.5"
        mock_settings.REDIS_PORT = 6379
        mock_settings.REDIS_DB = 0
        mock_settings.REDIS_PASSWORD = ""
        mock_settings.REDIS_TLS_ENABLED = True
        mock_settings.REDIS_SOCKET_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = 2.0
        mock_settings.REDIS_CACHE_TTL_SECONDS = 604800

        mock_client = MagicMock()
        mock_client.set.side_effect = TimeoutError("timed out")
        mock_redis_cls.return_value = mock_client

        cache = TranslationCache()
        # Must not raise.
        cache.set(
            "any-key",
            "value",
            provider="p",
            model="m",
            lang_in="en",
            lang_out="fr",
        )
