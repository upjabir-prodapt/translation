"""Redis-backed translation cache tests."""

import functools
import inspect
from unittest.mock import MagicMock
from unittest.mock import patch

from src.worker.doctranslator.translator.translation_cache import TranslationCache
from src.worker.doctranslator.translator.translation_cache import build_cache_key
from src.worker.services import redis_client as redis_client_module


def _reset_module_singletons():
    # The pooled client is no longer private to this module: it now lives in
    # the shared `redis_client` module that the cache and the job lease both
    # consume, so there is one connection pool per process rather than one
    # per Redis feature.
    redis_client_module.reset_for_tests()


def _patch_settings(func):
    """Patch `settings` in the cache module *and* the shared redis_client
    module with one shared mock.

    Connection parameters (host/port/TLS/timeouts) are read by redis_client
    now that the client factory moved there, while REDIS_CACHE_TTL_SECONDS is
    still read by translation_cache -- so a test that configures only one of
    the two would silently exercise the real settings for the other.

    Applied as the innermost decorator so its mock arrives last, matching the
    bottom-up argument order `unittest.mock.patch` uses.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mock_settings = MagicMock()
        with (
            patch(
                "src.worker.doctranslator.translator.translation_cache.settings",
                mock_settings,
            ),
            patch("src.worker.services.redis_client.settings", mock_settings),
        ):
            return func(*args, mock_settings, **kwargs)

    # pytest resolves a test's parameters through `__wrapped__`, so without
    # this it would see the trailing `mock_settings` parameter, find no
    # fixture by that name and error out at collection.
    del wrapper.__wrapped__
    wrapper.__signature__ = inspect.Signature(
        list(inspect.signature(func).parameters.values())[:-1]
    )
    return wrapper


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

    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    @_patch_settings
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

    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    @_patch_settings
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

    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    @_patch_settings
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

    @_patch_settings
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

    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    @_patch_settings
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

    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    @_patch_settings
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

    @patch("redis.Redis")
    @patch("redis.ConnectionPool")
    @_patch_settings
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
