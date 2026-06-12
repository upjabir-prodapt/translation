"""
Centralized test environment values.

Usage in a test:
    from tests.settings_env import apply_test_settings_env

    def test_something(monkeypatch):
        apply_test_settings_env(monkeypatch)
        # now import settings-dependent code

IMPORTANT: call apply_test_settings_env(monkeypatch) BEFORE importing any
module that triggers `from src.config.constants import settings`, because
_DOTENV_FILE = load_dotenv_file() runs at import time.
"""

from pytest import MonkeyPatch

TEST_SETTINGS_ENV: dict[str, str] = {
    # Disable all .env file loading — tests are fully self-contained
    "DOTENV_DISABLE": "true",
    # Runtime mode
    "IS_LOCAL": "true",
    # GCP Bootstrap
    "GOOGLE_CLOUD_PROJECT": "mock-project-id",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    # GCS
    "GCS_BUCKET_NAME": "mock-bucket",
    # BigQuery
    "BIGQUERY_DATASET": "mock-dataset",
    # Security
    "JWT_SECRET_KEY": "mock-secret-key-for-testing-only-12345",
    # Telemetry (disable export in tests; keep capture vars for Settings validation)
    "TRACE_ENABLED": "false",
    "OTEL_SERVICE_NAME": "translation_service_test",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "https://telemetry.googleapis.com/v1/traces",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    "OTEL_RESOURCE_ATTRIBUTES": "service.name=translation_service_test",
    "OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental",
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT",
    "OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED": "false",
}


def apply_test_settings_env(monkeypatch: MonkeyPatch) -> None:
    """Set all required env vars via monkeypatch before importing config."""
    for key, value in TEST_SETTINGS_ENV.items():
        monkeypatch.setenv(key, value)
