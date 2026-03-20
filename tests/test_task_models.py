"""
Unit tests for worker/models/task_models.py.

Covers: WatermarkOutputMode, TaskStatus, TranslationTaskConfig,
BabelDOCTranslationConfig (validators), TranslationTask.
"""

import pytest
from pydantic import ValidationError

from worker.models.task_models import (
    BabelDOCTranslationConfig,
    TaskStatus,
    TranslationTask,
    TranslationTaskConfig,
    WatermarkOutputMode,
)


# ---------------------------------------------------------------------------
# WatermarkOutputMode
# ---------------------------------------------------------------------------


class TestWatermarkOutputMode:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("watermarked", WatermarkOutputMode.WATERMARKED),
            ("no_watermark", WatermarkOutputMode.NO_WATERMARK),
            ("both", WatermarkOutputMode.BOTH),
        ],
    )
    def test_valid_values(self, value, expected):
        assert WatermarkOutputMode(value) == expected

    def test_is_str_enum(self):
        assert isinstance(WatermarkOutputMode.WATERMARKED, str)


# ---------------------------------------------------------------------------
# TaskStatus
# ---------------------------------------------------------------------------


class TestTaskStatus:
    @pytest.mark.parametrize(
        "value",
        ["queued", "processing", "completed", "failed", "cancelled"],
    )
    def test_all_statuses(self, value):
        ts = TaskStatus(value)
        assert str(ts) == value

    def test_is_str_enum(self):
        assert isinstance(TaskStatus.QUEUED, str)


# ---------------------------------------------------------------------------
# TranslationTaskConfig
# ---------------------------------------------------------------------------


class TestTranslationTaskConfig:
    def test_minimal_valid(self):
        cfg = TranslationTaskConfig(lang_in="auto", lang_out="es")
        assert cfg.lang_in == "auto"
        assert cfg.lang_out == "es"
        assert cfg.domain is None
        assert cfg.options == {}

    def test_with_domain_and_options(self):
        cfg = TranslationTaskConfig(
            lang_in="en",
            lang_out="fr",
            domain="legal",
            options={"qps": 2},
        )
        assert cfg.domain == "legal"
        assert cfg.options["qps"] == 2

    def test_frozen(self):
        cfg = TranslationTaskConfig(lang_in="auto", lang_out="es")
        with pytest.raises(Exception):
            cfg.lang_in = "en"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BabelDOCTranslationConfig — valid construction
# ---------------------------------------------------------------------------


def _minimal_babel_config(**overrides):
    defaults = {
        "input_file": "/tmp/doc.pdf",
        "lang_in": "en",
        "lang_out": "es",
        "model_list": ["gpt-4o-mini"],
    }
    defaults.update(overrides)
    return BabelDOCTranslationConfig(**defaults)


class TestBabelDOCTranslationConfig:
    def test_minimal_valid(self):
        cfg = _minimal_babel_config()
        assert cfg.lang_in == "en"
        assert cfg.lang_out == "es"
        assert cfg.model_list == ["gpt-4o-mini"]

    def test_lang_codes_lowercased(self):
        cfg = _minimal_babel_config(lang_in="EN", lang_out="ES")
        assert cfg.lang_in == "en"
        assert cfg.lang_out == "es"

    def test_lang_code_whitespace_stripped(self):
        cfg = _minimal_babel_config(lang_in=" fr ", lang_out=" de ")
        assert cfg.lang_in == "fr"
        assert cfg.lang_out == "de"

    def test_invalid_lang_code_raises(self):
        with pytest.raises(ValidationError, match="Language code"):
            _minimal_babel_config(lang_in="123")

    def test_non_alpha_lang_code_raises(self):
        with pytest.raises(ValidationError, match="Language code"):
            _minimal_babel_config(lang_out="e-s")

    def test_lang_code_too_long_raises(self):
        with pytest.raises(ValidationError, match="Language code"):
            _minimal_babel_config(lang_in="toolongcode")

    def test_empty_model_list_raises(self):
        with pytest.raises(ValidationError, match="model_list"):
            _minimal_babel_config(model_list=[])

    def test_model_list_strips_whitespace(self):
        cfg = _minimal_babel_config(model_list=["  gpt-4o  ", "gemini"])
        assert cfg.model_list == ["gpt-4o", "gemini"]

    def test_model_list_filters_blank_entries(self):
        cfg = _minimal_babel_config(model_list=["gpt-4o", "   ", "gemini"])
        assert cfg.model_list == ["gpt-4o", "gemini"]

    def test_default_watermark_mode(self):
        cfg = _minimal_babel_config()
        assert cfg.watermark_output_mode == WatermarkOutputMode.NO_WATERMARK

    def test_to_babeldoc_kwargs_excludes_none(self):
        cfg = _minimal_babel_config()
        kwargs = cfg.to_babeldoc_kwargs()
        assert "output_dir" not in kwargs  # None by default
        assert "lang_in" in kwargs

    def test_frozen(self):
        cfg = _minimal_babel_config()
        with pytest.raises(Exception):
            cfg.lang_in = "de"  # type: ignore[misc]

    def test_no_dual_default_false(self):
        cfg = _minimal_babel_config()
        assert cfg.no_dual is False

    def test_qps_default(self):
        cfg = _minimal_babel_config()
        assert cfg.qps == 4

    def test_add_cover_page_default_true(self):
        cfg = _minimal_babel_config()
        assert cfg.add_cover_page is True

    def test_auto_extract_glossary_default_true(self):
        cfg = _minimal_babel_config()
        assert cfg.auto_extract_glossary is True

    @pytest.mark.parametrize("lang", ["en", "fr", "auto", "zh"])
    def test_valid_lang_codes(self, lang):
        cfg = _minimal_babel_config(lang_in=lang)
        assert cfg.lang_in == lang


# ---------------------------------------------------------------------------
# TranslationTask
# ---------------------------------------------------------------------------


class TestTranslationTask:
    def test_valid_task(self):
        task = TranslationTask(
            job_id="job-abc",
            config=TranslationTaskConfig(lang_in="auto", lang_out="de"),
        )
        assert task.job_id == "job-abc"
        assert task.config.lang_out == "de"

    def test_missing_job_id_raises(self):
        with pytest.raises(ValidationError):
            TranslationTask(
                config=TranslationTaskConfig(lang_in="auto", lang_out="de")
            )

    def test_missing_config_raises(self):
        with pytest.raises(ValidationError):
            TranslationTask(job_id="job-123")

    def test_frozen(self):
        task = TranslationTask(
            job_id="job-abc",
            config=TranslationTaskConfig(lang_in="auto", lang_out="de"),
        )
        with pytest.raises(Exception):
            task.job_id = "other"  # type: ignore[misc]
