import asyncio
import importlib
import sys
from pathlib import Path


class DummyFirestore:
    def __init__(self):
        self.updates = []

    async def update_job(self, *_args, **_kwargs):
        self.updates.append((_args, _kwargs))
        return None

    async def get_job(self, _job_id):
        return {
            "original_filename": "input.pdf",
            "file_size_bytes": 128,
            "created_at": None,
        }


class DummyStorage:
    async def download_input_pdf(self, _job_id, input_path, filename="input.pdf"):
        Path(input_path).write_bytes(b"%PDF-1.4")

    async def upload_output_files(self, _job_id, _files):
        return {"mono": "gs://bucket/output.pdf"}


class DummyBigQuery:
    async def write_job_completion(self, *_args, **_kwargs):
        return None

    async def write_translation_report(self, *_args, **_kwargs):
        return None


class DummyProcessor:
    last_config = None

    def __init__(self, _progress_tracker):
        pass

    async def translate(self, config):
        DummyProcessor.last_config = config
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "mono.pdf"
        out_file.write_bytes(b"%PDF-1.4")
        return {"mono_pdf_path": out_file, "page_count": 1}


def test_worker_builds_translation_config_with_model_list(monkeypatch, tmp_path):
    import repository
    import worker.repository.worker_storage_repository as worker_storage_repository

    firestore = DummyFirestore()
    monkeypatch.setattr(repository, "get_firestore_repository", lambda: firestore)
    monkeypatch.setattr(repository, "get_bigquery_repository", lambda: DummyBigQuery())
    monkeypatch.setattr(
        worker_storage_repository,
        "get_worker_storage_repository",
        lambda: DummyStorage(),
    )

    sys.modules.pop("worker.handlers.translation_services", None)
    translation_services = importlib.import_module(
        "worker.handlers.translation_services"
    )

    monkeypatch.setattr(translation_services, "JobProcessor", DummyProcessor)
    monkeypatch.setattr(translation_services.settings, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(
        translation_services,
        "select_model_list",
        lambda _lang_in, _lang_out, _domain: [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
    )
    monkeypatch.setattr(
        DummyProcessor,
        "detect_source_language",
        lambda _self, _input_path: "en",
    )

    task_config = {
        "lang_in": "auto",
        "lang_out": "es",
        "domain": "hr",
        "options": {"debug": True},
    }

    result = asyncio.run(
        translation_services._process_translation_job("job-123", task_config)
    )
    assert result["success"] is True
    assert DummyProcessor.last_config["model_list"] == [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    assert DummyProcessor.last_config["add_cover_page"] is True
    assert DummyProcessor.last_config["job_id"] == "job-123"
    assert DummyProcessor.last_config["lang_in"] == "en"


def test_worker_marks_job_failed_when_language_detection_finds_too_many_languages(
    monkeypatch, tmp_path
):
    import repository
    import worker.repository.worker_storage_repository as worker_storage_repository

    firestore = DummyFirestore()
    monkeypatch.setattr(repository, "get_firestore_repository", lambda: firestore)
    monkeypatch.setattr(repository, "get_bigquery_repository", lambda: DummyBigQuery())
    monkeypatch.setattr(
        worker_storage_repository,
        "get_worker_storage_repository",
        lambda: DummyStorage(),
    )

    sys.modules.pop("worker.handlers.translation_services", None)
    translation_services = importlib.import_module(
        "worker.handlers.translation_services"
    )

    class FailingProcessor(DummyProcessor):
        def detect_source_language(self, _input_path):
            raise ValueError("Detected more than 2 languages on page 1: de, en, fr")

    monkeypatch.setattr(translation_services, "JobProcessor", FailingProcessor)
    monkeypatch.setattr(translation_services.settings, "TEMP_DIR", tmp_path)

    task_config = {
        "lang_in": "auto",
        "lang_out": "es",
        "domain": "hr",
        "options": {"debug": True},
    }

    try:
        asyncio.run(
            translation_services._process_translation_job("job-123", task_config)
        )
    except ValueError as exc:
        assert str(exc) == "Detected more than 2 languages on page 1: de, en, fr"
    else:
        raise AssertionError("Expected _process_translation_job to fail")

    failure_updates = [
        args[1]
        for args, _kwargs in firestore.updates
        if len(args) == 2 and args[1].get("status") == "failed"
    ]
    assert failure_updates
    assert (
        failure_updates[-1]["error_message"]
        == "Detected more than 2 languages on page 1: de, en, fr"
    )
