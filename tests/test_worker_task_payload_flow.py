import asyncio
import importlib
import sys
from pathlib import Path


class DummyFirestore:
    async def update_job(self, *_args, **_kwargs):
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

    monkeypatch.setattr(repository, "get_firestore_repository", lambda: DummyFirestore())
    monkeypatch.setattr(repository, "get_bigquery_repository", lambda: DummyBigQuery())
    monkeypatch.setattr(
        worker_storage_repository,
        "get_worker_storage_repository",
        lambda: DummyStorage(),
    )

    sys.modules.pop("worker.handlers.translation_services", None)
    translation_services = importlib.import_module("worker.handlers.translation_services")

    monkeypatch.setattr(translation_services, "JobProcessor", DummyProcessor)
    monkeypatch.setattr(translation_services.settings, "TEMP_DIR", tmp_path)

    task_config = {
        "lang_in": "en",
        "lang_out": "es",
        "model_list": ["gemini-2.5-pro", "gemini-2.5-flash"],
        "options": {"debug": True},
    }

    result = asyncio.run(translation_services._process_translation_job("job-123", task_config))
    assert result["success"] is True
    assert DummyProcessor.last_config["model_list"] == [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    assert DummyProcessor.last_config["job_id"] == "job-123"
