import itertools
import multiprocessing as mp
import os
import threading
from pathlib import Path

__version__ = "0.5.23"


def find_project_root(start_path: Path) -> Path:
    for parent in [start_path] + list(start_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Project root not found")


project_root = find_project_root(Path(__file__).resolve())

CACHE_FOLDER = project_root / "assests"
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


def get_cache_file_path(filename: str, sub_folder: str | None = None) -> Path:
    if sub_folder is not None:
        sub_folder = sub_folder.strip("/")
        sub_folder_path = CACHE_FOLDER / sub_folder
        sub_folder_path.mkdir(parents=True, exist_ok=True)
        return sub_folder_path / filename
    return CACHE_FOLDER / filename


WATERMARK_VERSION = "1.0"
GCS_BUCKET_NAME = "colt-ai-usecase"
GCS_ASSETS_PREFIX = "translation_service/artifacts"

# Asset Filenames
DOCLAYOUT_MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.onnx"
TABLE_DETECTION_MODEL_FILENAME = "ch_PP-OCRv4_det_infer.onnx"
FONT_METADATA_FILENAME = "font_metadata.json"
CMAP_METADATA_FILENAME = "cmap_metadata.json"

TIKTOKEN_CACHE_FOLDER = CACHE_FOLDER / "tiktoken"
TIKTOKEN_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_FOLDER)


_process_pool = None
_process_pool_lock = threading.Lock()
_ENABLE_PROCESS_POOL = False


def enable_process_pool():
    # Development and Testing ONLY API
    global _ENABLE_PROCESS_POOL
    _ENABLE_PROCESS_POOL = True


# macos & windows use spawn mode
# linux use forkserver mode


def get_process_pool():
    if not _ENABLE_PROCESS_POOL:
        return None
    global _process_pool
    with _process_pool_lock:
        if _process_pool is None:
            # Create pool only in main process
            if mp.current_process().name != "MainProcess":
                return None

            _process_pool = mp.Pool()
        return _process_pool


def close_process_pool():
    if not _ENABLE_PROCESS_POOL:
        return None
    global _process_pool
    with _process_pool_lock:
        if _process_pool:
            _process_pool.close()
            _process_pool.join()
            _process_pool = None


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch
