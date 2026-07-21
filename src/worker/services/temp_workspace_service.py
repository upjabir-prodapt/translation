"""Job-scoped temporary workspace management."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.config.constants import settings


@dataclass(slots=True)
class JobWorkspace:
    """Concrete directory layout for one translation job."""

    root: Path
    input_dir: Path
    masked_dir: Path
    attempts_dir: Path
    verification_dir: Path
    assembly_dir: Path
    final_dir: Path
    logs_dir: Path

    def attempt_dir(self, attempt_index: int, model_id: str) -> Path:
        safe_model = model_id.replace("/", "_").replace(":", "_")
        path = self.attempts_dir / f"attempt_{attempt_index}_{safe_model}"
        path.mkdir(parents=True, exist_ok=True)
        return path


class TempWorkspaceService:
    """Manage isolated temporary directories per job."""

    def __init__(self, base_dir: Path | None = None):
        root = base_dir or (settings.temp_root_path / settings.TEMP_JOBS_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        self._root = root

    def create(self, job_id: str) -> JobWorkspace:
        root = self._root / job_id
        workspace = JobWorkspace(
            root=root,
            input_dir=root / "input",
            masked_dir=root / "masked",
            attempts_dir=root / "attempts",
            verification_dir=root / "verification",
            assembly_dir=root / "assembly",
            final_dir=root / "final",
            logs_dir=root / "logs",
        )
        for directory in (
            workspace.input_dir,
            workspace.masked_dir,
            workspace.attempts_dir,
            workspace.verification_dir,
            workspace.assembly_dir,
            workspace.final_dir,
            workspace.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return workspace

    def cleanup(self, job_id: str) -> None:
        root = self._root / job_id
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
