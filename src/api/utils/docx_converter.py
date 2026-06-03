"""DOCX to PDF conversion using LibreOffice headless."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_docx_to_pdf(docx_path: Path, output_dir: Path) -> Path:
    """Convert a DOCX file to PDF using LibreOffice headless.

    Returns the path to the generated PDF file.
    Raises RuntimeError if conversion fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(output_dir),
                str(docx_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as e:
        raise RuntimeError("LibreOffice is not installed or not on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"DOCX to PDF conversion timed out for {docx_path.name}") from e

    if result.returncode != 0:
        raise RuntimeError(
            f"DOCX to PDF conversion failed for {docx_path.name}: {result.stderr.strip()}"
        )

    pdf_path = output_dir / (docx_path.stem + ".pdf")
    if not pdf_path.exists():
        raise RuntimeError(
            f"LibreOffice conversion succeeded but output PDF not found at {pdf_path}"
        )

    logger.info(f"Converted {docx_path.name} to PDF at {pdf_path}")
    return pdf_path
