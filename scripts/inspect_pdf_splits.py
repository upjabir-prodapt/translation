"""Inspect how StructureAwareSplitStrategy splits test PDF documents."""

import logging
import sys
from pathlib import Path

# Add project root and .venv site-packages if needed
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
for venv_site in root_dir.glob(".venv/lib/python*/site-packages"):
    sys.path.insert(0, str(venv_site))

from pymupdf import Document  # noqa: E402
from src.worker.doctranslator.format.pdf.split_manager import (  # noqa: E402
    StructureAwareSplitStrategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("split_inspector")


class DummyConfig:
    def __init__(self, file_path: Path):
        self.input_file = file_path


def inspect_document(file_path: Path):
    print("\n=======================================================")
    print(f"Inspecting Document: {file_path.name}")
    print(f"Path: {file_path}")
    print(f"Size: {file_path.stat().st_size / 1024:.1f} KB")
    print("=======================================================")

    doc = Document(str(file_path))
    total_pages = doc.page_count
    toc = doc.get_toc()

    print(f"Total Pages in PDF: {total_pages}")
    print(f"Total Raw TOC Entries: {len(toc)}")

    if toc:
        print("\nFirst 10 Raw TOC Entries (level, title, page):")
        for idx, entry in enumerate(toc[:10]):
            print(f"  {idx + 1}. Level {entry[0]}: {entry[1]!r} -> Page {entry[2]}")
        if len(toc) > 10:
            print(f"  ... and {len(toc) - 10} more TOC entries.")
    else:
        print("  (No TOC/Outline found in this PDF)")

    print(
        "\n--- Running StructureAwareSplitStrategy(min_pages_to_split=10, overlap_pages=2) ---"
    )
    strategy = StructureAwareSplitStrategy(min_pages_to_split=10, overlap_pages=2)
    cfg = DummyConfig(file_path)
    split_points = strategy.determine_split_points(cfg)

    print(f"\nResulting Split Parts: {len(split_points)}")
    total_pages_parsed = sum(sp.end_page - sp.start_page + 1 for sp in split_points)
    print(
        f"Total Pages That Will Be Parsed Across All Parts: {total_pages_parsed} (vs {total_pages} physical pages)"
    )

    print("\nDetailed Split Points:")
    for sp in split_points[:15]:
        span_len = sp.end_page - sp.start_page + 1
        print(
            f"  Part {sp.chunk_index:2d}: Pages [{sp.start_page:3d}..{sp.end_page:3d}] "
            f"(count={span_len:2d}, overlap={sp.overlap_pages:2d}) "
            f"Title: {sp.chapter_title!r}"
        )
    if len(split_points) > 15:
        print(f"  ... and {len(split_points) - 15} more parts.")
        last_sp = split_points[-1]
        span_len = last_sp.end_page - last_sp.start_page + 1
        print(
            f"  Part {last_sp.chunk_index:2d}: Pages [{last_sp.start_page:3d}..{last_sp.end_page:3d}] "
            f"(count={span_len:2d}, overlap={last_sp.overlap_pages:2d}) "
            f"Title: {last_sp.chapter_title!r}"
        )


def main():
    docs = [
        Path(
            "/home/jabir_mohammed_colt_net/Translation/docs/test_docs/22-promptengg.pdf"
        ),
        Path(
            "/home/jabir_mohammed_colt_net/Translation/docs/test_docs/HR Policy Manual 2023.pdf"
        ),
    ]

    for doc_path in docs:
        if not doc_path.exists():
            print(f"File not found: {doc_path}", file=sys.stderr)
            continue
        inspect_document(doc_path)


if __name__ == "__main__":
    main()
