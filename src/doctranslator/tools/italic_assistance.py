import argparse
import json
import re
from pathlib import Path

import orjson
from rich.console import Console
from rich.table import Table
from src.config.constants import settings
from src.doctranslator.format.pdf.document_il.utils.formular_helper import (
    is_formulas_font,
)
from src.doctranslator.format.pdf.translation_config import TranslationConfig

# Base folder containing per-job working directories.
WORKING_FOLDER = settings.temp_root_path / settings.TEMP_JOBS_ROOT


def find_latest_il_json(job_id: str | None = None) -> Path | None:
    """
    Find the latest il_translated.json file.

    Args:
        job_id: If provided, look only in this specific job's working directory.
                If None, search across all job directories (backward compatibility).

    Returns:
        Path to the most recently modified il_translated.json file, or None if not found.
    """
    base_dir = Path(WORKING_FOLDER)

    if job_id:
        # Look in specific job directory
        json_path = base_dir / job_id / "working" / "il_translated.json"
        if json_path.exists():
            return json_path
        return None
    else:
        # Search across all job directories (backward compatibility)
        json_files = list(base_dir.glob("*/working/il_translated.json"))
        if not json_files:
            return None
        # Sort by modification time (newest first)
        json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return json_files[0]


def _get_font_from_style(
    style: dict, page_font_map: dict[str, tuple[str, str]]
) -> tuple[str, str] | None:
    """Return (font_id, name) if style has a font_id present in page_font_map, else None."""
    font_id = style.get("font_id")
    if font_id and font_id in page_font_map:
        return page_font_map[font_id]
    return None


def _extract_font_from_char(
    char: dict, page_font_map: dict[str, tuple[str, str]]
) -> set[tuple[str, str]]:
    """Extract font from a single pdf_character composition."""
    fonts: set[tuple[str, str]] = set()
    if "pdf_style" in char:
        entry = _get_font_from_style(char["pdf_style"], page_font_map)
        if entry:
            fonts.add(entry)
    return fonts


def _extract_font_from_line(
    line: dict, page_font_map: dict[str, tuple[str, str]]
) -> set[tuple[str, str]]:
    """Extract fonts from all characters inside a pdf_line composition."""
    fonts: set[tuple[str, str]] = set()
    for char in line.get("pdf_character", []):
        if "pdf_style" in char:
            entry = _get_font_from_style(char["pdf_style"], page_font_map)
            if entry:
                fonts.add(entry)
    return fonts


def _extract_font_from_formula(
    formula: dict, page_font_map: dict[str, tuple[str, str]]
) -> set[tuple[str, str]]:
    """Extract fonts from all characters inside a pdf_formula composition."""
    return _extract_font_from_line(formula, page_font_map)


def _extract_font_from_same_style(
    same_style: dict, page_font_map: dict[str, tuple[str, str]]
) -> set[tuple[str, str]]:
    """Extract font from a pdf_same_style_characters composition."""
    fonts: set[tuple[str, str]] = set()
    if "pdf_style" in same_style:
        entry = _get_font_from_style(same_style["pdf_style"], page_font_map)
        if entry:
            fonts.add(entry)
    return fonts


def _extract_font_from_same_style_unicode(
    same_style_unicode: dict, page_font_map: dict[str, tuple[str, str]]
) -> set[tuple[str, str]]:
    """Extract font from a pdf_same_style_unicode_characters composition."""
    fonts: set[tuple[str, str]] = set()
    if (
        "pdf_style" in same_style_unicode
        and same_style_unicode["pdf_style"] is not None
    ):
        entry = _get_font_from_style(same_style_unicode["pdf_style"], page_font_map)
        if entry:
            fonts.add(entry)
    return fonts


def extract_fonts_from_paragraph(
    paragraph: dict, page_font_map: dict[str, tuple[str, str]]
) -> set[tuple[str, str]]:
    """
    Extract all font_ids and names used in a paragraph.

    Args:
        paragraph: The paragraph dictionary
        page_font_map: Dictionary mapping font_id to (font_id, name) tuples

    Returns:
        Set of (font_id, name) tuples
    """
    fonts: set[tuple[str, str]] = set()

    # Check if paragraph has a pdfStyle with font_id
    if "pdf_style" in paragraph and paragraph["pdf_style"]:
        entry = _get_font_from_style(paragraph["pdf_style"], page_font_map)
        if entry:
            fonts.add(entry)

    # Process paragraph compositions if present
    for comp in paragraph.get("pdf_paragraph_composition", []):
        if "pdf_character" in comp and comp["pdf_character"]:
            fonts |= _extract_font_from_char(comp["pdf_character"], page_font_map)
        elif "pdf_line" in comp and comp["pdf_line"]:
            fonts |= _extract_font_from_line(comp["pdf_line"], page_font_map)
        elif "pdf_formula" in comp and comp["pdf_formula"]:
            fonts |= _extract_font_from_formula(comp["pdf_formula"], page_font_map)
        elif "pdf_same_style_characters" in comp and comp["pdf_same_style_characters"]:
            fonts |= _extract_font_from_same_style(
                comp["pdf_same_style_characters"], page_font_map
            )
        elif (
            "pdf_same_style_unicode_characters" in comp
            and comp["pdf_same_style_unicode_characters"]
        ):
            fonts |= _extract_font_from_same_style_unicode(
                comp["pdf_same_style_unicode_characters"], page_font_map
            )

    return fonts


def find_fonts_by_debug_id(json_path: Path, debug_id_regex: str) -> dict[str, str]:
    """
    Find all fonts used in paragraphs with matching debug_id.

    Args:
        json_path: Path to the il_translated.json file
        debug_id_regex: Regular expression to match debug_id values

    Returns:
        Dictionary mapping font_ids to font names
    """
    # Load and parse JSON
    with json_path.open("rb") as f:
        doc_data = orjson.loads(f.read())

    # Compile regex pattern (case insensitive)
    pattern = re.compile(debug_id_regex.strip(" \"'"), re.IGNORECASE)

    # Set to collect all found font information
    found_fonts = set()

    # Process each page
    for page in doc_data.get("page", []):
        # Create a mapping of font_id to (font_id, name) tuples for this page
        page_font_map = {}
        for font in page.get("pdf_font", []):
            if "font_id" in font and "name" in font:
                page_font_map[font["font_id"]] = (font["font_id"], font["name"])

        # Check each paragraph
        for paragraph in page.get("pdf_paragraph", []):
            # Check if paragraph has debug_id and if it matches the pattern
            debug_id = paragraph.get("debug_id")
            if debug_id and pattern.search(debug_id):
                # Get all fonts used in this paragraph
                paragraph_fonts = extract_fonts_from_paragraph(paragraph, page_font_map)
                found_fonts.update(paragraph_fonts)

    # Convert set of tuples to dictionary
    return dict(found_fonts)


def _resolve_json_path(args) -> "tuple[Path | None, int]":
    """Resolve the JSON path from CLI args. Returns (path, error_code) where error_code != 0 means error."""
    if args.json_path:
        json_path = Path(args.json_path)
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            return None, 1
        return json_path, 0

    json_path = find_latest_il_json(job_id=args.job_id)
    if not json_path:
        if args.job_id:
            print(f"Error: Could not find il_translated.json for job_id: {args.job_id}")
        else:
            print("Error: Could not find any il_translated.json file")
        return None, 1
    return json_path, 0


def _collect_paragraph_fonts_from_file(json_path: Path) -> list:
    """Scan all paragraphs in *json_path* and return unique (page_idx, font_name, para_idx, debug_id) tuples."""
    fonts = []
    with json_path.open(encoding="utf-8") as f:
        pdf_data = json.load(f)

    for page_index, page in enumerate(pdf_data["page"]):
        page_font_map = {
            font["font_id"]: (font["font_id"], font["name"])
            for font in page["pdf_font"]
            if "font_id" in font and "name" in font
        }
        for paragraph_index, paragraph_content in enumerate(page["pdf_paragraph"]):
            font_debug_id = paragraph_content.get("debug_id")
            if not font_debug_id:
                continue
            paragraph_fonts = extract_fonts_from_paragraph(paragraph_content, page_font_map)
            existing_names = {entry[1] for entry in fonts}
            for _font_id, font_name in paragraph_fonts:
                if font_name not in existing_names:
                    fonts.append((page_index, font_name, paragraph_index, font_debug_id))
                    existing_names.add(font_name)
    return fonts


def _print_font_table(fonts: list) -> None:
    """Render and print a rich table of font recognition results."""
    _translation_config = TranslationConfig(
        *[None for _ in range(3)], lang_out="zh_cn", doc_layout_model=1
    )
    table = Table(title="Font Recognition Results")
    table.add_column("Page #", justify="center", style="cyan")
    table.add_column("Paragraph #", justify="center", style="cyan")
    table.add_column("DEBUG_ID", justify="center", style="cyan")
    table.add_column("Font Name", style="magenta")
    table.add_column("Recognition Result", justify="center")

    for page_index, font_name, paragraph_index, font_debug_id in fonts:
        if is_formulas_font(font_name, None):
            result_label = "[bold red]Formula Font[/bold red]"
        else:
            result_label = "[bold blue]Non-Formula Font[/bold blue]"
        table.add_row(
            str(page_index),
            str(paragraph_index),
            str(font_debug_id),
            font_name,
            result_label,
        )

    Console().print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Extract fonts from paragraphs with matching debug_id"
    )
    parser.add_argument(
        "debug_id_regex", nargs="+", help="Regular expression to match debug_id values"
    )
    parser.add_argument(
        "--json-path",
        help="Path to il_translated.json (if not provided, will use the latest file)",
    )
    parser.add_argument(
        "--working-folder",
        help="Path to the working folder containing il_translated.json files",
    )
    parser.add_argument(
        "--job-id",
        help="Specific job ID to analyze (instead of searching all jobs)",
    )

    args = parser.parse_args()

    if args.working_folder:
        global WORKING_FOLDER
        WORKING_FOLDER = Path(args.working_folder)
        if not WORKING_FOLDER.exists():
            print(f"Error: Working folder does not exist: {WORKING_FOLDER}")
            return 1

    json_path, err = _resolve_json_path(args)
    if err:
        return err

    print(f"Using JSON file: {json_path}")

    # Find fonts matching the debug_id pattern and report them
    found_fonts = find_fonts_by_debug_id(json_path, "|".join(args.debug_id_regex))
    if found_fonts:
        print(
            f"Found {len(found_fonts)} fonts in paragraphs matching debug_id pattern: {args.debug_id_regex}"
        )
        print(json.dumps(found_fonts, indent=2, ensure_ascii=False))
    else:
        print(
            f"No fonts found for paragraphs matching debug_id pattern: {args.debug_id_regex}"
        )

    # Collect and display all paragraph fonts in a table
    fonts = _collect_paragraph_fonts_from_file(json_path)
    _print_font_table(fonts)

    return 0


if __name__ == "__main__":
    exit(main())
