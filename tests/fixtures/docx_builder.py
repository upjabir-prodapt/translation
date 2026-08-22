"""Build minimal but valid .docx packages for tests."""

from __future__ import annotations

import io
import zipfile

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def run(text: str, *, bold: bool = False, space_preserve: bool = True) -> str:
    """One ``w:r`` holding one ``w:t``."""
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    space = ' xml:space="preserve"' if space_preserve else ""
    return f"<w:r>{rpr}<w:t{space}>{text}</w:t></w:r>"


def paragraph(*runs: str) -> str:
    """One ``w:p`` wrapping the given runs."""
    return f"<w:p>{''.join(runs)}</w:p>"


def table(*rows: list[str]) -> str:
    """A real ``w:tbl`` with grid and cell properties, one paragraph per cell."""
    grid = "".join('<w:gridCol w:w="4675"/>' for _ in (rows[0] if rows else []))
    body = ""
    for row in rows:
        cells = "".join(
            "<w:tc>"
            '<w:tcPr><w:tcW w:w="4675" w:type="dxa"/></w:tcPr>'
            f"{paragraph(run(cell))}"
            "</w:tc>"
            for cell in row
        )
        body += f"<w:tr>{cells}</w:tr>"
    return (
        "<w:tbl>"
        '<w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>"
        f"{body}"
        "</w:tbl>"
    )


def document_xml(body: str) -> str:
    """Wrap body XML in a document part, with a trailing section properties block."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}">'
        f"<w:body>{body}"
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/></w:sectPr>'
        "</w:body>"
        "</w:document>"
    )


def build_docx(body: str, extra_parts: dict[str, str] | None = None) -> bytes:
    """Assemble a .docx package around the given document body XML."""
    parts = {
        "[Content_Types].xml": CONTENT_TYPES,
        "_rels/.rels": ROOT_RELS,
        "word/document.xml": document_xml(body),
    }
    parts.update(extra_parts or {})

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in parts.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def simple_docx() -> bytes:
    """A document with a heading, a mixed-formatting paragraph, and a table."""
    body = (
        paragraph(run("Payment Terms"))
        + paragraph(
            run("Payment is due ", bold=False), run("within 30 days", bold=True)
        )
        + table(["Item", "Amount"], ["Licence fee", "1000"])
    )
    return build_docx(body)


def write_docx(path, content: bytes | None = None):
    """Write a .docx to ``path`` and return the path."""
    path.write_bytes(content if content is not None else simple_docx())
    return path
