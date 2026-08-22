"""Read and rewrite a .docx OPC package without disturbing untouched parts."""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

from lxml import etree

logger = logging.getLogger(__name__)

DOCUMENT_PART = "word/document.xml"

# Wordprocessing parts whose body text belongs to the document the user sees.
# Text boxes and table cells live inside these parts, so they need no separate
# handling — their paragraphs are nested in the same XML tree.
_TEXT_PART_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^word/document\d*\.xml$"),
    re.compile(r"^word/header\d*\.xml$"),
    re.compile(r"^word/footer\d*\.xml$"),
    re.compile(r"^word/footnotes\.xml$"),
    re.compile(r"^word/endnotes\.xml$"),
    re.compile(r"^word/comments\.xml$"),
)

_ENCRYPTED_MARKERS = ("EncryptedPackage", "EncryptionInfo")


class InvalidDocxError(ValueError):
    """Raised when a file is not a usable, unencrypted .docx package."""


def is_text_part(name: str) -> bool:
    """Return True when the named part holds translatable document body text."""
    return any(pattern.match(name) for pattern in _TEXT_PART_PATTERNS)


class DocxPackage:
    """A parsed .docx package.

    Parts are held as raw bytes and re-emitted verbatim on save. Only parts
    handed out by :meth:`text_parts` are parsed and re-serialized, so styles,
    numbering, media, tables, and relationships survive a round trip unchanged.
    """

    def __init__(
        self,
        *,
        infos: list[zipfile.ZipInfo],
        raw_parts: dict[str, bytes],
    ) -> None:
        self._infos = infos
        self._raw_parts = raw_parts
        self._parsed: dict[str, etree._Element] = {}

    @classmethod
    def open(cls, path: Path) -> DocxPackage:
        """Open and validate a .docx file."""
        if not path.exists():
            raise InvalidDocxError(f"DOCX file not found: {path}")
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                names = {info.filename for info in infos}
                cls._reject_unsupported(names, path)
                raw_parts = {
                    info.filename: archive.read(info.filename) for info in infos
                }
        except zipfile.BadZipFile as exc:
            raise InvalidDocxError(
                f"{path.name} is not a valid .docx package (corrupt or not a Word file)"
            ) from exc
        return cls(infos=infos, raw_parts=raw_parts)

    @staticmethod
    def _reject_unsupported(names: set[str], path: Path) -> None:
        if any(marker in names for marker in _ENCRYPTED_MARKERS):
            raise InvalidDocxError(
                f"{path.name} is password-protected. Remove the password and resubmit."
            )
        if DOCUMENT_PART not in names:
            if "word/document.bin" in names:
                raise InvalidDocxError(
                    f"{path.name} is a binary Word document (.doc). Save it as .docx and resubmit."
                )
            raise InvalidDocxError(
                f"{path.name} is missing {DOCUMENT_PART}; it is not a Word document"
            )

    def text_part_names(self) -> list[str]:
        """Names of body-text parts, document first then headers/footers/notes."""
        names = [name for name in self._raw_parts if is_text_part(name)]
        names.sort(key=lambda name: (name != DOCUMENT_PART, name))
        return names

    def part_root(self, name: str) -> etree._Element:
        """Parse (once) and return the root element of a part."""
        root = self._parsed.get(name)
        if root is None:
            root = etree.fromstring(self._raw_parts[name])
            self._parsed[name] = root
        return root

    def text_parts(self) -> list[tuple[str, etree._Element]]:
        """Return (name, root) for every body-text part in the package."""
        return [(name, self.part_root(name)) for name in self.text_part_names()]

    def save(self, dest: Path) -> Path:
        """Write the package to ``dest``, re-serializing only parsed parts."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as archive:
            for info in self._infos:
                payload = self._render_part(info.filename)
                entry = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                entry.compress_type = info.compress_type
                entry.external_attr = info.external_attr
                entry.internal_attr = info.internal_attr
                entry.create_system = info.create_system
                archive.writestr(entry, payload)
        logger.info(f"Wrote translated DOCX to {dest}")
        return dest

    def _render_part(self, name: str) -> bytes:
        root = self._parsed.get(name)
        if root is None:
            return self._raw_parts[name]
        return etree.tostring(
            root.getroottree(),
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        )
