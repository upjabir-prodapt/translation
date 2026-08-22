import itertools
import logging
from collections.abc import Container
from collections.abc import Iterator
from typing import Any
from typing import BinaryIO

from src.worker.doctranslator.pdfminer import settings
from src.worker.doctranslator.pdfminer.pdfdocument import PDFDocument
from src.worker.doctranslator.pdfminer.pdfdocument import PDFNoPageLabels
from src.worker.doctranslator.pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from src.worker.doctranslator.pdfminer.pdfexceptions import PDFObjectNotFound
from src.worker.doctranslator.pdfminer.pdfexceptions import PDFValueError
from src.worker.doctranslator.pdfminer.pdfparser import PDFParser
from src.worker.doctranslator.pdfminer.pdftypes import PDFObjRef
from src.worker.doctranslator.pdfminer.pdftypes import dict_value
from src.worker.doctranslator.pdfminer.pdftypes import int_value
from src.worker.doctranslator.pdfminer.pdftypes import list_value
from src.worker.doctranslator.pdfminer.pdftypes import resolve1
from src.worker.doctranslator.pdfminer.psparser import LIT
from src.worker.doctranslator.pdfminer.utils import Rect
from src.worker.doctranslator.pdfminer.utils import parse_rect

log = logging.getLogger(__name__)

# some predefined literals and keywords.
LITERAL_PAGE = LIT("Page")
LITERAL_PAGES = LIT("Pages")


class PDFPage:
    """An object that holds the information about a page.

    A PDFPage object is merely a convenience class that has a set
    of keys and values, which describe the properties of a page
    and point to its contents.

    Attributes
    ----------
      doc: a PDFDocument object.
      pageid: any Python object that can uniquely identify the page.
      attrs: a dictionary of page attributes.
      contents: a list of PDFStream objects that represents the page content.
      lastmod: the last modified time of the page.
      resources: a dictionary of resources used by the page.
      mediabox: the physical size of the page.
      cropbox: the crop rectangle of the page.
      rotate: the page rotation (in degree).
      annots: the page annotations.
      beads: a chain that represents natural reading order.
      label: the page's label (typically, the logical page number).

    """

    def __init__(
        self,
        doc: PDFDocument,
        pageid: object,
        attrs: object,
        label: str | None,
    ) -> None:
        """Initialize a page object.

        doc: a PDFDocument object.
        pageid: any Python object that can uniquely identify the page.
        attrs: a dictionary of page attributes.
        label: page label string.
        """
        self.doc = doc
        self.pageid = pageid
        self.attrs = dict_value(attrs)
        self.label = label
        self.lastmod = resolve1(self.attrs.get("LastModified"))
        self.resources: dict[object, object] = resolve1(
            self.attrs.get("Resources", dict()),
        )
        try:
            while isinstance(attrs["MediaBox"], PDFObjRef):
                attrs["MediaBox"] = resolve1(attrs["MediaBox"])
        except Exception:
            log.exception(f"try to fix mediabox failed: {attrs}")

        self.mediabox = self._parse_mediabox(self.attrs.get("MediaBox"))
        try:
            self.cropbox = self._parse_cropbox(self.attrs.get("CropBox"), self.mediabox)
        except Exception:
            self.cropbox = self.mediabox
        self.contents = self._parse_contents(self.attrs.get("Contents"))

        self.rotate = (int_value(self.attrs.get("Rotate", 0)) + 360) % 360
        self.annots = self.attrs.get("Annots")
        self.beads = self.attrs.get("B")

    def __repr__(self) -> str:
        return f"<PDFPage: Resources={self.resources!r}, MediaBox={self.mediabox!r}>"

    INHERITABLE_ATTRS = {"Resources", "MediaBox", "CropBox", "Rotate"}

    @classmethod
    def _resolve_page_obj(
        cls,
        obj: Any,
        document: PDFDocument,
    ) -> tuple[int, dict]:
        """Resolve *obj* to its object ID and properties dict."""
        if isinstance(obj, int):
            object_id: int = obj
            object_properties: dict = dict_value(document.getobj(object_id)).copy()
        else:
            # obj is PDFObjRef or PDFStream
            object_id = obj.objid  # type: ignore[attr-defined]
            object_properties = dict_value(obj).copy()
        return object_id, object_properties

    @classmethod
    def _depth_first_search_pages(
        cls,
        obj: Any,
        parent: dict[str, Any],
        document: PDFDocument,
        visited: set[Any] | None = None,
    ) -> Iterator[tuple[int, dict]]:
        """Recursively walk the page-tree rooted at *obj*, yielding (id, props) pairs."""
        object_id, object_properties = cls._resolve_page_obj(obj, document)

        # Avoid recursion errors by keeping track of visited nodes
        if visited is None:
            visited = set()
        if object_id in visited:
            return
        visited.add(object_id)

        for k, v in parent.items():
            if k in cls.INHERITABLE_ATTRS and k not in object_properties:
                object_properties[k] = v

        object_type = object_properties.get("Type")
        if object_type is None and not settings.STRICT:  # See #64
            object_type = object_properties.get("type")

        if object_type is LITERAL_PAGES and "Kids" in object_properties:
            log.debug("Pages: Kids=%r", object_properties["Kids"])
            for child in list_value(object_properties["Kids"]):
                yield from cls._depth_first_search_pages(
                    child, object_properties, document, visited
                )
        elif object_type is LITERAL_PAGE:
            log.debug("Page: %r", object_properties)
            yield (object_id, object_properties)

    @classmethod
    def _fallback_page_scan(
        cls,
        document: PDFDocument,
        page_labels: Iterator[str | None],
    ) -> Iterator["PDFPage"]:
        """Yield pages by scanning all xrefs when the /Pages tree is missing."""
        for xref in document.xrefs:
            for objid in xref.get_objids():
                try:
                    obj = document.getobj(objid)
                    if isinstance(obj, dict) and obj.get("Type") is LITERAL_PAGE:
                        yield cls(document, objid, obj, next(page_labels))
                except PDFObjectNotFound:
                    pass

    @classmethod
    def create_pages(cls, document: PDFDocument) -> Iterator["PDFPage"]:
        try:
            page_labels: Iterator[str | None] = document.get_page_labels()
        except PDFNoPageLabels:
            page_labels = itertools.repeat(None)

        pages = False
        if "Pages" in document.catalog:
            objects = cls._depth_first_search_pages(
                document.catalog["Pages"], document.catalog, document
            )
            for objid, tree in objects:
                yield cls(document, objid, tree, next(page_labels))
                pages = True
        if not pages:
            # Fallback when /Pages is missing.
            yield from cls._fallback_page_scan(document, page_labels)

    @classmethod
    def _check_text_extractable(
        cls,
        doc: PDFDocument,
        fp: BinaryIO,
        check_extractable: bool,
    ) -> None:
        """Raise or warn if the document prohibits text extraction."""
        if doc.is_extractable:
            return
        if check_extractable:
            error_msg = "Text extraction is not allowed: %r" % fp
            raise PDFTextExtractionNotAllowed(error_msg)
        warning_msg = (
            "The PDF %r contains a metadata field "
            "indicating that it should not allow "
            "text extraction. Ignoring this field "
            "and proceeding. Use the check_extractable "
            "if you want to raise an error in this case" % fp
        )
        log.warning(warning_msg)

    @classmethod
    def _iter_selected_pages(
        cls,
        doc: PDFDocument,
        pagenos: Container[int] | None,
        maxpages: int,
    ) -> Iterator["PDFPage"]:
        """Iterate pages from *doc*, filtering by pagenos and maxpages."""
        for pageno, page in enumerate(cls.create_pages(doc)):
            if pagenos and (pageno not in pagenos):
                continue
            yield page
            if maxpages and maxpages <= pageno + 1:
                break

    @classmethod
    def get_pages(
        cls,
        fp: BinaryIO,
        pagenos: Container[int] | None = None,
        maxpages: int = 0,
        password: str = "",
        caching: bool = True,
        check_extractable: bool = False,
    ) -> Iterator["PDFPage"]:
        # Create a PDF parser object associated with the file object.
        parser = PDFParser(fp)
        # Create a PDF document object that stores the document structure.
        doc = PDFDocument(parser, password=password, caching=caching)
        # Check if the document allows text extraction.
        # If not, warn the user and proceed.
        cls._check_text_extractable(doc, fp, check_extractable)
        # Process each page contained in the document.
        yield from cls._iter_selected_pages(doc, pagenos, maxpages)

    def _parse_mediabox(self, value: Any) -> Rect:
        us_letter = (0.0, 0.0, 612.0, 792.0)

        if value is None:
            log.warning(
                "MediaBox missing from /Page (and not inherited), "
                "defaulting to US Letter"
            )
            return us_letter

        try:
            return parse_rect(resolve1(val) for val in resolve1(value))

        except PDFValueError:
            log.warning("Invalid MediaBox in /Page, defaulting to US Letter")
            return us_letter

    def _parse_cropbox(self, value: Any, mediabox: Rect) -> Rect:
        if value is None:
            # CropBox is optional, and MediaBox is used if not specified.
            return mediabox

        try:
            return parse_rect(resolve1(val) for val in resolve1(value))

        except PDFValueError:
            log.warning("Invalid CropBox in /Page, defaulting to MediaBox")
            return mediabox

    def _parse_contents(self, value: Any) -> list[Any]:
        contents: list[Any] = []
        if value is not None:
            contents = resolve1(value)
            if not isinstance(contents, list):
                contents = [contents]
        return contents
