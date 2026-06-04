import logging
from io import BytesIO
from typing import TYPE_CHECKING
from typing import BinaryIO
from typing import Union

from src.doctranslator.pdfminer import settings
from src.doctranslator.pdfminer.casting import safe_int
from src.doctranslator.pdfminer.pdfexceptions import PDFException
from src.doctranslator.pdfminer.pdftypes import PDFObjRef
from src.doctranslator.pdfminer.pdftypes import PDFStream
from src.doctranslator.pdfminer.pdftypes import dict_value
from src.doctranslator.pdfminer.pdftypes import int_value
from src.doctranslator.pdfminer.psexceptions import PSEOF
from src.doctranslator.pdfminer.psparser import KWD
from src.doctranslator.pdfminer.psparser import PSKeyword
from src.doctranslator.pdfminer.psparser import PSStackParser

if TYPE_CHECKING:
    from src.doctranslator.pdfminer.pdfdocument import PDFDocument

log = logging.getLogger(__name__)


class PDFSyntaxError(PDFException):
    pass


# PDFParser stack holds all the base types plus PDFStream, PDFObjRef, and None
class PDFParser(PSStackParser[Union[PSKeyword, PDFStream, PDFObjRef, None]]):
    """PDFParser fetch PDF objects from a file stream.
    It can handle indirect references by referring to
    a PDF document set by set_document method.
    It also reads XRefs at the end of every PDF file.

    Typical usage:
      parser = PDFParser(fp)
      parser.read_xref()
      parser.read_xref(fallback=True) # optional
      parser.set_document(doc)
      parser.seek(offset)
      parser.nextobject()

    """

    def __init__(self, fp: BinaryIO) -> None:
        PSStackParser.__init__(self, fp)
        self.doc: PDFDocument | None = None
        self.fallback = False

    def set_document(self, doc: "PDFDocument") -> None:
        """Associates the parser with a PDFDocument object."""
        self.doc = doc

    KEYWORD_R = KWD(b"R")
    KEYWORD_NULL = KWD(b"null")
    KEYWORD_ENDOBJ = KWD(b"endobj")
    KEYWORD_STREAM = KWD(b"stream")
    KEYWORD_XREF = KWD(b"xref")
    KEYWORD_STARTXREF = KWD(b"startxref")

    def _handle_keyword_r(self, pos: int) -> None:
        """Handle the 'R' keyword: push an indirect object reference."""
        if len(self.curstack) >= 2:
            (_, _object_id), _ = self.pop(2)
            object_id = safe_int(_object_id)
            if object_id is not None:
                obj = PDFObjRef(self.doc, object_id)
                self.push((pos, obj))

    def _read_stream_length(self, dic: dict) -> int:
        """Return the declared stream length, or 0 when operating in fallback mode."""
        if self.fallback:
            return 0
        try:
            return int_value(dic["Length"])
        except KeyError:
            if settings.STRICT:
                raise PDFSyntaxError("/Length is undefined: %r" % dic)
            return 0

    def _seek_past_stream_header(self, pos: int) -> tuple[int, int]:
        """Seek to *pos* and skip the 'stream' line; return (new_pos, line_len)."""
        self.seek(pos)
        try:
            (_, line) = self.nextline()  # 'stream'
        except PSEOF:
            if settings.STRICT:
                raise PDFSyntaxError("Unexpected EOF")
            return pos, -1  # sentinel: caller should return
        return pos + len(line), len(line)

    def _read_stream_tail(
        self, pos: int, objlen: int, data: bytearray
    ) -> tuple[int, bytearray]:
        """Read lines after the declared stream body until 'endstream' is found."""
        while 1:
            try:
                (linepos, line) = self.nextline()
            except PSEOF:
                if settings.STRICT:
                    raise PDFSyntaxError("Unexpected EOF")
                break
            if b"endstream" in line:
                i = line.index(b"endstream")
                objlen += i
                if self.fallback:
                    data += line[:i]
                break
            objlen += len(line)
            if self.fallback:
                data += line
        return objlen, data

    def _handle_keyword_stream(self, pos: int) -> None:
        """Handle the 'stream' keyword: read stream data from the file."""
        ((_, dic),) = self.pop(1)
        dic = dict_value(dic)
        objlen = self._read_stream_length(dic)
        new_pos, line_len = self._seek_past_stream_header(pos)
        if line_len == -1:
            return
        pos = new_pos
        self.fp.seek(pos)
        data = bytearray(self.fp.read(objlen))
        self.seek(pos + objlen)
        objlen, data = self._read_stream_tail(pos, objlen, data)
        self.seek(pos + objlen)
        # XXX limit objlen not to exceed object boundary
        log.debug(
            "Stream: pos=%d, objlen=%d, dic=%r, data=%r...",
            pos,
            objlen,
            dic,
            data[:10],
        )
        if self.doc is None:
            raise RuntimeError("doc must be set before parsing streams")
        stream = PDFStream(dic, bytes(data), self.doc.decipher)
        self.push((pos, stream))

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        """Handles PDF-related keywords."""
        if token in (self.KEYWORD_XREF, self.KEYWORD_STARTXREF):
            self.add_results(*self.pop(1))

        elif token is self.KEYWORD_ENDOBJ:
            self.add_results(*self.pop(4))

        elif token is self.KEYWORD_NULL:
            # null object
            self.push((pos, None))

        elif token is self.KEYWORD_R:
            # reference to indirect object
            self._handle_keyword_r(pos)

        elif token is self.KEYWORD_STREAM:
            # stream object
            self._handle_keyword_stream(pos)

        else:
            # others
            self.push((pos, token))


class PDFStreamParser(PDFParser):
    """PDFStreamParser is used to parse PDF content streams
    that is contained in each page and has instructions
    for rendering the page. A reference to a PDF document is
    needed because a PDF content stream can also have
    indirect references to other objects in the same document.
    """

    def __init__(self, data: bytes) -> None:
        PDFParser.__init__(self, BytesIO(data))

    def flush(self) -> None:
        self.add_results(*self.popall())

    KEYWORD_OBJ = KWD(b"obj")

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        if token is self.KEYWORD_R:
            # reference to indirect object
            (_, _object_id), _ = self.pop(2)
            object_id = safe_int(_object_id)
            if object_id is not None:
                obj = PDFObjRef(self.doc, object_id)
                self.push((pos, obj))
            return

        elif token in (self.KEYWORD_OBJ, self.KEYWORD_ENDOBJ):
            if settings.STRICT:
                # See PDF Spec 3.4.6: Only the object values are stored in the
                # stream; the obj and endobj keywords are not used.
                raise PDFSyntaxError("Keyword endobj found in stream")
            return

        # others
        self.push((pos, token))
