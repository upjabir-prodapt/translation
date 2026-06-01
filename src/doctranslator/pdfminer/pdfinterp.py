import logging
import re
from collections.abc import Mapping
from collections.abc import Sequence
from io import BytesIO
from typing import Union
from typing import cast

from src.doctranslator.pdfminer import settings
from src.doctranslator.pdfminer.casting import safe_cmyk
from src.doctranslator.pdfminer.casting import safe_float
from src.doctranslator.pdfminer.casting import safe_int
from src.doctranslator.pdfminer.casting import safe_matrix
from src.doctranslator.pdfminer.casting import safe_rgb
from src.doctranslator.pdfminer.cmapdb import CMap
from src.doctranslator.pdfminer.cmapdb import CMapBase
from src.doctranslator.pdfminer.cmapdb import CMapDB
from src.doctranslator.pdfminer.pdfcolor import PREDEFINED_COLORSPACE
from src.doctranslator.pdfminer.pdfcolor import PDFColorSpace
from src.doctranslator.pdfminer.pdfdevice import PDFDevice
from src.doctranslator.pdfminer.pdfdevice import PDFTextSeq
from src.doctranslator.pdfminer.pdfexceptions import PDFException
from src.doctranslator.pdfminer.pdfexceptions import PDFValueError
from src.doctranslator.pdfminer.pdffont import PDFCIDFont
from src.doctranslator.pdfminer.pdffont import PDFFont
from src.doctranslator.pdfminer.pdffont import PDFFontError
from src.doctranslator.pdfminer.pdffont import PDFTrueTypeFont
from src.doctranslator.pdfminer.pdffont import PDFType1Font
from src.doctranslator.pdfminer.pdffont import PDFType3Font
from src.doctranslator.pdfminer.pdfpage import PDFPage
from src.doctranslator.pdfminer.pdftypes import LITERALS_ASCII85_DECODE
from src.doctranslator.pdfminer.pdftypes import PDFObjRef
from src.doctranslator.pdfminer.pdftypes import PDFStream
from src.doctranslator.pdfminer.pdftypes import dict_value
from src.doctranslator.pdfminer.pdftypes import list_value
from src.doctranslator.pdfminer.pdftypes import resolve1
from src.doctranslator.pdfminer.pdftypes import stream_value
from src.doctranslator.pdfminer.psexceptions import PSEOF
from src.doctranslator.pdfminer.psexceptions import PSTypeError
from src.doctranslator.pdfminer.psparser import KWD
from src.doctranslator.pdfminer.psparser import LIT
from src.doctranslator.pdfminer.psparser import PSKeyword
from src.doctranslator.pdfminer.psparser import PSLiteral
from src.doctranslator.pdfminer.psparser import PSStackParser
from src.doctranslator.pdfminer.psparser import PSStackType
from src.doctranslator.pdfminer.psparser import keyword_name
from src.doctranslator.pdfminer.psparser import literal_name
from src.doctranslator.pdfminer.utils import MATRIX_IDENTITY
from src.doctranslator.pdfminer.utils import Matrix
from src.doctranslator.pdfminer.utils import PathSegment
from src.doctranslator.pdfminer.utils import Point
from src.doctranslator.pdfminer.utils import Rect
from src.doctranslator.pdfminer.utils import choplist
from src.doctranslator.pdfminer.utils import mult_matrix

log = logging.getLogger(__name__)


class PDFResourceError(PDFException):
    pass


class PDFInterpreterError(PDFException):
    pass


LITERAL_PDF = LIT("PDF")
LITERAL_TEXT = LIT("Text")
LITERAL_FONT = LIT("Font")
LITERAL_FORM = LIT("Form")
LITERAL_IMAGE = LIT("Image")


class PDFTextState:
    matrix: Matrix
    linematrix: Point

    def __init__(self) -> None:
        self.font: PDFFont | None = None
        self.fontsize: float = 0
        self.charspace: float = 0
        self.wordspace: float = 0
        self.scaling: float = 100
        self.leading: float = 0
        self.render: int = 0
        self.rise: float = 0
        self.reset()

    def __repr__(self) -> str:
        return (
            "<PDFTextState: font=%r, fontsize=%r, charspace=%r, "
            "wordspace=%r, scaling=%r, leading=%r, render=%r, rise=%r, "
            "matrix=%r, linematrix=%r>"
            % (
                self.font,
                self.fontsize,
                self.charspace,
                self.wordspace,
                self.scaling,
                self.leading,
                self.render,
                self.rise,
                self.matrix,
                self.linematrix,
            )
        )

    def copy(self) -> "PDFTextState":
        obj = PDFTextState()
        obj.font = self.font
        obj.fontsize = self.fontsize
        obj.charspace = self.charspace
        obj.wordspace = self.wordspace
        obj.scaling = self.scaling
        obj.leading = self.leading
        obj.render = self.render
        obj.rise = self.rise
        obj.matrix = self.matrix
        obj.linematrix = self.linematrix
        obj.font_id = getattr(self, "font_id", None)
        return obj

    def reset(self) -> None:
        self.matrix = MATRIX_IDENTITY
        self.linematrix = (0, 0)


Color = Union[
    float,  # Greyscale
    tuple[float, float, float],  # R, G, B
    tuple[float, float, float, float],  # C, M, Y, K
]


class PDFGraphicState:
    def __init__(self) -> None:
        self.linewidth: float = 0
        self.linecap: object | None = None
        self.linejoin: object | None = None
        self.miterlimit: object | None = None
        self.dash: tuple[object, object] | None = None
        self.intent: object | None = None
        self.flatness: object | None = None

        # stroking color
        self.scolor: Color | None = None

        # non stroking color
        self.ncolor: Color | None = None

    def copy(self) -> "PDFGraphicState":
        obj = PDFGraphicState()
        obj.linewidth = self.linewidth
        obj.linecap = self.linecap
        obj.linejoin = self.linejoin
        obj.miterlimit = self.miterlimit
        obj.dash = self.dash
        obj.intent = self.intent
        obj.flatness = self.flatness
        obj.scolor = self.scolor
        obj.ncolor = self.ncolor
        return obj

    def __repr__(self) -> str:
        return (
            "<PDFGraphicState: linewidth=%r, linecap=%r, linejoin=%r, "
            " miterlimit=%r, dash=%r, intent=%r, flatness=%r, "
            " stroking color=%r, non stroking color=%r>"
            % (
                self.linewidth,
                self.linecap,
                self.linejoin,
                self.miterlimit,
                self.dash,
                self.intent,
                self.flatness,
                self.scolor,
                self.ncolor,
            )
        )


class PDFResourceManager:
    """Repository of shared resources.

    ResourceManager facilitates reuse of shared resources
    such as fonts and images so that large objects are not
    allocated multiple times.
    """

    def __init__(self, caching: bool = True) -> None:
        self.caching = caching
        self._cached_fonts: dict[object, PDFFont] = {}

    def get_procset(self, procs: Sequence[object]) -> None:
        for proc in procs:
            if proc is not LITERAL_PDF and proc is not LITERAL_TEXT:
                log.debug("Unknown procset: %r", proc)

    def get_cmap(self, cmapname: str, strict: bool = False) -> CMapBase:
        try:
            return CMapDB.get_cmap(cmapname)
        except CMapDB.CMapNotFound:
            if strict:
                raise
            return CMap()

    def _get_font_subtype(self, spec: Mapping[str, object]) -> str:
        """Determine the font subtype from a font spec dictionary."""
        if "Subtype" in spec:
            return literal_name(spec["Subtype"])
        if settings.STRICT:
            raise PDFFontError("Font Subtype is not specified.")
        return "Type1"

    def _create_type0_font(self, spec: Mapping[str, object]) -> PDFFont:
        """Build a Type0 (composite) font by delegating to its descendant font."""
        dfonts = list_value(spec["DescendantFonts"])
        if not dfonts:
            raise AssertionError
        subspec = dict_value(dfonts[0]).copy()
        for k in ("Encoding", "ToUnicode"):
            if k in spec:
                subspec[k] = resolve1(spec[k])
        return self.get_font(None, subspec)

    def _create_font(self, spec: Mapping[str, object]) -> PDFFont:
        """Create a new PDFFont instance from a font spec dictionary."""
        if settings.STRICT and spec["Type"] is not LITERAL_FONT:
            raise PDFFontError("Type is not /Font")

        subtype = self._get_font_subtype(spec)

        if subtype in ("Type1", "MMType1"):
            return PDFType1Font(self, spec)
        if subtype == "TrueType":
            return PDFTrueTypeFont(self, spec)
        if subtype == "Type3":
            return PDFType3Font(self, spec)
        if subtype in ("CIDFontType0", "CIDFontType2"):
            return PDFCIDFont(self, spec)
        if subtype == "Type0":
            return self._create_type0_font(spec)
        if settings.STRICT:
            raise PDFFontError("Invalid Font spec: %r" % spec)
        return PDFType1Font(self, spec)  # this is so wrong!

    def get_font(self, objid: object, spec: Mapping[str, object]) -> PDFFont:
        if objid and objid in self._cached_fonts:
            return self._cached_fonts[objid]
        log.debug("get_font: create: objid=%r, spec=%r", objid, spec)
        font = self._create_font(spec)
        if objid and self.caching:
            self._cached_fonts[objid] = font
        return font


class PDFContentParser(PSStackParser[Union[PSKeyword, PDFStream]]):
    def __init__(self, streams: Sequence[object]) -> None:
        self.streams = streams
        self.istream = 0
        # PSStackParser.__init__ is called with a cast placeholder because all
        # methods that access self.fp are overloaded here to call fillfp() first,
        # which lazily initialises self.fp from self.streams.
        PSStackParser.__init__(self, cast(BytesIO, None))

    def fillfp(self) -> None:
        if not self.fp:
            if self.istream < len(self.streams):
                strm = stream_value(self.streams[self.istream])
                self.istream += 1
            else:
                raise PSEOF("Unexpected EOF, file truncated?")
            self.fp = BytesIO(strm.get_data())

    def seek(self, pos: int) -> None:
        self.fillfp()
        PSStackParser.seek(self, pos)

    def fillbuf(self) -> None:
        if self.charpos < len(self.buf):
            return
        while 1:
            self.fillfp()
            self.bufpos = self.fp.tell()
            self.buf = self.fp.read(self.BUFSIZ)
            if self.buf:
                break
            self.fp = None  # type: ignore[assignment]
        self.charpos = 0

    def get_inline_data(self, pos: int, target: bytes = b"EI") -> tuple[int, bytes]:
        self.seek(pos)
        i = 0
        data = b""
        while i <= len(target):
            self.fillbuf()
            if i:
                ci = self.buf[self.charpos]
                c = bytes((ci,))
                data += c
                self.charpos += 1
                if (
                    len(target) <= i
                    and c.isspace()
                    or i < len(target)
                    and c == (bytes((target[i],)))
                ):
                    i += 1
                else:
                    i = 0
            else:
                try:
                    j = self.buf.index(target[0], self.charpos)
                    data += self.buf[self.charpos : j + 1]
                    self.charpos = j + 1
                    i = 1
                except ValueError:
                    data += self.buf[self.charpos :]
                    self.charpos = len(self.buf)
        data = data[: -(len(target) + 1)]  # strip the last part
        data = re.sub(rb"(\x0d\x0a|[\x0d\x0a])$", b"", data)
        return (pos, data)

    def flush(self) -> None:
        self.add_results(*self.popall())

    KEYWORD_BI = KWD(b"BI")
    KEYWORD_ID = KWD(b"ID")
    KEYWORD_EI = KWD(b"EI")

    def _get_inline_image_eos(self, d: dict) -> bytes:
        """Determine the end-of-stream marker for an inline image."""
        eos = b"EI"
        filter_val = d.get("F", None)
        if filter_val is None:
            return eos
        if isinstance(filter_val, PSLiteral):
            filter_val = [filter_val]
        if filter_val[0] in LITERALS_ASCII85_DECODE:
            eos = b"~>"
        return eos

    def _handle_inline_image(self, pos: int) -> None:
        """Handle the ID keyword by parsing the inline image data."""
        try:
            (_, objs) = self.end_type("inline")
            if len(objs) % 2 != 0:
                error_msg = f"Invalid dictionary construct: {objs!r}"
                raise PSTypeError(error_msg)
            d = {literal_name(k): resolve1(v) for (k, v) in choplist(2, objs)}
            eos = self._get_inline_image_eos(d)
            (pos, data) = self.get_inline_data(pos + len(b"ID "), target=eos)
            if eos != b"EI":  # it may be necessary for decoding
                data += eos
            obj = PDFStream(d, data)
            self.push((pos, obj))
            if eos == b"EI":  # otherwise it is still in the stream
                self.push((pos, self.KEYWORD_EI))
        except PSTypeError:
            if settings.STRICT:
                raise

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        if token is self.KEYWORD_BI:
            # inline image within a content stream
            self.start_type(pos, "inline")
        elif token is self.KEYWORD_ID:
            self._handle_inline_image(pos)
        else:
            self.push((pos, token))


PDFStackT = PSStackType[PDFStream]
"""Types that may appear on the PDF argument stack."""


class PDFPageInterpreter:
    """Processor for the content of a PDF page

    Reference: PDF Reference, Appendix A, Operator Summary
    """

    def __init__(self, rsrcmgr: PDFResourceManager, device: PDFDevice) -> None:
        self.rsrcmgr = rsrcmgr
        self.device = device

    def dup(self) -> "PDFPageInterpreter":
        return self.__class__(self.rsrcmgr, self.device)

    @staticmethod
    def _resolve_colorspace(spec: object) -> PDFColorSpace | None:
        """Resolve a colorspace spec to a PDFColorSpace instance."""
        if isinstance(spec, list):
            name = literal_name(spec[0])
        else:
            name = literal_name(spec)
        if name == "ICCBased" and isinstance(spec, list) and len(spec) >= 2:
            return PDFColorSpace(name, stream_value(spec[1])["N"])
        if name == "DeviceN" and isinstance(spec, list) and len(spec) >= 2:
            return PDFColorSpace(name, len(list_value(spec[1])))
        return PREDEFINED_COLORSPACE.get(name)

    def _load_font_resources(self, v: object) -> None:
        for fontid, spec in dict_value(v).items():
            objid = spec.objid if isinstance(spec, PDFObjRef) else None
            self.fontmap[fontid] = self.rsrcmgr.get_font(objid, dict_value(spec))

    def _load_colorspace_resources(self, v: object) -> None:
        for csid, spec in dict_value(v).items():
            colorspace = self._resolve_colorspace(resolve1(spec))
            if colorspace is not None:
                self.csmap[csid] = colorspace

    def _load_xobject_resources(self, v: object) -> None:
        for xobjid, xobjstrm in dict_value(v).items():
            self.xobjmap[xobjid] = xobjstrm

    def init_resources(self, resources: dict[object, object]) -> None:
        """Prepare the fonts and XObjects listed in the Resource attribute."""
        self.resources = resources
        self.fontmap: dict[object, PDFFont] = {}
        self.xobjmap = {}
        self.csmap: dict[str, PDFColorSpace] = PREDEFINED_COLORSPACE.copy()
        if not resources:
            return
        for k, v in dict_value(resources).items():
            log.debug("Resource: %r: %r", k, v)
            if k == "Font":
                self._load_font_resources(v)
            elif k == "ColorSpace":
                self._load_colorspace_resources(v)
            elif k == "ProcSet":
                self.rsrcmgr.get_procset(list_value(v))
            elif k == "XObject":
                self._load_xobject_resources(v)

    def init_state(self, ctm: Matrix) -> None:
        """Initialize the text and graphic states for rendering a page."""
        # gstack: stack for graphical states.
        self.gstack: list[tuple[Matrix, PDFTextState, PDFGraphicState]] = []
        self.ctm = ctm
        self.device.set_ctm(self.ctm)
        self.textstate = PDFTextState()
        self.graphicstate = PDFGraphicState()
        self.curpath: list[PathSegment] = []
        # argstack: stack for command arguments.
        self.argstack: list[PDFStackT] = []
        # set some global states.
        self.scs: PDFColorSpace | None = None
        self.ncs: PDFColorSpace | None = None
        if self.csmap:
            self.scs = self.ncs = next(iter(self.csmap.values()))

    def push(self, obj: PDFStackT) -> None:
        self.argstack.append(obj)

    def pop(self, n: int) -> list[PDFStackT]:
        if n == 0:
            return []
        x = self.argstack[-n:]
        self.argstack = self.argstack[:-n]
        return x

    def get_current_state(self) -> tuple[Matrix, PDFTextState, PDFGraphicState]:
        return (self.ctm, self.textstate.copy(), self.graphicstate.copy())

    def set_current_state(
        self,
        state: tuple[Matrix, PDFTextState, PDFGraphicState],
    ) -> None:
        (self.ctm, self.textstate, self.graphicstate) = state
        self.device.set_ctm(self.ctm)

    def do_q(self) -> None:
        """Save graphics state"""
        self.gstack.append(self.get_current_state())

    def do_q_upper(self) -> None:
        """Restore graphics state (PDF operator Q, uppercase)"""
        if self.gstack:
            self.set_current_state(self.gstack.pop())

    # PDF operator dispatch alias â€“ 'Q' maps to do_Q via getattr
    do_Q = do_q_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_cm(
        self,
        a1: PDFStackT,
        b1: PDFStackT,
        c1: PDFStackT,
        d1: PDFStackT,
        e1: PDFStackT,
        f1: PDFStackT,
    ) -> None:
        """Concatenate matrix to current transformation matrix"""
        matrix = safe_matrix(a1, b1, c1, d1, e1, f1)

        if matrix is None:
            log.warning(
                f"Cannot concatenate matrix to current transformation matrix because not all values in {(a1, b1, c1, d1, e1, f1)!r} can be parsed as floats"
            )
        else:
            self.ctm = mult_matrix(matrix, self.ctm)
            self.device.set_ctm(self.ctm)

    def do_w(self, linewidth: PDFStackT) -> None:
        """Set line width"""
        linewidth_f = safe_float(linewidth)
        if linewidth_f is None:
            log.warning(
                f"Cannot set line width because {linewidth!r} is an invalid float value"
            )
        else:
            self.graphicstate.linewidth = linewidth_f

    def do_j_upper(self, linecap: PDFStackT) -> None:
        """Set line cap style (PDF operator J, uppercase)"""
        self.graphicstate.linecap = linecap

    # PDF operator dispatch alias â€“ 'J' maps to do_J via getattr
    do_J = do_j_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_j(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, linejoin: PDFStackT
    ) -> None:
        """Set line join style"""
        self.graphicstate.linejoin = linejoin

    def do_m_upper(self, miterlimit: PDFStackT) -> None:
        """Set miter limit (PDF operator M, uppercase)"""
        self.graphicstate.miterlimit = miterlimit

    # PDF operator dispatch alias â€“ 'M' maps to do_M via getattr
    do_M = do_m_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_d(self, dash: PDFStackT, phase: PDFStackT) -> None:
        """Set line dash pattern"""
        self.graphicstate.dash = (dash, phase)

    def do_ri(self, intent: PDFStackT) -> None:
        """Set color rendering intent"""
        self.graphicstate.intent = intent

    def do_i(self, flatness: PDFStackT) -> None:
        """Set flatness tolerance"""
        self.graphicstate.flatness = flatness

    def do_gs(self, name: PDFStackT) -> None:
        """Set parameters from graphics state parameter dictionary.

        Intentionally not implemented: ExtGState dictionary handling (soft masks,
        blend modes, etc.) is not required for the text-extraction and translation
        pipeline.  PDF viewing fidelity may vary for documents that rely heavily on
        extended graphic-state parameters.
        """

    def do_m(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, x: PDFStackT, y: PDFStackT
    ) -> None:
        """Begin new subpath"""
        x_f = safe_float(x)
        y_f = safe_float(y)

        if x_f is None or y_f is None:
            point = ("m", x, y)
            log.warning(
                f"Cannot start new subpath because not all values in {point!r} can be parsed as floats"
            )
        else:
            point = ("m", x_f, y_f)
            self.curpath.append(point)

    def do_l(self, x: PDFStackT, y: PDFStackT) -> None:
        """Append straight line segment to path"""
        x_f = safe_float(x)
        y_f = safe_float(y)
        if x_f is None or y_f is None:
            point = ("l", x, y)
            log.warning(
                f"Cannot append straight line segment to path because not all values in {point!r} can be parsed as floats"
            )
        else:
            point = ("l", x_f, y_f)
            self.curpath.append(point)

    def do_c(
        self,
        x1: PDFStackT,
        y1: PDFStackT,
        x2: PDFStackT,
        y2: PDFStackT,
        x3: PDFStackT,
        y3: PDFStackT,
    ) -> None:
        """Append curved segment to path (three control points)"""
        x1_f = safe_float(x1)
        y1_f = safe_float(y1)
        x2_f = safe_float(x2)
        y2_f = safe_float(y2)
        x3_f = safe_float(x3)
        y3_f = safe_float(y3)
        if (
            x1_f is None
            or y1_f is None
            or x2_f is None
            or y2_f is None
            or x3_f is None
            or y3_f is None
        ):
            point = ("c", x1, y1, x2, y2, x3, y3)
            log.warning(
                f"Cannot append curved segment to path because not all values in {point!r} can be parsed as floats"
            )
        else:
            point = ("c", x1_f, y1_f, x2_f, y2_f, x3_f, y3_f)
            self.curpath.append(point)

    def do_v(self, x2: PDFStackT, y2: PDFStackT, x3: PDFStackT, y3: PDFStackT) -> None:
        """Append curved segment to path (initial point replicated)"""
        x2_f = safe_float(x2)
        y2_f = safe_float(y2)
        x3_f = safe_float(x3)
        y3_f = safe_float(y3)
        if x2_f is None or y2_f is None or x3_f is None or y3_f is None:
            point = ("v", x2, y2, x3, y3)
            log.warning(
                f"Cannot append curved segment to path because not all values in {point!r} can be parsed as floats"
            )
        else:
            point = ("v", x2_f, y2_f, x3_f, y3_f)
            self.curpath.append(point)

    def do_y(self, x1: PDFStackT, y1: PDFStackT, x3: PDFStackT, y3: PDFStackT) -> None:
        """Append curved segment to path (final point replicated)"""
        x1_f = safe_float(x1)
        y1_f = safe_float(y1)
        x3_f = safe_float(x3)
        y3_f = safe_float(y3)
        if x1_f is None or y1_f is None or x3_f is None or y3_f is None:
            point = ("y", x1, y1, x3, y3)
            log.warning(
                f"Cannot append curved segment to path because not all values in {point!r} can be parsed as floats"
            )
        else:
            point = ("y", x1_f, y1_f, x3_f, y3_f)
            self.curpath.append(point)

    def do_h(self) -> None:
        """Close subpath"""
        self.curpath.append(("h",))

    def do_re(self, x: PDFStackT, y: PDFStackT, w: PDFStackT, h: PDFStackT) -> None:
        """Append rectangle to path"""
        x_f = safe_float(x)
        y_f = safe_float(y)
        w_f = safe_float(w)
        h_f = safe_float(h)

        if x_f is None or y_f is None or w_f is None or h_f is None:
            values = (x, y, w, h)
            log.warning(
                f"Cannot append rectangle to path because not all values in {values!r} can be parsed as floats"
            )
        else:
            self.curpath.append(("m", x_f, y_f))
            self.curpath.append(("l", x_f + w_f, y_f))
            self.curpath.append(("l", x_f + w_f, y_f + h_f))
            self.curpath.append(("l", x_f, y_f + h_f))
            self.curpath.append(("h",))

    def do_s_upper(self) -> None:
        """Stroke path (PDF operator S, uppercase)"""
        self.device.paint_path(self.graphicstate, True, False, False, self.curpath)
        self.curpath = []

    # PDF operator dispatch alias â€“ 'S' maps to do_S via getattr
    do_S = do_s_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_s(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """Close and stroke path"""
        self.do_h()
        self.do_S()

    def do_f(self) -> None:
        """Fill path using nonzero winding number rule"""
        self.device.paint_path(self.graphicstate, False, True, False, self.curpath)
        self.curpath = []

    def do_f_upper(self) -> None:
        """Fill path using nonzero winding number rule (obsolete, PDF operator F, uppercase).

        Per the PDF specification, the F operator is obsolete and equivalent to f.
        The current path is cleared but no painting action is taken by this no-op handler.
        """

    # PDF operator dispatch alias â€“ 'F' maps to do_F via getattr
    do_F = do_f_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_f_a(self) -> None:
        """Fill path using even-odd rule"""
        self.device.paint_path(self.graphicstate, False, True, True, self.curpath)
        self.curpath = []

    def do_b_upper(self) -> None:
        """Fill and stroke path using nonzero winding number rule (PDF operator B, uppercase)"""
        self.device.paint_path(self.graphicstate, True, True, False, self.curpath)
        self.curpath = []

    # PDF operator dispatch alias â€“ 'B' maps to do_B via getattr
    do_B = do_b_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_b_upper_a(self) -> None:
        """Fill and stroke path using even-odd rule (PDF operator B*, uppercase)"""
        self.device.paint_path(self.graphicstate, True, True, True, self.curpath)
        self.curpath = []

    # PDF operator dispatch alias â€“ 'B*' maps to do_B_a via getattr
    do_B_a = do_b_upper_a  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_b(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """Close, fill, and stroke path using nonzero winding number rule"""
        self.do_h()
        self.do_B()

    def do_b_a(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """Close, fill, and stroke path using even-odd rule"""
        self.do_h()
        self.do_B_a()

    def do_n(self) -> None:
        """End path without filling or stroking"""
        self.curpath = []

    def do_w_upper(self) -> None:
        """Set clipping path using nonzero winding number rule (PDF operator W, uppercase).

        Clipping path operations are not needed for the text-extraction and translation
        pipeline; this is an intentional no-op to preserve dispatch compatibility.
        """

    # PDF operator dispatch alias â€“ 'W' maps to do_W via getattr
    do_W = do_w_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_w_upper_a(self) -> None:
        """Set clipping path using even-odd rule (PDF operator W*, uppercase).

        Clipping path operations are not needed for the text-extraction and translation
        pipeline; this is an intentional no-op to preserve dispatch compatibility.
        """

    # PDF operator dispatch alias â€“ 'W*' maps to do_W_a via getattr
    do_W_a = do_w_upper_a  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_cs_upper(self, name: PDFStackT) -> None:
        """Set color space for stroking operations (PDF operator CS, uppercase).

        Introduced in PDF 1.1
        """
        try:
            self.scs = self.csmap[literal_name(name)]
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError("Undefined ColorSpace: %r" % name)

    # PDF operator dispatch alias â€“ 'CS' maps to do_CS via getattr
    do_CS = do_cs_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_cs(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, name: PDFStackT
    ) -> None:
        """Set color space for nonstroking operations"""
        try:
            self.ncs = self.csmap[literal_name(name)]
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError("Undefined ColorSpace: %r" % name)

    def do_g_upper(self, gray: PDFStackT) -> None:
        """Set gray level for stroking operations (PDF operator G, uppercase)"""
        gray_f = safe_float(gray)

        if gray_f is None:
            log.warning(
                f"Cannot set gray level because {gray!r} is an invalid float value"
            )
        else:
            self.graphicstate.scolor = gray_f
            self.scs = self.csmap["DeviceGray"]

    # PDF operator dispatch alias â€“ 'G' maps to do_G via getattr
    do_G = do_g_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_g(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, gray: PDFStackT
    ) -> None:
        """Set gray level for nonstroking operations"""
        gray_f = safe_float(gray)

        if gray_f is None:
            log.warning(
                f"Cannot set gray level because {gray!r} is an invalid float value"
            )
        else:
            self.graphicstate.ncolor = gray_f
            self.ncs = self.csmap["DeviceGray"]

    def do_rg_upper(self, r: PDFStackT, g: PDFStackT, b: PDFStackT) -> None:
        """Set RGB color for stroking operations (PDF operator RG, uppercase)"""
        rgb = safe_rgb(r, g, b)

        if rgb is None:
            log.warning(
                f"Cannot set RGB stroke color because not all values in {(r, g, b)!r} can be parsed as floats"
            )
        else:
            self.graphicstate.scolor = rgb
            self.scs = self.csmap["DeviceRGB"]

    # PDF operator dispatch alias â€“ 'RG' maps to do_RG via getattr
    do_RG = do_rg_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_rg(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, r: PDFStackT, g: PDFStackT, b: PDFStackT
    ) -> None:
        """Set RGB color for nonstroking operations"""
        rgb = safe_rgb(r, g, b)

        if rgb is None:
            log.warning(
                f"Cannot set RGB non-stroke color because not all values in {(r, g, b)!r} can be parsed as floats"
            )
        else:
            self.graphicstate.ncolor = rgb
            self.ncs = self.csmap["DeviceRGB"]

    def do_k_upper(
        self, c: PDFStackT, m: PDFStackT, y: PDFStackT, k: PDFStackT
    ) -> None:
        """Set CMYK color for stroking operations (PDF operator K, uppercase)"""
        cmyk = safe_cmyk(c, m, y, k)

        if cmyk is None:
            log.warning(
                f"Cannot set CMYK stroke color because not all values in {(c, m, y, k)!r} can be parsed as floats"
            )
        else:
            self.graphicstate.scolor = cmyk
            self.scs = self.csmap["DeviceCMYK"]

    # PDF operator dispatch alias â€“ 'K' maps to do_K via getattr
    do_K = do_k_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_k(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, c: PDFStackT, m: PDFStackT, y: PDFStackT, k: PDFStackT
    ) -> None:
        """Set CMYK color for nonstroking operations"""
        cmyk = safe_cmyk(c, m, y, k)

        if cmyk is None:
            log.warning(
                f"Cannot set CMYK non-stroke color because not all values in {(c, m, y, k)!r} can be parsed as floats"
            )
        else:
            self.graphicstate.ncolor = cmyk
            self.ncs = self.csmap["DeviceCMYK"]

    def _set_color_value(self, color: "Color", is_stroke: bool) -> None:
        """Assign the resolved color to stroking or nonstroking graphic state."""
        if is_stroke:
            self.graphicstate.scolor = color
        else:
            self.graphicstate.ncolor = color

    def _apply_gray_color(self, is_stroke: bool, label: str) -> None:
        """Pop one value and apply as grayscale color."""
        gray = self.pop(1)[0]
        gray_f = safe_float(gray)
        if gray_f is None:
            log.warning(
                f"Cannot set gray {label} color because {gray!r} is an invalid float value"
            )
        else:
            self._set_color_value(gray_f, is_stroke)

    def _apply_rgb_color(self, is_stroke: bool, label: str) -> None:
        """Pop three values and apply as RGB color."""
        values = self.pop(3)
        rgb = safe_rgb(*values)
        if rgb is None:
            log.warning(
                f"Cannot set RGB {label} color because not all values in {values!r} can be parsed as floats"
            )
        else:
            self._set_color_value(rgb, is_stroke)

    def _apply_cmyk_color(self, is_stroke: bool, label: str) -> None:
        """Pop four values and apply as CMYK color."""
        values = self.pop(4)
        cmyk = safe_cmyk(*values)
        if cmyk is None:
            log.warning(
                f"Cannot set CMYK {label} color because not all values in {values!r} can be parsed as floats"
            )
        else:
            self._set_color_value(cmyk, is_stroke)

    def _apply_color(self, n: int, is_stroke: bool, label: str) -> None:
        """Apply a color value with n components to stroking or nonstroking state."""
        if n == 1:
            self._apply_gray_color(is_stroke, label)
        elif n == 3:
            self._apply_rgb_color(is_stroke, label)
        elif n == 4:
            self._apply_cmyk_color(is_stroke, label)
        else:
            log.warning(
                f"Cannot set {label} color because {n} components are specified "
                "but only 1 (grayscale), 3 (rgb) and 4 (cmyk) are supported"
            )

    def do_scn_upper(self) -> None:
        """Set color for stroking operations (PDF operator SCN, uppercase)."""
        if self.scs:
            n = self.scs.ncomponents
        elif settings.STRICT:
            raise PDFInterpreterError("No colorspace specified!")
        else:
            n = 1
        self._apply_color(n, is_stroke=True, label="stroke")

    # PDF operator dispatch alias â€“ 'SCN' maps to do_SCN via getattr
    do_SCN = do_scn_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_scn(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """Set color for nonstroking operations"""
        if self.ncs:
            n = self.ncs.ncomponents
        elif settings.STRICT:
            raise PDFInterpreterError("No colorspace specified!")
        else:
            n = 1
        self._apply_color(n, is_stroke=False, label="non-stroke")

    def do_sc_upper(self) -> None:
        """Set color for stroking operations (PDF operator SC, uppercase)"""
        self.do_SCN()

    # PDF operator dispatch alias â€“ 'SC' maps to do_SC via getattr
    do_SC = do_sc_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_sc(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """Set color for nonstroking operations"""
        self.do_scn()

    def do_sh(self, name: object) -> None:
        """Paint area defined by shading pattern.

        Shading-pattern rendering is not implemented in the base interpreter;
        this is an intentional no-op.  Shading patterns are ignored during
        text extraction and translation without loss of textual content.
        """

    def do_BT(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """Begin text object

        Initializing the text matrix, Tm, and the text line matrix, Tlm, to
        the identity matrix. Text objects cannot be nested; a second BT cannot
        appear before an ET.
        """
        self.textstate.reset()

    def do_ET(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """End a text object.

        Text state clean-up after a BT/ET block is handled at the device level
        via the renderer; the base interpreter needs no additional action here.
        """

    def do_BX(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """Begin compatibility section.

        Content inside a BX/EX pair is implementation-defined and may be
        ignored by conforming readers that do not recognise the extension.
        The base interpreter intentionally ignores the BX marker.
        """

    def do_EX(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """End compatibility section.

        Marks the end of a BX/EX compatibility block; ignored by the base
        interpreter as it does not process implementation-specific extensions.
        """

    def do_MP(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, tag: PDFStackT
    ) -> None:
        """Define marked-content point"""
        if isinstance(tag, PSLiteral):
            self.device.do_tag(tag)
        else:
            log.warning(
                f"Cannot define marked-content point because {tag!r} is not a PSLiteral"
            )

    def do_DP(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, tag: PDFStackT, props: PDFStackT
    ) -> None:
        """Define marked-content point with property list"""
        if isinstance(tag, PSLiteral):
            self.device.do_tag(tag, props)
        else:
            log.warning(
                f"Cannot define marked-content point with property list because {tag!r} is not a PSLiteral"
            )

    def do_BMC(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, tag: PDFStackT
    ) -> None:
        """Begin marked-content sequence"""
        if isinstance(tag, PSLiteral):
            self.device.begin_tag(tag)
        else:
            log.warning(
                f"Cannot begin marked-content sequence because {tag!r} is not a PSLiteral"
            )

    def do_BDC(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, tag: PDFStackT, props: PDFStackT
    ) -> None:
        """Begin marked-content sequence with property list"""
        if isinstance(tag, PSLiteral):
            self.device.begin_tag(tag, props)
        else:
            log.warning(
                f"Cannot begin marked-content sequence with property list because {tag!r} is not a PSLiteral"
            )

    def do_EMC(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """End marked-content sequence"""
        self.device.end_tag()

    def do_Tc(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, space: PDFStackT
    ) -> None:
        """Set character spacing.

        Character spacing is used by the Tj, TJ, and ' operators.

        :param space: a number expressed in unscaled text space units.
        """
        charspace = safe_float(space)
        if charspace is None:
            log.warning(
                f"Could not set character spacing because {space!r} is an invalid float value"
            )
        else:
            self.textstate.charspace = charspace

    def do_Tw(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, space: PDFStackT
    ) -> None:
        """Set the word spacing.

        Word spacing is used by the Tj, TJ, and ' operators.

        :param space: a number expressed in unscaled text space units
        """
        wordspace = safe_float(space)
        if wordspace is None:
            log.warning(
                f"Could not set word spacing becuase {space!r} is an invalid float value"
            )
        else:
            self.textstate.wordspace = wordspace

    def do_Tz(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, scale: PDFStackT
    ) -> None:
        """Set the horizontal scaling.

        :param scale: is a number specifying the percentage of the normal width
        """
        scale_f = safe_float(scale)

        if scale_f is None:
            log.warning(
                f"Could not set horizontal scaling because {scale!r} is an invalid float value"
            )
        else:
            self.textstate.scaling = scale_f

    def do_TL(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, leading: PDFStackT
    ) -> None:
        """Set the text leading.

        Text leading is used only by the T*, ', and " operators.

        :param leading: a number expressed in unscaled text space units
        """
        leading_f = safe_float(leading)
        if leading_f is None:
            log.warning(
                f"Could not set text leading because {leading!r} is an invalid float value"
            )
        else:
            self.textstate.leading = -leading_f

    def do_Tf(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, fontid: PDFStackT, fontsize: PDFStackT
    ) -> None:
        """Set the text font

        :param fontid: the name of a font resource in the Font subdictionary
            of the current resource dictionary
        :param fontsize: size is a number representing a scale factor.
        """
        try:
            self.textstate.font = self.fontmap[literal_name(fontid)]
            self.textstate.font_id = literal_name(fontid)
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError("Undefined Font id: %r" % fontid)
            self.textstate.font = self.rsrcmgr.get_font(None, {})

        fontsize_f = safe_float(fontsize)
        if fontsize_f is None:
            log.warning(
                f"Could not set text font because {fontsize!r} is an invalid float value"
            )
        else:
            self.textstate.fontsize = fontsize_f

    def do_Tr(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, render: PDFStackT
    ) -> None:
        """Set the text rendering mode"""
        render_i = safe_int(render)

        if render_i is None:
            log.warning(
                f"Could not set text rendering mode because {render!r} is an invalid int value"
            )
        else:
            self.textstate.render = render_i

    def do_Ts(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, rise: PDFStackT
    ) -> None:
        """Set the text rise

        :param rise: a number expressed in unscaled text space units
        """
        rise_f = safe_float(rise)

        if rise_f is None:
            log.warning(
                f"Could not set text rise because {rise!r} is an invalid float value"
            )
        else:
            self.textstate.rise = rise_f

    def do_Td(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, tx: PDFStackT, ty: PDFStackT
    ) -> None:
        """Move to the start of the next line

        Offset from the start of the current line by (tx , ty).
        """
        tx_ = safe_float(tx)
        ty_ = safe_float(ty)
        if tx_ is not None and ty_ is not None:
            (a, b, c, d, e, f) = self.textstate.matrix
            e_new = tx_ * a + ty_ * c + e
            f_new = tx_ * b + ty_ * d + f
            self.textstate.matrix = (a, b, c, d, e_new, f_new)

        elif settings.STRICT:
            raise PDFValueError(f"Invalid offset ({tx!r}, {ty!r}) for Td")

        self.textstate.linematrix = (0, 0)

    def do_td_upper(self, tx: PDFStackT, ty: PDFStackT) -> None:
        """Move to the start of the next line (PDF operator TD, uppercase).

        offset from the start of the current line by (tx , ty). As a side effect, this
        operator sets the leading parameter in the text state.
        """
        tx_ = safe_float(tx)
        ty_ = safe_float(ty)

        if tx_ is not None and ty_ is not None:
            (a, b, c, d, e, f) = self.textstate.matrix
            e_new = tx_ * a + ty_ * c + e
            f_new = tx_ * b + ty_ * d + f
            self.textstate.matrix = (a, b, c, d, e_new, f_new)
        elif settings.STRICT:
            raise PDFValueError("Invalid offset ({tx}, {ty}) for TD")

        if ty_ is not None:
            self.textstate.leading = ty_

        self.textstate.linematrix = (0, 0)

    # PDF operator dispatch alias â€“ 'TD' maps to do_TD via getattr
    do_TD = do_td_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_Tm(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
        a: PDFStackT,
        b: PDFStackT,
        c: PDFStackT,
        d: PDFStackT,
        e: PDFStackT,
        f: PDFStackT,
    ) -> None:
        """Set text matrix and text line matrix"""
        values = (a, b, c, d, e, f)
        matrix = safe_matrix(*values)

        if matrix is None:
            log.warning(
                f"Could not set text matrix because not all values in {values!r} can be parsed as floats"
            )
        else:
            self.textstate.matrix = matrix
            self.textstate.linematrix = (0, 0)

    def do_T_a(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """Move to start of next text line"""
        (a, b, c, d, e, f) = self.textstate.matrix
        self.textstate.matrix = (
            a,
            b,
            c,
            d,
            self.textstate.leading * c + e,
            self.textstate.leading * d + f,
        )
        self.textstate.linematrix = (0, 0)

    def do_tj_upper(self, seq: PDFStackT) -> None:
        """Show text, allowing individual glyph positioning (PDF operator TJ, uppercase)"""
        if self.textstate.font is None:
            if settings.STRICT:
                raise PDFInterpreterError("No font specified!")
            return
        if self.ncs is None:
            raise AssertionError
        self.device.render_string(
            self.textstate,
            cast(PDFTextSeq, seq),
            self.ncs,
            self.graphicstate.copy(),
        )

    # PDF operator dispatch alias â€“ 'TJ' maps to do_TJ via getattr
    do_TJ = do_tj_upper  # NOSONAR - PDF spec mandates this name; dispatch relies on exact method name

    def do_Tj(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, s: PDFStackT
    ) -> None:
        """Show text"""
        self.do_TJ([s])

    def do__q(self, s: PDFStackT) -> None:
        """Move to next line and show text

        The ' (single quote) operator.
        """
        self.do_T_a()
        self.do_TJ([s])

    def do__w(self, aw: PDFStackT, ac: PDFStackT, s: PDFStackT) -> None:
        """Set word and character spacing, move to next line, and show text

        The " (double quote) operator.
        """
        self.do_Tw(aw)
        self.do_Tc(ac)
        self.do_TJ([s])

    def do_BI(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """Begin inline image object.

        The inline image data is parsed by PDFContentParser before this method
        is called; the base interpreter needs no additional action here.
        """

    def do_ID(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self,
    ) -> None:
        """Begin inline image data.

        Inline image data is handled entirely by PDFContentParser._handle_inline_image;
        this dispatch handler is an intentional no-op in the base interpreter.
        """

    def do_EI(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, obj: PDFStackT
    ) -> None:
        """End inline image object"""
        if isinstance(obj, PDFStream) and "W" in obj and "H" in obj:
            iobjid = str(id(obj))
            self.device.begin_figure(iobjid, (0, 0, 1, 1), MATRIX_IDENTITY)
            self.device.render_image(iobjid, obj)
            self.device.end_figure(iobjid)

    def do_Do(  # NOSONAR - PDF operator dispatch requires this exact case-sensitive name
        self, xobjid_arg: PDFStackT
    ) -> None:
        """Invoke named XObject"""
        xobjid = literal_name(xobjid_arg)
        try:
            xobj = stream_value(self.xobjmap[xobjid])
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError("Undefined xobject id: %r" % xobjid)
            return
        log.debug("Processing xobj: %r", xobj)
        subtype = xobj.get("Subtype")
        if subtype is LITERAL_FORM and "BBox" in xobj:
            interpreter = self.dup()
            bbox = cast(Rect, list_value(xobj["BBox"]))
            matrix = cast(Matrix, list_value(xobj.get("Matrix", MATRIX_IDENTITY)))
            # According to PDF reference 1.7 section 4.9.1, XObjects in
            # earlier PDFs (prior to v1.2) use the page's Resources entry
            # instead of having their own Resources entry.
            xobjres = xobj.get("Resources")
            if xobjres:
                resources = dict_value(xobjres)
            else:
                resources = self.resources.copy()
            self.device.begin_figure(xobjid, bbox, matrix)
            interpreter.render_contents(
                resources,
                [xobj],
                ctm=mult_matrix(matrix, self.ctm),
            )
            self.device.end_figure(xobjid)
        elif subtype is LITERAL_IMAGE and "Width" in xobj and "Height" in xobj:
            self.device.begin_figure(xobjid, (0, 0, 1, 1), MATRIX_IDENTITY)
            self.device.render_image(xobjid, xobj)
            self.device.end_figure(xobjid)
        else:
            # Unsupported xobject type (e.g. PostScript XObject): intentionally ignored.
            pass

    def process_page(self, page: PDFPage) -> None:
        log.debug("Processing page: %r", page)
        (x0, y0, x1, y1) = page.mediabox
        if page.rotate == 90:
            ctm = (0, -1, 1, 0, -y0, x1)
        elif page.rotate == 180:
            ctm = (-1, 0, 0, -1, x1, y1)
        elif page.rotate == 270:
            ctm = (0, 1, -1, 0, y1, -x0)
        else:
            ctm = (1, 0, 0, 1, -x0, -y0)
        self.device.begin_page(page, ctm)
        self.render_contents(page.resources, page.contents, ctm=ctm)
        self.device.end_page(page)

    def render_contents(
        self,
        resources: dict[object, object],
        streams: Sequence[object],
        ctm: Matrix = MATRIX_IDENTITY,
    ) -> None:
        """Render the content streams.

        This method may be called recursively.
        """
        log.debug(
            "render_contents: resources=%r, streams=%r, ctm=%r",
            resources,
            streams,
            ctm,
        )
        self.init_resources(resources)
        self.init_state(ctm)
        self.execute(list_value(streams))

    @staticmethod
    def _operator_to_method_name(name: str) -> str:
        """Convert a PDF operator name to the corresponding do_* method name."""
        return "do_%s" % name.replace("*", "_a").replace('"', "_w").replace("'", "_q")

    def _dispatch_operator(self, name: str) -> None:
        """Dispatch a PDF operator to its handler method."""
        method = self._operator_to_method_name(name)
        if not hasattr(self, method):
            if settings.STRICT:
                raise PDFInterpreterError("Unknown operator: %r" % name)
            return
        func = getattr(self, method)
        nargs = func.__code__.co_argcount - 1
        if nargs:
            args = self.pop(nargs)
            log.debug("exec: %s %r", name, args)
            if len(args) == nargs:
                func(*args)
        else:
            log.debug("exec: %s", name)
            func()

    def execute(self, streams: Sequence[object]) -> None:
        try:
            parser = PDFContentParser(streams)
        except PSEOF:
            # empty page
            return
        while True:
            try:
                (_, obj) = parser.nextobject()
            except PSEOF:
                break
            if isinstance(obj, PSKeyword):
                self._dispatch_operator(keyword_name(obj))
            else:
                self.push(obj)
