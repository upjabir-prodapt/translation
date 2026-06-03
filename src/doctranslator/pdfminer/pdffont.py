import logging
import struct
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from io import BytesIO
from typing import TYPE_CHECKING
from typing import Any
from typing import BinaryIO
from typing import cast

import freetype

from src.doctranslator.format.pdf.babelpdf.cmap import CharacterMap
from src.doctranslator.pdfminer import settings
from src.doctranslator.pdfminer.casting import safe_float
from src.doctranslator.pdfminer.casting import safe_rect_list
from src.doctranslator.pdfminer.cmapdb import CMap
from src.doctranslator.pdfminer.cmapdb import CMapBase
from src.doctranslator.pdfminer.cmapdb import CMapDB
from src.doctranslator.pdfminer.cmapdb import CMapParser
from src.doctranslator.pdfminer.cmapdb import FileUnicodeMap
from src.doctranslator.pdfminer.cmapdb import IdentityUnicodeMap
from src.doctranslator.pdfminer.cmapdb import UnicodeMap
from src.doctranslator.pdfminer.encodingdb import EncodingDB
from src.doctranslator.pdfminer.encodingdb import name2unicode
from src.doctranslator.pdfminer.fontmetrics import FONT_METRICS
from src.doctranslator.pdfminer.pdfexceptions import PDFException
from src.doctranslator.pdfminer.pdfexceptions import PDFKeyError
from src.doctranslator.pdfminer.pdfexceptions import PDFValueError
from src.doctranslator.pdfminer.pdftypes import PDFStream
from src.doctranslator.pdfminer.pdftypes import dict_value
from src.doctranslator.pdfminer.pdftypes import int_value
from src.doctranslator.pdfminer.pdftypes import list_value
from src.doctranslator.pdfminer.pdftypes import num_value
from src.doctranslator.pdfminer.pdftypes import resolve1
from src.doctranslator.pdfminer.pdftypes import resolve_all
from src.doctranslator.pdfminer.pdftypes import stream_value
from src.doctranslator.pdfminer.psexceptions import PSEOF
from src.doctranslator.pdfminer.psparser import KWD
from src.doctranslator.pdfminer.psparser import LIT
from src.doctranslator.pdfminer.psparser import PSKeyword
from src.doctranslator.pdfminer.psparser import PSLiteral
from src.doctranslator.pdfminer.psparser import PSStackParser
from src.doctranslator.pdfminer.psparser import literal_name
from src.doctranslator.pdfminer.utils import Matrix
from src.doctranslator.pdfminer.utils import Point
from src.doctranslator.pdfminer.utils import Rect
from src.doctranslator.pdfminer.utils import apply_matrix_norm
from src.doctranslator.pdfminer.utils import choplist
from src.doctranslator.pdfminer.utils import nunpack

if TYPE_CHECKING:
    from src.doctranslator.pdfminer.pdfinterp import PDFResourceManager

log = logging.getLogger(__name__)


def _apply_width_list(widths: dict[int, float], char1: float, v: list[object]) -> None:
    """Apply a list of widths starting from char1 into the widths dict."""
    for idx, w in enumerate(v):
        widths[cast(int, char1) + idx] = w  # type: ignore[assignment]


def _apply_width_range(
    widths: dict[int, float], char1: float, char2: float, w: float
) -> None:
    """Apply a single width w to all chars in the range [char1, char2]."""
    if isinstance(char1, int) and isinstance(char2, int):
        for code in range(cast(int, char1), cast(int, char2) + 1):
            widths[code] = w
    else:
        log.warning(
            f"Skipping invalid font width specification for {char1} to {char2} because either of them is not an int"
        )


def get_widths(seq: Iterable[object]) -> dict[str | int, float]:
    """Build a mapping of character widths for horizontal writing."""
    widths: dict[int, float] = {}
    r: list[float] = []
    for v in seq:
        v = resolve1(v)
        if isinstance(v, list):
            if r:
                char1 = r[-1]
                _apply_width_list(widths, char1, v)
                r = []
        elif isinstance(v, (int, float)):  # == utils.isnumber(v)
            r.append(v)
            if len(r) == 3:
                (char1, char2, w) = r
                _apply_width_range(widths, char1, char2, w)
                r = []
        else:
            log.warning(
                f"Skipping invalid font width specification for {v} because it is not a number or a list"
            )
    return cast(dict[str | int, float], widths)


def _apply_width2_list(
    widths: dict[int, tuple[float, Point]], char1: float, v: list[object]
) -> None:
    """Apply a list of (w, vx, vy) triplets starting from char1 into widths dict."""
    for idx, (w, vx, vy) in enumerate(choplist(3, v)):
        widths[cast(int, char1) + idx] = (w, (vx, vy))


def _apply_width2_range(
    widths: dict[int, tuple[float, Point]],
    char1: float,
    char2: float,
    w: float,
    vx: float,
    vy: float,
) -> None:
    """Apply a single vertical width entry to all chars in [char1, char2]."""
    for code in range(cast(int, char1), cast(int, char2) + 1):
        widths[code] = (w, (vx, vy))


def get_widths2(seq: Iterable[object]) -> dict[int, tuple[float, Point]]:
    """Build a mapping of character widths for vertical writing."""
    widths: dict[int, tuple[float, Point]] = {}
    r: list[float] = []
    for v in seq:
        if isinstance(v, list):
            if r:
                char1 = r[-1]
                _apply_width2_list(widths, char1, v)
                r = []
        elif isinstance(v, (int, float)):  # == utils.isnumber(v)
            r.append(v)
            if len(r) == 5:
                (char1, char2, w, vx, vy) = r
                _apply_width2_range(widths, char1, char2, w, vx, vy)
                r = []
    return widths


class FontMetricsDB:
    @classmethod
    def get_metrics(cls, fontname: str) -> tuple[dict[str, object], dict[str, int]]:
        return FONT_METRICS[fontname]


# int here means that we're not extending PSStackParser with additional types.
class Type1FontHeaderParser(PSStackParser[int]):
    KEYWORD_BEGIN = KWD(b"begin")
    KEYWORD_END = KWD(b"end")
    KEYWORD_DEF = KWD(b"def")
    KEYWORD_PUT = KWD(b"put")
    KEYWORD_DICT = KWD(b"dict")
    KEYWORD_ARRAY = KWD(b"array")
    KEYWORD_READONLY = KWD(b"readonly")
    KEYWORD_FOR = KWD(b"for")

    def __init__(self, data: BinaryIO) -> None:
        PSStackParser.__init__(self, data)
        self._cid2unicode: dict[int, str] = {}

    def get_encoding(self) -> dict[int, str]:
        """Parse the font encoding.

        The Type1 font encoding maps character codes to character names. These
        character names could either be standard Adobe glyph names, or
        character names associated with custom CharStrings for this font. A
        CharString is a sequence of operations that describe how the character
        should be drawn. Currently, this function returns '' (empty string)
        for character names that are associated with a CharStrings.

        Reference: Adobe Systems Incorporated, Adobe Type 1 Font Format

        :returns mapping of character identifiers (cid's) to unicode characters
        """
        while 1:
            try:
                (cid, name) = self.nextobject()
            except PSEOF:
                break
            try:
                self._cid2unicode[cid] = name2unicode(cast(str, name))
            except KeyError as e:
                log.debug(str(e))
        return self._cid2unicode

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        if token is self.KEYWORD_PUT:
            ((_, key), (_, value)) = self.pop(2)
            if isinstance(key, int) and isinstance(value, PSLiteral):
                self.add_results((key, literal_name(value)))


NIBBLES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "e", "e-", None, "-")

# Mapping of cmap names. Original cmap name is kept if not in the mapping.
# (missing reference for why DLIdent is mapped to Identity)
IDENTITY_ENCODER = {
    "DLIdent-H": "Identity-H",
    "DLIdent-V": "Identity-V",
}


def _parse_real_number(fp: BytesIO) -> float:
    """Parse a real number (b0==30) from the CFF font stream."""
    s = ""
    loop = True
    while loop:
        b = ord(fp.read(1))
        for n in (b >> 4, b & 15):
            if n == 15:
                loop = False
            else:
                nibble = NIBBLES[n]
                if nibble is None:
                    raise AssertionError
                s += nibble
    return float(s)


def _parse_multi_byte_number(fp: BytesIO, b0: int) -> int | float:
    """Parse a multi-byte integer from the CFF font stream."""
    b1 = ord(fp.read(1))
    if b0 >= 247 and b0 <= 250:
        return ((b0 - 247) << 8) + b1 + 108
    if b0 >= 251 and b0 <= 254:
        return -((b0 - 251) << 8) - b1 - 108
    # b0 == 28 or b0 == 29
    b2 = ord(fp.read(1))
    if b1 >= 128:
        b1 -= 256
    if b0 == 28:
        return b1 << 8 | b2
    return b1 << 24 | b2 << 16 | struct.unpack(">H", fp.read(2))[0]


def getdict(data: bytes) -> dict[int, list[float | int]]:
    d: dict[int, list[float | int]] = {}
    fp = BytesIO(data)
    stack: list[float | int] = []
    while 1:
        c = fp.read(1)
        if not c:
            break
        b0 = ord(c)
        if b0 <= 21:
            d[b0] = stack
            stack = []
            continue
        if b0 == 30:
            value: float | int = _parse_real_number(fp)
        elif b0 >= 32 and b0 <= 246:
            value = b0 - 139
        else:
            value = _parse_multi_byte_number(fp, b0)
        stack.append(value)
    return d


class CFFFont:
    STANDARD_STRINGS = (
        ".notdef",
        "space",
        "exclam",
        "quotedbl",
        "numbersign",
        "dollar",
        "percent",
        "ampersand",
        "quoteright",
        "parenleft",
        "parenright",
        "asterisk",
        "plus",
        "comma",
        "hyphen",
        "period",
        "slash",
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "colon",
        "semicolon",
        "less",
        "equal",
        "greater",
        "question",
        "at",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "bracketleft",
        "backslash",
        "bracketright",
        "asciicircum",
        "underscore",
        "quoteleft",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "braceleft",
        "bar",
        "braceright",
        "asciitilde",
        "exclamdown",
        "cent",
        "sterling",
        "fraction",
        "yen",
        "florin",
        "section",
        "currency",
        "quotesingle",
        "quotedblleft",
        "guillemotleft",
        "guilsinglleft",
        "guilsinglright",
        "fi",
        "fl",
        "endash",
        "dagger",
        "daggerdbl",
        "periodcentered",
        "paragraph",
        "bullet",
        "quotesinglbase",
        "quotedblbase",
        "quotedblright",
        "guillemotright",
        "ellipsis",
        "perthousand",
        "questiondown",
        "grave",
        "acute",
        "circumflex",
        "tilde",
        "macron",
        "breve",
        "dotaccent",
        "dieresis",
        "ring",
        "cedilla",
        "hungarumlaut",
        "ogonek",
        "caron",
        "emdash",
        "AE",
        "ordfeminine",
        "Lslash",
        "Oslash",
        "OE",
        "ordmasculine",
        "ae",
        "dotlessi",
        "lslash",
        "oslash",
        "oe",
        "germandbls",
        "onesuperior",
        "logicalnot",
        "mu",
        "trademark",
        "Eth",
        "onehalf",
        "plusminus",
        "Thorn",
        "onequarter",
        "divide",
        "brokenbar",
        "degree",
        "thorn",
        "threequarters",
        "twosuperior",
        "registered",
        "minus",
        "eth",
        "multiply",
        "threesuperior",
        "copyright",
        "Aacute",
        "Acircumflex",
        "Adieresis",
        "Agrave",
        "Aring",
        "Atilde",
        "Ccedilla",
        "Eacute",
        "Ecircumflex",
        "Edieresis",
        "Egrave",
        "Iacute",
        "Icircumflex",
        "Idieresis",
        "Igrave",
        "Ntilde",
        "Oacute",
        "Ocircumflex",
        "Odieresis",
        "Ograve",
        "Otilde",
        "Scaron",
        "Uacute",
        "Ucircumflex",
        "Udieresis",
        "Ugrave",
        "Yacute",
        "Ydieresis",
        "Zcaron",
        "aacute",
        "acircumflex",
        "adieresis",
        "agrave",
        "aring",
        "atilde",
        "ccedilla",
        "eacute",
        "ecircumflex",
        "edieresis",
        "egrave",
        "iacute",
        "icircumflex",
        "idieresis",
        "igrave",
        "ntilde",
        "oacute",
        "ocircumflex",
        "odieresis",
        "ograve",
        "otilde",
        "scaron",
        "uacute",
        "ucircumflex",
        "udieresis",
        "ugrave",
        "yacute",
        "ydieresis",
        "zcaron",
        "exclamsmall",
        "Hungarumlautsmall",
        "dollaroldstyle",
        "dollarsuperior",
        "ampersandsmall",
        "Acutesmall",
        "parenleftsuperior",
        "parenrightsuperior",
        "twodotenleader",
        "onedotenleader",
        "zerooldstyle",
        "oneoldstyle",
        "twooldstyle",
        "threeoldstyle",
        "fouroldstyle",
        "fiveoldstyle",
        "sixoldstyle",
        "sevenoldstyle",
        "eightoldstyle",
        "nineoldstyle",
        "commasuperior",
        "threequartersemdash",
        "periodsuperior",
        "questionsmall",
        "asuperior",
        "bsuperior",
        "centsuperior",
        "dsuperior",
        "esuperior",
        "isuperior",
        "lsuperior",
        "msuperior",
        "nsuperior",
        "osuperior",
        "rsuperior",
        "ssuperior",
        "tsuperior",
        "ff",
        "ffi",
        "ffl",
        "parenleftinferior",
        "parenrightinferior",
        "Circumflexsmall",
        "hyphensuperior",
        "Gravesmall",
        "Asmall",
        "Bsmall",
        "Csmall",
        "Dsmall",
        "Esmall",
        "Fsmall",
        "Gsmall",
        "Hsmall",
        "Ismall",
        "Jsmall",
        "Ksmall",
        "Lsmall",
        "Msmall",
        "Nsmall",
        "Osmall",
        "Psmall",
        "Qsmall",
        "Rsmall",
        "Ssmall",
        "Tsmall",
        "Usmall",
        "Vsmall",
        "Wsmall",
        "Xsmall",
        "Ysmall",
        "Zsmall",
        "colonmonetary",
        "onefitted",
        "rupiah",
        "Tildesmall",
        "exclamdownsmall",
        "centoldstyle",
        "Lslashsmall",
        "Scaronsmall",
        "Zcaronsmall",
        "Dieresissmall",
        "Brevesmall",
        "Caronsmall",
        "Dotaccentsmall",
        "Macronsmall",
        "figuredash",
        "hypheninferior",
        "Ogoneksmall",
        "Ringsmall",
        "Cedillasmall",
        "questiondownsmall",
        "oneeighth",
        "threeeighths",
        "fiveeighths",
        "seveneighths",
        "onethird",
        "twothirds",
        "zerosuperior",
        "foursuperior",
        "fivesuperior",
        "sixsuperior",
        "sevensuperior",
        "eightsuperior",
        "ninesuperior",
        "zeroinferior",
        "oneinferior",
        "twoinferior",
        "threeinferior",
        "fourinferior",
        "fiveinferior",
        "sixinferior",
        "seveninferior",
        "eightinferior",
        "nineinferior",
        "centinferior",
        "dollarinferior",
        "periodinferior",
        "commainferior",
        "Agravesmall",
        "Aacutesmall",
        "Acircumflexsmall",
        "Atildesmall",
        "Adieresissmall",
        "Aringsmall",
        "AEsmall",
        "Ccedillasmall",
        "Egravesmall",
        "Eacutesmall",
        "Ecircumflexsmall",
        "Edieresissmall",
        "Igravesmall",
        "Iacutesmall",
        "Icircumflexsmall",
        "Idieresissmall",
        "Ethsmall",
        "Ntildesmall",
        "Ogravesmall",
        "Oacutesmall",
        "Ocircumflexsmall",
        "Otildesmall",
        "Odieresissmall",
        "OEsmall",
        "Oslashsmall",
        "Ugravesmall",
        "Uacutesmall",
        "Ucircumflexsmall",
        "Udieresissmall",
        "Yacutesmall",
        "Thornsmall",
        "Ydieresissmall",
        "001.000",
        "001.001",
        "001.002",
        "001.003",
        "Black",
        "Bold",
        "Book",
        "Light",
        "Medium",
        "Regular",
        "Roman",
        "Semibold",
    )

    class INDEX:
        def __init__(self, fp: BinaryIO) -> None:
            self.fp = fp
            self.offsets: list[int] = []
            (count, offsize) = struct.unpack(">HB", self.fp.read(3))
            for _ in range(count + 1):
                self.offsets.append(nunpack(self.fp.read(offsize)))
            self.base = self.fp.tell() - 1
            self.fp.seek(self.base + self.offsets[-1])

        def __repr__(self) -> str:
            return "<INDEX: size=%d>" % len(self)

        def __len__(self) -> int:
            return len(self.offsets) - 1

        def __getitem__(self, i: int) -> bytes:
            self.fp.seek(self.base + self.offsets[i])
            return self.fp.read(self.offsets[i + 1] - self.offsets[i])

        def __iter__(self) -> Iterator[bytes]:
            return iter(self[i] for i in range(len(self)))

    def __init__(self, name: str, fp: BinaryIO) -> None:
        self.name = name
        self.fp = fp
        # Header
        (_major, _minor, hdrsize, offsize) = struct.unpack("BBBB", self.fp.read(4))
        self.fp.read(hdrsize - 4)
        # Name INDEX
        self.name_index = self.INDEX(self.fp)
        # Top DICT INDEX
        self.dict_index = self.INDEX(self.fp)
        # String INDEX
        self.string_index = self.INDEX(self.fp)
        # Global Subr INDEX
        self.subr_index = self.INDEX(self.fp)
        # Top DICT DATA
        self.top_dict = getdict(self.dict_index[0])
        (charset_pos,) = self.top_dict.get(15, [0])
        (encoding_pos,) = self.top_dict.get(16, [0])
        (charstring_pos,) = self.top_dict.get(17, [0])
        # CharStrings
        self.fp.seek(cast(int, charstring_pos))
        self.charstring = self.INDEX(self.fp)
        self.nglyphs = len(self.charstring)
        # Encodings
        self.code2gid = {}
        self.gid2code = {}
        self.fp.seek(cast(int, encoding_pos))
        self._parse_encodings()
        # Charsets
        self.name2gid = {}
        self.gid2name = {}
        self.fp.seek(cast(int, charset_pos))
        self._parse_charsets()

    def _parse_encodings(self) -> None:
        """Parse CFF encoding table and populate code2gid/gid2code maps."""
        encoding_format = self.fp.read(1)
        if encoding_format == b"\x00":
            # Format 0
            (n,) = struct.unpack("B", self.fp.read(1))
            for code, gid in enumerate(struct.unpack("B" * n, self.fp.read(n))):
                self.code2gid[code] = gid
                self.gid2code[gid] = code
        elif encoding_format == b"\x01":
            # Format 1
            (n,) = struct.unpack("B", self.fp.read(1))
            code = 0
            for _ in range(n):
                (first, nleft) = struct.unpack("BB", self.fp.read(2))
                for gid in range(first, first + nleft + 1):
                    self.code2gid[code] = gid
                    self.gid2code[gid] = code
                    code += 1
        else:
            raise PDFValueError("unsupported encoding format: %r" % encoding_format)

    def _parse_charsets(self) -> None:
        """Parse CFF charset table and populate name2gid/gid2name maps."""
        charset_format = self.fp.read(1)
        if charset_format == b"\x00":
            # Format 0
            n = self.nglyphs - 1
            for gid, sid in enumerate(
                cast(
                    tuple[int, ...], struct.unpack(">" + "H" * n, self.fp.read(2 * n))
                ),
            ):
                gid += 1
                sidname = self.getstr(sid)
                self.name2gid[sidname] = gid
                self.gid2name[gid] = sidname
        elif charset_format == b"\x01":
            # Format 1
            (n,) = struct.unpack("B", self.fp.read(1))
            sid = 0
            for _ in range(n):
                (first, nleft) = struct.unpack("BB", self.fp.read(2))
                for gid in range(first, first + nleft + 1):
                    sidname = self.getstr(sid)
                    self.name2gid[sidname] = gid
                    self.gid2name[gid] = sidname
                    sid += 1
        elif charset_format == b"\x02":
            # Format 2
            raise AssertionError(str(("Unhandled", charset_format)))
        else:
            raise PDFValueError("unsupported charset format: %r" % charset_format)

    def getstr(self, sid: int) -> str | bytes:
        # This returns str for one of the STANDARD_STRINGS but bytes otherwise,
        # and appears to be a needless source of type complexity.
        if sid < len(self.STANDARD_STRINGS):
            return self.STANDARD_STRINGS[sid]
        return self.string_index[sid - len(self.STANDARD_STRINGS)]


class TrueTypeFont:
    class CMapNotFound(PDFException):
        pass

    def __init__(self, name: str, fp: BinaryIO) -> None:
        self.name = name
        self.fp = fp
        self.tables: dict[bytes, tuple[int, int]] = {}
        self.fonttype = fp.read(4)
        try:
            (ntables, _1, _2, _3) = cast(
                tuple[int, int, int, int],
                struct.unpack(">HHHH", fp.read(8)),
            )
            for _ in range(ntables):
                (name_bytes, tsum, offset, length) = cast(
                    tuple[bytes, int, int, int],
                    struct.unpack(">4sLLL", fp.read(16)),
                )
                self.tables[name_bytes] = (offset, length)
        except struct.error:
            # Do not fail if there are not enough bytes to read. Even for
            # corrupted PDFs we would like to get as much information as
            # possible, so continue.
            pass

    def create_unicode_map(self) -> FileUnicodeMap:
        if b"cmap" not in self.tables:
            raise TrueTypeFont.CMapNotFound
        fp = self.fp
        char2gid = []
        try:
            face = freetype.Face(fp)
            char2gid = list(face.get_chars())
        except Exception:
            raise TrueTypeFont.CMapNotFound
        # create unicode map
        unicode_map = FileUnicodeMap()
        for char, gid in char2gid:
            unicode_map.add_cid2unichr(gid, char)
        return unicode_map


class PDFFontError(PDFException):
    pass


class PDFUnicodeNotDefined(PDFFontError):
    pass


LITERAL_STANDARD_ENCODING = LIT("StandardEncoding")
LITERAL_TYPE1C = LIT("Type1C")

# Font widths are maintained in a dict type that maps from *either* unicode
# chars or integer character IDs.
FontWidthDict = dict[int | str, float]


class PDFFont:
    def __init__(
        self,
        descriptor: Mapping[str, Any],
        widths: FontWidthDict,
        default_width: float | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.widths: FontWidthDict = resolve_all(widths)
        self.fontname = resolve1(descriptor.get("FontName", "unknown"))
        if isinstance(self.fontname, PSLiteral):
            self.fontname = literal_name(self.fontname)
        self.flags = int_value(descriptor.get("Flags", 0))
        self.ascent = num_value(descriptor.get("Ascent", 0))
        self.descent = num_value(descriptor.get("Descent", 0))
        self.italic_angle = num_value(descriptor.get("ItalicAngle", 0))
        if default_width is None:
            self.default_width = num_value(descriptor.get("MissingWidth", 0))
        else:
            self.default_width = default_width
        self.default_width = resolve1(self.default_width)
        self.leading = num_value(descriptor.get("Leading", 0))
        self.bbox = self._parse_bbox(descriptor)
        self.hscale = self.vscale = 0.001

        # PDF RM 9.8.1 specifies /Descent should always be a negative number.
        # PScript5.dll seems to produce Descent with a positive number, but
        # text analysis will be wrong if this is taken as correct. So force
        # descent to negative.
        if self.descent > 0:
            self.descent = -self.descent

    def __repr__(self) -> str:
        return "<PDFFont>"

    def is_vertical(self) -> bool:
        return False

    def is_multibyte(self) -> bool:
        return False

    def decode(self, bytes: bytes) -> Iterable[int]:
        return bytearray(bytes)  # map(ord, bytes)

    def get_ascent(self) -> float:
        """Ascent above the baseline, in text space units"""
        return self.ascent * self.vscale

    def get_descent(self) -> float:
        """Descent below the baseline, in text space units; always negative"""
        return self.descent * self.vscale

    def get_width(self) -> float:
        w = self.bbox[2] - self.bbox[0]
        if w == 0:
            w = -self.default_width
        return w * self.hscale

    def get_height(self) -> float:
        h = self.bbox[3] - self.bbox[1]
        if h == 0:
            h = self.ascent - self.descent
        return h * self.vscale

    def char_width(self, cid: int) -> float:
        # Because character widths may be mapping either IDs or strings,
        # we try to lookup the character ID first, then its str equivalent.
        cid_width = safe_float(self.widths.get(cid))
        if cid_width is not None:
            return cid_width * self.hscale

        try:
            str_cid = self.to_unichr(cid)
            cid_width = safe_float(self.widths.get(str_cid))
            if cid_width is not None:
                return cid_width * self.hscale

        except PDFUnicodeNotDefined:
            pass

        return self.default_width * self.hscale

    def char_disp(self, _cid: int) -> float | tuple[float | None, float]:
        """Returns an integer for horizontal fonts, a tuple for vertical fonts."""
        return 0

    def string_width(self, s: bytes) -> float:
        return sum(self.char_width(cid) for cid in self.decode(s))

    def to_unichr(self, cid: int) -> str:
        raise NotImplementedError

    @staticmethod
    def _parse_bbox(descriptor: Mapping[str, Any]) -> Rect:
        """Parse FontBBox from the fonts descriptor"""
        font_bbox = resolve_all(descriptor.get("FontBBox"))
        bbox = safe_rect_list(font_bbox)
        if bbox is None:
            log.warning(
                f"Could get FontBBox from font descriptor because {font_bbox!r} cannot be parsed as 4 floats"
            )
            return 0.0, 0.0, 0.0, 0.0
        return bbox


class PDFSimpleFont(PDFFont):
    def __init__(
        self,
        descriptor: Mapping[str, Any],
        widths: FontWidthDict,
        spec: Mapping[str, Any],
    ) -> None:
        # Font encoding is specified either by a name of
        # built-in encoding or a dictionary that describes
        # the differences.
        if "Encoding" in spec:
            encoding = resolve1(spec["Encoding"])
        else:
            encoding = LITERAL_STANDARD_ENCODING
        if isinstance(encoding, dict):
            name = literal_name(encoding.get("BaseEncoding", LITERAL_STANDARD_ENCODING))
            diff = list_value(encoding.get("Differences", []))
            self.cid2unicode = EncodingDB.get_encoding(name, diff)
        else:
            self.cid2unicode = EncodingDB.get_encoding(literal_name(encoding))
        self.unicode_map: UnicodeMap | None = None
        if "ToUnicode" in spec:
            strm = stream_value(spec["ToUnicode"])
            self.unicode_map = FileUnicodeMap()
            CMapParser(self.unicode_map, BytesIO(strm.get_data())).run()
        PDFFont.__init__(self, descriptor, widths)

    def to_unichr(self, cid: int) -> str:
        if self.unicode_map:
            try:
                return self.unicode_map.get_unichr(cid)
            except KeyError:
                pass
        try:
            return self.cid2unicode[cid]
        except KeyError:
            raise PDFUnicodeNotDefined(None, cid)


class PDFType1Font(PDFSimpleFont):
    def __init__(self, rsrcmgr: "PDFResourceManager", spec: Mapping[str, Any]) -> None:
        try:
            self.basefont = literal_name(spec["BaseFont"])
        except KeyError:
            if settings.STRICT:
                raise PDFFontError("BaseFont is missing")
            self.basefont = "unknown"

        widths: FontWidthDict
        try:
            (descriptor, int_widths) = FontMetricsDB.get_metrics(self.basefont)
            widths = cast(dict[str | int, float], int_widths)  # implicit int->float
        except KeyError:
            descriptor = dict_value(spec.get("FontDescriptor", {}))
            firstchar = int_value(spec.get("FirstChar", 0))
            width_list = list_value(spec.get("Widths", [0] * 256))
            widths = {i + firstchar: resolve1(w) for (i, w) in enumerate(width_list)}
        PDFSimpleFont.__init__(self, descriptor, widths, spec)
        if "Encoding" not in spec and "FontFile" in descriptor:
            # try to recover the missing encoding info from the font file.
            self.fontfile = stream_value(descriptor.get("FontFile"))
            length1 = int_value(self.fontfile["Length1"])
            data = self.fontfile.get_data()[:length1]
            # awcm: quickfix for type 1 font which contains bad string literals
            offset = 0
            if enc_offset := data.index(b"/Encoding"):
                offset = enc_offset
            parser = Type1FontHeaderParser(BytesIO(data[offset:]))
            self.cid2unicode = parser.get_encoding()

    def __repr__(self) -> str:
        return "<PDFType1Font: basefont=%r>" % self.basefont


class PDFTrueTypeFont(PDFType1Font):
    def __repr__(self) -> str:
        return "<PDFTrueTypeFont: basefont=%r>" % self.basefont


class PDFType3Font(PDFSimpleFont):
    def __init__(self, rsrcmgr: "PDFResourceManager", spec: Mapping[str, Any]) -> None:
        firstchar = int_value(spec.get("FirstChar", 0))
        width_list = list_value(spec.get("Widths", [0] * 256))
        widths: dict[str | int, float] = {
            i + firstchar: w for (i, w) in enumerate(width_list)
        }
        if "FontDescriptor" in spec:
            descriptor = dict_value(spec["FontDescriptor"])
        else:
            descriptor = {"Ascent": 0, "Descent": 0, "FontBBox": spec["FontBBox"]}
        PDFSimpleFont.__init__(self, descriptor, widths, spec)
        self.matrix = cast(Matrix, tuple(list_value(spec.get("FontMatrix"))))
        (_, self.descent, _, self.ascent) = self.bbox
        (self.hscale, self.vscale) = apply_matrix_norm(self.matrix, (1, 1))

    def __repr__(self) -> str:
        return "<PDFType3Font>"


class PDFCIDFont(PDFFont):
    default_disp: float | tuple[float | None, float]

    def _load_cid_encoding(self, spec: Mapping[str, Any]) -> None:
        """Load optional CID encoding from font spec."""
        self.has_encoding = False
        self.cid_encoding = None
        try:
            if "Encoding" in spec:
                encoding_part = resolve1(spec["Encoding"])
                if isinstance(encoding_part, PDFStream):
                    self.has_encoding = True
                    self.cid_encoding = CharacterMap(
                        encoding_part.get_data().decode("U8")
                    )
        except Exception as e:
            log.error(f"Error get cid_encoding from spec: {e}")
            self.has_encoding = False
            self.cid_encoding = None

    def _resolve_unicode_map(
        self,
        spec: Mapping[str, Any],
        cid_ordering: str,
        ttf: "TrueTypeFont | None",
    ) -> None:
        """Resolve and assign self.unicode_map from spec or TTF."""
        self.unicode_map: UnicodeMap | None = None
        if "ToUnicode" in spec:
            self._resolve_unicode_map_from_to_unicode(spec, cid_ordering)
        elif self.cidcoding in ("Adobe-Identity", "Adobe-UCS"):
            if ttf:
                try:
                    self.unicode_map = ttf.create_unicode_map()
                except TrueTypeFont.CMapNotFound:
                    pass
        else:
            try:
                self.unicode_map = CMapDB.get_unicode_map(
                    self.cidcoding,
                    self.cmap.is_vertical(),
                )
            except CMapDB.CMapNotFound:
                pass

    def _resolve_unicode_map_from_to_unicode(
        self, spec: Mapping[str, Any], cid_ordering: str
    ) -> None:
        """Resolve unicode map from the ToUnicode entry in the font spec."""
        if isinstance(spec["ToUnicode"], PDFStream):
            strm = stream_value(spec["ToUnicode"])
            self.unicode_map = FileUnicodeMap()
            CMapParser(self.unicode_map, BytesIO(strm.get_data())).run()
        else:
            cmap_name = literal_name(spec["ToUnicode"])
            encoding = literal_name(spec["Encoding"])
            if (
                "Identity" in cid_ordering
                or "Identity" in cmap_name
                or "Identity" in encoding
            ):
                self.unicode_map = IdentityUnicodeMap()

    def __init__(
        self,
        rsrcmgr: "PDFResourceManager",
        spec: Mapping[str, Any],
        strict: bool = settings.STRICT,
    ) -> None:
        try:
            self.basefont = literal_name(spec["BaseFont"])
        except KeyError:
            if strict:
                raise PDFFontError("BaseFont is missing")
            self.basefont = "unknown"
        self.cidsysteminfo = dict_value(spec.get("CIDSystemInfo", {}))
        cid_registry = resolve1(self.cidsysteminfo.get("Registry", b"unknown")).decode(
            "latin1",
        )
        cid_ordering = resolve1(self.cidsysteminfo.get("Ordering", b"unknown")).decode(
            "latin1",
        )
        self.cidcoding = f"{cid_registry.strip()}-{cid_ordering.strip()}"
        self.cmap: CMapBase = self.get_cmap_from_spec(spec, strict)

        try:
            descriptor = dict_value(spec["FontDescriptor"])
        except KeyError:
            if strict:
                raise PDFFontError("FontDescriptor is missing")
            descriptor = {}
        ttf = None
        self._load_cid_encoding(spec)
        if "FontFile2" in descriptor:
            self.fontfile = stream_value(descriptor.get("FontFile2"))
            ttf = TrueTypeFont(self.basefont, BytesIO(self.fontfile.get_data()))
        self._resolve_unicode_map(spec, cid_ordering, ttf)

        self.vertical = self.cmap.is_vertical()
        if self.vertical:
            # writing mode: vertical
            widths2 = get_widths2(list_value(spec.get("W2", [])))
            self.disps = {cid: (vx, vy) for (cid, (_, (vx, vy))) in widths2.items()}
            (vy, w) = resolve1(spec.get("DW2", [880, -1000]))
            self.default_disp = (None, vy)
            widths: dict[str | int, float] = {
                cid: w for (cid, (w, _)) in widths2.items()
            }
            default_width = w
        else:
            # writing mode: horizontal
            self.disps = {}
            self.default_disp = 0
            widths = get_widths(list_value(spec.get("W", [])))
            default_width = spec.get("DW", 1000)
        PDFFont.__init__(self, descriptor, widths, default_width=default_width)

    def get_cmap_from_spec(self, spec: Mapping[str, Any], strict: bool) -> CMapBase:
        """Get cmap from font specification

        For certain PDFs, Encoding Type isn't mentioned as an attribute of
        Encoding but as an attribute of CMapName, where CMapName is an
        attribute of spec['Encoding'].
        The horizontal/vertical modes are mentioned with different name
        such as 'DLIdent-H/V','OneByteIdentityH/V','Identity-H/V'.
        """
        cmap_name = self._get_cmap_name(spec, strict)

        try:
            return CMapDB.get_cmap(cmap_name)
        except CMapDB.CMapNotFound as e:
            if strict:
                raise PDFFontError(e)
            return CMap()

    @staticmethod
    def _get_cmap_name(spec: Mapping[str, Any], strict: bool) -> str:
        """Get cmap name from font specification"""
        cmap_name = "unknown"  # default value

        try:
            spec_encoding = spec["Encoding"]
            if hasattr(spec_encoding, "name"):
                cmap_name = literal_name(spec["Encoding"])
            else:
                cmap_name = literal_name(spec_encoding["CMapName"])
        except KeyError:
            if strict:
                raise PDFFontError("Encoding is unspecified")

        if type(cmap_name) is PDFStream:  # type: ignore[comparison-overlap]
            cmap_name_stream: PDFStream = cast(PDFStream, cmap_name)
            if "CMapName" in cmap_name_stream:
                cmap_name = cmap_name_stream.get("CMapName").name
            elif strict:
                raise PDFFontError("CMapName unspecified for encoding")

        return IDENTITY_ENCODER.get(cmap_name, cmap_name)

    def __repr__(self) -> str:
        return f"<PDFCIDFont: basefont={self.basefont!r}, cidcoding={self.cidcoding!r}>"

    def is_vertical(self) -> bool:
        return self.vertical

    def is_multibyte(self) -> bool:
        return True

    def decode(self, bytes: bytes) -> Iterable[int]:
        try:
            if self.has_encoding:
                res = self.cid_encoding.decode(bytes)

                if res is not None and all(x > 0 for x in res):
                    return res
        except Exception as e:
            log.error(f"Error use cid_encoding to decode bytes: {e}")
        return self.cmap.decode(bytes)

    def char_disp(self, cid: int) -> float | tuple[float | None, float]:
        """Returns an integer for horizontal fonts, a tuple for vertical fonts."""
        return self.disps.get(cid, self.default_disp)

    def to_unichr(self, cid: int) -> str:
        try:
            if not self.unicode_map:
                raise PDFKeyError(cid)
            return self.unicode_map.get_unichr(cid)
        except KeyError:
            raise PDFUnicodeNotDefined(self.cidcoding, cid)
