import base64
import functools
import logging
import math
import re
import unicodedata
from io import BytesIO
from typing import Literal

import freetype
import pymupdf
import tiktoken

import src.doctranslator.pdfminer.pdfinterp
from src.doctranslator.format.pdf.babelpdf.base14 import get_base14_bbox
from src.doctranslator.format.pdf.babelpdf.cidfont import get_cidfont_bbox
from src.doctranslator.format.pdf.babelpdf.cidfont import get_glyph_bbox
from src.doctranslator.format.pdf.babelpdf.encoding import WinAnsiEncoding
from src.doctranslator.format.pdf.babelpdf.encoding import get_type1_encoding
from src.doctranslator.format.pdf.babelpdf.type3 import get_type3_bbox
from src.doctranslator.format.pdf.babelpdf.utils import guarded_bbox
from src.doctranslator.format.pdf.document_il import il_version_1
from src.doctranslator.format.pdf.document_il.utils import zstd_helper
from src.doctranslator.format.pdf.document_il.utils.fontmap import FontMapper
from src.doctranslator.format.pdf.document_il.utils.matrix_helper import decompose_ctm
from src.doctranslator.format.pdf.document_il.utils.style_helper import BLACK
from src.doctranslator.format.pdf.document_il.utils.style_helper import YELLOW
from src.doctranslator.format.pdf.translation_config import TranslationConfig
from src.doctranslator.pdfminer.layout import LTChar
from src.doctranslator.pdfminer.layout import LTFigure
from src.doctranslator.pdfminer.pdffont import PDFCIDFont
from src.doctranslator.pdfminer.pdffont import PDFFont
from src.doctranslator.pdfminer.psparser import PSLiteral
from src.doctranslator.pdfminer.utils import apply_matrix_pt
from src.doctranslator.pdfminer.utils import get_bound
from src.doctranslator.pdfminer.utils import mult_matrix
from src.doctranslator.utils.common import batched


def invert_matrix(
    ctm: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """
    Calculate the inverse of a 2D transformation matrix.
    Matrix format: (a, b, c, d, e, f) representing:
    [a c e]
    [b d f]
    [0 0 1]
    """
    a, b, c, d, e, f = ctm

    # Calculate determinant
    det = a * d - b * c

    if abs(det) < 1e-10:
        # Matrix is singular, return identity matrix
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    # Calculate inverse matrix elements
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    inv_e = (c * f - d * e) / det
    inv_f = (b * e - a * f) / det

    return (inv_a, inv_b, inv_c, inv_d, inv_e, inv_f)


logger = logging.getLogger(__name__)


def indirect(obj):
    if isinstance(obj, tuple) and obj[0] == "xref":
        return int(obj[1].split(" ")[0])


def get_char_cbox(face, idx):
    g = face.get_char_index(idx)
    return get_glyph_bbox(face, g)


def get_name_cbox(face, name):
    if name:
        if isinstance(name, str):
            name = name.encode("utf-8")
        g = face.get_name_index(name)
        return get_glyph_bbox(face, g)
    return (0, 0, 0, 0)


def font_encoding_lookup(doc, idx, key):
    obj = doc.xref_get_key(idx, key)
    if obj[0] == "name":
        enc_name = obj[1][1:]
        if enc_vector := get_type1_encoding(enc_name):
            return enc_name, enc_vector


def parse_font_encoding(doc, idx):
    if encoding := font_encoding_lookup(doc, idx, "Encoding/BaseEncoding"):
        return encoding
    if encoding := font_encoding_lookup(doc, idx, "Encoding"):
        return encoding
    return ("Custom", get_type1_encoding("StandardEncoding"))


def get_truetype_ansi_bbox_list(face):
    scale = 1000 / face.units_per_EM
    bbox_list = [get_char_cbox(face, code) for code in WinAnsiEncoding]
    bbox_list = [[v * scale for v in bbox] for bbox in bbox_list]
    return bbox_list


def collect_face_cmap(face):
    umap = []  # unicode maps
    lmap = []  # legacy maps
    for cmap in face.charmaps:
        if cmap.encoding_name == "FT_ENCODING_UNICODE":
            umap.append(cmap)
        else:
            lmap.append(cmap)
    return umap, lmap


def get_truetype_custom_bbox_list(face):
    umap, lmap = collect_face_cmap(face)
    if umap:
        face.set_charmap(umap[0])
    elif lmap:
        face.set_charmap(lmap[0])
    else:
        return []
    scale = 1000 / face.units_per_EM
    bbox_list = [get_char_cbox(face, code) for code in range(256)]
    bbox_list = [[v * scale for v in bbox] for bbox in bbox_list]
    return bbox_list


def parse_font_file(doc, idx, encoding, differences):
    bbox_list = []
    data = doc.xref_stream(idx)
    face = freetype.Face(BytesIO(data))
    if face.get_format() == b"TrueType":
        if encoding[0] == "WinAnsiEncoding":
            return get_truetype_ansi_bbox_list(face)
        elif encoding[0] == "Custom":
            return get_truetype_custom_bbox_list(face)
    glyph_name_set = set()
    for x in range(0, face.num_glyphs):
        glyph_name_set.add(face.get_glyph_name(x).decode("U8"))
    scale = 1000 / face.units_per_EM
    enc_name, enc_vector = encoding
    _, lmap = collect_face_cmap(face)
    abbr = enc_name.removesuffix("Encoding")
    if lmap and abbr in ["Custom", "MacRoman", "Standard", "WinAnsi", "MacExpert"]:
        face.set_charmap(lmap[0])
    for i, x in enumerate(enc_vector):
        if x in glyph_name_set:
            v = get_name_cbox(face, x.encode("U8"))
        else:
            v = get_char_cbox(face, i)
        bbox_list.append(v)
    if differences:
        for code, name in differences:
            bbox_list[code] = get_name_cbox(face, name.encode("U8"))
    norm_bbox_list = [[v * scale for v in box] for box in bbox_list]
    return norm_bbox_list


def parse_encoding(obj_str):
    delta = []
    current = 0
    for x in re.finditer(
        r"(?P<p>[\[\]])|(?P<c>\d+)|(?P<n>/[^\s/\[\]()<>]+)|(?P<s>.)", obj_str
    ):
        key = x.lastgroup
        val = x.group()
        if key == "c":
            current = int(val)
        if key == "n":
            delta.append((current, val[1:]))
            current += 1
    return delta


def parse_mapping(text):
    mapping = []
    for x in re.finditer(r"<(?P<num>[a-fA-F0-9]+)>", text):
        mapping.append(x.group("num"))
    return mapping


def update_cmap_pair(cmap, data):
    for start_str, stop_str, value_str in batched(data, 3):
        start = int(start_str, 16)
        stop = int(stop_str, 16)
        try:
            value = base64.b16decode(value_str, True).decode("UTF-16-BE")
            for code in range(start, stop + 1):
                cmap[code] = value
        except Exception:
            pass  # to skip surrogate pairs (D800-DFFF)


def update_cmap_code(cmap, data):
    for code_str, value_str in batched(data, 2):
        code = int(code_str, 16)
        try:
            value = base64.b16decode(value_str, True).decode("UTF-16-BE")
            cmap[code] = value
        except Exception:
            pass  # to skip surrogate pairs (D800-DFFF)


def parse_cmap(cmap_str):
    cmap = {}
    for x in re.finditer(
        r"\s+beginbfrange\s*(?P<r>(<[0-9a-fA-F]+>\s*)+)endbfrange\s+", cmap_str
    ):
        update_cmap_pair(cmap, parse_mapping(x.group("r")))
    for x in re.finditer(
        r"\s+beginbfchar\s*(?P<c>(<[0-9a-fA-F]+>\s*)+)endbfchar", cmap_str
    ):
        update_cmap_code(cmap, parse_mapping(x.group("c")))
    return cmap


def get_code(cmap, c):
    for k, v in cmap.items():
        if v == c:
            return k
    return -1


def get_bbox(bbox, size, c, x, y):
    x_min, y_min, x_max, y_max = bbox[c]
    factor = 1 / 1000 * size
    x_min = x_min * factor
    y_min = -y_min * factor
    x_max = x_max * factor
    y_max = -y_max * factor
    ll = (x + x_min, y + y_min)
    lr = (x + x_max, y + y_min)
    ul = (x + x_min, y + y_max)
    ur = (x + x_max, y + y_max)
    return pymupdf.Quad(ll, lr, ul, ur)


# 常见 Unicode 空格字符的代码点
unicode_spaces = [
    "\u0020",  # 半角空格
    "\u00a0",  # 不间断空格
    "\u1680",  # Ogham 空格标记
    "\u2000",  # En Quad
    "\u2001",  # Em Quad
    "\u2002",  # En Space
    "\u2003",  # Em Space
    "\u2004",  # 三分之一 Em 空格
    "\u2005",  # 四分之一 Em 空格
    "\u2006",  # 六分之一 Em 空格
    "\u2007",  # 数样间距
    "\u2008",  # 行首前导空格
    "\u2009",  # 瘦弱空格
    "\u200a",  # hair space
    "\u202f",  # 窄不间断空格
    "\u205f",  # 数学中等空格
    "\u3000",  # 全角空格
    "\u200b",  # 零宽度空格
    "\u2060",  # 零宽度非断空格
    "\t",  # 水平制表符
]

# 构建正则表达式
pattern = "^[" + "".join(unicode_spaces) + "]+$"

# 编译正则
space_regex = re.compile(pattern)


def get_rotation_angle(matrix):
    """
    根据 PDF 的字符矩阵计算旋转角度（单位：度）
    matrix: tuple/list, 格式 (a, b, c, d, e, f)
    """
    a, b, c, d, e, f = matrix
    # 旋转角度：arctan2(b, a)
    angle_rad = math.atan2(b, a)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


class ILCreater:
    stage_name = "Parse PDF and Create Intermediate Representation"

    def __init__(self, translation_config: TranslationConfig):
        self.progress = None
        self.current_page: il_version_1.Page = None
        self.mupdf: pymupdf.Document = None
        self.model = translation_config.doc_layout_model
        self.docs = il_version_1.Document(page=[])
        self.stroking_color_space_name = None
        self.non_stroking_color_space_name = None
        self.passthrough_per_char_instruction: list[tuple[str, str]] = []
        self.translation_config = translation_config
        self.passthrough_per_char_instruction_stack: list[list[tuple[str, str]]] = []
        self.xobj_id = 0
        self.xobj_inc = 0
        self.xobj_map: dict[int, il_version_1.PdfXobject] = {}
        self.xobj_stack = []
        self.current_page_font_name_id_map = {}
        self.current_page_font_char_bounding_box_map = {}
        self.current_available_fonts = {}
        self.mupdf_font_map: dict[int, pymupdf.Font] = {}
        self.graphic_state_pool = {}
        self.enable_graphic_element_process = (
            translation_config.enable_graphic_element_process
        )
        self.render_order = 0
        self.current_clip_paths: list[tuple] = []
        self.clip_paths_stack: list[list[tuple]] = []
        # For valid character collection
        self.font_mapper = FontMapper(translation_config)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self._page_valid_chars_buffer: list[str] | None = None

    def transform_clip_path(
        self,
        clip_path,
        source_ctm: tuple[float, float, float, float, float, float],
        target_ctm: tuple[float, float, float, float, float, float],
    ):
        """Transform clip path coordinates from source CTM to target CTM."""
        if source_ctm == target_ctm:
            return clip_path

        # Calculate transformation matrix: inverse(target_ctm) * source_ctm
        inv_target_ctm = invert_matrix(target_ctm)
        transform_matrix = mult_matrix(source_ctm, inv_target_ctm)

        transformed_path = []
        for path_element in clip_path:
            if len(path_element) == 1:
                # Path operation without coordinates (e.g., 'h' for close path)
                transformed_path.append(path_element)
            else:
                # Path operation with coordinates
                op = path_element[0]
                coords = path_element[1:]
                transformed_coords = []

                # Transform coordinate pairs
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        x, y = coords[i], coords[i + 1]
                        transformed_point = apply_matrix_pt(transform_matrix, (x, y))
                        transformed_coords.extend(transformed_point)
                    else:
                        # Handle odd number of coordinates (shouldn't happen in well-formed paths)
                        transformed_coords.append(coords[i])

                transformed_path.append([op] + transformed_coords)

        return transformed_path

    def get_render_order_and_increase(self):
        self.render_order += 1
        return self.render_order

    def get_render_order(self):
        return self.render_order

    def on_finish(self):
        self.progress.__exit__(None, None, None)

    def is_graphic_operation(self, operator: str):
        if not self.enable_graphic_element_process:
            return False

        return re.match(
            r"^(m|l|c|v|y|re|h|S|s|f\*?|F|B\*?|b\*?|n|Do)$",
            operator,
        )

    def is_passthrough_per_char_operation(self, operator: str):
        return re.match(
            "^(sc|SC|sh|scn|SCN|g|G|rg|RG|k|K|cs|CS|gs|ri|w|J|j|M|i)$",
            operator,
        )

    def can_remove_old_passthrough_per_char_instruction(self, operator: str):
        return re.match(
            "^(sc|SC|sh|scn|SCN|g|G|rg|RG|k|K|cs|CS|ri|w|J|j|M|i|d)$",
            operator,
        )

    def on_line_dash(self, dash, phase):
        dash_str = f"[{' '.join(f'{arg}' for arg in dash)}]"
        self.on_passthrough_per_char("d", [dash_str, str(phase)])

    def on_passthrough_per_char(self, operator: str, args: list[str]):
        if not self.is_passthrough_per_char_operation(operator) and operator not in (
            "W n",
            "W* n",
            "d",
            "W",
            "W*",
        ):
            logger.error(f"Unknown passthrough_per_char operation: {operator}")
            return
        # logger.debug("xobj_id: %d, on_passthrough_per_char: %s ( %s )", self.xobj_id, operator, args)
        args = [self.parse_arg(arg) for arg in args]
        if self.can_remove_old_passthrough_per_char_instruction(operator):
            for _i, value in enumerate(self.passthrough_per_char_instruction.copy()):
                op, arg = value
                if op == operator:
                    self.passthrough_per_char_instruction.remove(value)
                    break
        self.passthrough_per_char_instruction.append((operator, " ".join(args)))

    def remove_latest_passthrough_per_char_instruction(self):
        if self.passthrough_per_char_instruction:
            self.passthrough_per_char_instruction.pop()

    def parse_arg(self, arg: str):
        if isinstance(arg, PSLiteral):
            return f"/{arg.name}"
        elif isinstance(arg, float):
            return f"{arg:f}"
        elif not isinstance(arg, str):
            return str(arg)
        return arg

    def pop_passthrough_per_char_instruction(self):
        if self.passthrough_per_char_instruction_stack:
            self.passthrough_per_char_instruction = (
                self.passthrough_per_char_instruction_stack.pop()
            )
        else:
            self.passthrough_per_char_instruction = []
            logging.error(
                f"pop_passthrough_per_char_instruction error on page: {self.current_page.page_number}"
            )

        if self.clip_paths_stack:
            self.current_clip_paths = self.clip_paths_stack.pop()
        else:
            self.current_clip_paths = []

    def push_passthrough_per_char_instruction(self):
        self.passthrough_per_char_instruction_stack.append(
            self.passthrough_per_char_instruction.copy(),
        )
        self.clip_paths_stack.append(self.current_clip_paths.copy())

    # pdf32000 page 171
    def on_stroking_color_space(self, color_space_name):
        self.stroking_color_space_name = color_space_name

    def on_non_stroking_color_space(self, color_space_name):
        self.non_stroking_color_space_name = color_space_name

    def on_new_stream(self):
        self.stroking_color_space_name = None
        self.non_stroking_color_space_name = None
        self.passthrough_per_char_instruction = []
        self.current_clip_paths = []

    def push_xobj(self):
        self.xobj_stack.append(
            (
                self.xobj_id,
                self.current_clip_paths.copy(),
                self.current_available_fonts.copy(),
            ),
        )
        self.current_clip_paths = []

    def pop_xobj(self):
        (self.xobj_id, self.current_clip_paths, self.current_available_fonts) = (
            self.xobj_stack.pop()
        )

    def on_xobj_begin(self, bbox, xref_id):
        logger.debug(f"on_xobj_begin: {bbox} @ {xref_id}")
        self.push_passthrough_per_char_instruction()
        self.push_xobj()
        self.xobj_inc += 1
        self.xobj_id = self.xobj_inc
        xobject = il_version_1.PdfXobject(
            box=il_version_1.Box(
                x=float(bbox[0]),
                y=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
            ),
            xobj_id=self.xobj_id,
            xref_id=xref_id,
            pdf_font=[],
        )
        self.current_page.pdf_xobject.append(xobject)
        self.xobj_map[self.xobj_id] = xobject
        xobject.pdf_font.extend(self.current_available_fonts.values())
        return self.xobj_id

    def on_xobj_end(self, xobj_id, base_op):
        self.pop_passthrough_per_char_instruction()
        self.pop_xobj()
        xobj = self.xobj_map[xobj_id]
        base_op = zstd_helper.zstd_compress(base_op)
        xobj.base_operations = il_version_1.BaseOperations(value=base_op)
        self.xobj_inc += 1

    def on_page_start(self):
        self.current_page = il_version_1.Page(
            pdf_font=[],
            pdf_character=[],
            page_layout=[],
            pdf_curve=[],
            pdf_form=[],
            # currently don't support UserUnit page parameter
            # pdf32000 page 79
            unit="point",
        )
        self.current_page_font_name_id_map = {}
        self.current_page_font_char_bounding_box_map = {}
        self.passthrough_per_char_instruction_stack = []
        self.xobj_stack = []
        self.non_stroking_color_space_name = None
        self.stroking_color_space_name = None
        self.current_clip_paths = []
        self.clip_paths_stack = []
        self.docs.page.append(self.current_page)
        # Prepare per-page buffer for valid characters on translated pages
        self._page_valid_chars_buffer = []

    def on_page_end(self):
        # Accumulate this page's valid characters and tokens into shared context
        try:
            if (
                self._page_valid_chars_buffer is not None
                and len(self._page_valid_chars_buffer) > 0
            ):
                page_text = "".join(self._page_valid_chars_buffer)
                char_count = len(page_text)
                try:
                    token_count = len(
                        self.tokenizer.encode(page_text, disallowed_special=())
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute token count for page: {e}")
                    token_count = 0
                self.translation_config.shared_context_cross_split_part.add_valid_counts(
                    char_count, token_count
                )
        except Exception as e:
            logger.warning(f"Failed to accumulate page valid stats: {e}")
        finally:
            self._page_valid_chars_buffer = []
        self.progress.advance(1)

    def on_page_crop_box(
        self,
        x0: float | int,
        y0: float | int,
        x1: float | int,
        y1: float | int,
    ):
        box = il_version_1.Box(x=float(x0), y=float(y0), x2=float(x1), y2=float(y1))
        self.current_page.cropbox = il_version_1.Cropbox(box=box)

    def on_page_media_box(
        self,
        x0: float | int,
        y0: float | int,
        x1: float | int,
        y1: float | int,
    ):
        box = il_version_1.Box(x=float(x0), y=float(y0), x2=float(x1), y2=float(y1))
        self.current_page.mediabox = il_version_1.Mediabox(box=box)

    def on_page_number(self, page_number: int):
        if not isinstance(page_number, int):
            raise ValueError(
                f"page_number must be an int, got {type(page_number).__name__}"
            )
        if page_number < 0:
            raise ValueError(f"page_number must be non-negative, got {page_number}")
        self.current_page.page_number = page_number

    def on_page_base_operation(self, operation: str):
        operation = zstd_helper.zstd_compress(operation)
        self.current_page.base_operations = il_version_1.BaseOperations(value=operation)

    def _decode_font_name(self, font_name):
        """Decode font name bytes to string."""
        if isinstance(font_name, bytes):
            try:
                return font_name.decode("utf-8")
            except UnicodeDecodeError:
                return "BASE64:" + base64.b64encode(font_name).decode("utf-8")
        return font_name

    def _cidfont_encoding_length_from_encoding(self, xref_id: int) -> int | None:
        """Return encoding length from the Encoding key, or None if undetermined."""
        _, encoding = self.mupdf.xref_get_key(xref_id, "Encoding")
        if encoding in ("/Identity-H", "/Identity-V"):
            return 2
        if encoding == "/WinAnsiEncoding":
            return 1
        return None

    def _cidfont_encoding_length_from_tounicode(self, xref_id: int) -> int | None:
        """Return encoding length from the ToUnicode stream, or None if unavailable."""
        _, to_unicode_id = self.mupdf.xref_get_key(xref_id, "ToUnicode")
        if to_unicode_id is None:
            return None
        to_unicode_bytes = self.mupdf.xref_stream(int(to_unicode_id.split(" ")[0]))
        code_range = re.search(
            b"begincodespacerange\n?.*<(\\d+?)>.*",
            to_unicode_bytes,
        ).group(1)
        return len(code_range) // 2

    def _cidfont_encoding_length_from_unicode_map(self, font) -> int:
        """Return encoding length inferred from the font's unicode map."""
        if (
            font.unicode_map
            and font.unicode_map.cid2unichr
            and max(font.unicode_map.cid2unichr.keys()) > 255
        ):
            return 2
        return 1

    def _get_cidfont_encoding_length(self, xref_id: int, font) -> int:
        """Determine encoding length for a CID font."""
        try:
            length = self._cidfont_encoding_length_from_encoding(xref_id)
            if length is not None:
                return length
            length = self._cidfont_encoding_length_from_tounicode(xref_id)
            if length is not None:
                return length
        except Exception:
            pass
        return self._cidfont_encoding_length_from_unicode_map(font)

    def _get_mupdf_font_properties(self, xref_id: int):
        """Get bold/italic/monospaced/serif properties from mupdf font."""
        try:
            if xref_id in self.mupdf_font_map:
                mupdf_font = self.mupdf_font_map[xref_id]
            else:
                mupdf_font = pymupdf.Font(
                    fontbuffer=self.mupdf.extract_font(xref_id)[3]
                )
                mupdf_font.has_glyph = functools.lru_cache(maxsize=10240, typed=True)(
                    mupdf_font.has_glyph,
                )
            bold = mupdf_font.is_bold
            italic = mupdf_font.is_italic
            monospaced = mupdf_font.is_monospaced
            serif = mupdf_font.is_serif
            self.mupdf_font_map[xref_id] = mupdf_font
            return bold, italic, monospaced, serif
        except Exception:
            return None, None, None, None

    def _build_font_bbox_map(self, xref_id: int, il_font_metadata, font_name: str):
        """Build character bounding box map and update il_font_metadata."""
        if xref_id is None:
            logger.warning(f"xref_id is None for font {font_name}")
            raise ValueError("xref_id is None for font %s", font_name)
        bbox_list, cmap = self.parse_font_xobj_id(xref_id)
        font_char_bounding_box_map = {}
        if not cmap:
            cmap = {x: x for x in range(257)}
        for char_id, char_bbox in enumerate(bbox_list):
            font_char_bounding_box_map[char_id] = char_bbox
        for char_id in cmap:
            if char_id < 0 or char_id >= len(bbox_list):
                continue
            bbox = bbox_list[char_id]
            x, y, x2, y2 = bbox
            is_default_bbox = (x == 0 and y == 0 and x2 == 500 and y2 == 698) or (
                x == 0 and y == 0 and x2 == 0 and y2 == 0
            )
            if is_default_bbox:
                continue
            il_font_metadata.pdf_font_char_bounding_box.append(
                il_version_1.PdfFontCharBoundingBox(
                    x=x,
                    y=y,
                    x2=x2,
                    y2=y2,
                    char_id=char_id,
                )
            )
            font_char_bounding_box_map[char_id] = bbox
        return font_char_bounding_box_map

    def _store_font_bbox_map(self, xref_id: int, font_char_bounding_box_map: dict):
        """Store the font bounding box map in the appropriate scope."""
        if self.xobj_id in self.xobj_map:
            if self.xobj_id not in self.current_page_font_char_bounding_box_map:
                self.current_page_font_char_bounding_box_map[self.xobj_id] = {}
            self.current_page_font_char_bounding_box_map[self.xobj_id][xref_id] = (
                font_char_bounding_box_map
            )
        else:
            self.current_page_font_char_bounding_box_map[xref_id] = (
                font_char_bounding_box_map
            )

    def _build_il_font_metadata(
        self, font: PDFFont, xref_id: int, font_id: str, font_name: str
    ) -> "il_version_1.PdfFont":
        """Build a PdfFont IL object from a parsed PDF font."""
        encoding_length = 1
        if isinstance(font, PDFCIDFont):
            encoding_length = self._get_cidfont_encoding_length(xref_id, font)
        bold, italic, monospaced, serif = self._get_mupdf_font_properties(xref_id)
        return il_version_1.PdfFont(
            name=font_name,
            xref_id=xref_id,
            font_id=font_id,
            encoding_length=encoding_length,
            bold=bold,
            italic=italic,
            monospace=monospaced,
            serif=serif,
            ascent=font.ascent,
            descent=font.descent,
            pdf_font_char_bounding_box=[],
        )

    def _register_font_in_page(self, font_id: str, il_font_metadata) -> None:
        """Register il_font_metadata into the correct font list (page or xobj)."""
        fonts = self.current_page.pdf_font
        if self.xobj_id in self.xobj_map:
            fonts = self.xobj_map[self.xobj_id].pdf_font
        fonts[:] = [f for f in fonts if f.font_id != font_id]
        fonts.append(il_font_metadata)

    def on_page_resource_font(self, font: PDFFont, xref_id: int, font_id: str):
        font_name = self._decode_font_name(font.fontname)
        logger.debug(f"handle font {font_name} @ {xref_id} in {self.xobj_id}")
        il_font_metadata = self._build_il_font_metadata(
            font, xref_id, font_id, font_name
        )
        try:
            font_char_bounding_box_map = self._build_font_bbox_map(
                xref_id, il_font_metadata, font_name
            )
            self._store_font_bbox_map(xref_id, font_char_bounding_box_map)
        except Exception as e:
            xref_label = "None" if xref_id is None else str(xref_id)
            logger.error(f"failed to parse font xobj id {xref_label}: {e}")
        self.current_page_font_name_id_map[xref_id] = font_id
        self.current_available_fonts[font_id] = il_font_metadata
        self._register_font_in_page(font_id, il_font_metadata)

    def _load_bbox_from_font_files(self, xobj_id: int, encoding, differences) -> list:
        """Try each FontFile key and return the first non-empty bbox_list found."""
        for file_key in ["FontFile", "FontFile2", "FontFile3"]:
            font_file = self.mupdf.xref_get_key(xobj_id, f"FontDescriptor/{file_key}")
            if file_idx := indirect(font_file):
                return parse_font_file(self.mupdf, file_idx, encoding, differences)
        return []

    def _load_cmap(self, xobj_id: int) -> dict:
        """Load the ToUnicode CMap for a font xref, returning {} if absent."""
        to_unicode = self.mupdf.xref_get_key(xobj_id, "ToUnicode")
        if to_unicode_idx := indirect(to_unicode):
            return parse_cmap(self.mupdf.xref_stream(to_unicode_idx).decode("U8"))
        return {}

    def _resolve_bbox_fallbacks(self, xobj_id: int, bbox_list: list) -> list:
        """Apply Base14, CID-font, and Type3 fallbacks to arrive at a final bbox_list."""
        if not bbox_list:
            obj_type, obj_val = self.mupdf.xref_get_key(xobj_id, "BaseFont")
            if obj_type == "name":
                bbox_list = get_base14_bbox(obj_val[1:])
        if cid_bbox := get_cidfont_bbox(self.mupdf, xobj_id):
            bbox_list = cid_bbox
        if self.mupdf.xref_get_key(xobj_id, "Subtype")[1] == "/Type3":
            bbox_list = get_type3_bbox(self.mupdf, xobj_id)
        return bbox_list

    def parse_font_xobj_id(self, xobj_id: int):
        if xobj_id is None:
            return [], {}

        encoding = parse_font_encoding(self.mupdf, xobj_id)
        differences = []
        font_differences = self.mupdf.xref_get_key(xobj_id, "Encoding/Differences")
        if font_differences:
            differences = parse_encoding(font_differences[1])
        bbox_list = self._load_bbox_from_font_files(xobj_id, encoding, differences)
        cmap = self._load_cmap(xobj_id)
        bbox_list = self._resolve_bbox_fallbacks(xobj_id, bbox_list)
        return bbox_list, cmap

    def _build_clip_path_instruction(self, clip_path, source_ctm, target_ctm, evenodd):
        """Build a single clipping path instruction string."""
        transformed_path = self.transform_clip_path(clip_path, source_ctm, target_ctm)
        op = "W* n" if evenodd else "W n"
        args = []
        for p in transformed_path:
            if len(p) == 1:
                args.append(p[0])
            elif len(p) > 1:
                args.extend([f"{x:F}" for x in p[1:]])
                args.append(p[0])
        if args:
            return f"{' '.join(args)} {op}"
        return None

    def _append_clipping_instructions(self, parts, clip_paths, target_ctm):
        """Append transformed clipping path instructions to parts list."""
        for clip_path, source_ctm, evenodd in clip_paths:
            try:
                instruction = self._build_clip_path_instruction(
                    clip_path, source_ctm, target_ctm, evenodd
                )
                if instruction:
                    parts.append(instruction)
            except Exception as e:
                logger.warning(f"Error transforming clip path: {e}")

    def create_graphic_state(
        self,
        gs: src.doctranslator.pdfminer.pdfinterp.PDFGraphicState
        | list[tuple[str, str]],
        include_clipping: bool = False,
        target_ctm: tuple[float, float, float, float, float, float] = None,
        clip_paths=None,
    ):
        if clip_paths is None:
            clip_paths = self.current_clip_paths
        passthrough_instruction = getattr(gs, "passthrough_instruction", gs)

        if include_clipping:
            instruction_parts = [f"{arg} {op}" for op, arg in passthrough_instruction]
        else:
            instruction_parts = [
                f"{arg} {op}"
                for op, arg in passthrough_instruction
                if op not in ("W n", "W* n")
            ]

        # Add transformed clipping paths if requested and target CTM is provided
        if include_clipping and target_ctm and clip_paths:
            self._append_clipping_instructions(
                instruction_parts, clip_paths, target_ctm
            )

        passthrough_per_char_instruction = " ".join(instruction_parts)

        # Pool graphic states to reduce memory usage
        if passthrough_per_char_instruction not in self.graphic_state_pool:
            self.graphic_state_pool[passthrough_per_char_instruction] = (
                il_version_1.GraphicState(
                    passthrough_per_char_instruction=passthrough_per_char_instruction
                )
            )

        return self.graphic_state_pool[passthrough_per_char_instruction]

    def _get_char_font(self, char: LTChar):
        """Find the PdfFont for a character from current page or xobject."""
        for pdf_font in self.xobj_map.get(char.xobj_id, self.current_page).pdf_font:
            if pdf_font.font_id == char.aw_font_id:
                return pdf_font
        return None

    def _get_char_bounding_box(self, char: LTChar, font, char_id: int):
        """Retrieve character bounding box from font bounding box map."""
        try:
            font_bounding_box_map = self.current_page_font_char_bounding_box_map.get(
                char.xobj_id, self.current_page_font_char_bounding_box_map
            ).get(font.xref_id)
            if font_bounding_box_map:
                return font_bounding_box_map.get(char_id, None)
        except Exception:
            pass
        return None

    def _compute_visual_bbox(self, char: LTChar, descent: float):
        """Compute vertical flag and visual bounding box for a character."""
        if char.matrix[0] == 0 and char.matrix[3] == 0:
            vertical = True
            box = il_version_1.Box(
                x=char.bbox[0] - descent,
                y=char.bbox[1],
                x2=char.bbox[2] - descent,
                y2=char.bbox[3],
            )
        else:
            vertical = False
            box = il_version_1.Box(
                x=char.bbox[0],
                y=char.bbox[1] + descent,
                x2=char.bbox[2],
                y2=char.bbox[3] + descent,
            )
        return vertical, il_version_1.VisualBbox(box=box)

    def _refine_visual_bbox(self, pdf_char, char: LTChar, char_bounding_box, font_size):
        """Refine visual bbox using per-character bounding box data if available."""
        if not (char_bounding_box and len(char_bounding_box) == 4):
            return
        x_min, y_min, x_max, y_max = char_bounding_box
        factor = font_size / 1000
        ll = (char.bbox[0] + x_min * factor, char.bbox[1] + y_min * factor)
        ur = (char.bbox[0] + x_max * factor, char.bbox[1] + y_max * factor)
        if (ur[0] - ll[0]) * (ur[1] - ll[1]) > 1:
            pdf_char.visual_bbox = il_version_1.VisualBbox(
                il_version_1.Box(ll[0], ll[1], ur[0], ur[1])
            )

    def _check_rotation_angle(self, char: LTChar) -> bool:
        """Return False if the character's rotation angle is outside accepted ranges."""
        try:
            rotation_angle = get_rotation_angle(char.matrix)
            return -0.1 <= rotation_angle <= 0.1 or 89.9 <= rotation_angle <= 90.1
        except Exception:
            logger.warning("Failed to get rotation angle for char %s", char.get_text())
            return True  # allow through on error, consistent with original behaviour

    def _build_char_bbox(self, char: LTChar, char_unicode: str) -> "il_version_1.Box":
        """Build and validate an il_version_1.Box for the character."""
        bbox = il_version_1.Box(
            x=char.bbox[0],
            y=char.bbox[1],
            x2=char.bbox[2],
            y2=char.bbox[3],
        )
        if bbox.x2 < bbox.x or bbox.y2 < bbox.y:
            logger.warning(
                "Invalid bounding box for character %s: %s", char_unicode, bbox
            )
        return bbox

    def _build_pdf_char(
        self,
        char: LTChar,
        char_id: int,
        char_unicode: str,
        advance,
        bbox,
        vertical,
        visual_bbox,
        gs,
    ) -> "il_version_1.PdfCharacter":
        """Construct the PdfCharacter IL object."""
        pdf_style = il_version_1.PdfStyle(
            font_id=char.aw_font_id,
            font_size=char.size,
            graphic_state=gs,
        )
        pdf_char = il_version_1.PdfCharacter(
            box=bbox,
            pdf_character_id=char_id,
            advance=advance,
            char_unicode=char_unicode,
            vertical=vertical,
            pdf_style=pdf_style,
            xobj_id=char.xobj_id,
            visual_bbox=visual_bbox,
            render_order=char.render_order,
            sub_render_order=0,
        )
        if self.translation_config.ocr_workaround:
            pdf_char.pdf_style.graphic_state = BLACK
            pdf_char.render_order = None
        return pdf_char

    def _maybe_add_char_box_rect(self, pdf_char) -> None:
        """Append a debug rectangle for the character visual bbox if show_char_box is set."""
        if self.translation_config.show_char_box:
            self.current_page.pdf_rectangle.append(
                il_version_1.PdfRectangle(
                    box=pdf_char.visual_bbox.box,
                    graphic_state=YELLOW,
                    debug_info=True,
                    line_width=0.2,
                )
            )

    def on_lt_char(self, char: LTChar):
        if char.aw_font_id is None:
            return
        if not self._check_rotation_angle(char):
            return
        try:
            self._collect_valid_char(char.get_text())
        except Exception as e:
            logger.warning(f"Error collecting valid char: {e}")

        gs = self.create_graphic_state(char.graphicstate)
        font = self._get_char_font(char)
        descent = (
            font.descent * char.size / 1000 if font and hasattr(font, "descent") else 0
        )
        char_id = char.cid
        char_bounding_box = (
            self._get_char_bounding_box(char, font, char_id) if font else None
        )

        char_unicode = char.get_text()
        if space_regex.match(char_unicode):
            char_unicode = " "
        advance = char.adv
        bbox = self._build_char_bbox(char, char_unicode)
        vertical, visual_bbox = self._compute_visual_bbox(char, descent)
        pdf_char = self._build_pdf_char(
            char, char_id, char_unicode, advance, bbox, vertical, visual_bbox, gs
        )
        if pdf_char.pdf_style.font_size == 0.0:
            logger.warning("Font size is 0.0 for character %s. Skip it.", char_unicode)
            return

        self._refine_visual_bbox(
            pdf_char, char, char_bounding_box, pdf_char.pdf_style.font_size
        )
        self.current_page.pdf_character.append(pdf_char)
        self._maybe_add_char_box_rect(pdf_char)

    def _is_valid_char(self, ch: str) -> bool:
        """Return True if ch is a valid translatable character."""
        if not ch or "(cid:" in ch:
            return False
        try:
            if self.font_mapper.has_char(ch):
                return True
            return len(ch) > 1 and all(self.font_mapper.has_char(x) for x in ch)
        except Exception:
            return False

    def _collect_valid_char(self, ch: str):
        """Append a valid character into the current page buffer according to rules.
        Rules:
        - Include whitespace matched by space_regex directly.
        - Ignore categories that are never normal text: {Cc, Cs, Co, Cn}.
        - Apply inverted criteria from formular_helper.py (21-28):
          empty -> invalid, contains '(cid:' -> invalid,
          not has_char(ch) -> invalid unless len(ch) > 1 and all(has_char(x)).
        """
        if self._page_valid_chars_buffer is None:
            return
        if space_regex.match(ch):
            self._page_valid_chars_buffer.append(ch)
            return
        try:
            cat = unicodedata.category(ch[0]) if ch else None
        except Exception:
            cat = None
        if cat in {"Cc", "Cs", "Co", "Cn"}:
            return
        if self._is_valid_char(ch):
            self._page_valid_chars_buffer.append(ch)

    def _build_transformed_paths(self, original_path) -> list:
        """Build transformed PdfPath list from original path points."""
        paths = []
        for point in original_path:
            op = point[0]
            if len(point) == 1:
                paths.append(il_version_1.PdfPath(op=op, x=None, y=None, has_xy=False))
                continue
            for p in point[1:-1]:
                paths.append(il_version_1.PdfPath(op="", x=p[0], y=p[1], has_xy=True))
            paths.append(
                il_version_1.PdfPath(op=op, x=point[-1][0], y=point[-1][1], has_xy=True)
            )
        return paths

    def _build_raw_pdf_paths(self, raw_path) -> list | None:
        """Build PdfOriginalPath list from raw path data."""
        if raw_path is None:
            return None
        raw_pdf_paths = []
        for path in raw_path:
            if path[0] == "h":
                raw_pdf_paths.append(
                    il_version_1.PdfOriginalPath(
                        pdf_path=il_version_1.PdfPath(
                            x=0.0, y=0.0, op="h", has_xy=False
                        )
                    )
                )
            else:
                for p in batched(path[1:-2], 2, strict=True):
                    raw_pdf_paths.append(
                        il_version_1.PdfOriginalPath(
                            pdf_path=il_version_1.PdfPath(
                                x=float(p[0]),
                                y=float(p[1]),
                                op="",
                                has_xy=True,
                            )
                        )
                    )
                raw_pdf_paths.append(
                    il_version_1.PdfOriginalPath(
                        pdf_path=il_version_1.PdfPath(
                            x=float(path[-2]),
                            y=float(path[-1]),
                            op=path[0],
                            has_xy=True,
                        )
                    )
                )
        return raw_pdf_paths

    def on_lt_curve(self, curve: src.doctranslator.pdfminer.layout.LTCurve):
        if not self.enable_graphic_element_process:
            return
        bbox = il_version_1.Box(
            x=curve.bbox[0],
            y=curve.bbox[1],
            x2=curve.bbox[2],
            y2=curve.bbox[3],
        )
        ctm = getattr(curve, "ctm", None)
        gs = self.create_graphic_state(
            curve.passthrough_instruction,
            include_clipping=True,
            target_ctm=ctm,
            clip_paths=curve.clip_paths,
        )
        paths = self._build_transformed_paths(curve.original_path)
        raw_pdf_paths = self._build_raw_pdf_paths(getattr(curve, "raw_path", None))

        curve_obj = il_version_1.PdfCurve(
            box=bbox,
            graphic_state=gs,
            pdf_path=paths,
            fill_background=curve.fill,
            stroke_path=curve.stroke,
            evenodd=curve.evenodd,
            debug_info="a",
            xobj_id=curve.xobj_id,
            render_order=curve.render_order,
            ctm=list(ctm) if ctm is not None else None,
            pdf_original_path=raw_pdf_paths,
        )
        self.current_page.pdf_curve.append(curve_obj)

    def on_xobj_form(
        self,
        ctm: tuple[float, float, float, float, float, float],
        xobj_id: int,
        xref_id: int,
        form_type: Literal["image", "form"],
        do_args: str,
        bbox: tuple[float, float, float, float],
        matrix: tuple[float, float, float, float, float, float],
    ):
        logger.debug(f"on_xobj_form: {do_args}[{bbox}] @ {xref_id} in {self.xobj_id}")
        matrix = mult_matrix(matrix, ctm)
        (x, y, w, h) = guarded_bbox(bbox)
        bounds = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
        bbox = get_bound(apply_matrix_pt(matrix, (p, q)) for (p, q) in bounds)

        gs = self.create_graphic_state(
            self.passthrough_per_char_instruction, include_clipping=True, target_ctm=ctm
        )

        figure_bbox = il_version_1.Box(
            x=bbox[0],
            y=bbox[1],
            x2=bbox[2],
            y2=bbox[3],
        )
        pdf_matrix = il_version_1.PdfMatrix(
            a=ctm[0],
            b=ctm[1],
            c=ctm[2],
            d=ctm[3],
            e=ctm[4],
            f=ctm[5],
        )
        affine_transform = decompose_ctm(ctm)
        xobj_form = il_version_1.PdfXobjForm(
            xref_id=xref_id,
            do_args=do_args,
        )
        pdf_form_subtype = il_version_1.PdfFormSubtype(
            pdf_xobj_form=xobj_form,
        )
        new_form = il_version_1.PdfForm(
            xobj_id=xobj_id,
            box=figure_bbox,
            pdf_matrix=pdf_matrix,
            graphic_state=gs,
            pdf_affine_transform=affine_transform,
            render_order=self.get_render_order_and_increase(),
            form_type=form_type,
            pdf_form_subtype=pdf_form_subtype,
            ctm=list(ctm),
        )
        self.current_page.pdf_form.append(new_form)

    def on_pdf_clip_path(
        self,
        clip_path,
        evenodd: bool,
        ctm: tuple[float, float, float, float, float, float],
    ):
        try:
            self.current_clip_paths.append((clip_path.copy(), ctm, evenodd))
        except Exception as e:
            logger.warning(f"Error in on_pdf_clip_path: {e}")

    def create_il(self):
        pages = [
            page
            for page in self.docs.page
            if self.translation_config.should_translate_page(page.page_number + 1)
        ]
        self.docs.page = pages
        return self.docs

    def on_total_pages(self, total_pages: int):
        if not isinstance(total_pages, int):
            raise ValueError(
                f"total_pages must be an int, got {type(total_pages).__name__}"
            )
        if total_pages <= 0:
            raise ValueError(f"total_pages must be positive, got {total_pages}")
        self.docs.total_pages = total_pages
        total = 0
        for page in range(total_pages):
            if self.translation_config.should_translate_page(page + 1) is False:
                continue
            total += 1
        self.progress = self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total,
        )

    def on_pdf_figure(self, figure: LTFigure):
        box = il_version_1.Box(
            figure.bbox[0],
            figure.bbox[1],
            figure.bbox[2],
            figure.bbox[3],
        )
        self.current_page.pdf_figure.append(il_version_1.PdfFigure(box=box))

    def on_inline_image_begin(self):
        """Begin processing inline image"""
        # Store current state for inline image processing
        self._inline_image_state = {
            "ctm": None,
            "parameters": {},
        }

    def _extract_inline_image_data(self, stream_obj) -> str:
        """Return base64-encoded image data from an inline image stream object."""
        import base64

        if hasattr(stream_obj, "data") and stream_obj.data is not None:
            return base64.b64encode(stream_obj.data).decode("ascii")
        if hasattr(stream_obj, "rawdata") and stream_obj.rawdata is not None:
            return base64.b64encode(stream_obj.rawdata).decode("ascii")
        return ""

    def _build_inline_image_parameters(self, stream_obj) -> dict:
        """Build a plain dict of image parameters from a stream object's attrs."""
        image_dict = stream_obj.attrs if hasattr(stream_obj, "attrs") else {}
        parameters = {}
        for key, value in image_dict.items():
            parameters[key] = value.name if hasattr(value, "name") else str(value)
        return parameters

    def _compute_inline_image_bbox(self, ctm):
        """Compute the final bounding box of an inline image given its CTM."""
        from src.doctranslator.format.pdf.babelpdf.utils import guarded_bbox
        from src.doctranslator.pdfminer.utils import apply_matrix_pt
        from src.doctranslator.pdfminer.utils import get_bound

        (x, y, w, h) = guarded_bbox((0, 0, 1, 1))
        bounds = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
        return get_bound(apply_matrix_pt(ctm, (p, q)) for (p, q) in bounds)

    def on_inline_image_end(self, stream_obj, ctm):
        """End processing inline image and create PdfForm"""
        import json

        from src.doctranslator.format.pdf.document_il.utils.matrix_helper import (
            decompose_ctm,
        )

        parameters = self._build_inline_image_parameters(stream_obj)
        image_data = self._extract_inline_image_data(stream_obj)
        inline_form = il_version_1.PdfInlineForm(
            form_data=image_data, image_parameters=json.dumps(parameters)
        )
        final_bbox = self._compute_inline_image_bbox(ctm)
        gs = self.create_graphic_state(
            self.passthrough_per_char_instruction, include_clipping=True, target_ctm=ctm
        )
        pdf_matrix = il_version_1.PdfMatrix(
            a=ctm[0], b=ctm[1], c=ctm[2], d=ctm[3], e=ctm[4], f=ctm[5]
        )
        affine_transform = decompose_ctm(ctm)
        pdf_form_subtype = il_version_1.PdfFormSubtype(pdf_inline_form=inline_form)
        pdf_form = il_version_1.PdfForm(
            box=il_version_1.Box(
                x=final_bbox[0],
                y=final_bbox[1],
                x2=final_bbox[2],
                y2=final_bbox[3],
            ),
            graphic_state=gs,
            pdf_matrix=pdf_matrix,
            pdf_affine_transform=affine_transform,
            pdf_form_subtype=pdf_form_subtype,
            xobj_id=self.xobj_id,
            ctm=list(ctm),
            render_order=self.get_render_order_and_increase(),
            form_type="image",
        )
        self.current_page.pdf_form.append(pdf_form)
