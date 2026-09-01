import io
import logging
import os
import re
import time
import unicodedata
from abc import ABC
from abc import abstractmethod
from multiprocessing import Process
from pathlib import Path

import freetype
import pymupdf
from bitstring import BitStream

from src.worker.doctranslator.format.pdf.document_il import PdfOriginalPath
from src.worker.doctranslator.format.pdf.document_il import il_version_1
from src.worker.doctranslator.format.pdf.document_il.utils.fontmap import FontMapper
from src.worker.doctranslator.format.pdf.document_il.utils.matrix_helper import (
    matrix_to_bytes,
)
from src.worker.doctranslator.format.pdf.document_il.utils.zstd_helper import (
    zstd_decompress,
)
from src.worker.doctranslator.format.pdf.translation_config import TranslateResult
from src.worker.doctranslator.format.pdf.translation_config import TranslationConfig
from src.worker.doctranslator.format.pdf.translation_config import WatermarkOutputMode
from src.worker.doctranslator.utils.common import batched
from src.worker.loaders import font_names

logger = logging.getLogger(__name__)

SUBSET_FONT_STAGE_NAME = "Subset font"
SAVE_PDF_STAGE_NAME = "Save PDF"

# Upper bound on the paragraph text echoed into diagnostic error logs.
_UNICODE_LOG_PREVIEW_CHARS = 200


def _describe_box(box) -> str:
    """Compact `WxH@(x,y)` description of a box for diagnostic logs."""
    if box is None:
        return "none"
    try:
        return f"{box.x2 - box.x:.1f}x{box.y2 - box.y:.1f}@({box.x:.1f},{box.y:.1f})"
    except Exception:
        return "unknown"


class RenderUnit(ABC):
    """Abstract base class for all renderable units."""

    def __init__(
        self,
        render_order: int,
        sub_render_order: int = 0,
        xobj_id: str | None = None,
    ):
        self.render_order = render_order
        self.sub_render_order = sub_render_order
        self.xobj_id = xobj_id
        if self.render_order is None:
            self.render_order = 9999999999999999
        if self.sub_render_order is None:
            self.sub_render_order = 9999999999999999

    @abstractmethod
    def render(
        self,
        draw_op: BitStream,
        context: "RenderContext",
    ) -> None:
        """Render this unit to the draw_op BitStream."""
        pass

    def get_sort_key(self) -> tuple[int, int]:
        """Get the sort key for ordering render units."""
        return (self.render_order, self.sub_render_order)


class CharacterRenderUnit(RenderUnit):
    """Render unit for PDF characters."""

    def __init__(
        self,
        char: il_version_1.PdfCharacter,
        render_order: int,
        sub_render_order: int = 0,
    ):
        super().__init__(render_order, sub_render_order, char.xobj_id)
        self.char = char

    def _is_font_available(self, font_id: str, context: "RenderContext") -> bool:
        """Check whether the font is available in the current rendering context."""
        if self.xobj_id in context.xobj_available_fonts:
            return font_id in context.xobj_available_fonts[self.xobj_id]
        return font_id in context.available_font_list

    def _resolve_encoding_length(
        self, font_id: str, encoding_length_map: dict, context: "RenderContext"
    ):
        """Resolve encoding length for a font, falling back to all_encoding_length_map."""
        length = encoding_length_map.get(font_id)
        if length is not None:
            return length
        if font_id in context.all_encoding_length_map:
            return context.all_encoding_length_map[font_id]
        logger.debug(
            f"Font {font_id} not found in encoding length map for page {context.page.page_number}"
        )
        return None

    def render(self, draw_op: BitStream, context: "RenderContext") -> None:
        char = self.char
        if char.char_unicode == "\n":
            return
        if char.pdf_character_id is None:
            return

        char_size = char.pdf_style.font_size
        font_id = char.pdf_style.font_id

        encoding_length_map = (
            context.xobj_encoding_length_map[self.xobj_id]
            if self.xobj_id in context.xobj_encoding_length_map
            else context.page_encoding_length_map
        )

        if context.check_font_exists and not self._is_font_available(font_id, context):
            return

        draw_op.append(b"q ")
        context.pdf_creator.render_graphic_state(draw_op, char.pdf_style.graphic_state)

        if char.vertical:
            draw_op.append(
                f"BT /{font_id} {char_size:f} Tf 0 1 -1 0 {char.box.x2:f} {char.box.y:f} Tm ".encode(),
            )
        else:
            draw_op.append(
                f"BT /{font_id} {char_size:f} Tf 1 0 0 1 {char.box.x:f} {char.box.y:f} Tm ".encode(),
            )

        encoding_length = self._resolve_encoding_length(
            font_id, encoding_length_map, context
        )
        if encoding_length is None:
            return

        draw_op.append(
            f"<{char.pdf_character_id:0{encoding_length * 2}x}>".upper().encode(),
        )
        draw_op.append(b" Tj ET Q \n")


class FormRenderUnit(RenderUnit):
    """Render unit for PDF forms."""

    def __init__(
        self,
        form: il_version_1.PdfForm,
        render_order: int,
        sub_render_order: int = 0,
    ):
        super().__init__(render_order, sub_render_order, form.xobj_id)
        self.form = form

    def _append_inline_image_params(self, draw_op: BitStream, image_parameters: str):
        """Append inline image parameters to the draw stream."""
        import json

        try:
            params = json.loads(image_parameters)
            for key, value in params.items():
                key = key.lstrip("/")
                if value is True or value == "True":
                    value = "true"
                elif value is False or value == "False":
                    value = "false"
                draw_op.append(f"/{key} {value} ".encode())
        except json.JSONDecodeError:
            pass

    def _append_inline_form(self, draw_op: BitStream, inline_form):
        """Append an inline image (BI...ID...EI) to the draw stream."""
        import base64

        draw_op.append(b" BI ")
        if inline_form.image_parameters:
            self._append_inline_image_params(draw_op, inline_form.image_parameters)
        draw_op.append(b"ID ")
        if inline_form.form_data:
            try:
                draw_op.append(base64.b64decode(inline_form.form_data))
            except Exception:
                pass
        draw_op.append(b" EI ")

    def render(self, draw_op: BitStream, context: "RenderContext") -> None:
        form = self.form
        draw_op.append(b"q ")

        if form.pdf_matrix is None:
            raise RuntimeError("form.pdf_matrix must be set before rendering")
        if form.relocation_transform and len(form.relocation_transform) == 6:
            try:
                relocation_matrix = tuple(float(x) for x in form.relocation_transform)
                draw_op.append(matrix_to_bytes(relocation_matrix))
            except (ValueError, TypeError):
                pass

        draw_op.append(matrix_to_bytes(form.pdf_matrix))
        draw_op.append(b" ")
        draw_op.append(form.graphic_state.passthrough_per_char_instruction.encode())
        draw_op.append(b" ")

        if form.pdf_form_subtype is None:
            raise RuntimeError("form.pdf_form_subtype must be set before rendering")
        if form.pdf_form_subtype.pdf_xobj_form:
            draw_op.append(
                f" /{form.pdf_form_subtype.pdf_xobj_form.do_args} Do ".encode()
            )
        elif form.pdf_form_subtype.pdf_inline_form:
            self._append_inline_form(draw_op, form.pdf_form_subtype.pdf_inline_form)
        draw_op.append(b" Q\n")


class RectangleRenderUnit(RenderUnit):
    """Render unit for PDF rectangles."""

    def __init__(
        self,
        rectangle: il_version_1.PdfRectangle,
        render_order: int,
        sub_render_order: int = 0,
        line_width: float = 0.4,
    ):
        super().__init__(render_order, sub_render_order, rectangle.xobj_id)
        self.rectangle = rectangle
        self.line_width = line_width

    def render(self, draw_op: BitStream, context: "RenderContext") -> None:
        rectangle = self.rectangle
        x1 = rectangle.box.x
        y1 = rectangle.box.y
        x2 = rectangle.box.x2
        y2 = rectangle.box.y2
        width = x2 - x1
        height = y2 - y1

        draw_op.append(b"q n ")
        draw_op.append(
            rectangle.graphic_state.passthrough_per_char_instruction.encode(),
        )

        line_width = self.line_width
        if rectangle.line_width is not None:
            line_width = rectangle.line_width
        if line_width > 0:
            draw_op.append(f" {line_width:.6f} w ".encode())

        draw_op.append(f"{x1:.6f} {y1:.6f} {width:.6f} {height:.6f} re ".encode())
        if rectangle.fill_background:
            draw_op.append(b" f ")
        else:
            draw_op.append(b" S ")

        draw_op.append(b"Q\n")


class CurveRenderUnit(RenderUnit):
    """Render unit for PDF curves."""

    def __init__(
        self,
        curve: il_version_1.PdfCurve,
        render_order: int,
        sub_render_order: int = 0,
    ):
        super().__init__(render_order, sub_render_order, curve.xobj_id)
        self.curve = curve

    def _build_path_op(self, curve) -> BitStream:
        """Build path drawing operations from curve paths."""
        path_op = BitStream(b" ")
        path_to_use = (
            curve.pdf_original_path
            if curve.pdf_original_path is not None
            else curve.pdf_path
        )
        for path in path_to_use:
            if isinstance(path, PdfOriginalPath):
                path = path.pdf_path
            if path.has_xy:
                path_op.append(f"{path.x:F} {path.y:F} {path.op} ".encode())
            else:
                path_op.append(f"{path.op} ".encode())
        return path_op

    def render(self, draw_op: BitStream, context: "RenderContext") -> None:
        curve = self.curve
        draw_op.append(b"q n ")

        if curve.relocation_transform and len(curve.relocation_transform) == 6:
            try:
                relocation_matrix = tuple(float(x) for x in curve.relocation_transform)
                draw_op.append(matrix_to_bytes(relocation_matrix))
            except (ValueError, TypeError):
                pass

        draw_op.append(b" ")

        if curve.ctm and len(curve.ctm) == 6:
            ctm = curve.ctm
            draw_op.append(
                f"{ctm[0]:.6f} {ctm[1]:.6f} {ctm[2]:.6f} {ctm[3]:.6f} {ctm[4]:.6f} {ctm[5]:.6f} cm ".encode()
            )

        draw_op.append(b" ")
        draw_op.append(curve.graphic_state.passthrough_per_char_instruction.encode())
        draw_op.append(b" ")

        path_op = self._build_path_op(curve)

        if curve.fill_background:
            draw_op.append(path_op)
            draw_op.append(b" f")
        draw_op.append(b"* " if curve.evenodd else b" ")
        if curve.stroke_path:
            draw_op.append(path_op)
            draw_op.append(b"S ")

        draw_op.append(b" n Q\n")


class RenderContext:
    """Context object containing shared state for rendering."""

    def __init__(
        self,
        pdf_creator: "PDFCreater",
        page: il_version_1.Page,
        available_font_list: set[str],
        page_encoding_length_map: dict[str, int],
        all_encoding_length_map: dict[str, int],
        xobj_available_fonts: dict[str, set[str]],
        xobj_encoding_length_map: dict[str, dict[str, int]],
        ctm_for_ops: bytes,
        check_font_exists: bool = False,
    ):
        self.pdf_creator = pdf_creator
        self.page = page
        self.available_font_list = available_font_list
        self.page_encoding_length_map = page_encoding_length_map
        self.all_encoding_length_map = all_encoding_length_map
        self.xobj_available_fonts = xobj_available_fonts
        self.xobj_encoding_length_map = xobj_encoding_length_map
        self.ctm_for_ops = ctm_for_ops
        self.check_font_exists = check_font_exists


def to_int(src):
    return int(re.search(r"\d+", src).group(0))


def parse_mapping(text):
    mapping = []
    for x in re.finditer(rb"<(?P<num>[a-fA-F0-9]+)>", text):
        mapping.append(int(x.group("num"), 16))
    return mapping


def apply_normalization(cmap, gid, code):
    need = False
    if 0x2F00 <= code <= 0x2FD5:  # Kangxi Radicals
        need = True
    if 0xF900 <= code <= 0xFAFF:  # CJK Compatibility Ideographs
        need = True
    if need:
        norm = unicodedata.normalize("NFD", chr(code))
        cmap[gid] = ord(norm)
    else:
        cmap[gid] = code


def update_tounicode_cmap_pair(cmap, data):
    for start, stop, value in batched(data, 3):
        for gid in range(start, stop + 1):
            code = value + gid - start
            apply_normalization(cmap, gid, code)


def update_tounicode_cmap_code(cmap, data):
    for gid, code in batched(data, 2):
        apply_normalization(cmap, gid, code)


def parse_tounicode_cmap(data):
    cmap = {}
    for x in re.finditer(
        rb"\s+beginbfrange\s*(?P<r>(<[0-9a-fA-F]+>\s*)+)endbfrange\s+", data
    ):
        update_tounicode_cmap_pair(cmap, parse_mapping(x.group("r")))
    for x in re.finditer(
        rb"\s+beginbfchar\s*(?P<c>(<[0-9a-fA-F]+>\s*)+)endbfchar", data
    ):
        update_tounicode_cmap_code(cmap, parse_mapping(x.group("c")))
    return cmap


def parse_truetype_data(data):
    glyph_in_use = []
    face = freetype.Face(io.BytesIO(data))
    for i in range(face.num_glyphs):
        face.load_glyph(i)
        if face.glyph.outline.contours:
            glyph_in_use.append(i)
    return glyph_in_use


TOUNICODE_HEAD = """\
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo <</Registry(Adobe)/Ordering(UCS)/Supplement 0>> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange"""
TOUNICODE_TAIL = """\
endcmap
CMapName currentdict /CMap defineresource pop
end
end"""


def make_tounicode(cmap, used):
    short = []
    for x in used:
        if x in cmap:
            short.append((x, cmap[x]))
    line = [TOUNICODE_HEAD]
    for block in batched(short, 100):
        line.append(f"{len(block)} beginbfchar")
        for glyph, code in block:
            if code < 0x10000:
                line.append(f"<{glyph:04x}><{code:04x}>")
            else:
                code -= 0x10000
                high = 0xD800 + (code >> 10)
                low = 0xDC00 + (code & 0b1111111111)
                line.append(f"<{glyph:04x}><{high:04x}{low:04x}>")
        line.append("endbfchar")
    line.append(TOUNICODE_TAIL)
    return "\n".join(line)


def reproduce_one_font(doc, index):
    m = doc.xref_get_key(index, "ToUnicode")
    f = doc.xref_get_key(index, "DescendantFonts")
    if m[0] == "xref" and f[0] == "array":
        mi = to_int(m[1])
        fi = to_int(f[1])
        ff = doc.xref_get_key(fi, "FontDescriptor/FontFile2")
        ms = doc.xref_stream(mi)
        fs = doc.xref_stream(to_int(ff[1]))
        cmap = parse_tounicode_cmap(ms)
        used = parse_truetype_data(fs)
        text = make_tounicode(cmap, used)
        doc.update_stream(mi, bytes(text, "U8"))


def reproduce_cmap(doc):
    if not doc:
        raise ValueError("doc must not be None or empty")
    font_set = set()
    for page in doc:
        try:
            font_list = page.get_fonts()
            for font in font_list:
                if font[1] == "ttf" and font[3] in font_names() and ".ttf" in font[4]:
                    font_set.add(font)
        except Exception as e:
            logger.error(f"Error in getting page fonts: {e}")
    for font in font_set:
        reproduce_one_font(doc, font[0])
    return doc


def _subset_fonts_process(pdf_path, output_path):
    """Function to run in subprocess for font subsetting.

    Args:
        pdf_path: Path to the PDF file to subset
        output_path: Path where to save the result
    """
    try:
        pdf = pymupdf.open(pdf_path)
        pdf.subset_fonts(fallback=False)
        pdf.save(output_path)
        # 返回 0 表示成功
        os._exit(0)
    except Exception as e:
        logger.error(f"Error in font subsetting subprocess: {e}")
        # 返回 1 表示失败
        os._exit(1)


def _save_pdf_clean_process(
    pdf_path,
    output_path,
    garbage=1,
    deflate=True,
    clean=True,
    deflate_fonts=True,
    linear=False,
):
    """Function to run in subprocess for saving PDF with clean=True which can be time-consuming.

    Args:
        pdf_path: Path to the PDF file to save
        output_path: Path where to save the result
        garbage: Garbage collection level (0, 1, 2, 3, 4)
        deflate: Whether to deflate the PDF
        clean: Whether to clean the PDF
        deflate_fonts: Whether to deflate fonts
        linear: Whether to linearize the PDF
    """
    try:
        pdf = pymupdf.open(pdf_path)
        pdf.save(
            output_path,
            garbage=garbage,
            deflate=deflate,
            clean=clean,
            deflate_fonts=deflate_fonts,
            linear=linear,
        )
        # 返回 0 表示成功
        os._exit(0)
    except Exception as e:
        logger.error(f"Error in save PDF with clean=True subprocess: {e}")
        # 返回 1 表示失败
        os._exit(1)


class PDFCreater:
    stage_name = "Generate drawing instructions"

    def __init__(
        self,
        original_pdf_path: str,
        document: il_version_1.Document,
        translation_config: TranslationConfig,
        mediabox_data: dict,
    ):
        self.original_pdf_path = original_pdf_path
        self.docs = document
        self.font_path = translation_config.font
        self.font_mapper = FontMapper(translation_config)
        self.translation_config = translation_config
        self.mediabox_data = mediabox_data

    def render_graphic_state(
        self,
        draw_op: BitStream,
        graphic_state: il_version_1.GraphicState,
    ):
        if graphic_state is None:
            return
        if graphic_state.passthrough_per_char_instruction:
            draw_op.append(
                f"{graphic_state.passthrough_per_char_instruction} \n".encode(),
            )

    def render_paragraph_to_char(
        self,
        paragraph: il_version_1.PdfParagraph,
    ) -> list[il_version_1.PdfCharacter]:
        chars = []
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_character:
                chars.append(composition.pdf_character)
            elif composition.pdf_formula:
                # Flatten formula: extract all characters from the formula
                chars.extend(composition.pdf_formula.pdf_character)
            else:
                logger.error(
                    f"Unknown composition type. "
                    f"This type only appears in the IL "
                    f"after the translation is completed."
                    f"During pdf rendering, this type is not supported."
                    f"Composition: {composition}. "
                    f"Paragraph: {paragraph}. ",
                )
        if not chars and paragraph.unicode and paragraph.debug_id:
            # Log identifying fields rather than the whole PdfParagraph
            # repr: the repr includes every character/style object and ran
            # to multiple kilobytes per occurrence, which during an incident
            # (hundreds of dropped paragraphs) is itself a memory and
            # log-cost amplifier.
            unicode_text = paragraph.unicode or ""
            preview = unicode_text[:_UNICODE_LOG_PREVIEW_CHARS].replace("\n", "\\n")
            truncated = "..." if len(unicode_text) > _UNICODE_LOG_PREVIEW_CHARS else ""
            logger.error(
                "Unable to export paragraphs that have not yet been formatted: "
                f"debug_id={paragraph.debug_id} "
                f"layout_label={paragraph.layout_label} "
                f"unicode_chars={len(unicode_text)} "
                f"box={_describe_box(paragraph.box)} "
                f"scale={paragraph.scale} "
                f"unicode_preview={preview!r}{truncated}",
            )
            return chars
        return chars

    def _collect_formula_items(self, page: il_version_1.Page, attr: str) -> list:
        """Collect items from formula compositions across all page paragraphs."""
        items = []
        for paragraph in page.pdf_paragraph:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_formula:
                    items.extend(getattr(composition.pdf_formula, attr))
        return items

    def _char_render_units(self, page: il_version_1.Page) -> list[RenderUnit]:
        """Build character render units from page characters and paragraphs."""
        chars = list(page.pdf_character) if page.pdf_character else []
        for paragraph in page.pdf_paragraph:
            chars.extend(self.render_paragraph_to_char(paragraph))
        return [
            CharacterRenderUnit(
                char,
                getattr(char, "render_order", 100),
                getattr(char, "sub_render_order", i),
            )
            for i, char in enumerate(chars)
        ]

    def _form_render_units(
        self, page: il_version_1.Page, translation_config: TranslationConfig
    ) -> list[RenderUnit]:
        """Build form render units if form rendering is not skipped."""
        if translation_config.skip_form_render:
            return []
        all_forms = list(page.pdf_form) + self._collect_formula_items(page, "pdf_form")
        return [
            FormRenderUnit(
                form,
                getattr(form, "render_order", 50),
                getattr(form, "sub_render_order", i),
            )
            for i, form in enumerate(all_forms)
        ]

    def _rect_render_units(
        self, page: il_version_1.Page, translation_config: TranslationConfig
    ) -> list[RenderUnit]:
        """Build rectangle render units for OCR workaround or debug mode."""
        units = []
        line_width = 0.1 if translation_config.ocr_workaround else 0.4
        for i, rect in enumerate(page.pdf_rectangle):
            include = (
                translation_config.ocr_workaround
                and not rect.debug_info
                and rect.fill_background
            ) or (translation_config.debug and rect.debug_info)
            if include:
                units.append(
                    RectangleRenderUnit(
                        rect,
                        getattr(rect, "render_order", 10),
                        getattr(rect, "sub_render_order", i),
                        line_width,
                    )
                )
        return units

    def _curve_render_units(
        self, page: il_version_1.Page, translation_config: TranslationConfig
    ) -> list[RenderUnit]:
        """Build curve render units if curve rendering is not skipped."""
        if translation_config.skip_curve_render:
            return []
        all_curves = list(page.pdf_curve) + self._collect_formula_items(
            page, "pdf_curve"
        )
        return [
            CurveRenderUnit(
                curve,
                getattr(curve, "render_order", 20),
                getattr(curve, "sub_render_order", i),
            )
            for i, curve in enumerate(all_curves)
            if curve.debug_info or translation_config.debug
        ]

    def create_render_units_for_page(
        self,
        page: il_version_1.Page,
        translation_config: TranslationConfig,
    ) -> list[RenderUnit]:
        """Convert all renderable objects in a page to render units."""
        render_units = self._char_render_units(page)
        render_units.extend(self._form_render_units(page, translation_config))
        render_units.extend(self._rect_render_units(page, translation_config))
        render_units.extend(self._curve_render_units(page, translation_config))
        return render_units

    def render_units_to_stream(
        self,
        render_units: list[RenderUnit],
        context: RenderContext,
        page_op: BitStream,
        xobj_draw_ops: dict[str, BitStream],
    ) -> None:
        """Render sorted render units to appropriate draw streams."""
        # Sort render units by (render_order, sub_render_order)
        sorted_units = sorted(render_units, key=lambda unit: unit.get_sort_key())

        for unit in sorted_units:
            # Determine which draw_op to use based on xobj_id
            if unit.xobj_id in xobj_draw_ops:
                draw_op = xobj_draw_ops[unit.xobj_id]
            else:
                draw_op = page_op

            # Render the unit
            unit.render(draw_op, context)

    def get_available_font_list(self, pdf, page):
        page_xref_id = pdf[page.page_number].xref
        return self.get_xobj_available_fonts(page_xref_id, pdf)

    def _resolve_xref_resources(self, pdf, resources_type, r_id):
        """Resolve resource dict from an xref or inline dict, returning (type, value)."""
        if resources_type == "xref":
            resource_xref_id = re.search("(\\d+) 0 R", r_id).group(1)
            r_id = pdf.xref_object(int(resource_xref_id))
            resources_type = "dict"
        return resources_type, r_id

    def _extract_font_dict_from_resources(self, pdf, resources_type, r_id):
        """Extract raw font dictionary string from resolved resources."""
        if resources_type == "dict":
            xref_id = re.search("/Font (\\d+) 0 R", r_id)
            if xref_id is not None:
                return pdf.xref_object(int(xref_id.group(1)))
            search = re.search("/Font *<<(.+?)>>", r_id.replace("\n", " "))
            if search is None:
                return None
            return search.group(1)
        # resources_type is something else — treat r_id as direct xref
        r_id_int = int(r_id.split(" ")[0])
        _, font_dict = pdf.xref_get_key(r_id_int, "Font")
        return font_dict

    def get_xobj_available_fonts(self, page_xref_id, pdf):
        try:
            resources_type, r_id = pdf.xref_get_key(page_xref_id, "Resources")
            resources_type, r_id = self._resolve_xref_resources(
                pdf, resources_type, r_id
            )
            font_dict = self._extract_font_dict_from_resources(
                pdf, resources_type, r_id
            )
            if font_dict is None:
                return set()
            fonts = re.findall("/([^ ]+?) ", font_dict)
            return set(fonts)
        except Exception:
            return set()

    def _render_rectangle(
        self,
        draw_op: BitStream,
        rectangle: il_version_1.PdfRectangle,
        line_width: float = 0.4,
    ):
        """Draw a rectangle in PDF for visualization purposes.

        Args:
            draw_op: BitStream to append PDF drawing operations
            rectangle: Rectangle object containing position information
            line_width: Line width
        """
        x1 = rectangle.box.x
        y1 = rectangle.box.y
        x2 = rectangle.box.x2
        y2 = rectangle.box.y2
        width = x2 - x1
        height = y2 - y1
        # Save graphics state
        draw_op.append(b"q ")

        # Set green color for debug visibility
        draw_op.append(
            rectangle.graphic_state.passthrough_per_char_instruction.encode(),
        )  # Green stroke
        if rectangle.line_width is not None:
            line_width = rectangle.line_width
        if line_width > 0:
            draw_op.append(f" {line_width:.6f} w ".encode())  # Line width
        draw_op.append(f"{x1:.6f} {y1:.6f} {width:.6f} {height:.6f} re ".encode())
        if rectangle.fill_background:
            draw_op.append(b" f ")
        else:
            draw_op.append(b" S ")

        # Restore graphics state
        draw_op.append(b" n Q\n")

    def _compute_side_by_side_layout(self, orig_page, trans_page, dual_translate_first):
        """Compute dimensions and rect pair for a side-by-side dual page."""
        rotate_angle = orig_page.rotation
        total_width = orig_page.rect.width + trans_page.rect.width
        max_height = max(orig_page.rect.height, trans_page.rect.height)
        left_width = (
            trans_page.rect.width if dual_translate_first else orig_page.rect.width
        )
        rect_left = pymupdf.Rect(0, 0, left_width, max_height)
        rect_right = pymupdf.Rect(left_width, 0, total_width, max_height)
        if dual_translate_first:
            rect_left, rect_right = rect_right, rect_left
        return rotate_angle, total_width, max_height, rect_left, rect_right

    def _show_dual_page_side(
        self,
        dual_page,
        rect,
        src_pdf,
        page_id: int,
        rotate_angle: float,
        side_label: str,
    ):
        """Render one side of a dual page, logging a warning on failure."""
        try:
            dual_page.show_pdf_page(
                rect,
                src_pdf,
                page_id,
                keep_proportion=True,
                rotate=-rotate_angle,
            )
        except Exception as e:
            logger.warning(
                f"Failed to show {side_label}. "
                f"Page ID: {page_id}. "
                f"Original PDF: {self.original_pdf_path}. ",
                exc_info=e,
            )

    def create_side_by_side_dual_pdf(
        self,
        original_pdf: pymupdf.Document,
        translated_pdf: pymupdf.Document,
        translation_config: TranslationConfig,
    ) -> pymupdf.Document:
        """Create a dual PDF with side-by-side pages (original and translation).

        Args:
            original_pdf: Original PDF document
            translated_pdf: Translated PDF document
            translation_config: Translation configuration

        Returns:
            The created dual PDF document
        """
        dual = pymupdf.open()
        page_count = min(original_pdf.page_count, translated_pdf.page_count)

        for page_id in range(page_count):
            orig_page = original_pdf[page_id]
            trans_page = translated_pdf[page_id]
            rotate_angle, total_width, max_height, rect_left, rect_right = (
                self._compute_side_by_side_layout(
                    orig_page, trans_page, translation_config.dual_translate_first
                )
            )
            orig_page.set_rotation(0)
            trans_page.set_rotation(0)
            dual_page = dual.new_page(width=total_width, height=max_height)
            self._show_dual_page_side(
                dual_page,
                rect_left,
                original_pdf,
                page_id,
                rotate_angle,
                "original page on left",
            )
            self._show_dual_page_side(
                dual_page,
                rect_right,
                translated_pdf,
                page_id,
                rotate_angle,
                "translated page on right",
            )
        return dual

    def create_alternating_pages_dual_pdf(
        self,
        original_pdf: pymupdf.Document,
        translated_pdf: pymupdf.Document,
        translation_config: TranslationConfig,
    ) -> pymupdf.Document:
        """Create a dual PDF with alternating pages (original and translation).

        Args:
            original_pdf_path: Path to the original PDF
            translated_pdf: Translated PDF document
            translation_config: Translation configuration

        Returns:
            The created dual PDF document
        """
        # Open the original PDF and insert translated PDF
        dual = original_pdf
        dual.insert_file(translated_pdf)

        # Rearrange pages to alternate between original and translated
        page_count = translated_pdf.page_count
        for page_id in range(page_count):
            if translation_config.dual_translate_first:
                dual.move_page(page_count + page_id, page_id * 2)
            else:
                dual.move_page(page_count + page_id, page_id * 2 + 1)

        return dual

    def _build_debug_page_op(self, pdf, page, base_op) -> BitStream:
        """Build the full debug page BitStream for one page."""
        page_op = BitStream()
        page_op.append(b"q ")
        if base_op is not None:
            page_op.append(base_op)
        page_op.append(b" Q ")
        page_op.append(
            f"q Q 1 0 0 1 {page.cropbox.box.x:.6f} {page.cropbox.box.y:.6f} cm \n".encode()
        )
        available_font_list = self.get_available_font_list(pdf, page)
        page_encoding_length_map = {f.font_id: f.encoding_length for f in page.pdf_font}
        chars = list(page.pdf_character) if page.pdf_character else []
        for paragraph in page.pdf_paragraph:
            chars.extend(self.render_paragraph_to_char(paragraph))
        self._render_debug_chars(
            chars, page_op, available_font_list, page_encoding_length_map
        )
        for rect in page.pdf_rectangle:
            if rect.debug_info:
                self._render_rectangle(page_op, rect)
        return page_op

    def write_debug_info(
        self,
        pdf: pymupdf.Document,
        translation_config: TranslationConfig,
    ):
        self.font_mapper.add_font(pdf, self.docs)

        for page in self.docs.page:
            _, r_id = pdf.xref_get_key(pdf[page.page_number].xref, "Contents")
            resource_xref_id = re.search("(\\d+) 0 R", r_id).group(1)
            base_op = pdf.xref_stream(int(resource_xref_id))
            translation_config.raise_if_cancelled()
            page_op = self._build_debug_page_op(pdf, page, base_op)
            pdf.update_stream(int(resource_xref_id), page_op.tobytes())
        translation_config.raise_if_cancelled()

        if not translation_config.skip_clean:
            pdf = self.subset_fonts_in_subprocess(pdf, translation_config, tag="debug")
        return pdf

    def _render_debug_chars(
        self, chars, page_op, available_font_list, encoding_length_map
    ):
        """Render characters marked for debug output into page_op."""
        for char in chars:
            if not getattr(char, "debug_info", False):
                continue
            if char.char_unicode == "\n" or char.pdf_character_id is None:
                continue
            font_id = char.pdf_style.font_id
            if font_id not in available_font_list:
                continue
            char_size = char.pdf_style.font_size
            page_op.append(b"q ")
            self.render_graphic_state(page_op, char.pdf_style.graphic_state)
            if char.vertical:
                page_op.append(
                    f"BT /{font_id} {char_size:f} Tf 0 1 -1 0 {char.box.x2:f} {char.box.y:f} Tm ".encode(),
                )
            else:
                page_op.append(
                    f"BT /{font_id} {char_size:f} Tf 1 0 0 1 {char.box.x:f} {char.box.y:f} Tm ".encode(),
                )
            encoding_length = encoding_length_map[font_id]
            page_op.append(
                f"<{char.pdf_character_id:0{encoding_length * 2}x}>".upper().encode(),
            )
            page_op.append(b" Tj ET Q \n")

    @staticmethod
    def _wait_for_subprocess_with_timeout(process, timeout: int, label: str) -> bool:
        """Poll *process* until done or *timeout* seconds elapsed.

        Returns True if the process finished on its own, False if it timed out.
        """
        start_time = time.time()
        while process.is_alive():
            if time.time() - start_time > timeout:
                logger.warning(
                    f"{label} timeout after {timeout} seconds, terminating subprocess"
                )
                process.terminate()
                try:
                    process.join(5)
                    if process.is_alive():
                        logger.warning("Subprocess did not terminate, killing it")
                        process.kill()
                        process.terminate()
                        process.kill()
                        process.terminate()
                        process.kill()
                        process.terminate()
                except Exception as e:
                    logger.error(f"Error terminating subprocess: {e}")
                return False
            time.sleep(0.5)
        return True

    @staticmethod
    def _is_output_valid(temp_output: str) -> bool:
        """Return True if the subprocess output file exists and is non-empty."""
        return Path(temp_output).exists() and Path(temp_output).stat().st_size > 0

    @staticmethod
    def subset_fonts_in_subprocess(
        pdf: pymupdf.Document, translation_config: TranslationConfig, tag: str
    ) -> pymupdf.Document:
        """Run font subsetting in a subprocess with timeout.

        Args:
            pdf: The PDF document object
            translation_config: Translation configuration

        Returns:
            Path to the PDF with subsetted fonts, or original path if subsetting failed or timed out
        """
        original_pdf = pdf
        temp_input = str(
            translation_config.get_working_file_path(f"temp_subset_input_{tag}.pdf")
        )
        temp_output = str(
            translation_config.get_working_file_path(f"temp_subset_output_{tag}.pdf")
        )

        pdf.save(temp_input)

        process = Process(target=_subset_fonts_process, args=(temp_input, temp_output))
        process.start()

        timeout = 60
        finished = PDFCreater._wait_for_subprocess_with_timeout(
            process, timeout, "Font subsetting"
        )
        if not finished:
            return original_pdf

        exit_code = process.exitcode
        if exit_code == 0 and PDFCreater._is_output_valid(temp_output):
            logger.info("Font subsetting completed successfully")
            return pymupdf.open(temp_output)

        logger.warning(
            f"Font subsetting failed with exit code {exit_code} or produced empty file"
        )
        return original_pdf

    @staticmethod
    def _save_pdf_fallback(pdf, output_path, garbage, deflate, deflate_fonts, linear):
        """Save PDF without clean=True as fallback."""
        try:
            pdf.save(
                output_path,
                garbage=garbage,
                deflate=deflate,
                clean=False,
                deflate_fonts=deflate_fonts,
                linear=linear,
            )
        except Exception as e:
            logger.error(f"Error in fallback save: {e}")
            pdf.save(output_path)

    @staticmethod
    def _terminate_subprocess(process):
        """Attempt to terminate and then kill a subprocess."""
        try:
            process.join(5)
            if process.is_alive():
                logger.warning("Subprocess did not terminate, killing it")
                process.kill()
                process.terminate()
                process.kill()
                process.terminate()
                process.kill()
                process.terminate()
        except Exception as e:
            logger.error(f"Error terminating PDF save process: {e}")

    @staticmethod
    def _copy_saved_pdf(
        temp_input: str, temp_output: str, output_path: str, pdf
    ) -> bool:
        """Copy temp_output to output_path, falling back to pdf.save on error."""
        try:
            import shutil

            shutil.copy2(temp_output, output_path)
            return True
        except Exception as e:
            logger.error(f"Error copying saved PDF: {e}")
            pdf.save(output_path)
            return False
        finally:
            Path(temp_input).unlink(missing_ok=True)
            Path(temp_output).unlink(missing_ok=True)

    @staticmethod
    def save_pdf_with_timeout(
        pdf: pymupdf.Document,
        output_path: str,
        translation_config: TranslationConfig,
        garbage: int = 1,
        deflate: bool = True,
        clean: bool = True,
        deflate_fonts: bool = True,
        linear: bool = False,
        timeout: int = 120,
        tag: str = "",
    ) -> bool:
        """Save a PDF document with a timeout for the clean=True operation.

        Args:
            pdf: The PDF document object
            output_path: Path where to save the PDF
            translation_config: Translation configuration
            garbage: Garbage collection level (0, 1, 2, 3, 4)
            deflate: Whether to deflate the PDF
            clean: Whether to clean the PDF
            deflate_fonts: Whether to deflate fonts
            linear: Whether to linearize the PDF
            timeout: Timeout in seconds (default: 2 minutes)

        Returns:
            True if saved with clean=True successfully, False if fallback to clean=False was used
        """
        temp_input = str(
            translation_config.get_working_file_path(f"temp_save_input_{tag}.pdf")
        )
        temp_output = str(
            translation_config.get_working_file_path(f"temp_save_output_{tag}.pdf")
        )
        pdf.save(temp_input)
        process = Process(
            target=_save_pdf_clean_process,
            args=(
                temp_input,
                temp_output,
                garbage,
                deflate,
                clean,
                deflate_fonts,
                linear,
            ),
        )
        process.start()

        finished = PDFCreater._wait_for_subprocess_with_timeout(
            process, timeout, f"PDF save with clean={clean}"
        )
        if not finished:
            logger.info("Falling back to save with clean=False")
            PDFCreater._save_pdf_fallback(
                pdf, output_path, garbage, deflate, deflate_fonts, linear
            )
            return False

        exit_code = process.exitcode
        if exit_code == 0 and PDFCreater._is_output_valid(temp_output):
            logger.info(f"PDF save with clean={clean} completed successfully")
            return PDFCreater._copy_saved_pdf(temp_input, temp_output, output_path, pdf)

        logger.warning(
            f"PDF save with clean={clean} failed with exit code {exit_code} or produced empty file"
        )
        PDFCreater._save_pdf_fallback(
            pdf, output_path, garbage, deflate, deflate_fonts, linear
        )
        return False

    def restore_media_box(self, doc: pymupdf.Document, mediabox_data: dict) -> None:
        for xref, page_box_data in mediabox_data.items():
            for name, box in page_box_data.items():
                try:
                    doc.xref_set_key(xref, name, box)
                except Exception:
                    logger.debug(f"Error restoring media box {name} from PDF")

    def _save_mono_pdf(self, pdf, mono_out_path, gc_level, translation_config):
        """Save the mono (translated) PDF with optional debug decompressed copy."""
        if translation_config.debug:
            translation_config.raise_if_cancelled()
            pdf.save(f"{mono_out_path}.decompressed.pdf", expand=True, pretty=True)
        translation_config.raise_if_cancelled()
        self.save_pdf_with_timeout(
            pdf,
            mono_out_path,
            translation_config,
            garbage=gc_level,
            deflate=True,
            clean=not translation_config.skip_clean,
            deflate_fonts=True,
            linear=False,
            tag="mono",
        )

    def _build_dual_pdf(
        self,
        pdf,
        translation_config,
        basename,
        debug_suffix,
        gc_level,
        should_removed_page,
    ):
        """Create and save the dual (side-by-side or alternating) PDF."""
        dual_out_path = translation_config.get_output_file_path(
            f"{basename}{debug_suffix}.{translation_config.lang_out}.dual.pdf",
        )
        translation_config.raise_if_cancelled()
        original_pdf = pymupdf.open(self.original_pdf_path)
        if translation_config.debug:
            translation_config.raise_if_cancelled()
            try:
                original_pdf = self.write_debug_info(original_pdf, translation_config)
            except Exception:
                logger.warning("Failed to write debug info to dual PDF", exc_info=True)
        if self.translation_config.only_include_translated_page and should_removed_page:
            original_pdf.delete_pages(should_removed_page)
        if translation_config.use_alternating_pages_dual:
            dual = self.create_alternating_pages_dual_pdf(
                original_pdf, pdf, translation_config
            )
        else:
            dual = self.create_side_by_side_dual_pdf(
                original_pdf, pdf, translation_config
            )
        self.save_pdf_with_timeout(
            dual,
            dual_out_path,
            translation_config,
            garbage=gc_level,
            deflate=True,
            clean=not translation_config.skip_clean,
            deflate_fonts=True,
            linear=False,
            tag="dual",
        )
        if translation_config.debug:
            translation_config.raise_if_cancelled()
            dual.save(f"{dual_out_path}.decompressed.pdf", expand=True, pretty=True)
        return dual_out_path

    def _save_glossary(self, basename, debug_suffix, translation_config):
        """Save auto-extracted glossary if available and configured."""
        if not (
            self.translation_config.save_auto_extracted_glossary
            and self.translation_config.shared_context_cross_split_part.auto_extracted_glossary
        ):
            return None
        path = self.translation_config.get_output_file_path(
            f"{basename}{debug_suffix}.{translation_config.lang_out}.glossary.csv"
        )
        with path.open("w", encoding="utf-8") as f:
            logger.info(f"save auto extracted glossary to {path}")
            f.write(
                self.translation_config.shared_context_cross_split_part.auto_extracted_glossary.to_csv()
            )
        return path

    def _build_debug_suffix(self, translation_config: TranslationConfig) -> str:
        """Return the debug/watermark filename suffix."""
        suffix = ".debug" if translation_config.debug else ""
        if translation_config.watermark_output_mode != WatermarkOutputMode.Watermarked:
            suffix += ".no_watermark"
        return suffix

    def _render_all_pages(self, pdf, translation_config, check_font_exists):
        """Render every page's content stream into *pdf*."""
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name, len(self.docs.page)
        ) as pbar:
            for page in self.docs.page:
                self.update_page_content_stream(
                    check_font_exists, page, pdf, translation_config
                )
                pbar.advance()

    def _subset_and_restore(self, pdf, translation_config):
        """Optionally subset fonts and restore the media box."""
        with self.translation_config.progress_monitor.stage_start(
            SUBSET_FONT_STAGE_NAME, 1
        ) as pbar:
            if not translation_config.skip_clean:
                pdf = self.subset_fonts_in_subprocess(
                    pdf, translation_config, tag="mono"
                )
            pbar.advance()
        try:
            self.restore_media_box(pdf, self.mediabox_data)
        except Exception:
            logger.exception("restore media box failed")
        return pdf

    def _remove_untranslated_pages(self, pdf, translation_config) -> list[int]:
        """Delete non-translated pages from *pdf* when only_include_translated_page is set."""
        if not translation_config.only_include_translated_page:
            return []
        pages_to_translate = {
            page.page_number
            for page in self.docs.page
            if self.translation_config.should_translate_page(page.page_number + 1)
        }
        should_removed_page = list(set(range(len(pdf))) - pages_to_translate)
        pdf.delete_pages(should_removed_page)
        return should_removed_page

    def _save_output_pdfs(
        self,
        pdf,
        mono_out_path,
        translation_config,
        basename,
        debug_suffix,
        gc_level,
        should_removed_page,
    ):
        """Save mono and dual output PDFs under progress monitor tracking."""
        dual_out_path = None
        with self.translation_config.progress_monitor.stage_start(
            SAVE_PDF_STAGE_NAME, 2
        ) as pbar:
            if not translation_config.no_mono:
                self._save_mono_pdf(pdf, mono_out_path, gc_level, translation_config)
            pbar.advance()
            if not translation_config.no_dual:
                dual_out_path = self._build_dual_pdf(
                    pdf,
                    translation_config,
                    basename,
                    debug_suffix,
                    gc_level,
                    should_removed_page,
                )
            pbar.advance()
        return dual_out_path

    def write(
        self,
        translation_config: TranslationConfig,
        check_font_exists: bool = False,
    ) -> TranslateResult:
        try:
            basename = Path(translation_config.input_file).stem
            debug_suffix = self._build_debug_suffix(translation_config)
            mono_out_path = translation_config.get_output_file_path(
                f"{basename}{debug_suffix}.{translation_config.lang_out}.mono.pdf",
            )
            pdf = pymupdf.open(self.original_pdf_path)
            self.font_mapper.add_font(pdf, self.docs)
            self._render_all_pages(pdf, translation_config, check_font_exists)
            translation_config.raise_if_cancelled()
            gc_level = 4 if self.translation_config.ocr_workaround else 1
            pdf = self._subset_and_restore(pdf, translation_config)
            should_removed_page = self._remove_untranslated_pages(
                pdf, translation_config
            )
            dual_out_path = self._save_output_pdfs(
                pdf,
                mono_out_path,
                translation_config,
                basename,
                debug_suffix,
                gc_level,
                should_removed_page,
            )
            if self.translation_config.no_mono:
                mono_out_path = None
            if self.translation_config.no_dual:
                dual_out_path = None
            auto_extracted_glossary_path = self._save_glossary(
                basename, debug_suffix, translation_config
            )
            return TranslateResult(
                mono_out_path, dual_out_path, auto_extracted_glossary_path
            )
        except Exception:
            logger.exception("Failed to create PDF: %s", translation_config.input_file)
            if not check_font_exists:
                return self.write(translation_config, True)
            raise

    def _build_xobj_maps(
        self, page, pdf, available_font_list, page_encoding_length_map
    ):
        """Build xobj font/encoding/draw-op maps for all xobjects on the page."""
        xobj_available_fonts = {}
        xobj_draw_ops = {}
        xobj_encoding_length_map = {}
        all_encoding_length_map = page_encoding_length_map.copy()
        for xobj in page.pdf_xobject:
            xobj_available_fonts[xobj.xobj_id] = available_font_list.copy()
            try:
                xobj_available_fonts[xobj.xobj_id].update(
                    self.get_xobj_available_fonts(xobj.xref_id, pdf)
                )
            except Exception:
                pass
            xobj_encoding_length_map[xobj.xobj_id] = {
                f.font_id: f.encoding_length for f in xobj.pdf_font
            }
            all_encoding_length_map.update(xobj_encoding_length_map[xobj.xobj_id])
            xobj_encoding_length_map[xobj.xobj_id].update(page_encoding_length_map)
            xobj_op = BitStream()
            base_op = zstd_decompress(xobj.base_operations.value)
            xobj_op.append(base_op.encode())
            xobj_draw_ops[xobj.xobj_id] = xobj_op
        return (
            xobj_available_fonts,
            xobj_encoding_length_map,
            all_encoding_length_map,
            xobj_draw_ops,
        )

    def _flush_xobj_streams(self, page, pdf, xobj_draw_ops):
        """Write each xobject's draw BitStream back into the PDF."""
        for xobj in page.pdf_xobject:
            draw_op = xobj_draw_ops[xobj.xobj_id]
            try:
                pdf.update_stream(xobj.xref_id, draw_op.tobytes())
            except Exception:
                logger.warning(f"update xref {xobj.xref_id} stream fail, continue")

    def update_page_content_stream(
        self, check_font_exists, page, pdf, translation_config, skip_char: bool = False
    ):
        if not (page.cropbox is not None and page.cropbox.box is not None):
            raise RuntimeError(
                "page.cropbox and page.cropbox.box must be set before updating content stream"
            )
        page_crop_box = page.cropbox.box
        ctm_for_ops = (1, 0, 0, 1, -page_crop_box.x, -page_crop_box.y)
        ctm_for_ops = f" {' '.join(f'{x:f}' for x in ctm_for_ops)} cm ".encode()
        translation_config.raise_if_cancelled()
        available_font_list = self.get_available_font_list(pdf, page)
        page_encoding_length_map: dict[str | None, int | None] = {
            f.font_id: f.encoding_length for f in page.pdf_font
        }
        (
            xobj_available_fonts,
            xobj_encoding_length_map,
            all_encoding_length_map,
            xobj_draw_ops,
        ) = self._build_xobj_maps(
            page, pdf, available_font_list, page_encoding_length_map
        )
        page_op = BitStream()
        page_op.append(ctm_for_ops)
        page_op.append(b" \n")
        context = RenderContext(
            pdf_creator=self,
            page=page,
            available_font_list=available_font_list,
            page_encoding_length_map=page_encoding_length_map,
            all_encoding_length_map=all_encoding_length_map,
            xobj_available_fonts=xobj_available_fonts,
            xobj_encoding_length_map=xobj_encoding_length_map,
            ctm_for_ops=ctm_for_ops,
            check_font_exists=check_font_exists,
        )
        render_units = self.create_render_units_for_page(page, translation_config)
        if skip_char:
            render_units = [
                unit
                for unit in render_units
                if not isinstance(unit, CharacterRenderUnit)
            ]
        self.render_units_to_stream(render_units, context, page_op, xobj_draw_ops)
        self._flush_xobj_streams(page, pdf, xobj_draw_ops)
        op_container = pdf.get_new_xref()
        pdf.update_object(op_container, "<<>>")
        pdf.update_stream(op_container, page_op.tobytes())
        pdf[page.page_number].set_contents(op_container)
