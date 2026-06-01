import logging
import re
import secrets

import numpy as np

from src.doctranslator.doctranslator_exception.DocTranslatorException import (
    ExtractTextError,
)
from src.doctranslator.format.pdf.document_il import Box
from src.doctranslator.format.pdf.document_il import Document
from src.doctranslator.format.pdf.document_il import Page
from src.doctranslator.format.pdf.document_il import PdfCharacter
from src.doctranslator.format.pdf.document_il import PdfLine
from src.doctranslator.format.pdf.document_il import PdfParagraph
from src.doctranslator.format.pdf.document_il import PdfParagraphComposition
from src.doctranslator.format.pdf.document_il import PdfRectangle
from src.doctranslator.format.pdf.document_il.utils.fontmap import FontMapper
from src.doctranslator.format.pdf.document_il.utils.formular_helper import (
    collect_page_formula_font_ids,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    HEIGHT_NOT_USFUL_CHAR_IN_CHAR,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import SPACE_REGEX
from src.doctranslator.format.pdf.document_il.utils.layout_helper import Layout
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    add_space_dummy_chars,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    build_layout_index,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    calculate_iou_for_boxes,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    get_char_unicode_string,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    get_character_layout,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import is_bullet_point
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    is_character_in_formula_layout,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import is_text_layout
from src.doctranslator.format.pdf.document_il.utils.paragraph_helper import (
    is_cid_paragraph,
)
from src.doctranslator.format.pdf.document_il.utils.style_helper import INDIGO
from src.doctranslator.format.pdf.document_il.utils.style_helper import WHITE
from src.doctranslator.format.pdf.translation_config import TranslationConfig

logger = logging.getLogger(__name__)

# Base58 alphabet (Bitcoin style, without numbers 0, O, I, l)
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def generate_base58_id(length: int = 5) -> str:
    """Generate a random base58 ID of specified length."""
    return "".join(secrets.choice(BASE58_ALPHABET) for _ in range(length))


class ParagraphFinder:
    stage_name = "Parse Paragraphs"

    # å®šä¹‰é¡¹ç›®ç¬¦å·çš„æ­£åˆ™è¡¨è¾¾å¼æ¨¡å¼

    def __init__(self, translation_config: TranslationConfig):
        self.translation_config = translation_config
        self.font_mapper = FontMapper(translation_config)

    def _preprocess_formula_layouts(self, page: Page):
        """
        Identifies 'formula' layouts that do not significantly overlap with any text layouts
        and re-labels them as 'isolate_formula'.
        """
        # Use a simplified Layout object for is_text_layout check
        text_layouts = [
            layout
            for layout in page.page_layout
            if is_text_layout(Layout(layout.id, layout.class_name))
        ]
        formula_layouts = [
            layout for layout in page.page_layout if layout.class_name == "formula"
        ]

        if not text_layouts or not formula_layouts:
            return

        for formula_layout in formula_layouts:
            is_isolated = True
            for text_layout in text_layouts:
                iou = calculate_iou_for_boxes(formula_layout.box, text_layout.box)
                if iou >= 0.5:
                    is_isolated = False
                    break

            if is_isolated:
                formula_layout.class_name = "isolate_formula"

    def _expand_box_to_layout(self, para_box, layout_box) -> tuple:
        """Return (x1, y1, x2, y2) expanded to include both boxes."""
        x1 = min(para_box.x, layout_box.x)
        y1 = min(para_box.y, layout_box.y)
        x2 = max(para_box.x2, layout_box.x2)
        y2 = max(para_box.y2, layout_box.y2)
        return x1, y1, x2, y2

    def _make_fill_rectangle(self, x1, y1, x2, y2, xobj_id) -> PdfRectangle:
        """Create a fill-background rectangle."""
        return PdfRectangle(
            box=Box(x1, y1, x2, y2),
            fill_background=True,
            graphic_state=WHITE,
            debug_info=False,
            xobj_id=xobj_id,
        )

    def add_text_fill_background(self, page: Page):
        layout_map = {layout.id: layout for layout in page.page_layout}
        for paragraph in page.pdf_paragraph:
            layout_id = paragraph.layout_id
            if layout_id is None:
                continue
            layout = layout_map[layout_id]
            if paragraph.box is None:
                continue
            x1, y1, x2, y2 = self._expand_box_to_layout(paragraph.box, layout.box)
            if not (x2 > x1 and y2 > y1):
                raise AssertionError
            page.pdf_rectangle.append(
                self._make_fill_rectangle(x1, y1, x2, y2, paragraph.xobj_id)
            )

    def _collect_chars_from_compositions(self, paragraph: PdfParagraph) -> list:
        """Collect all characters from a paragraph's compositions."""
        chars = []
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_line:
                chars.extend(composition.pdf_line.pdf_character)
            elif composition.pdf_formula:
                chars.extend(composition.pdf_formula.pdf_character)
            elif composition.pdf_character:
                chars.append(composition.pdf_character)
            elif composition.pdf_same_style_unicode_characters:
                # pdf_same_style_unicode_characters holds pre-translated text, not raw chars; skip
                pass
            else:
                logger.error(
                    "Unexpected composition type"
                    " in PdfParagraphComposition. "
                    "This type only appears in the IL "
                    "after the translation is completed.",
                )
                # no action needed; continue iterating over remaining compositions
        return chars

    def _update_paragraph_bbox(self, paragraph: PdfParagraph, chars: list) -> None:
        """Update bounding box, vertical and xobj_id from character list."""
        min_x = min(char.visual_bbox.box.x for char in chars)
        min_y = min(char.visual_bbox.box.y for char in chars)
        max_x = max(char.visual_bbox.box.x2 for char in chars)
        max_y = max(char.visual_bbox.box.y2 for char in chars)
        paragraph.box = Box(min_x, min_y, max_x, max_y)
        paragraph.vertical = chars[0].vertical
        paragraph.xobj_id = chars[0].xobj_id

    def _detect_first_line_indent(self, paragraph: PdfParagraph) -> bool:
        """Return True if the first line of the paragraph is indented."""
        if not paragraph.pdf_paragraph_composition:
            return False
        first_comp = paragraph.pdf_paragraph_composition[0]
        if not first_comp.pdf_line:
            return False
        first_char_x = first_comp.pdf_line.pdf_character[0].visual_bbox.box.x
        return first_char_x - paragraph.box.x > 1

    def update_paragraph_data(self, paragraph: PdfParagraph, update_unicode=False):
        if not paragraph.pdf_paragraph_composition:
            return

        chars = self._collect_chars_from_compositions(paragraph)

        if update_unicode and chars:
            paragraph.unicode = get_char_unicode_string(chars)
        if not chars:
            return

        # æ›´æ–°è¾¹ç•Œæ¡†
        self._update_paragraph_bbox(paragraph, chars)

        paragraph.first_line_indent = self._detect_first_line_indent(paragraph)

    def update_line_data(self, line: PdfLine):
        min_x = min(char.visual_bbox.box.x for char in line.pdf_character)
        min_y = min(char.visual_bbox.box.y for char in line.pdf_character)
        max_x = max(char.visual_bbox.box.x2 for char in line.pdf_character)
        max_y = max(char.visual_bbox.box.y2 for char in line.pdf_character)
        line.box = Box(min_x, min_y, max_x, max_y)

    def add_debug_info(self, page: Page):
        if not self.translation_config.debug:
            return
        for paragraph in page.pdf_paragraph:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    page.pdf_rectangle.append(
                        PdfRectangle(
                            box=line.box,
                            fill_background=False,
                            graphic_state=INDIGO,
                            debug_info=True,
                            line_width=0.2,
                        )
                    )

    def process(self, document):
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            len(document.page),
        ) as pbar:
            if not document.page:
                return
            for page in document.page:
                self.translation_config.raise_if_cancelled()
                self.process_page(page)
                pbar.advance()

            total_paragraph_count = 0
            for page in document.page:
                total_paragraph_count += len(page.pdf_paragraph)
            if total_paragraph_count == 0:
                raise ExtractTextError("The document contains no paragraphs.")

            if self.check_cid_paragraph(document):
                raise ExtractTextError("The document contains too many CID paragraphs.")

    def check_cid_paragraph(self, doc: Document):
        cid_para_count = 0
        para_total = 0
        for page in doc.page:
            para_total += len(page.pdf_paragraph)
            for para in page.pdf_paragraph:
                if is_cid_paragraph(para):
                    cid_para_count += 1
        return cid_para_count / para_total > 0.8

    def bbox_overlap(self, bbox1: Box, bbox2: Box) -> bool:
        return (
            bbox1.x < bbox2.x2
            and bbox1.x2 > bbox2.x
            and bbox1.y < bbox2.y2
            and bbox1.y2 > bbox2.y
        )

    def process_page(self, page: Page):
        layout_index, layout_map = build_layout_index(page)
        # é¢„å¤„ç†å…¬å¼å¸ƒå±€çš„æ ‡ç­¾
        self._preprocess_formula_layouts(page)

        # ç¬¬ä¸€æ­¥ï¼šæ ¹æ® layout åˆ›å»º paragraphs
        # åœ¨è¿™ä¸€æ­¥ä¸­ï¼Œpage.pdf_character ä¸­çš„å­—ç¬¦ä¼šè¢«ç§»é™¤
        paragraphs = self._group_characters_into_paragraphs(
            page, layout_index, layout_map
        )
        page.pdf_paragraph = paragraphs

        page_level_formula_font_ids, xobj_specific_formula_font_ids = (
            collect_page_formula_font_ids(
                page, self.translation_config.formular_font_pattern
            )
        )

        # ç¬¬äºŒæ­¥ï¼šå°†æ®µè½å†…çš„å­—ç¬¦æ‹†åˆ†ä¸ºè¡Œ
        for paragraph in paragraphs:
            if (
                paragraph.xobj_id
                and paragraph.xobj_id in xobj_specific_formula_font_ids
            ):
                current_formula_font_ids = xobj_specific_formula_font_ids[
                    paragraph.xobj_id
                ]
            else:
                current_formula_font_ids = page_level_formula_font_ids
            self._split_paragraph_into_lines(paragraph, current_formula_font_ids)

        # ç¬¬ä¸‰æ­¥ï¼šå¤„ç†æ®µè½ä¸­çš„ç©ºæ ¼
        for paragraph in paragraphs:
            add_space_dummy_chars(paragraph)
            self.process_paragraph_spacing(paragraph)
            self.update_paragraph_data(paragraph)

        # ç¬¬å››æ­¥ï¼šè®¡ç®—æ‰€æœ‰è¡Œå®½åº¦çš„ä¸­ä½æ•°
        median_width = self.calculate_median_line_width(paragraphs)

        # ç¬¬äº”æ­¥ï¼šå¤„ç†ç‹¬ç«‹æ®µè½
        self.process_independent_paragraphs(paragraphs, median_width)

        # æ–°å¢žåŽå¤„ç†ï¼šåˆå¹¶å¸¦è¡Œå·äº¤æ›¿çš„æ­£æ–‡æ®µè½ï¼ˆa æ­£æ–‡ã€b è¡Œå·ã€c æ­£æ–‡ -> åˆå¹¶ a ä¸Ž cï¼Œä¿ç•™ bï¼‰
        if getattr(self.translation_config, "merge_alternating_line_numbers", True):
            self.merge_alternating_line_number_paragraphs(paragraphs)

        for paragraph in paragraphs:
            self.update_paragraph_data(paragraph, update_unicode=True)

        if self.translation_config.ocr_workaround:
            self.add_text_fill_background(page)
            # since this is ocr file,
            # image characters are not needed
            page.pdf_character = []

        self.fix_overlapping_paragraphs(page)

        # ç¬¬å…­æ­¥ï¼šå¯¹æ¯ä¸€è¡Œçš„å­—ç¬¦è¿›è¡ŒæŽ’åº
        # self._sort_characters_in_lines(page)

        self.add_debug_info(page)

        # æ–°é˜¶æ®µï¼šè®¾ç½®æ®µè½çš„ renderorder ä¸ºæ‰€æœ‰ç»„æˆéƒ¨åˆ†ä¸­ renderorder æœ€å°çš„
        self._set_paragraph_render_order(page)

    @staticmethod
    def _min_render_order_for_chars(chars) -> int:
        """Return the minimum render_order among a list of characters."""
        result = 9999999999999999
        for char in chars:
            if hasattr(char, "render_order") and char.render_order is not None:
                result = min(result, char.render_order)
        return result

    def _min_render_order_for_composition(self, composition) -> int:
        """Return the minimum render_order for a single composition."""
        if composition.pdf_line:
            return self._min_render_order_for_chars(composition.pdf_line.pdf_character)
        if composition.pdf_character:
            char = composition.pdf_character
            if hasattr(char, "render_order") and char.render_order is not None:
                return char.render_order
        if composition.pdf_formula:
            return self._min_render_order_for_chars(
                composition.pdf_formula.pdf_character
            )
        return 9999999999999999

    def _set_paragraph_render_order(self, page: Page):
        """
        è®¾ç½®æ®µè½çš„ renderorder ä¸ºæ®µè½æ‰€æœ‰ç»„æˆéƒ¨åˆ†ä¸­ renderorder æœ€å°çš„å€¼
        """
        for paragraph in page.pdf_paragraph:
            min_render_order = 9999999999999999

            for composition in paragraph.pdf_paragraph_composition:
                comp_min = self._min_render_order_for_composition(composition)
                min_render_order = min(min_render_order, comp_min)

            # å¦‚æžœæ‰¾åˆ°äº†æœ‰æ•ˆçš„ renderorderï¼Œè®¾ç½®æ®µè½çš„ renderorder
            if min_render_order != 9999999999999999:
                paragraph.render_order = min_render_order

    def is_isolated_formula(self, char: PdfCharacter):
        return char.char_unicode in (
            "(cid:122)",
            "(cid:123)",
            "(cid:124)",
            "(cid:125)",
        )

    def _paragraph_text_ascii(self, p: PdfParagraph) -> str:
        parts: list[str] = []
        for comp in p.pdf_paragraph_composition or []:
            if comp.pdf_line:
                for ch in comp.pdf_line.pdf_character or []:
                    if ch.char_unicode is not None:
                        parts.append(ch.char_unicode)
            elif comp.pdf_character and comp.pdf_character.char_unicode is not None:
                parts.append(comp.pdf_character.char_unicode)
        return "".join(parts)

    def _is_ascii_digit_or_space_paragraph(self, p: PdfParagraph) -> bool:
        text = self._paragraph_text_ascii(p)
        if not text:
            return True
        has_digit = False
        for c in text:
            if c.isdigit() and ord(c) < 128:
                has_digit = True
                continue
            if c.isspace():
                continue
            return False
        return True if has_digit or text.strip() == "" else False

    @staticmethod
    def _same_layout_and_xobj(a: PdfParagraph, c: PdfParagraph) -> bool:
        return (
            a.layout_id is not None
            and c.layout_id is not None
            and a.layout_id == c.layout_id
            and a.xobj_id is not None
            and c.xobj_id is not None
            and a.xobj_id == c.xobj_id
        )

    def merge_alternating_line_number_paragraphs(self, paragraphs: list[PdfParagraph]):
        # a ä»£è¡¨æ­£æ–‡
        # l ä»£è¡¨è¡Œå·
        if not paragraphs or len(paragraphs) < 3:
            return
        i = 0
        while i < len(paragraphs) - 2:
            a = paragraphs[i]
            # åžæŽ‰ä¸€ä¸ªæˆ–å¤šä¸ªè¿žç»­çš„è¡Œå·æ®µ l
            j = i + 1
            saw_l = False
            while j < len(paragraphs) and self._is_ascii_digit_or_space_paragraph(
                paragraphs[j]
            ):
                saw_l = True
                j += 1
            # çŽ°åœ¨ j æŒ‡å‘å€™é€‰çš„ c
            if saw_l and j < len(paragraphs):
                c = paragraphs[j]
                if self._same_layout_and_xobj(a, c):
                    a.pdf_paragraph_composition.extend(c.pdf_paragraph_composition)
                    self.update_paragraph_data(a)
                    del paragraphs[j]
                    # ä¸ç§»åŠ¨ iï¼Œç»§ç»­å°è¯•æŠŠæ›´å¤šæ­£æ–‡æŽ¥åˆ° aï¼Œå®žçŽ° a l+ a l+ a ... é“¾å¼åˆå¹¶
                    continue
            i += 1

    @staticmethod
    def _compute_median_char_area(chars) -> float:
        """Compute the median area of visual bounding boxes for a list of characters."""
        char_areas = [
            (char.visual_bbox.box.x2 - char.visual_bbox.box.x)
            * (char.visual_bbox.box.y2 - char.visual_bbox.box.y)
            for char in chars
        ]
        if not char_areas:
            return 0.0
        char_areas.sort()
        mid = len(char_areas) // 2
        return (
            char_areas[mid]
            if len(char_areas) % 2 == 1
            else (char_areas[mid - 1] + char_areas[mid]) / 2
        )

    def _is_small_char_same_layout(
        self,
        is_small_char: bool,
        current_paragraph: "PdfParagraph | None",
        char_layout: Layout,
        current_layout: "Layout | None",
    ) -> bool:
        """Return True if char is a small char staying in the current paragraph."""
        return (
            is_small_char
            and current_paragraph is not None
            and current_paragraph.pdf_paragraph_composition
            and char_layout.id == current_layout.id
        )

    def _is_different_layout_non_space(
        self, char, char_layout: Layout, current_layout: "Layout | None"
    ) -> bool:
        """Return True if char is in a different layout and is not whitespace."""
        return char_layout.id != current_layout.id and not SPACE_REGEX.match(
            char.char_unicode
        )

    def _is_different_xobj(
        self, char, current_paragraph: "PdfParagraph | None"
    ) -> bool:
        """Return True if char belongs to a different xobject than the last composition."""
        if not current_paragraph or not current_paragraph.pdf_paragraph_composition:
            return False
        return (
            current_paragraph.pdf_paragraph_composition[-1].pdf_character.xobj_id
            != char.xobj_id
        )

    def _should_start_new_paragraph(
        self,
        char,
        char_layout: Layout,
        current_paragraph: "PdfParagraph | None",
        current_layout: "Layout | None",
        is_small_char: bool,
    ) -> bool:
        """Return True if the current character should begin a new paragraph."""
        if current_paragraph is None:
            return True
        # Small chars in the same layout stay in the current paragraph
        if self._is_small_char_same_layout(
            is_small_char, current_paragraph, char_layout, current_layout
        ):
            return False
        if char.char_unicode in HEIGHT_NOT_USFUL_CHAR_IN_CHAR:
            return False
        # Different layout (ignoring spaces)
        if self._is_different_layout_non_space(char, char_layout, current_layout):
            return True
        # Different xobject
        if self._is_different_xobj(char, current_paragraph):
            return True
        # Bullet point starts new paragraph
        if is_bullet_point(char) and not current_paragraph.pdf_paragraph_composition:
            return True
        return False

    def _make_new_paragraph_for_layout(self, char_layout: Layout) -> PdfParagraph:
        """Create a fresh PdfParagraph for the given layout."""
        return PdfParagraph(
            pdf_paragraph_composition=[],
            layout_id=char_layout.id,
            debug_id=generate_base58_id(),
            layout_label=char_layout.name,
        )

    def _process_char_into_paragraph(
        self,
        char,
        page: Page,
        layout_index,
        layout_map,
        median_char_area: float,
        paragraphs: list,
        skip_chars: list,
        current_paragraph: "PdfParagraph | None",
        current_layout: "Layout | None",
    ) -> tuple:
        """Process one character: assign to paragraph or skip list. Returns updated (current_paragraph, current_layout)."""
        char_layout = get_character_layout(char, layout_index, layout_map)
        char.formula_layout_id = is_character_in_formula_layout(
            char, page, layout_index, layout_map
        )
        if not is_text_layout(char_layout) or self.is_isolated_formula(char):
            skip_chars.append(char)
            return current_paragraph, current_layout
        char_box = char.visual_bbox.box
        char_area = (char_box.x2 - char_box.x) * (char_box.y2 - char_box.y)
        is_small_char = char_area < median_char_area * 0.05
        if self._should_start_new_paragraph(
            char, char_layout, current_paragraph, current_layout, is_small_char
        ):
            current_layout = char_layout
            current_paragraph = self._make_new_paragraph_for_layout(current_layout)
            paragraphs.append(current_paragraph)
        current_paragraph.pdf_paragraph_composition.append(
            PdfParagraphComposition(pdf_character=char)
        )
        return current_paragraph, current_layout

    def _group_characters_into_paragraphs(
        self, page: Page, layout_index, layout_map
    ) -> list[PdfParagraph]:
        paragraphs: list[PdfParagraph] = []
        if page.pdf_paragraph:
            paragraphs.extend(page.pdf_paragraph)
            page.pdf_paragraph = []

        median_char_area = self._compute_median_char_area(page.pdf_character)

        current_paragraph: PdfParagraph | None = None
        current_layout: Layout | None = None
        skip_chars: list = []

        for char in page.pdf_character:
            current_paragraph, current_layout = self._process_char_into_paragraph(
                char,
                page,
                layout_index,
                layout_map,
                median_char_area,
                paragraphs,
                skip_chars,
                current_paragraph,
                current_layout,
            )

        page.pdf_character = skip_chars
        for para in paragraphs:
            self.update_paragraph_data(para)
        return paragraphs

    @staticmethod
    def _compute_cluster_ranges_and_midlines(
        lines: dict[int, list[PdfCharacter]],
    ) -> tuple[dict, dict]:
        """Compute y-axis range and midline for each cluster."""
        cluster_ranges = {}
        cluster_midlines = {}
        for label, chars in lines.items():
            y_values = [char.visual_bbox.box.y for char in chars] + [
                char.visual_bbox.box.y2 for char in chars
            ]
            y_min, y_max = min(y_values), max(y_values)
            cluster_ranges[label] = (y_min, y_max)
            cluster_midlines[label] = (y_min + y_max) / 2
        return cluster_ranges, cluster_midlines

    @staticmethod
    def _should_merge_clusters(
        y1_min: float,
        y1_max: float,
        y2_min: float,
        y2_max: float,
        midline_distance: float,
        char_height_average: float,
    ) -> bool:
        """Return True if two clusters should be merged based on overlap or midline proximity."""
        if midline_distance < char_height_average:
            return True
        intersection_start = max(y1_min, y2_min)
        intersection_end = min(y1_max, y2_max)
        if intersection_end <= intersection_start:
            return False
        intersection_height = intersection_end - intersection_start
        min_height = min(y1_max - y1_min, y2_max - y2_min)
        return min_height > 0 and intersection_height / min_height > 0.3

    def _try_merge_cluster_pair(
        self,
        label1: int,
        label2: int,
        lines: dict,
        cluster_ranges: dict,
        cluster_midlines: dict,
        char_height_average: float,
    ) -> bool:
        """Attempt to merge cluster label2 into label1. Returns True if merged."""
        if label1 not in lines or label2 not in lines:
            return False
        y1_min, y1_max = cluster_ranges[label1]
        y2_min, y2_max = cluster_ranges[label2]
        midline_distance = abs(cluster_midlines[label1] - cluster_midlines[label2])
        if not self._should_merge_clusters(
            y1_min, y1_max, y2_min, y2_max, midline_distance, char_height_average
        ):
            return False
        lines[label1].extend(lines[label2])
        del lines[label2]
        new_y_min = min(y1_min, y2_min)
        new_y_max = max(y1_max, y2_max)
        cluster_ranges[label1] = (new_y_min, new_y_max)
        cluster_midlines[label1] = (new_y_min + new_y_max) / 2
        del cluster_ranges[label2]
        del cluster_midlines[label2]
        return True

    def _merge_overlapping_clusters(
        self, lines: dict[int, list[PdfCharacter]], char_height_average: float
    ) -> dict[int, list[PdfCharacter]]:
        """
        Merge clusters that have significant y-axis overlap.
        If y_intersection / min_height > 0.5 or the distance between y-midlines is less than char_height_average, merge the two clusters.
        """
        if len(lines) <= 1:
            return lines

        cluster_ranges, cluster_midlines = self._compute_cluster_ranges_and_midlines(
            lines
        )

        # Keep merging until no more merges are possible
        changed = True
        while changed:
            changed = False
            labels_to_check = list(lines.keys())
            for i in range(len(labels_to_check)):
                if changed:
                    break
                for j in range(i + 1, len(labels_to_check)):
                    label1, label2 = labels_to_check[i], labels_to_check[j]
                    if self._try_merge_cluster_pair(
                        label1,
                        label2,
                        lines,
                        cluster_ranges,
                        cluster_midlines,
                        char_height_average,
                    ):
                        changed = True
                        break

        return lines

    def _get_effective_y_bounds(self, char: PdfCharacter) -> tuple[float, float]:
        """
        Determines the effective vertical boundaries (y1, y2) for a character.

        It prioritizes the visual bounding box if its Intersection over Union (IoU)
        with the PDF bounding box is high (>= 0.5), otherwise, it falls back to the
        PDF bounding box. This helps use more accurate layout information when available.
        """
        visual_box = char.visual_bbox.box
        pdf_box = char.box
        if calculate_iou_for_boxes(visual_box, pdf_box) >= 0.5:
            return visual_box.y, visual_box.y2
        return pdf_box.y, pdf_box.y2

    @staticmethod
    def _compute_collision_counts_histogram(
        y1_arr: np.ndarray,
        y2_arr: np.ndarray,
        para_y_min: float,
        para_y_max: float,
        step: float,
    ) -> np.ndarray:
        """Compute overlap counts at each scan line using a difference-array histogram.

        Args:
            y1_arr: 1-D array with lower y bounds of characters (inclusive).
            y2_arr: 1-D array with upper y bounds of characters (exclusive).
            para_y_min: Minimum y of the paragraph.
            para_y_max: Maximum y of the paragraph.
            step: Scan step size.

        Returns:
            1-D NumPy int32 array where index i corresponds to y = para_y_max - i Ã— step.
        """
        # Number of scan positions
        m = int(np.ceil((para_y_max - para_y_min) / step))
        if m <= 0:
            return np.array([], dtype=np.int32)

        # Map character bounds to discrete indices (top inclusive, bottom exclusive)
        starts = np.floor((para_y_max - y2_arr) / step).astype(np.int32)
        ends = np.floor((para_y_max - y1_arr) / step).astype(np.int32) + 1
        # Clip ends to the valid range [0, m]
        np.clip(ends, 0, m, out=ends)

        hist = np.zeros(m + 1, dtype=np.int32)
        np.add.at(hist, starts, 1)
        np.add.at(hist, ends, -1)

        return np.cumsum(hist[:-1])

    def _extract_chars_and_other_compositions(
        self, paragraph: PdfParagraph
    ) -> tuple[list[PdfCharacter], list[PdfParagraphComposition]]:
        """Separate a paragraph's compositions into plain characters and everything else."""
        all_chars: list[PdfCharacter] = []
        other_compositions: list[PdfParagraphComposition] = []
        for comp in paragraph.pdf_paragraph_composition:
            if comp.pdf_character:
                all_chars.append(comp.pdf_character)
            else:
                other_compositions.append(comp)
        return all_chars, other_compositions

    def _build_char_y_bounds(self, all_chars: list[PdfCharacter]) -> list[dict]:
        """Return per-character dicts with effective y1/y2 bounds."""
        return [
            {"char": char, "y1": y1, "y2": y2}
            for char in all_chars
            for y1, y2 in [self._get_effective_y_bounds(char)]
        ]

    @staticmethod
    def _find_histogram_gaps(collision_counts) -> list[tuple[int, int]]:
        """Identify contiguous zero-count regions in a collision histogram."""
        gaps = []
        in_gap = False
        gap_start_index = 0
        for i, count in enumerate(collision_counts):
            if count < 1 and not in_gap:
                in_gap = True
                gap_start_index = i
            elif count >= 1 and in_gap:
                in_gap = False
                gaps.append((gap_start_index, i - 1))
        if in_gap:
            gaps.append((gap_start_index, len(collision_counts) - 1))
        return gaps

    def _assign_chars_to_line_buckets(
        self, char_y_bounds: list[dict], separator_y_coords: list[float]
    ) -> list[list[PdfCharacter]]:
        """Assign each character to a line bucket based on separator y-coordinates."""
        lines: list[list[PdfCharacter]] = [
            [] for _ in range(len(separator_y_coords) + 1)
        ]
        for b in char_y_bounds:
            char_y_center = (b["y1"] + b["y2"]) / 2
            line_idx = 0
            for sep_y in separator_y_coords:
                if char_y_center > sep_y:
                    break
                line_idx += 1
            lines[line_idx].append(b["char"])
        return lines

    def _split_paragraph_into_lines(
        self, paragraph: PdfParagraph, _formula_font_ids: set[str]
    ):
        """
        Splits a paragraph into lines using a "line-threading" method.

        This method works by scanning vertically across the paragraph's bounding
        box and counting how many characters intersect with a horizontal line
        at each y-coordinate. The regions with a low number of intersections
        (less than 2) are identified as gaps between lines. The characters
        are then partitioned into lines based on these identified gaps.
        """
        if not paragraph.pdf_paragraph_composition:
            return

        # 1. Extract all characters and other compositions from the paragraph.
        all_chars, other_compositions = self._extract_chars_and_other_compositions(
            paragraph
        )

        if not all_chars:
            return

        # 2. Determine effective y-bounds for each character and the paragraph's total vertical range.
        char_y_bounds = self._build_char_y_bounds(all_chars)

        if not char_y_bounds:
            paragraph.pdf_paragraph_composition = other_compositions
            self.update_paragraph_data(paragraph)
            return

        para_y_min = min(b["y1"] for b in char_y_bounds)
        para_y_max = max(b["y2"] for b in char_y_bounds)

        # If the paragraph is vertically flat, treat it as a single line.
        if (para_y_max - para_y_min) < 5:  # Using a small threshold
            single_line_composition = self.create_line(all_chars)
            paragraph.pdf_paragraph_composition = [
                single_line_composition
            ] + other_compositions
            self.update_paragraph_data(paragraph)
            return

        # 3. Perform "threading" scan to create a collision histogram.
        step = 0.25
        y_coordinates = np.arange(para_y_max, para_y_min, -step)

        # Compute collision counts using NumPy histogram (O(m + n))
        y1_arr = np.array([b["y1"] for b in char_y_bounds], dtype=np.float32)
        y2_arr = np.array([b["y2"] for b in char_y_bounds], dtype=np.float32)
        collision_counts = self._compute_collision_counts_histogram(
            y1_arr,
            y2_arr,
            para_y_min,
            para_y_max,
            step,
        )

        # 4. Find gaps (regions with low collision count) from the histogram.
        gaps = self._find_histogram_gaps(collision_counts)

        # If no significant gaps are found, treat it as a single line.
        if not gaps:
            single_line_composition = self.create_line(all_chars)
            paragraph.pdf_paragraph_composition = [
                single_line_composition
            ] + other_compositions
            self.update_paragraph_data(paragraph)
            return

        # 5. Assign characters to lines based on the identified gaps.
        separator_y_coords = sorted(
            [y_coordinates[start_idx] for start_idx, _end_idx in gaps],
            reverse=True,
        )

        lines = self._assign_chars_to_line_buckets(char_y_bounds, separator_y_coords)

        # 6. Rebuild the paragraph's composition list from the new lines.
        new_line_compositions = [
            self.create_line(line_chars) for line_chars in lines if line_chars
        ]

        # The lines are already sorted vertically due to the scanning process.
        paragraph.pdf_paragraph_composition = new_line_compositions + other_compositions
        self.update_paragraph_data(paragraph)

    @staticmethod
    def _strip_trailing_spaces(chars: list) -> list:
        """Remove trailing whitespace characters from a character list."""
        result = []
        for char in chars:
            if not char.char_unicode.isspace():
                result = result + [char]
            elif result:  # åªæœ‰åœ¨æœ‰éžç©ºæ ¼å­—ç¬¦åŽæ‰è€ƒè™‘ä¿ç•™ç©ºæ ¼
                result.append(char)
        while result and result[-1].char_unicode.isspace():
            result.pop()
        return result

    def _process_line_spacing(self, composition) -> "PdfParagraphComposition | None":
        """Process a single line composition for spacing. Returns None to discard."""
        if not composition.pdf_line:
            return composition
        line = composition.pdf_line
        if not "".join(x.char_unicode for x in line.pdf_character).strip():
            return None  # è·³è¿‡å®Œå…¨ç©ºç™½çš„è¡Œ
        processed_chars = self._strip_trailing_spaces(line.pdf_character)
        if not processed_chars:
            return None
        return self.create_line(processed_chars)

    def process_paragraph_spacing(self, paragraph: PdfParagraph):
        if not paragraph.pdf_paragraph_composition:
            return

        # å¤„ç†è¡Œçº§åˆ«çš„ç©ºæ ¼
        processed_lines = []
        for composition in paragraph.pdf_paragraph_composition:
            result = self._process_line_spacing(composition)
            if result is not None:
                processed_lines.append(result)

        paragraph.pdf_paragraph_composition = processed_lines
        self.update_paragraph_data(paragraph)

    def create_line(self, chars: list[PdfCharacter]) -> PdfParagraphComposition:
        if not chars:
            raise AssertionError

        line = PdfLine(pdf_character=chars)
        self.update_line_data(line)
        return PdfParagraphComposition(pdf_line=line)

    def calculate_median_line_width(self, paragraphs: list[PdfParagraph]) -> float:
        # æ”¶é›†æ‰€æœ‰è¡Œçš„å®½åº¦
        line_widths = []
        for paragraph in paragraphs:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    line_widths.append(line.box.x2 - line.box.x)

        if not line_widths:
            return 0.0

        # è®¡ç®—ä¸­ä½æ•°
        line_widths.sort()
        mid = len(line_widths) // 2
        if len(line_widths) % 2 == 0:
            return (line_widths[mid - 1] + line_widths[mid]) / 2
        return line_widths[mid]

    def _split_off_tail_as_paragraph(
        self,
        paragraph: PdfParagraph,
        paragraphs: list[PdfParagraph],
        i: int,
        j: int,
    ) -> None:
        """Split compositions[j:] out of paragraph into a new paragraph inserted at i+1."""
        new_paragraph = PdfParagraph(
            box=Box(0, 0, 0, 0),
            pdf_paragraph_composition=paragraph.pdf_paragraph_composition[j:],
            unicode="",
            debug_id=generate_base58_id(),
            layout_label=paragraph.layout_label,
            layout_id=paragraph.layout_id,
        )
        paragraph.pdf_paragraph_composition = paragraph.pdf_paragraph_composition[:j]
        self.update_paragraph_data(paragraph)
        self.update_paragraph_data(new_paragraph)
        paragraphs.insert(i + 1, new_paragraph)

    def _is_short_line_split(self, prev_width: float, median_width: float) -> bool:
        """Return True if the previous line is short enough to trigger a split."""
        return (
            self.translation_config.split_short_lines
            and prev_width
            < median_width * self.translation_config.short_line_split_factor
        )

    @staticmethod
    def _next_line_starts_with_bullet(paragraph: PdfParagraph, j: int) -> bool:
        """Return True if the composition at index j starts with a bullet point."""
        current_line = paragraph.pdf_paragraph_composition[j]
        line = current_line.pdf_line
        if not line or not line.pdf_character:
            return False
        return is_bullet_point(line.pdf_character[0])

    @staticmethod
    def _is_toc_entry_line(prev_text: str) -> bool:
        """Return True if the line looks like a table-of-contents entry (20+ consecutive dots)."""
        return bool(re.search(r"\.{20,}", prev_text))

    def _should_split_at_composition(
        self,
        paragraph: PdfParagraph,
        j: int,
        median_width: float,
    ) -> bool:
        """
        Return True if the paragraph should be split at composition index j.
        Checks the previous line for TOC dots, short-line split, or bullet start.
        Returns None when the composition at j-1 is not a text line (skip it).
        """
        prev_composition = paragraph.pdf_paragraph_composition[j - 1]
        if not prev_composition.pdf_line:
            return False
        prev_line = prev_composition.pdf_line
        prev_width = prev_line.box.x2 - prev_line.box.x
        prev_text = "".join([c.char_unicode for c in prev_line.pdf_character])
        if self._is_toc_entry_line(prev_text):
            return True
        return self._is_short_line_split(
            prev_width, median_width
        ) or self._next_line_starts_with_bullet(paragraph, j)

    def _process_single_paragraph_splits(
        self,
        paragraph: PdfParagraph,
        paragraphs: list[PdfParagraph],
        i: int,
        median_width: float,
    ) -> None:
        """Scan one paragraph for split points and split off tails as needed."""
        j = 1
        while j < len(paragraph.pdf_paragraph_composition):
            prev_composition = paragraph.pdf_paragraph_composition[j - 1]
            if not prev_composition.pdf_line:
                j += 1
                continue
            if self._should_split_at_composition(paragraph, j, median_width):
                self._split_off_tail_as_paragraph(paragraph, paragraphs, i, j)
                break
            j += 1

    def process_independent_paragraphs(
        self,
        paragraphs: list[PdfParagraph],
        median_width: float,
    ):
        i = 0
        while i < len(paragraphs):
            paragraph = paragraphs[i]
            if (
                len(paragraph.pdf_paragraph_composition) <= 1
            ):  # è·³è¿‡åªæœ‰ä¸€è¡Œçš„æ®µè½
                i += 1
                continue
            self._process_single_paragraph_splits(
                paragraph, paragraphs, i, median_width
            )
            i += 1

    @staticmethod
    def is_bbox_contain_in_vertical(bbox1: Box, bbox2: Box) -> bool:
        """Check if one bounding box is completely contained within the other."""
        # Check if bbox1 is contained in bbox2
        bbox1_in_bbox2 = bbox1.y >= bbox2.y and bbox1.y2 <= bbox2.y2
        # Check if bbox2 is contained in bbox1
        bbox2_in_bbox1 = bbox2.y >= bbox1.y and bbox2.y2 <= bbox1.y2
        return bbox1_in_bbox2 or bbox2_in_bbox1

    def _resolve_paragraph_overlap(self, para1, para2) -> bool:
        """
        Attempt to resolve vertical overlap between two paragraphs by adjusting
        their bounding boxes to the midpoint of the overlap.
        Returns True if an overlap was found and processed (even if unresolvable).
        """
        if para1.box is None or para2.box is None:
            return False
        if para1.xobj_id != para2.xobj_id:
            return False
        if not self.bbox_overlap(para1.box, para2.box):
            return False
        if self.is_bbox_contain_in_vertical(para1.box, para2.box):
            return False

        overlap_y_start = max(para1.box.y, para2.box.y)
        overlap_y_end = min(para1.box.y2, para2.box.y2)
        overlap_height = overlap_y_end - overlap_y_start
        overlap_x_start = max(para1.box.x, para2.box.x)
        overlap_x_end = min(para1.box.x2, para2.box.x2)
        overlap_width = overlap_x_end - overlap_x_start

        if overlap_height <= 1e-6 or overlap_width <= 1e-6:
            return False

        # Determine which paragraph is visually higher
        if para1.box.y2 > para2.box.y and para1.box.y < para2.box.y:
            higher_para, lower_para = para2, para1
        elif para1.box.y2 < para2.box.y2:
            higher_para, lower_para = para2, para1
        else:
            higher_para, lower_para = para1, para2

        mid_y = overlap_y_start + overlap_height / 2
        if mid_y > higher_para.box.y and mid_y < lower_para.box.y2:
            higher_para.box.y = mid_y + 1
            lower_para.box.y2 = mid_y - 1
        else:
            logger.warning(
                "Could not resolve overlap between paragraphs"
                f" {higher_para.debug_id} and {lower_para.debug_id}"
                " using simple midpoint strategy."
                f" Midpoint: {mid_y},"
                f" Higher Box: {higher_para.box},"
                f" Lower Box: {lower_para.box}"
            )
        return True

    def fix_overlapping_paragraphs(self, page: Page):
        """
        Adjusts the bounding boxes of paragraphs on a page to resolve vertical overlaps.

        Iteratively checks pairs of paragraphs and adjusts their vertical boundaries
        (y and y2) if they overlap, aiming to place the boundary at the midpoint
        of the vertical overlap.
        """
        paragraphs = page.pdf_paragraph
        if not paragraphs or len(paragraphs) < 2:
            return

        max_iterations = len(paragraphs) * len(paragraphs)  # Safety break
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            overlap_found_in_pass = False

            for i in range(len(paragraphs)):
                for j in range(i + 1, len(paragraphs)):
                    if self._resolve_paragraph_overlap(paragraphs[i], paragraphs[j]):
                        overlap_found_in_pass = True

            # If no overlaps were found and adjusted in this pass, we're done.
            if not overlap_found_in_pass:
                break

        if iterations == max_iterations:
            logger.warning(
                f"Maximum iterations ({max_iterations}) reached in"
                f" fix_overlapping_paragraphs for page {page.page_number}."
                " Some overlaps might remain."
            )

    def _sort_characters_in_lines(self, page: Page):
        """Sort characters in each line from left to right, top to bottom."""
        for paragraph in page.pdf_paragraph:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    line.pdf_character.sort(key=self._get_char_sort_key)

    def _get_char_sort_key(self, char: PdfCharacter):
        """Get sort key for character positioning (top to bottom, left to right)."""
        visual_box = char.visual_bbox.box
        pdf_box = char.box

        # Use visual box if IoU with bbox is >= 0.1, otherwise use bbox
        if calculate_iou_for_boxes(visual_box, pdf_box) >= 0.1:
            box = visual_box
        else:
            box = pdf_box

        # Sort by y coordinate first (top to bottom), then x coordinate (left to right)
        # Note: In PDF coordinate system, y increases upward, so we negate y for top-to-bottom sorting
        return (box.x, -box.y)
