import math
import re

from src.doctranslator.format.pdf.document_il.il_version_1 import Box
from src.doctranslator.format.pdf.document_il.il_version_1 import Document
from src.doctranslator.format.pdf.document_il.il_version_1 import GraphicState
from src.doctranslator.format.pdf.document_il.il_version_1 import Page
from src.doctranslator.format.pdf.document_il.il_version_1 import PdfCharacter
from src.doctranslator.format.pdf.document_il.il_version_1 import PdfFormula
from src.doctranslator.format.pdf.document_il.il_version_1 import PdfLine
from src.doctranslator.format.pdf.document_il.il_version_1 import (
    PdfParagraphComposition,
)
from src.doctranslator.format.pdf.document_il.il_version_1 import PdfSameStyleCharacters
from src.doctranslator.format.pdf.document_il.il_version_1 import PdfStyle
from src.doctranslator.format.pdf.document_il.utils.fontmap import FontMapper
from src.doctranslator.format.pdf.document_il.utils.formular_helper import (
    collect_page_formula_font_ids,
)
from src.doctranslator.format.pdf.document_il.utils.formular_helper import (
    is_formulas_middle_char,
)
from src.doctranslator.format.pdf.document_il.utils.formular_helper import (
    is_formulas_start_char,
)
from src.doctranslator.format.pdf.document_il.utils.formular_helper import (
    update_formula_data,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import LEFT_BRACKET
from src.doctranslator.format.pdf.document_il.utils.layout_helper import RIGHT_BRACKET
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    build_layout_index,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    calculate_iou_for_boxes,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    calculate_y_true_iou_for_boxes,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import is_bullet_point
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    is_curve_in_figure_table_layout,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import (
    is_curve_overlapping_with_paragraphs,
)
from src.doctranslator.format.pdf.document_il.utils.layout_helper import is_same_style
from src.doctranslator.format.pdf.document_il.utils.spatial_analyzer import (
    is_element_contained_in_formula,
)
from src.doctranslator.format.pdf.translation_config import TranslationConfig


class StylesAndFormulas:
    stage_name = "Parse Formulas and Styles"

    def __init__(self, translation_config: TranslationConfig):
        self.translation_config = translation_config
        self.font_mapper = FontMapper(translation_config)

    def update_formula_data(self, formula: PdfFormula):
        update_formula_data(formula)

    def process(self, document: Document):
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            len(document.page),
        ) as pbar:
            for page in document.page:
                self.translation_config.raise_if_cancelled()
                self.process_page(page)
                pbar.advance()

    def update_all_formula_data(self, page: Page):
        for para in page.pdf_paragraph:
            for comp in para.pdf_paragraph_composition:
                if comp.pdf_formula:
                    self.update_formula_data(comp.pdf_formula)

    def _calculate_element_formula_iou(
        self, element_box: Box, formula_box: Box, tolerance: float = 2.0
    ) -> float:
        """Calculate precise IoU between an element and a formula with tolerance.

        Args:
            element_box: Bounding box of the element (curve/form)
            formula_box: Bounding box of the formula
            tolerance: Tolerance to expand formula box for containment check

        Returns:
            IoU value between element and expanded formula box
        """
        if element_box is None or formula_box is None:
            return 0.0

        # Expand formula box by tolerance for more lenient containment check
        expanded_formula_box = Box(
            x=formula_box.x - tolerance,
            y=formula_box.y - tolerance,
            x2=formula_box.x2 + tolerance,
            y2=formula_box.y2 + tolerance,
        )

        return calculate_iou_for_boxes(element_box, expanded_formula_box)

    def _is_element_contained_exact(
        self,
        element_box: Box,
        formula_box: Box,
        containment_threshold: float = 0.95,
    ) -> bool:
        """Check if an element is contained within a formula with zero tolerance.

        Args:
            element_box: Bounding box of the element (curve/form)
            formula_box: Bounding box of the formula
            containment_threshold: Minimum IoU ratio to consider as contained

        Returns:
            True if the element is contained within the formula (exact match)
        """
        if element_box is None or formula_box is None:
            return False

        # Use formula box without any tolerance expansion
        iou = calculate_iou_for_boxes(element_box, formula_box)
        return iou >= containment_threshold

    def _calculate_element_formula_distance(
        self, element_box: Box, formula_box: Box
    ) -> float:
        """Calculate the shortest distance between an element and a formula.

        Args:
            element_box: Bounding box of the element (curve/form)
            formula_box: Bounding box of the formula

        Returns:
            Shortest distance between the element and formula boxes
        """
        if element_box is None or formula_box is None:
            return float("inf")

        # Calculate horizontal distance
        if element_box.x2 < formula_box.x:
            # Element is to the left of formula
            dx = formula_box.x - element_box.x2
        elif element_box.x > formula_box.x2:
            # Element is to the right of formula
            dx = element_box.x - formula_box.x2
        else:
            # Horizontal overlap
            dx = 0.0

        # Calculate vertical distance
        if element_box.y2 < formula_box.y:
            # Element is above formula
            dy = formula_box.y - element_box.y2
        elif element_box.y > formula_box.y2:
            # Element is below formula
            dy = element_box.y - formula_box.y2
        else:
            # Vertical overlap
            dy = 0.0

        # Return Euclidean distance
        return (dx * dx + dy * dy) ** 0.5

    def _score_element_against_formulas(
        self,
        element_box,
        element_xobj_id,
        all_formulas: list,
        max_tolerant_distance: float = 100.0,
    ) -> list:
        """Score *element* against each formula and return a candidates list.

        Each entry is a (formula_idx, score, match_type) tuple.
        """
        candidates = []
        for formula_idx, (formula, paragraph_xobj_id) in enumerate(all_formulas):
            if not formula.box:
                continue
            if paragraph_xobj_id is not None and element_xobj_id != paragraph_xobj_id:
                continue
            if self._is_element_contained_exact(element_box, formula.box):
                iou = calculate_iou_for_boxes(element_box, formula.box)
                candidates.append((formula_idx, iou, "iou_exact"))
            elif is_element_contained_in_formula(element_box, formula.box):
                distance = self._calculate_element_formula_distance(
                    element_box, formula.box
                )
                distance_factor = max(0.0, 1.0 - distance / max_tolerant_distance)
                score = 0.5 + 0.4 * distance_factor
                candidates.append((formula_idx, score, "iou_tolerant"))
        return candidates

    def _match_elements_to_formulas(self, elements: list, all_formulas: list) -> dict:
        """Score each element in a list against all formulas and return a candidates dict."""
        candidates_map = {}
        for idx, element in enumerate(elements):
            if not element.box:
                continue
            candidates = self._score_element_against_formulas(
                element.box, element.xobj_id, all_formulas
            )
            if candidates:
                candidates_map[idx] = (element, candidates)
        return candidates_map

    def _collect_element_formula_candidates(
        self, page: Page
    ) -> tuple[list, dict, dict]:
        """Collect all potential assignments of elements to formulas.

        Uses two-level IoU matching strategy:
        1. Exact IoU matching (zero tolerance) - highest priority
        2. Tolerant IoU matching (2.0 tolerance, distance-sorted) - second priority

        Returns:
            Tuple of (all_formulas, curve_candidates, form_candidates) where:
            - all_formulas: list of (formula, paragraph_xobj_id) tuples
            - curve_candidates: dict mapping curve index to (curve, candidates) tuples
            - form_candidates: dict mapping form index to (form, candidates) tuples
            where candidates is a list of (formula_index, score, match_type) tuples
        """
        if not page.pdf_paragraph:
            return [], {}, {}

        # Collect all formulas from all paragraphs with their index
        all_formulas = [
            (composition.pdf_formula, paragraph.xobj_id)
            for paragraph in page.pdf_paragraph
            for composition in paragraph.pdf_paragraph_composition
            if composition.pdf_formula
        ]

        curve_candidates = self._match_elements_to_formulas(
            page.pdf_curve, all_formulas
        )
        form_candidates = self._match_elements_to_formulas(page.pdf_form, all_formulas)

        return all_formulas, curve_candidates, form_candidates

    @staticmethod
    def _candidate_sort_key(candidate) -> tuple:
        """Sort key: Exact IoU (priority 1) before tolerant IoU (priority 2), then by descending score."""
        formula_idx, score, match_type = candidate
        priority = 1 if match_type == "iou_exact" else 2
        return (priority, -score)

    def _get_best_candidate(self, candidates: list):
        """Return the highest-priority candidate or None if the list is empty."""
        if not candidates:
            return None
        return sorted(candidates, key=self._candidate_sort_key)[0]

    def _assign_element_to_formula(
        self,
        element,
        candidates: list,
        formula_assignments: dict,
        elements_to_remove: list,
        slot_index: int,
    ) -> None:
        """Assign element to its best-matching formula and record it for removal."""
        best_candidate = self._get_best_candidate(candidates)
        if not best_candidate:
            return
        best_formula_idx = best_candidate[0]
        if best_formula_idx not in formula_assignments:
            formula_assignments[best_formula_idx] = ([], [])
        formula_assignments[best_formula_idx][slot_index].append(element)
        elements_to_remove.append(element)

    def _resolve_assignment_conflicts(
        self, curve_candidates: dict, form_candidates: dict
    ) -> tuple[dict, list, list]:
        """Resolve assignment conflicts using prioritized matching strategy.

        Args:
            curve_candidates: dict mapping curve index to (curve, candidates) tuples
            form_candidates: dict mapping form index to (form, candidates) tuples
            where candidates is a list of (formula_index, score, match_type) tuples

        Returns:
            Tuple of (formula_assignments, curves_to_remove, forms_to_remove) where:
            - formula_assignments: dict mapping formula_index to (curves, forms) tuples
            - curves_to_remove: list of curves to remove from page level
            - forms_to_remove: list of forms to remove from page level
        """
        formula_assignments: dict = {}
        curves_to_remove: list = []
        forms_to_remove: list = []

        # Resolve curve assignments (slot 0)
        for _curve_idx, (curve, candidates) in curve_candidates.items():
            if candidates:
                self._assign_element_to_formula(
                    curve, candidates, formula_assignments, curves_to_remove, 0
                )

        # Resolve form assignments (slot 1)
        for _form_idx, (form, candidates) in form_candidates.items():
            if candidates:
                self._assign_element_to_formula(
                    form, candidates, formula_assignments, forms_to_remove, 1
                )

        return formula_assignments, curves_to_remove, forms_to_remove

    def collect_contained_elements(self, page: Page):
        """Collect curves and forms that are contained within formulas.

        Uses two-phase assignment strategy to ensure each element is assigned
        to only one formula based on highest IoU value.
        """
        if not page.pdf_paragraph:
            return

        # Phase 1: Collect all potential element-formula assignments
        all_formulas, curve_candidates, form_candidates = (
            self._collect_element_formula_candidates(page)
        )

        # Phase 2: Resolve conflicts using IoU maximization
        formula_assignments, curves_to_remove, forms_to_remove = (
            self._resolve_assignment_conflicts(curve_candidates, form_candidates)
        )

        # Apply the resolved assignments using formula indices
        for formula_idx, (
            assigned_curves,
            assigned_forms,
        ) in formula_assignments.items():
            formula = all_formulas[formula_idx][0]  # Extract formula from tuple
            formula.pdf_curve.extend(assigned_curves)
            formula.pdf_form.extend(assigned_forms)

        # Remove assigned elements from page level
        for curve in curves_to_remove:
            if curve in page.pdf_curve:
                page.pdf_curve.remove(curve)

        for form in forms_to_remove:
            if form in page.pdf_form:
                page.pdf_form.remove(form)

    def process_page(self, page: Page):
        """å¤„ç†é¡µé¢ï¼ŒåŒ…æ‹¬å…¬å¼è¯†åˆ«å’Œåç§»é‡è®¡ç®—"""
        self.process_page_formulas(page)
        # self.process_page_offsets(page)
        self.process_comma_formulas(page)
        self.merge_overlapping_formulas(page)
        if not self.translation_config.skip_formula_offset_calculation:
            self.process_page_offsets(page)
        self.process_translatable_formulas(page)
        self.update_all_formula_data(page)
        if not self.translation_config.ocr_workaround:
            self.collect_contained_elements(page)

        # Process remaining non-formula lines after formula assignment is complete
        if self.translation_config.remove_non_formula_lines:
            self.remove_non_formula_lines_from_paragraphs(page)

        if not self.translation_config.skip_formula_offset_calculation:
            self.process_page_offsets(page)
        self.update_all_formula_data(page)
        self.process_page_styles(page)

    def update_line_data(self, line: PdfLine):
        min_x = min(char.visual_bbox.box.x for char in line.pdf_character)
        min_y = min(char.visual_bbox.box.y for char in line.pdf_character)
        max_x = max(char.visual_bbox.box.x2 for char in line.pdf_character)
        max_y = max(char.visual_bbox.box.y2 for char in line.pdf_character)
        line.box = Box(min_x, min_y, max_x, max_y)

    @staticmethod
    def _char_bbox_is_disjoint(char: PdfCharacter) -> bool:
        """Return True if the PDF bbox and visual bbox of a character do not overlap."""
        return (
            char.box.x > char.visual_bbox.box.x2
            or char.box.x2 < char.visual_bbox.box.x
            or char.box.y > char.visual_bbox.box.y2
            or char.box.y2 < char.visual_bbox.box.y
        )

    def _is_formula_char_by_context(
        self, char: PdfCharacter, in_formula_state: bool
    ) -> bool:
        """Return True if char qualifies as formula via start/middle/null-in-formula rules."""
        if char.char_unicode is None:
            return in_formula_state
        if in_formula_state:
            return is_formulas_middle_char(
                char.char_unicode, self.font_mapper, self.translation_config
            )
        return is_formulas_start_char(
            char.char_unicode, self.font_mapper, self.translation_config
        )

    def _is_formula_char(
        self,
        char: PdfCharacter,
        in_formula_state: bool,
        formula_font_ids: set[int],
    ) -> bool:
        """Return True if `char` should be classified as formula (first-pass, no corner-mark logic)."""
        return (
            bool(char.formula_layout_id)
            or self._is_formula_char_by_context(char, in_formula_state)
            or char.pdf_style.font_id in formula_font_ids
            or char.vertical
            or self._char_bbox_is_disjoint(char)
        )

    @staticmethod
    def _is_smaller_than_prev(char, previous_char, in_corner_mark_state: bool) -> bool:
        """Return True if char is small relative to previous_char in a way that indicates a corner mark."""
        threshold = 1.1 if in_corner_mark_state else 0.79
        return char.pdf_style.font_size < previous_char.pdf_style.font_size * threshold

    @staticmethod
    def _is_smaller_than_next(char, next_char) -> bool:
        """Return True if char is smaller than next_char at the start of a segment (no state)."""
        return char.pdf_style.font_size < next_char.pdf_style.font_size * 0.79

    @staticmethod
    def _is_corner_mark(
        char: PdfCharacter,
        previous_char: "PdfCharacter | None",
        next_char: "PdfCharacter | None",
        isspace: bool,
        prev_is_space: bool,
        first_is_bullet: bool,
        in_corner_mark_state: bool,
    ) -> bool:
        """Return True if `char` is a superscript/subscript corner mark."""
        # Guard: bullet points, spaces, and first chars are never corner marks
        if isspace or prev_is_space or first_is_bullet:
            return False
        if previous_char is not None:
            return StylesAndFormulas._is_smaller_than_prev(
                char, previous_char, in_corner_mark_state
            )
        if next_char is not None and not in_corner_mark_state:
            return StylesAndFormulas._is_smaller_than_next(char, next_char)
        return False

    @staticmethod
    def _char_is_space(char) -> bool:
        return bool(char.char_unicode and char.char_unicode.isspace())

    def _classify_single_char(
        self,
        char,
        i: int,
        line,
        is_formula_tags: list,
        in_formula_state: bool,
        in_corner_mark_state: bool,
        first_is_bullet: bool,
        formula_font_ids: set,
    ) -> tuple[bool, bool, bool]:
        """Classify one character. Returns (is_formula, is_corner_mark, updated_first_is_bullet)."""
        is_start_of_segment = i == 0 or (
            len(is_formula_tags) > 0 and is_formula_tags[-1] != in_formula_state
        )
        if not first_is_bullet and is_start_of_segment and is_bullet_point(char):
            first_is_bullet = True

        is_formula = self._is_formula_char(char, in_formula_state, formula_font_ids)

        previous_char = line.pdf_character[i - 1] if i > 0 else None
        next_char = (
            line.pdf_character[i + 1] if i < len(line.pdf_character) - 1 else None
        )
        isspace = self._char_is_space(char)
        prev_is_space = self._char_is_space(previous_char) if previous_char else False

        is_corner_mark = self._is_corner_mark(
            char,
            previous_char,
            next_char,
            isspace,
            prev_is_space,
            first_is_bullet,
            in_corner_mark_state,
        )

        is_formula = is_formula or is_corner_mark
        if char.char_unicode == " ":
            is_formula = in_formula_state

        return is_formula, is_corner_mark, first_is_bullet

    def _classify_characters_in_composition(
        self,
        composition: PdfParagraphComposition,
        formula_font_ids: set[int],
        first_is_bullet_so_far: bool,
    ) -> tuple[list[tuple[PdfCharacter, bool]], bool]:
        """
        Phase 1: Classify every character in a composition as either formula or text.
        This preserves the original logic, including the sticky `first_is_bullet` flag.
        """
        line = composition.pdf_line
        if not line or not line.pdf_character:
            return [], first_is_bullet_so_far

        first_is_bullet = first_is_bullet_so_far
        in_formula_state = False
        in_corner_mark_state = False
        is_formula_tags: list = []
        corner_mark_info: list = []

        for i, char in enumerate(line.pdf_character):
            is_formula, is_corner_mark, first_is_bullet = self._classify_single_char(
                char,
                i,
                line,
                is_formula_tags,
                in_formula_state,
                in_corner_mark_state,
                first_is_bullet,
                formula_font_ids,
            )
            if is_formula != in_formula_state:
                in_formula_state = is_formula
            in_corner_mark_state = is_corner_mark
            is_formula_tags.append(is_formula)
            corner_mark_info.append(is_corner_mark)

        tagged_chars = [
            (char, is_formula, is_corner_mark)
            for char, is_formula, is_corner_mark in zip(
                line.pdf_character, is_formula_tags, corner_mark_info, strict=False
            )
        ]
        return tagged_chars, first_is_bullet

    def _group_classified_characters(
        self,
        tagged_chars: list[tuple[PdfCharacter, bool, bool]],
    ) -> list[PdfParagraphComposition]:
        """
        Phase 2: Group consecutive characters with the same tag into new compositions.
        """
        if not tagged_chars:
            return []

        new_compositions = []
        current_chars = []
        current_tag = tagged_chars[0][1]
        current_corner_mark_flags = []

        for char, is_formula_tag, is_corner_mark in tagged_chars:
            if is_formula_tag == current_tag:
                current_chars.append(char)
                current_corner_mark_flags.append(is_corner_mark)
            else:
                # Check if any character in current group is a corner mark
                has_corner_mark = any(current_corner_mark_flags)
                new_compositions.append(
                    self.create_composition(
                        current_chars, current_tag, has_corner_mark
                    ),
                )
                current_chars = [char]
                current_tag = is_formula_tag
                current_corner_mark_flags = [is_corner_mark]

        if current_chars:
            # Check if any character in final group is a corner mark
            has_corner_mark = any(current_corner_mark_flags)
            new_compositions.append(
                self.create_composition(current_chars, current_tag, has_corner_mark),
            )

        return new_compositions

    def process_page_formulas(self, page: Page):
        if not page.pdf_paragraph:
            return

        page_level_formula_font_ids, xobj_specific_formula_font_ids = (
            collect_page_formula_font_ids(
                page, self.translation_config.formular_font_pattern
            )
        )

        for paragraph in page.pdf_paragraph:
            if not paragraph.pdf_paragraph_composition:
                continue

            current_formula_font_ids: set[int]
            if (
                paragraph.xobj_id
                and paragraph.xobj_id in xobj_specific_formula_font_ids
            ):
                current_formula_font_ids = xobj_specific_formula_font_ids[
                    paragraph.xobj_id
                ]
            else:
                current_formula_font_ids = page_level_formula_font_ids

            new_paragraph_compositions = []
            # This flag is carried through all compositions in a paragraph, as in the original implementation.
            first_is_bullet = False

            for composition in paragraph.pdf_paragraph_composition:
                (
                    tagged_chars,
                    first_is_bullet,
                ) = self._classify_characters_in_composition(
                    composition,
                    current_formula_font_ids,
                    first_is_bullet,
                )

                if not tagged_chars:
                    new_paragraph_compositions.append(composition)
                    continue

                grouped_compositions = self._group_classified_characters(tagged_chars)
                new_paragraph_compositions.extend(grouped_compositions)

            paragraph.pdf_paragraph_composition = new_paragraph_compositions

    def process_translatable_formulas(self, page: Page):
        """å°†éœ€è¦æ­£å¸¸ç¿»è¯‘çš„å…¬å¼ï¼ˆå¦‚çº¯æ•°å­—ã€æ•°å­—åŠ é€—å·ç­‰ï¼‰è½¬æ¢ä¸ºæ™®é€šæ–‡æœ¬è¡Œ"""
        if not page.pdf_paragraph:
            return

        for paragraph in page.pdf_paragraph:
            if not paragraph.pdf_paragraph_composition:
                continue

            new_compositions = []
            for composition in paragraph.pdf_paragraph_composition:
                if (
                    composition.pdf_formula is not None
                    and not composition.pdf_formula.is_corner_mark
                    and self.is_translatable_formula(
                        composition.pdf_formula,
                    )
                ):
                    # å°†å¯ç¿»è¯‘å…¬å¼è½¬æ¢ä¸ºæ™®é€šæ–‡æœ¬è¡Œ
                    new_line = PdfLine(
                        pdf_character=composition.pdf_formula.pdf_character,
                    )
                    self.update_line_data(new_line)
                    new_compositions.append(PdfParagraphComposition(pdf_line=new_line))
                else:
                    new_compositions.append(composition)

            paragraph.pdf_paragraph_composition = new_compositions

    def _flush_style_group(
        self,
        current_chars: list,
        current_style,
        new_compositions: list,
    ) -> list:
        """Flush the accumulated same-style characters into a composition. Returns empty list."""
        if current_chars:
            new_comp = self._create_same_style_composition(current_chars, current_style)
            new_compositions.append(new_comp)
        return []

    def _accumulate_char_by_style(
        self,
        char,
        current_chars: list,
        current_style,
        new_compositions: list,
    ) -> tuple[list, object]:
        """Add char to current group or flush and start a new group. Returns (chars, style)."""
        char_style = char.pdf_style
        if current_style is None:
            return [char], char_style
        if is_same_style(char_style, current_style):
            current_chars.append(char)
            return current_chars, current_style
        # Style changed â€” flush and start new group
        self._flush_style_group(current_chars, current_style, new_compositions)
        return [char], char_style

    def _regroup_paragraph_by_style(self, paragraph) -> list:
        """Rebuild a paragraph's compositions grouped by text style."""
        new_compositions: list = []
        current_chars: list = []
        current_style = None

        for comp in paragraph.pdf_paragraph_composition:
            if comp.pdf_formula is not None:
                current_chars = self._flush_style_group(
                    current_chars, current_style, new_compositions
                )
                new_compositions.append(comp)
                continue
            if not comp.pdf_line:
                new_compositions.append(comp)
                continue
            for char in comp.pdf_line.pdf_character:
                current_chars, current_style = self._accumulate_char_by_style(
                    char, current_chars, current_style, new_compositions
                )

        self._flush_style_group(current_chars, current_style, new_compositions)
        return new_compositions

    def process_page_styles(self, page: Page):
        """å¤„ç†é¡µé¢ä¸­çš„æ–‡æœ¬æ ·å¼ï¼Œè¯†åˆ«ç›¸åŒæ ·å¼çš„æ–‡æœ¬"""
        if not page.pdf_paragraph:
            return

        for paragraph in page.pdf_paragraph:
            if not paragraph.pdf_paragraph_composition:
                continue

            # è®¡ç®—åŸºå‡†æ ·å¼ï¼ˆé™¤å…¬å¼å¤–æ‰€æœ‰æ–‡å­—æ ·å¼çš„äº¤é›†ï¼‰
            base_style = self._calculate_base_style(paragraph)
            paragraph.pdf_style = base_style

            # é‡æ–°ç»„ç»‡æ®µè½ä¸­çš„æ–‡æœ¬ï¼Œå°†ç›¸åŒæ ·å¼çš„æ–‡æœ¬ç»„åˆåœ¨ä¸€èµ·
            paragraph.pdf_paragraph_composition = self._regroup_paragraph_by_style(
                paragraph
            )

    @staticmethod
    def _collect_line_char_styles(paragraph) -> list:
        """Collect pdf_style for every non-formula line character in the paragraph."""
        styles = []
        for comp in paragraph.pdf_paragraph_composition:
            if isinstance(comp, PdfFormula):
                continue
            if not comp.pdf_line:
                continue
            for char in comp.pdf_line.pdf_character:
                styles.append(char.pdf_style)
        return styles

    def _fill_missing_style_fields(self, base_style, styles: list) -> None:
        """Fill in None font_id / font_size with mode values from the full style list."""
        if base_style.font_id is None:
            base_style.font_id = self._get_mode_value([s.font_id for s in styles])
        if base_style.font_size is None:
            base_style.font_size = self._get_mode_value([s.font_size for s in styles])

    def _calculate_base_style(self, paragraph) -> PdfStyle | None:
        """è®¡ç®—æ®µè½çš„åŸºå‡†æ ·å¼ï¼ˆé™¤å…¬å¼å¤–æ‰€æœ‰æ–‡å­—æ ·å¼çš„äº¤é›†ï¼‰"""
        styles = self._collect_line_char_styles(paragraph)
        if not styles:
            return None

        # è¿”å›žæ‰€æœ‰æ ·å¼çš„äº¤é›†
        base_style = styles[0]
        for style in styles[1:]:
            # æ›´æ–°åŸºå‡†æ ·å¼ä¸ºæ‰€æœ‰æ ·å¼çš„äº¤é›†
            base_style = self._merge_styles(base_style, style)

        # å¦‚æžœ font_id æˆ– font_size ä¸º Noneï¼Œåˆ™ä½¿ç”¨ä¼—æ•°
        self._fill_missing_style_fields(base_style, styles)
        return base_style

    def _get_mode_value(self, values):
        """è®¡ç®—åˆ—è¡¨ä¸­çš„ä¼—æ•°"""
        if not values:
            return None
        from collections import Counter

        counter = Counter(values)
        return counter.most_common(1)[0][0]

    def _merge_styles(self, style1, style2):
        """åˆå¹¶ä¸¤ä¸ªæ ·å¼ï¼Œè¿”å›žå®ƒä»¬çš„äº¤é›†"""
        if style1 is None or style1.font_size is None:
            return style2
        if style2 is None or style2.font_size is None:
            return style1

        return PdfStyle(
            font_id=style1.font_id if style1.font_id == style2.font_id else None,
            font_size=(
                style1.font_size
                if math.fabs(style1.font_size - style2.font_size) < 0.02
                else None
            ),
            graphic_state=self._merge_graphic_states(
                style1.graphic_state,
                style2.graphic_state,
            ),
        )

    def _merge_graphic_states(self, state1, state2):
        """åˆå¹¶ä¸¤ä¸ª GraphicStateï¼Œè¿”å›žå®ƒä»¬çš„äº¤é›†"""
        if state1 is None:
            return state2
        if state2 is None:
            return state1

        return GraphicState(
            passthrough_per_char_instruction=(
                state1.passthrough_per_char_instruction
                if state1.passthrough_per_char_instruction
                == state2.passthrough_per_char_instruction
                else None
            ),
        )

    def _create_same_style_composition(
        self,
        chars: list[PdfCharacter],
        style,
    ) -> PdfParagraphComposition | None:
        """åˆ›å»ºå…·æœ‰ç›¸åŒæ ·å¼çš„æ–‡æœ¬ç»„åˆ"""
        if not chars:
            return None

        # è®¡ç®—è¾¹ç•Œæ¡†
        min_x = min(char.visual_bbox.box.x for char in chars)
        min_y = min(char.visual_bbox.box.y for char in chars)
        max_x = max(char.visual_bbox.box.x2 for char in chars)
        max_y = max(char.visual_bbox.box.y2 for char in chars)
        box = Box(min_x, min_y, max_x, max_y)

        return PdfParagraphComposition(
            pdf_same_style_characters=PdfSameStyleCharacters(
                box=box,
                pdf_style=style,
                pdf_character=chars,
            ),
        )

    def _find_left_char_for_formula(
        self, formula, compositions: list, formula_idx: int
    ):
        """Return the nearest text character to the left of *formula* on the same line."""
        for j in range(formula_idx - 1, -1, -1):
            comp = compositions[j]
            if comp.pdf_line:
                for char in reversed(comp.pdf_line.pdf_character):
                    if not char.pdf_character_id:
                        continue
                    iou = calculate_y_true_iou_for_boxes(formula.box, char.box)
                    if iou > 0.6:
                        return char, iou
            else:
                # No pdf_line in this composition; stop searching
                pass
            break
        return None, 0

    def _find_right_char_for_formula(
        self, formula, compositions: list, formula_idx: int
    ):
        """Return the nearest text character to the right of *formula* on the same line."""
        for j in range(formula_idx + 1, len(compositions)):
            comp = compositions[j]
            if comp.pdf_line:
                for char in comp.pdf_line.pdf_character:
                    if not char.pdf_character_id:
                        continue
                    iou = calculate_y_true_iou_for_boxes(formula.box, char.box)
                    if iou > 0.6:
                        return char, iou
            else:
                # No pdf_line in this composition; stop searching
                pass
            break
        return None, 0

    @staticmethod
    def _select_dominant_anchor(left_char, right_char, left_iou, right_iou):
        """Keep only the anchor with the higher IOU when both exist."""
        if not (left_char and right_char):
            return left_char, right_char
        if left_iou < right_iou:
            return None, right_char
        if right_iou < left_iou:
            return left_char, None
        # Equal IOUs â€” keep both
        return left_char, right_char

    @staticmethod
    def _compute_x_offset(formula, left_char) -> float:
        """Return x_offset for formula relative to left anchor (0 if out of range)."""
        if not left_char:
            return 0.0
        offset = formula.box.x - left_char.box.x2
        if abs(offset) < 0.1 or offset > 10 or offset < -5:
            return 0.0
        return offset

    @staticmethod
    def _compute_y_offset(formula, left_char, right_char) -> float:
        """Return y_offset for formula relative to nearest anchor."""
        if left_char:
            offset = formula.box.y - left_char.box.y
        elif right_char:
            offset = formula.box.y - right_char.box.y
        else:
            return 0.0
        return 0.0 if abs(offset) < 0.1 else offset

    def _apply_formula_offsets(
        self, formula, left_char, right_char, left_iou, right_iou
    ) -> None:
        """Calculate and assign x_offset and y_offset on *formula* from its neighbours."""
        # If both text segments exist, keep the one with higher IOU
        left_char, right_char = self._select_dominant_anchor(
            left_char, right_char, left_iou, right_iou
        )
        formula.x_offset = self._compute_x_offset(formula, left_char)
        formula.y_offset = self._compute_y_offset(formula, left_char, right_char)

    def process_page_offsets(self, page: Page):
        """è®¡ç®—å…¬å¼çš„ x å’Œ y åç§»é‡"""
        if not page.pdf_paragraph:
            return

        for paragraph in page.pdf_paragraph:
            if paragraph.debug_id is None:
                continue
            if not paragraph.pdf_paragraph_composition:
                continue

            for i, composition in enumerate(paragraph.pdf_paragraph_composition):
                if not composition.pdf_formula:
                    continue

                formula = composition.pdf_formula
                comps = paragraph.pdf_paragraph_composition

                left_char, left_iou = self._find_left_char_for_formula(
                    formula, comps, i
                )
                right_char, right_iou = self._find_right_char_for_formula(
                    formula, comps, i
                )
                self._apply_formula_offsets(
                    formula, left_char, right_char, left_iou, right_iou
                )

    def calculate_line_spacing(self, paragraph) -> float:
        """è®¡ç®—æ®µè½ä¸­çš„å¹³å‡è¡Œé—´è·"""
        if not paragraph.pdf_paragraph_composition:
            return 0.0

        # æ”¶é›†æ‰€æœ‰æ–‡æœ¬è¡Œçš„ y åæ ‡
        line_y_positions = []
        for comp in paragraph.pdf_paragraph_composition:
            if comp.pdf_line:
                line_y_positions.append(comp.pdf_line.box.y)

        if len(line_y_positions) < 2:
            return 10.0  # å¦‚æžœåªæœ‰ä¸€è¡Œæˆ–æ²¡æœ‰è¡Œï¼Œè¿”å›žä¸€ä¸ªé»˜è®¤å€¼

        # è®¡ç®—ç›¸é‚»è¡Œä¹‹é—´çš„ y å·®å€¼
        line_spacings = []
        for i in range(len(line_y_positions) - 1):
            spacing = abs(line_y_positions[i] - line_y_positions[i + 1])
            if spacing > 0:  # å¿½ç•¥é‡å çš„è¡Œ
                line_spacings.append(spacing)

        if not line_spacings:
            return 10.0  # å¦‚æžœæ²¡æœ‰æœ‰æ•ˆçš„è¡Œé—´è·ï¼Œè¿”å›žé»˜è®¤å€¼

        # ä½¿ç”¨ä¸­ä½æ•°æ¥é¿å…å¼‚å¸¸å€¼çš„å½±å“
        median_spacing = sorted(line_spacings)[len(line_spacings) // 2]
        return median_spacing

    def create_composition(
        self,
        chars: list[PdfCharacter],
        is_formula: bool,
        is_corner_mark: bool = False,
    ) -> PdfParagraphComposition:
        if is_formula:
            formula = PdfFormula(pdf_character=chars)
            formula.is_corner_mark = is_corner_mark
            self.update_formula_data(formula)
            return PdfParagraphComposition(pdf_formula=formula)
        else:
            new_line = PdfLine(pdf_character=chars)
            self.update_line_data(new_line)
            return PdfParagraphComposition(pdf_line=new_line)

    def is_translatable_formula(self, formula: PdfFormula) -> bool:
        """åˆ¤æ–­å…¬å¼æ˜¯å¦åªåŒ…å«éœ€è¦æ­£å¸¸ç¿»è¯‘çš„å­—ç¬¦ï¼ˆæ•°å­—ã€ç©ºæ ¼å’Œè‹±æ–‡é€—å·ï¼‰"""
        if all(char.formula_layout_id for char in formula.pdf_character):
            return False

        text = "".join(char.char_unicode for char in formula.pdf_character)
        if formula.y_offset > 0.1:
            return False
        return bool(re.match(r"^[0-9, .]+$", text))

    def should_split_formula(self, formula: PdfFormula) -> bool:
        """åˆ¤æ–­å…¬å¼æ˜¯å¦éœ€è¦æŒ‰é€—å·æ‹†åˆ†ï¼ˆåŒ…å«é€—å·ä¸”æœ‰å…¶ä»–ç‰¹æ®Šç¬¦å·ï¼‰"""

        if all(x.formula_layout_id for x in formula.pdf_character):
            return False

        text = "".join(char.char_unicode for char in formula.pdf_character)
        # å¿…é¡»åŒ…å«é€—å·
        if "," not in text:
            return False
        # æ£€æŸ¥æ˜¯å¦åŒ…å«é™¤äº†æ•°å­—å’Œ [] ä¹‹å¤–çš„å…¶ä»–ç¬¦å·
        text_without_basic = re.sub(r"[0-9\[\],\s]", "", text)
        return bool(text_without_basic)

    def split_formula_by_comma(
        self,
        formula: PdfFormula,
    ) -> list[tuple[list[PdfCharacter], PdfCharacter]]:
        """æŒ‰é€—å·æ‹†åˆ†å…¬å¼å­—ç¬¦ï¼Œè¿”å›ž (å­—ç¬¦ç»„ï¼Œé€—å·å­—ç¬¦) çš„åˆ—è¡¨ï¼Œæœ€åŽä¸€ç»„çš„é€—å·å­—ç¬¦ä¸º Noneã€‚
        åªæœ‰ä¸åœ¨æ‹¬å·å†…çš„é€—å·æ‰ä¼šè¢«ç”¨ä½œåˆ†éš”ç¬¦ã€‚æ”¯æŒçš„æ‹¬å·å¯¹åŒ…æ‹¬ï¼š
        - (cid:8) å’Œ (cid:9)
        - ( å’Œ )
        - (cid:16) å’Œ (cid:17)
        """
        result = []
        current_chars = []
        bracket_level = 0  # è·Ÿè¸ªæ‹¬å·çš„å±‚æ•°

        for char in formula.pdf_character:
            # æ£€æŸ¥æ˜¯å¦æ˜¯å·¦æ‹¬å·
            if char.char_unicode in LEFT_BRACKET:
                bracket_level += 1
                current_chars.append(char)
            # æ£€æŸ¥æ˜¯å¦æ˜¯å³æ‹¬å·
            elif char.char_unicode in RIGHT_BRACKET:
                bracket_level = max(0, bracket_level - 1)  # é˜²æ­¢æ‹¬å·ä¸åŒ¹é…çš„æƒ…å†µ
                current_chars.append(char)
            # æ£€æŸ¥æ˜¯å¦æ˜¯é€—å·ï¼Œä¸”ä¸åœ¨æ‹¬å·å†…
            elif char.char_unicode == "," and bracket_level == 0:
                if current_chars:
                    result.append((current_chars, char))
                    current_chars = []
            else:
                current_chars.append(char)

        if current_chars:
            result.append((current_chars, None))  # æœ€åŽä¸€ç»„æ²¡æœ‰é€—å·

        return result

    def merge_formulas(self, formula1: PdfFormula, formula2: PdfFormula) -> PdfFormula:
        """åˆå¹¶ä¸¤ä¸ªå…¬å¼ï¼Œä¿æŒå­—ç¬¦çš„ç›¸å¯¹ä½ç½®"""
        # åˆå¹¶æ‰€æœ‰å­—ç¬¦
        all_chars = formula1.pdf_character + formula2.pdf_character
        # ç»§æ‰¿ç¬¬ä¸€ä¸ªå…¬å¼çš„è¡Œ ID
        merged_formula = PdfFormula(pdf_character=all_chars, line_id=formula1.line_id)
        self.update_formula_data(merged_formula)
        return merged_formula

    def is_x_axis_contained(self, box1: Box, box2: Box) -> bool:
        """åˆ¤æ–­ box1 çš„ x è½´æ˜¯å¦å®Œå…¨åŒ…å«åœ¨ box2 çš„ x è½´å†…ï¼Œæˆ–åä¹‹"""
        return (box1.x >= box2.x and box1.x2 <= box2.x2) or (
            box2.x >= box1.x and box2.x2 <= box1.x2
        )

    def has_y_intersection(self, box1: Box, box2: Box) -> bool:
        """åˆ¤æ–­ä¸¤ä¸ª box çš„ y è½´æ˜¯å¦æœ‰äº¤é›†"""
        tolerance = 1.0
        return not (box1.y2 < box2.y - tolerance or box2.y2 < box1.y - tolerance)

    def is_x_axis_adjacent(self, box1: Box, box2: Box, tolerance: float = 2.0) -> bool:
        """åˆ¤æ–­ä¸¤ä¸ª box åœ¨ x è½´ä¸Šæ˜¯å¦ç›¸é‚»æˆ–æœ‰äº¤é›†"""
        # æ£€æŸ¥æ˜¯å¦æœ‰äº¤é›†
        has_intersection = not (box1.x2 < box2.x or box2.x2 < box1.x)

        # æ£€æŸ¥ box1 æ˜¯å¦åœ¨ box2 å·¦è¾¹ä¸”ç›¸é‚»
        left_adjacent = abs(box1.x2 - box2.x) <= tolerance
        # æ£€æŸ¥ box2 æ˜¯å¦åœ¨ box1 å·¦è¾¹ä¸”ç›¸é‚»
        right_adjacent = abs(box2.x2 - box1.x) <= tolerance

        return has_intersection or left_adjacent or right_adjacent

    def calculate_y_iou(self, box1: Box, box2: Box) -> float:
        """è®¡ç®—ä¸¤ä¸ª box åœ¨ y è½´ä¸Šçš„ IOU (Intersection over Union)"""
        # è®¡ç®—äº¤é›†
        intersection_start = max(box1.y, box2.y)
        intersection_end = min(box1.y2, box2.y2)
        intersection_length = max(0, intersection_end - intersection_start)

        # è®¡ç®—å¹¶é›†
        box1_height = box1.y2 - box1.y
        box2_height = box2.y2 - box2.y
        union_length = box1_height + box2_height - intersection_length

        # é¿å…é™¤é›¶é”™è¯¯
        if union_length <= 0:
            return 0.0

        return intersection_length / union_length

    def _should_merge_formula_pair(
        self,
        formula1,
        formula2,
        comp_idx_delta: int,
        page: Page,
    ) -> bool:
        """Return True if *formula1* and *formula2* satisfy the merge criteria."""
        if formula1.line_id != formula2.line_id:
            return False
        if comp_idx_delta == 1 and (
            (
                self.is_x_axis_contained(formula1.box, formula2.box)
                and self.has_y_intersection(formula1.box, formula2.box)
            )
            or (
                self.is_x_axis_adjacent(formula1.box, formula2.box)
                and self.calculate_y_iou(formula1.box, formula2.box) > 0.5
            )
        ):
            return True
        if self._have_same_layout_ids(formula1, formula2, page):
            return True
        if calculate_iou_for_boxes(formula1.box, formula2.box) > 0.8:
            return True
        if calculate_iou_for_boxes(formula2.box, formula1.box) > 0.8:
            return True
        return False

    def _find_merge_candidate(self, paragraph, i: int, page: Page):
        """Find the first formula after index i that can be merged with composition i.

        Returns (j, formula2) if a candidate is found, else (None, None).
        """
        comp1 = paragraph.pdf_paragraph_composition[i]
        if comp1.pdf_formula is None:
            return None, None
        formula1 = comp1.pdf_formula
        for j in range(i + 1, len(paragraph.pdf_paragraph_composition)):
            comp2 = paragraph.pdf_paragraph_composition[j]
            if comp2.pdf_formula is None:
                continue
            if self._should_merge_formula_pair(
                formula1, comp2.pdf_formula, j - i, page
            ):
                return j, comp2.pdf_formula
        return None, None

    def _merge_paragraph_formulas_once(self, paragraph, page: Page) -> bool:
        """Perform one pass of formula merging in a paragraph. Returns True if any merge occurred."""
        for i in range(len(paragraph.pdf_paragraph_composition)):
            j, formula2 = self._find_merge_candidate(paragraph, i, page)
            if j is not None:
                formula1 = paragraph.pdf_paragraph_composition[i].pdf_formula
                merged_formula = self.merge_formulas(formula1, formula2)
                paragraph.pdf_paragraph_composition[i] = PdfParagraphComposition(
                    pdf_formula=merged_formula
                )
                del paragraph.pdf_paragraph_composition[j]
                return True
        return False

    def merge_overlapping_formulas(self, page: Page):
        """
        åˆå¹¶ç¬¦åˆä»¥ä¸‹æ¡ä»¶çš„å…¬å¼ï¼š
        1. x è½´é‡å ä¸” y è½´æœ‰äº¤é›†çš„ç›¸é‚»å…¬å¼ï¼Œæˆ–è€…
        2. x è½´ç›¸é‚»ä¸” y è½´ IOU > 0.5 çš„ç›¸é‚»å…¬å¼ï¼Œæˆ–è€…
        3. æ‰€æœ‰å­—ç¬¦çš„ layout id éƒ½ç›¸åŒçš„ç›¸é‚»å…¬å¼ï¼Œæˆ–è€…
        4. ä»»æ„ä¸¤ä¸ªå…¬å¼çš„ IOU > 0.8
        è§’æ ‡å¯èƒ½ä¼šè¢«è¯†åˆ«æˆå•ç‹¬çš„å…¬å¼ï¼Œéœ€è¦åˆå¹¶
        """
        if not page.pdf_paragraph:
            return

        for paragraph in page.pdf_paragraph:
            if not paragraph.pdf_paragraph_composition:
                continue

            merged = self._merge_paragraph_formulas_once(paragraph, page)
            while merged:
                merged = self._merge_paragraph_formulas_once(paragraph, page)

    def _have_same_layout_ids(
        self, formula1: PdfFormula, formula2: PdfFormula, _page: Page
    ) -> bool:
        """æ£€æŸ¥ä¸¤ä¸ªå…¬å¼çš„æ‰€æœ‰å­—ç¬¦æ˜¯å¦å…·æœ‰ç›¸åŒçš„ layout id"""
        # èŽ·å– formula1 ä¸­æ‰€æœ‰å­—ç¬¦çš„ layout id
        formula1_layout_ids = set()
        for char in formula1.pdf_character:
            if char.char_unicode == " ":
                continue
            layout = char.formula_layout_id
            if layout:
                formula1_layout_ids.add(layout)

        # èŽ·å– formula2 ä¸­æ‰€æœ‰å­—ç¬¦çš„ layout id
        formula2_layout_ids = set()
        for char in formula2.pdf_character:
            if char.char_unicode == " ":
                continue
            layout = char.formula_layout_id
            if layout:
                formula2_layout_ids.add(layout)

        # å¦‚æžœä»»ä¸€å…¬å¼æ²¡æœ‰æœ‰æ•ˆçš„ layout idï¼Œåˆ™ä¸åˆå¹¶
        if not (len(formula1_layout_ids) == len(formula2_layout_ids) == 1):
            return False

        # æ£€æŸ¥ä¸¤ä¸ªå…¬å¼çš„ layout id é›†åˆæ˜¯å¦ç›¸åŒ
        return formula1_layout_ids == formula2_layout_ids

    def _expand_comma_formula_composition(
        self, composition, new_compositions: list
    ) -> None:
        """Split a formula by commas and append the resulting compositions."""
        char_groups = self.split_formula_by_comma(composition.pdf_formula)
        for chars, comma in char_groups:
            if not chars:  # å¿½ç•¥ç©ºç»„ï¼ˆè¿žç»­çš„é€—å·ï¼‰
                continue
            # ç»§æ‰¿åŽŸå…¬å¼çš„è¡Œ ID
            formula = PdfFormula(
                pdf_character=chars,
                line_id=composition.pdf_formula.line_id,
            )
            self.update_formula_data(formula)
            new_compositions.append(PdfParagraphComposition(pdf_formula=formula))
            # å¦‚æžœæœ‰é€—å·ï¼Œæ·»åŠ ä¸ºæ–‡æœ¬è¡Œ
            if comma:
                comma_line = PdfLine(pdf_character=[comma])
                self.update_line_data(comma_line)
                new_compositions.append(PdfParagraphComposition(pdf_line=comma_line))

    def _split_paragraph_comma_formulas(self, paragraph) -> list:
        """Return a new composition list with splittable comma-formulas expanded."""
        new_compositions: list = []
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_formula is not None and self.should_split_formula(
                composition.pdf_formula,
            ):
                # æŒ‰é€—å·æ‹†åˆ†å…¬å¼
                self._expand_comma_formula_composition(composition, new_compositions)
            else:
                new_compositions.append(composition)
        return new_compositions

    def process_comma_formulas(self, page: Page):
        """å¤„ç†åŒ…å«é€—å·çš„å¤æ‚å…¬å¼ï¼Œå°†å…¶æŒ‰é€—å·æ‹†åˆ†"""
        if not page.pdf_paragraph:
            return

        for paragraph in page.pdf_paragraph:
            if not paragraph.pdf_paragraph_composition:
                continue
            paragraph.pdf_paragraph_composition = self._split_paragraph_comma_formulas(
                paragraph
            )

    def remove_non_formula_lines_from_paragraphs(self, page: Page):
        """Remove non-formula lines from paragraphs.

        This method processes curves that remain in page.pdf_curve after
        collect_contained_elements() has assigned formula-related curves to formulas.
        All remaining curves are non-formula lines, but we need to be careful
        not to remove lines from figure/table areas.

        Args:
            page: The page to process
        """
        if not page.pdf_curve:
            return

        # Build layout index for efficient spatial queries
        layout_index, layout_map = build_layout_index(page)

        curves_to_remove = []

        # Get configuration thresholds
        protection_threshold = getattr(
            self.translation_config, "figure_table_protection_threshold", 0.9
        )
        overlap_threshold = getattr(
            self.translation_config, "non_formula_line_iou_threshold", 0.9
        )

        for curve in page.pdf_curve:
            # Skip if curve is in figure/table layout areas
            if is_curve_in_figure_table_layout(
                curve, layout_index, layout_map, protection_threshold
            ):
                continue

            # Only remove if curve overlaps with text paragraph areas
            if is_curve_overlapping_with_paragraphs(
                curve, page.pdf_paragraph, overlap_threshold
            ):
                curves_to_remove.append(curve)

        # Remove identified curves
        removed_count = 0
        for curve in curves_to_remove:
            if curve in page.pdf_curve:
                page.pdf_curve.remove(curve)
                removed_count += 1

        if removed_count > 0:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Removed {removed_count} non-formula lines from paragraphs")
