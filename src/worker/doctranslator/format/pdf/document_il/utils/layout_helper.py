import logging
import math
import re
import unicodedata
from typing import Literal

import regex
from pymupdf import Font

from src.worker.doctranslator.format.pdf.document_il import GraphicState
from src.worker.doctranslator.format.pdf.document_il import il_version_1
from src.worker.doctranslator.format.pdf.document_il.il_version_1 import Box
from src.worker.doctranslator.format.pdf.document_il.il_version_1 import PdfCharacter
from src.worker.doctranslator.format.pdf.document_il.il_version_1 import PdfParagraph
from src.worker.doctranslator.format.pdf.document_il.il_version_1 import (
    PdfParagraphComposition,
)

logger = logging.getLogger(__name__)
HEIGHT_NOT_USFUL_CHAR_IN_CHAR = (None,)


LEFT_BRACKET = ("(cid:8)", "(", "(cid:16)", "{", "[", "(cid:104)", "(cid:2)")
RIGHT_BRACKET = ("(cid:9)", ")", "(cid:17)", "}", "]", "(cid:105)", "(cid:3)")

BULLET_POINT_PATTERN = re.compile(
    r"[■•⚫⬤◆◇○●◦‣⁃▪▫∗†‡¹²³⁴⁵⁶⁷⁸⁹⁰₁₂₃₄₅₆₇₈₉₀ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ¶※⁑⁂⁕⁎⁜❧☙⁋‖‽·]"
)


def is_bullet_point(char: PdfCharacter) -> bool:
    """Check if the character is a bullet point.

    Args:
        char: The character to check

    Returns:
        bool: True if the character is a bullet point
    """
    is_bullet = bool(BULLET_POINT_PATTERN.match(char.char_unicode))
    return is_bullet


def calculate_box_iou(box1: Box, box2: Box) -> float:
    """Calculate the Intersection over Union (IOU) between two boxes.

    Args:
        box1: First box
        box2: Second box

    Returns:
        float: IOU value between 0 and 1
    """
    if box1 is None or box2 is None:
        return 0.0

    # Calculate intersection
    x_left = max(box1.x, box2.x)
    y_top = max(box1.y, box2.y)
    x_right = min(box1.x2, box2.x2)
    y_bottom = min(box1.y2, box2.y2)

    # Check if there's no intersection
    if x_left >= x_right or y_top >= y_bottom:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas of both boxes
    box1_area = (box1.x2 - box1.x) * (box1.y2 - box1.y)
    box2_area = (box2.x2 - box2.x) * (box2.y2 - box2.y)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Avoid division by zero
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def formular_height_ignore_char(char: PdfCharacter):
    return (
        char.pdf_character_id is None
        or char.char_unicode in HEIGHT_NOT_USFUL_CHAR_IN_CHAR
    )


def box_to_tuple(box: Box) -> tuple[float, float, float, float]:
    """Converts a Box object to a tuple of its coordinates."""
    if box is None:
        return (0, 0, 0, 0)
    return (box.x, box.y, box.x2, box.y2)


class Layout:
    def __init__(self, layout_id, name):
        self.id = layout_id
        self.name = name

    @staticmethod
    def is_newline(prev_char: PdfCharacter, curr_char: PdfCharacter) -> bool:
        # å¦‚æžœæ²¡æœ‰å‰ä¸€ä¸ªå­—ç¬¦ï¼Œä¸æ˜¯æ¢è¡Œ
        if prev_char is None:
            return False

        # å¦‚æžœå½“å‰å­—ç¬¦çš„ y åæ ‡æ˜Žæ˜¾ä½ŽäºŽå‰ä¸€ä¸ªå­—ç¬¦ï¼Œè¯´æ˜Žæ¢è¡Œäº†
        # è¿™é‡Œä½¿ç”¨å­—ç¬¦é«˜åº¦çš„ä¸€åŠä½œä¸ºé˜ˆå€¼
        char_width = max(
            curr_char.box.x2 - curr_char.box.x,
            prev_char.box.x2 - prev_char.box.x,
        )
        should_new_line = (
            curr_char.box.y2 < prev_char.box.y
            or curr_char.box.x2 < prev_char.box.x - char_width * 10
        )
        if should_new_line and (
            formular_height_ignore_char(curr_char)
            or formular_height_ignore_char(prev_char)
        ):
            return False
        return should_new_line


def _composition_length_except(
    composition: "PdfParagraphComposition",
    paragraph: "PdfParagraph",
    except_chars: str,
    font: Font,
) -> float:
    """Return the pixel width contributed by *composition*, skipping *except_chars*."""
    if composition.pdf_character:
        return composition.pdf_character[0].box.x2 - composition.pdf_character[0].box.x
    if composition.pdf_same_style_characters:
        return sum(
            pdf_char.box.x2 - pdf_char.box.x
            for pdf_char in composition.pdf_same_style_characters.pdf_character
            if pdf_char.char_unicode not in except_chars
        )
    if composition.pdf_same_style_unicode_characters:
        return sum(
            font.char_lengths(
                char_unicode,
                composition.pdf_same_style_unicode_characters.pdf_style.font_size,
            )[0]
            for char_unicode in composition.pdf_same_style_unicode_characters.unicode
            if char_unicode not in except_chars
        )
    if composition.pdf_line:
        return sum(
            pdf_char.box.x2 - pdf_char.box.x
            for pdf_char in composition.pdf_line.pdf_character
            if pdf_char.char_unicode not in except_chars
        )
    if composition.pdf_formula:
        return composition.pdf_formula.box.x2 - composition.pdf_formula.box.x
    logger.error(
        f"Unknown composition type. "
        f"Composition: {composition}. "
        f"Paragraph: {paragraph}. ",
    )
    return 0


def get_paragraph_length_except(
    paragraph: PdfParagraph,
    except_chars: str,
    font: Font,
) -> int:
    return sum(
        _composition_length_except(composition, paragraph, except_chars, font)
        for composition in paragraph.pdf_paragraph_composition
    )


def get_paragraph_unicode(paragraph: PdfParagraph) -> str:
    chars = []
    for composition in paragraph.pdf_paragraph_composition:
        if composition.pdf_line:
            chars.extend(composition.pdf_line.pdf_character)
        elif composition.pdf_same_style_characters:
            chars.extend(composition.pdf_same_style_characters.pdf_character)
        elif composition.pdf_same_style_unicode_characters:
            chars.extend(composition.pdf_same_style_unicode_characters.unicode)
        elif composition.pdf_formula:
            chars.extend(composition.pdf_formula.pdf_character)
        elif composition.pdf_character:
            chars.append(composition.pdf_character)
        else:
            logger.error(
                f"Unknown composition type. "
                f"Composition: {composition}. "
                f"Paragraph: {paragraph}. ",
            )
    return get_char_unicode_string(chars)


SPACE_REGEX = regex.compile(r"\s+", regex.UNICODE)


def _compute_median_char_distance(chars: list) -> float:
    """Return the second-smallest distinct positive inter-character gap (or 1 if none)."""
    distances = []
    for i in range(len(chars) - 1):
        if not (
            isinstance(chars[i], PdfCharacter)
            and isinstance(chars[i + 1], PdfCharacter)
        ):
            continue
        distance = chars[i + 1].box.x - chars[i].box.x2
        if distance > 1:
            distances.append(distance)
    distinct_distances = sorted(set(distances))
    if not distinct_distances:
        return 1
    if len(distinct_distances) == 1:
        return distinct_distances[0]
    return distinct_distances[1]


def _normalize_char_unicode(char: PdfCharacter) -> str:
    """Return NFKC-normalised, space-collapsed unicode for *char*."""
    return regex.sub(r"\s+", " ", unicodedata.normalize("NFKC", char.char_unicode))


def _needs_space_after(chars: list, i: int, median_distance: float) -> bool:
    """Return True if a space should be inserted after chars[i]."""
    if chars[i].char_unicode == " ":
        return False
    if i >= len(chars) - 1:
        return False
    if not isinstance(chars[i + 1], PdfCharacter):
        return False
    distance = chars[i + 1].box.x - chars[i].box.x2
    return distance >= median_distance or Layout.is_newline(chars[i], chars[i + 1])


def get_char_unicode_string(chars: list[PdfCharacter | str]) -> str:
    """
    å°†å­—ç¬¦åˆ—è¡¨è½¬æ¢ä¸º Unicode å­—ç¬¦ä¸²ï¼Œæ ¹æ®å­—ç¬¦é—´è·è‡ªåŠ¨æ’å…¥ç©ºæ ¼ã€‚
    æœ‰äº› PDF ä¸ä¼šæ˜¾å¼ç¼–ç ç©ºæ ¼ï¼Œè¿™æ—¶éœ€è¦æ ¹æ®é—´è·è‡ªåŠ¨æ’å…¥ç©ºæ ¼ã€‚

    Args:
        chars: å­—ç¬¦åˆ—è¡¨ï¼Œå¯ä»¥æ˜¯ PdfCharacter å¯¹è±¡æˆ–å­—ç¬¦ä¸²

    Returns:
        str: å¤„ç†åŽçš„ Unicode å­—ç¬¦ä¸²
    """
    median_distance = _compute_median_char_distance(chars)

    # æž„å»º unicode å­—ç¬¦ä¸²ï¼Œæ ¹æ®é—´è·æ’å…¥ç©ºæ ¼
    unicode_chars = []
    for i in range(len(chars)):
        # å¦‚æžœä¸æ˜¯å­—ç¬¦å¯¹è±¡ï¼Œç›´æŽ¥æ·»åŠ ï¼Œä¸€èˆ¬æ¥è¯´è¿™ä¸ªæ—¶å€™ chars[i] æ˜¯å­—ç¬¦ä¸²
        if not isinstance(chars[i], PdfCharacter):
            unicode_chars.append(chars[i])
            continue

        # use unicode regex to replace all space with " "
        unicode_chars.append(_normalize_char_unicode(chars[i]))

        # å¦‚æžœä¸¤ä¸ªå­—ç¬¦éƒ½æ˜¯ PdfCharacterï¼Œæ£€æŸ¥é—´è·
        if _needs_space_after(chars, i, median_distance):
            unicode_chars.append(" ")  # æ·»åŠ ç©ºæ ¼

    result = "".join(unicode_chars)
    # use unicode regex to replace all space with " "
    normalize = unicodedata.normalize("NFKC", result)
    result = SPACE_REGEX.sub(" ", normalize).strip()
    return result


def _composition_max_height(
    composition: "PdfParagraphComposition", paragraph: "PdfParagraph"
) -> float:
    """Return the maximum character/formula height within a single *composition*."""
    if composition.pdf_character:
        return composition.pdf_character[0].box.y2 - composition.pdf_character[0].box.y
    if composition.pdf_same_style_characters:
        return (
            max(
                (pdf_char.box.y2 - pdf_char.box.y)
                for pdf_char in composition.pdf_same_style_characters.pdf_character
            )
            if composition.pdf_same_style_characters.pdf_character
            else 0.0
        )
    if composition.pdf_same_style_unicode_characters:
        # å¯¹äºŽçº¯ Unicode å­—ç¬¦ï¼Œæˆ‘ä»¬ä½¿ç”¨å…¶æ ·å¼ä¸­çš„å­—ä½“å¤§å°ä½œä¸ºé«˜åº¦ä¼°è®¡
        return composition.pdf_same_style_unicode_characters.pdf_style.font_size
    if composition.pdf_line:
        return (
            max(
                (pdf_char.box.y2 - pdf_char.box.y)
                for pdf_char in composition.pdf_line.pdf_character
            )
            if composition.pdf_line.pdf_character
            else 0.0
        )
    if composition.pdf_formula:
        return composition.pdf_formula.box.y2 - composition.pdf_formula.box.y
    logger.error(
        f"Unknown composition type. "
        f"Composition: {composition}. "
        f"Paragraph: {paragraph}. ",
    )
    return 0.0


def get_paragraph_max_height(paragraph: PdfParagraph) -> float:
    """
    èŽ·å–æ®µè½ä¸­æœ€é«˜çš„æŽ’ç‰ˆå•å…ƒé«˜åº¦ã€‚

    Args:
        paragraph: PDF æ®µè½å¯¹è±¡

    Returns:
        float: æœ€å¤§é«˜åº¦å€¼
    """
    max_height = 0.0
    for composition in paragraph.pdf_paragraph_composition:
        if composition is None:
            continue
        max_height = max(max_height, _composition_max_height(composition, paragraph))
    return max_height


def is_same_style(style1, style2) -> bool:
    """åˆ¤æ–­ä¸¤ä¸ªæ ·å¼æ˜¯å¦ç›¸åŒ"""
    if style1 is None or style2 is None:
        return style1 is style2

    return (
        style1.font_id == style2.font_id
        and math.fabs(style1.font_size - style2.font_size) < 0.02
        and is_same_graphic_state(style1.graphic_state, style2.graphic_state)
    )


def is_same_style_except_size(style1, style2) -> bool:
    """åˆ¤æ–­ä¸¤ä¸ªæ ·å¼æ˜¯å¦ç›¸åŒ"""
    if style1 is None or style2 is None:
        return style1 is style2

    return (
        style1.font_id == style2.font_id
        and 0.7 < math.fabs(style1.font_size / style2.font_size) < 1.3
        and is_same_graphic_state(style1.graphic_state, style2.graphic_state)
    )


def is_same_style_except_font(style1, style2) -> bool:
    """åˆ¤æ–­ä¸¤ä¸ªæ ·å¼æ˜¯å¦ç›¸åŒ"""
    if style1 is None or style2 is None:
        return style1 is style2

    return math.fabs(
        style1.font_size - style2.font_size,
    ) < 0.02 and is_same_graphic_state(style1.graphic_state, style2.graphic_state)


def is_same_graphic_state(state1: GraphicState, state2: GraphicState) -> bool:
    """åˆ¤æ–­ä¸¤ä¸ª GraphicState æ˜¯å¦ç›¸åŒ"""
    if state1 is None or state2 is None:
        return state1 is state2

    return (
        state1.passthrough_per_char_instruction
        == state2.passthrough_per_char_instruction
    )


def _add_intra_composition_spaces(paragraph: PdfParagraph) -> None:
    """Insert space dummies within each composition of *paragraph*."""
    for composition in paragraph.pdf_paragraph_composition:
        if composition.pdf_line:
            _add_space_dummy_chars_to_list(composition.pdf_line.pdf_character)
        elif composition.pdf_same_style_characters:
            _add_space_dummy_chars_to_list(
                composition.pdf_same_style_characters.pdf_character
            )
        elif composition.pdf_same_style_unicode_characters:
            # å¯¹äºŽ unicode å­—ç¬¦ï¼Œä¸éœ€è¦å¤„ç†ã€‚
            # è¿™ç§ç±»åž‹åªä¼šå‡ºçŽ°åœ¨ç¿»è¯‘å¥½çš„ç»“æžœä¸­
            continue
        elif composition.pdf_formula:
            _add_space_dummy_chars_to_list(composition.pdf_formula.pdf_character)


def _append_space_char_to_composition(
    comp: "PdfParagraphComposition", space_char: PdfCharacter
) -> None:
    """Append *space_char* to the appropriate character list in *comp*."""
    if comp.pdf_line:
        comp.pdf_line.pdf_character.append(space_char)
    elif comp.pdf_same_style_characters:
        comp.pdf_same_style_characters.pdf_character.append(space_char)
    elif comp.pdf_formula:
        comp.pdf_formula.pdf_character.append(space_char)


def _add_inter_composition_spaces(paragraph: PdfParagraph) -> None:
    """Insert space dummies between adjacent compositions of *paragraph*."""
    for i in range(len(paragraph.pdf_paragraph_composition) - 1):
        curr_comp = paragraph.pdf_paragraph_composition[i]
        next_comp = paragraph.pdf_paragraph_composition[i + 1]

        curr_last_char = _get_last_char_from_composition(curr_comp)
        if not curr_last_char:
            continue
        next_first_char = _get_first_char_from_composition(next_comp)
        if not next_first_char:
            continue

        distance = next_first_char.box.x - curr_last_char.box.x2
        if distance <= 1:
            continue

        space_box = Box(
            x=curr_last_char.box.x2,
            y=curr_last_char.box.y,
            x2=curr_last_char.box.x2 + distance,
            y2=curr_last_char.box.y2,
        )
        space_char = PdfCharacter(
            pdf_style=curr_last_char.pdf_style,
            box=space_box,
            char_unicode=" ",
            scale=curr_last_char.scale,
            advance=space_box.x2 - space_box.x,
            visual_bbox=il_version_1.VisualBbox(box=space_box),
        )
        _append_space_char_to_composition(curr_comp, space_char)


def add_space_dummy_chars(paragraph: PdfParagraph) -> None:
    """
    åœ¨ PDF æ®µè½ä¸­æ·»åŠ è¡¨ç¤ºç©ºæ ¼çš„ dummy å­—ç¬¦ã€‚
    è¿™ä¸ªå‡½æ•°ä¼šç›´æŽ¥ä¿®æ”¹ä¼ å…¥çš„ paragraph å¯¹è±¡ï¼Œåœ¨éœ€è¦ç©ºæ ¼çš„åœ°æ–¹æ·»åŠ  dummy å­—ç¬¦ã€‚
    åŒæ—¶ä¹Ÿä¼šå¤„ç†ä¸åŒç»„æˆéƒ¨åˆ†ä¹‹é—´çš„ç©ºæ ¼ã€‚

    Args:
        paragraph: éœ€è¦å¤„ç†çš„ PDF æ®µè½å¯¹è±¡
    """
    # é¦–å…ˆå¤„ç†æ¯ä¸ªç»„æˆéƒ¨åˆ†å†…éƒ¨çš„ç©ºæ ¼
    _add_intra_composition_spaces(paragraph)
    # ç„¶åŽå¤„ç†ç»„æˆéƒ¨åˆ†ä¹‹é—´çš„ç©ºæ ¼
    _add_inter_composition_spaces(paragraph)


def _get_first_char_from_composition(
    comp: PdfParagraphComposition,
) -> PdfCharacter | None:
    """èŽ·å–ç»„æˆéƒ¨åˆ†çš„ç¬¬ä¸€ä¸ªå­—ç¬¦"""
    if comp.pdf_line and comp.pdf_line.pdf_character:
        return comp.pdf_line.pdf_character[0]
    elif (
        comp.pdf_same_style_characters and comp.pdf_same_style_characters.pdf_character
    ):
        return comp.pdf_same_style_characters.pdf_character[0]
    elif comp.pdf_formula and comp.pdf_formula.pdf_character:
        return comp.pdf_formula.pdf_character[0]
    elif comp.pdf_character:
        return comp.pdf_character
    return None


def _get_last_char_from_composition(
    comp: PdfParagraphComposition,
) -> PdfCharacter | None:
    """èŽ·å–ç»„æˆéƒ¨åˆ†çš„æœ€åŽä¸€ä¸ªå­—ç¬¦"""
    if comp.pdf_line and comp.pdf_line.pdf_character:
        return comp.pdf_line.pdf_character[-1]
    elif (
        comp.pdf_same_style_characters and comp.pdf_same_style_characters.pdf_character
    ):
        return comp.pdf_same_style_characters.pdf_character[-1]
    elif comp.pdf_formula and comp.pdf_formula.pdf_character:
        return comp.pdf_formula.pdf_character[-1]
    elif comp.pdf_character:
        return comp.pdf_character
    return None


def _add_space_dummy_chars_to_list(chars: list[PdfCharacter]) -> None:
    """
    åœ¨å­—ç¬¦åˆ—è¡¨ä¸­çš„é€‚å½“ä½ç½®æ·»åŠ è¡¨ç¤ºç©ºæ ¼çš„ dummy å­—ç¬¦ã€‚

    Args:
        chars: PdfCharacter å¯¹è±¡åˆ—è¡¨
    """
    if not chars:
        return

    # è®¡ç®—å­—ç¬¦é—´è·çš„ä¸­ä½æ•°
    distances = []
    for i in range(len(chars) - 1):
        distance = chars[i + 1].box.x - chars[i].box.x2
        if distance > 1:  # åªè€ƒè™‘æ­£å‘è·ç¦»
            distances.append(distance)

    # åŽ»é‡åŽçš„è·ç¦»
    distinct_distances = sorted(set(distances))

    if not distinct_distances:
        median_distance = 1
    elif len(distinct_distances) == 1:
        median_distance = distinct_distances[0]
    else:
        median_distance = distinct_distances[1]

    # åœ¨éœ€è¦çš„åœ°æ–¹æ’å…¥ç©ºæ ¼å­—ç¬¦
    i = 0
    while i < len(chars) - 1:
        curr_char = chars[i]
        next_char = chars[i + 1]

        distance = next_char.box.x - curr_char.box.x2
        if distance >= median_distance or Layout.is_newline(curr_char, next_char):
            if distance < 0:
                distance = -distance
            # åˆ›å»ºä¸€ä¸ª dummy å­—ç¬¦ä½œä¸ºç©ºæ ¼
            space_box = Box(
                x=curr_char.box.x2,
                y=curr_char.box.y,
                x2=curr_char.box.x2 + min(distance, median_distance),
                y2=curr_char.box.y2,
            )

            space_char = PdfCharacter(
                pdf_style=curr_char.pdf_style,
                box=space_box,
                char_unicode=" ",
                scale=curr_char.scale,
                advance=space_box.x2 - space_box.x,
                visual_bbox=il_version_1.VisualBbox(box=space_box),
            )

            # åœ¨å½“å‰ä½ç½®åŽæ’å…¥ç©ºæ ¼å­—ç¬¦
            chars.insert(i + 1, space_char)
            i += 2  # è·³è¿‡åˆšæ’å…¥çš„ç©ºæ ¼
        else:
            i += 1


def build_layout_index(page):
    """Builds an R-tree index for all layouts on the page."""
    from rtree import index

    layout_index = index.Index()
    layout_map = {}
    for i, layout in enumerate(page.page_layout):
        layout_map[i] = layout
        if layout.box:
            layout_index.insert(i, box_to_tuple(layout.box))
    return layout_index, layout_map


def calculate_iou_for_boxes(box1: Box, box2: Box) -> float:
    """Calculate the intersection area divided by the first box area."""
    x_left = max(box1.x, box2.x)
    y_bottom = max(box1.y, box2.y)
    x_right = min(box1.x2, box2.x2)
    y_top = min(box1.y2, box2.y2)

    if x_right <= x_left or y_top <= y_bottom:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_top - y_bottom)

    # Calculate area of first box
    first_box_area = (box1.x2 - box1.x) * (box1.y2 - box1.y)

    # Return intersection divided by first box area, handle division by zero
    if first_box_area <= 0:
        return 0.0

    return intersection_area / first_box_area


def calculate_y_iou_for_boxes(box1: Box, box2: Box) -> float:
    """Calculate the intersection ratio in y-axis direction divided by the first box height.

    Args:
        box1: First box
        box2: Second box

    Returns:
        float: Intersection ratio in y-axis direction between 0 and 1
    """
    y_bottom = max(box1.y, box2.y)
    y_top = min(box1.y2, box2.y2)

    if y_top <= y_bottom:
        return 0.0

    # Calculate intersection height
    intersection_height = y_top - y_bottom

    # Calculate height of first box
    first_box_height = box1.y2 - box1.y

    # Return intersection divided by first box height, handle division by zero
    if first_box_height <= 0:
        return 0.0

    return intersection_height / first_box_height


def calculate_y_true_iou_for_boxes(box1: Box, box2: Box) -> float:
    """Calculate the intersection ratio in y-axis direction divided by the first box height.

    Args:
        box1: First box
        box2: Second box

    Returns:
        float: Intersection ratio in y-axis direction between 0 and 1
    """
    y_bottom = max(box1.y, box2.y)
    y_top = min(box1.y2, box2.y2)

    if y_top <= y_bottom:
        return 0.0

    # Calculate intersection height
    intersection_height = y_top - y_bottom

    # Calculate height of first box
    first_box_height = box1.y2 - box1.y
    second_box_height = box2.y2 - box2.y

    min_height = min(first_box_height, second_box_height)

    # Return intersection divided by first box height, handle division by zero
    if first_box_height <= 0:
        return 0.0

    return intersection_height / min_height


def get_character_layout(
    char,
    layout_index,
    layout_map,
    layout_priority=None,
    _bbox_mode: Literal["auto", "visual", "box"] = "auto",
):
    """Get the layout for a character based on priority and IoU."""
    if layout_priority is None:
        layout_priority = [
            "number",
            "reference",
            "reference_content",
            "algorithm",
            "formula_caption",
            "isolate_formula",
            "table_footnote",
            "table_caption",
            "figure_caption",
            "figure_title",
            "chart_title",
            "table_title",
            "table_cell_hybrid",
            "table_text",
            "wireless_table_cell",
            "wired_table_cell",
            "abandon",
            "title",
            "abstract",
            "paragraph_title",
            "content",
            "doc_title",
            "footnote",
            "header",
            "footer",
            "seal",
            "plain text",
            "tiny text",
            "author_info_hybrid",
            "list_item_hybrid",
            "text",
            "paragraph_hybrid",
            "paragraph",
            "table_cell",
            "figure_text",
            "list_item",
            "title",
            "caption",
            "footnote_hybrid",
            "footnote",
            "formula",
            "formula_hybrid",
            "page_header",
            "page_footer",
            # --- hybrid labels ---
            "reference_hybrid",
            "document_hybrid",
            "academic_paper_hybrid",
            "form_or_table_hybrid",
            "presentation_slide_hybrid",
            "webpage_screenshot_hybrid",
            "manga_or_comic_hybrid",
            "advertisement_hybrid",
            "magazine_or_newspaper_hybrid",
            "other_hybrid",
            "table_cell_hybrid",
            "figure_text_hybrid",
            "title_hybrid",
            "caption_hybrid",
            "code_algo_hybrid",
            "line_number_hybrid",
            "page_header_hybrid",
            "page_footer_hybrid",
            "page_number_hybrid",
            "unknown_hybrid",
            "fallback_line",
            "table",
            "figure",
            "image",
        ]

    char_box = char.visual_bbox.box

    # Collect all intersecting layouts and their IoU values
    matching_layouts = []
    candidate_ids = list(layout_index.intersection(box_to_tuple(char_box)))
    candidate_layouts = [layout_map[i] for i in candidate_ids]

    for layout in candidate_layouts:
        # Calculate IoU
        intersection_area = max(
            0, min(char_box.x2, layout.box.x2) - max(char_box.x, layout.box.x)
        ) * max(0, min(char_box.y2, layout.box.y2) - max(char_box.y, layout.box.y))
        char_area = (char_box.x2 - char_box.x) * (char_box.y2 - char_box.y)

        if char_area > 0:
            iou = intersection_area / char_area
            if iou > 0:
                matching_layouts.append(
                    {
                        "layout": Layout(layout.id, layout.class_name),
                        "priority": (
                            layout_priority.index(layout.class_name)
                            if layout.class_name in layout_priority
                            else len(layout_priority)
                        ),
                        "iou": iou,
                    }
                )

    if not matching_layouts:
        return None

    # Sort by priority (ascending) and IoU value (descending)
    matching_layouts.sort(key=lambda x: (x["priority"], -x["iou"]))

    return matching_layouts[0]["layout"]


def is_text_layout(layout: Layout):
    """Check if a layout is a text layout."""
    return layout is not None and layout.name in [
        "plain text",
        "tiny text",
        "title",
        "abandon",
        "figure_caption",
        "table_caption",
        "table_text",
        "table_footnote",
        "title",
        "paragraph_title",
        "abstract",
        "content",
        "figure_title",
        "table_title",
        "doc_title",
        "footnote",
        "header",
        "footer",
        "seal",
        "text",
        "chart_title",
        "paragraph",
        "table_cell",
        "figure_text",
        "list_item",
        "title",
        "caption",
        "footnote",
        "page_header",
        "page_footer",
        "wired_table_cell",
        "wireless_table_cell",
        "paragraph_hybrid",
        "table_cell_hybrid",
        "caption_hybrid",
        "unknown_hybrid",
        "figure_text_hybrid",
        "list_item_hybrid",
        "title_hybrid",
        "fallback_line",
        "author_info_hybrid",
        "page_header_hybrid",
        "page_footer_hybrid",
        "footnote_hybrid",
    ]


def is_character_in_formula_layout(
    char: il_version_1.PdfCharacter,
    _page: il_version_1.Page,
    layout_index,
    layout_map,
) -> int | None:
    """Check if character is contained within any formula-related layout."""
    formula_layout_types = {"formula"}

    char_box = char.visual_bbox.box
    char_box2 = char.box

    if calculate_iou_for_boxes(char_box, char_box2) < 0.2:
        char_box = char_box2

    # Get all candidate layouts that intersect with the character
    candidate_ids = list(layout_index.intersection(box_to_tuple(char_box)))
    candidate_layouts: list[il_version_1.PageLayout] = [
        layout_map[i] for i in candidate_ids
    ]

    # Check if any intersecting layout is a formula type
    for layout in candidate_layouts:
        if layout.class_name in formula_layout_types:
            iou = calculate_iou_for_boxes(char_box, layout.box)
            if iou > 0.4:  # Character has overlap with formula layout
                return layout.id

    return None


def is_curve_in_figure_table_layout(
    curve, layout_index, layout_map, protection_threshold: float = 0.3
) -> bool:
    """Check if curve is within figure/table layout areas.

    Args:
        curve: The curve object to check
        layout_index: Spatial index for layouts
        layout_map: Mapping from layout IDs to layout objects
        protection_threshold: IoU threshold for figure/table protection

    Returns:
        True if curve is within figure/table layout areas
    """
    if not curve.box:
        return False

    # Figure/table related layout types
    figure_table_layouts = {
        "figure",
        "table",
        "figure_text",
        "table_text",
        "figure_caption",
        "table_caption",
        "figure_title",
        "table_title",
        "chart_title",
        "table_cell",
        "table_cell_hybrid",
        "wired_table_cell",
        "wireless_table_cell",
        "table_footnote",
    }

    # Get candidate layouts that intersect with curve
    candidate_ids = list(layout_index.intersection(box_to_tuple(curve.box)))
    candidate_layouts = [layout_map[i] for i in candidate_ids]

    for layout in candidate_layouts:
        if layout.class_name in figure_table_layouts:
            # Check if curve has significant overlap with figure/table layout
            iou = calculate_iou_for_boxes(curve.box, layout.box)
            if iou > protection_threshold:
                return True

    return False


def is_curve_overlapping_with_paragraphs(
    curve, paragraphs: list, overlap_threshold: float = 0.2
) -> bool:
    """Check if curve overlaps with text paragraph areas.

    Args:
        curve: The curve object to check
        paragraphs: List of paragraph objects
        overlap_threshold: IoU threshold for paragraph overlap detection

    Returns:
        True if curve overlaps with any paragraph area
    """
    if not curve.box:
        return False

    for paragraph in paragraphs:
        para_box = get_paragraph_bounding_box(paragraph)
        if para_box:
            iou = calculate_iou_for_boxes(curve.box, para_box)
            if iou > overlap_threshold:
                return True

    return False


def _get_composition_box(composition: "PdfParagraphComposition") -> "Box | None":
    """Extract a bounding Box from a single *composition*, or return None."""
    if composition.pdf_line and composition.pdf_line.box:
        return composition.pdf_line.box
    if composition.pdf_formula and composition.pdf_formula.box:
        return composition.pdf_formula.box
    if (
        composition.pdf_same_style_characters
        and composition.pdf_same_style_characters.box
    ):
        return composition.pdf_same_style_characters.box
    if composition.pdf_character and len(composition.pdf_character) > 0:
        char_boxes = [
            char.visual_bbox.box
            for char in composition.pdf_character
            if char.visual_bbox and char.visual_bbox.box
        ]
        if char_boxes:
            return Box(
                min(box.x for box in char_boxes),
                min(box.y for box in char_boxes),
                max(box.x2 for box in char_boxes),
                max(box.y2 for box in char_boxes),
            )
    return None


def get_paragraph_bounding_box(paragraph) -> Box | None:
    """Calculate the bounding box of a paragraph from its compositions.

    Args:
        paragraph: The paragraph object

    Returns:
        Box object representing the paragraph bounds, or None if no valid bounds
    """
    if not paragraph.pdf_paragraph_composition:
        return None

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    has_valid_box = False

    for composition in paragraph.pdf_paragraph_composition:
        comp_box = _get_composition_box(composition)
        if comp_box:
            min_x = min(min_x, comp_box.x)
            min_y = min(min_y, comp_box.y)
            max_x = max(max_x, comp_box.x2)
            max_y = max(max_y, comp_box.y2)
            has_valid_box = True

    if not has_valid_box:
        return None

    return Box(min_x, min_y, max_x, max_y)
