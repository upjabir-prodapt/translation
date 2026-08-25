import copy
import json
import logging
import re
from pathlib import Path
from string import Template

import Levenshtein
import orjson
import tiktoken
from tqdm import tqdm

from src.config.domain_prompts import get_domain_role_block
from src.worker.doctranslator.batching import compute_batch_plan
from src.worker.doctranslator.batching import log_batch_plan
from src.worker.doctranslator.format.pdf.document_il import Document
from src.worker.doctranslator.format.pdf.document_il import Page
from src.worker.doctranslator.format.pdf.document_il import PdfFont
from src.worker.doctranslator.format.pdf.document_il import PdfParagraph
from src.worker.doctranslator.format.pdf.document_il.midend.il_translator import (
    DocumentTranslateTracker,
)
from src.worker.doctranslator.format.pdf.document_il.midend.il_translator import (
    ILTranslator,
)
from src.worker.doctranslator.format.pdf.document_il.midend.il_translator import (
    PageTranslateTracker,
)
from src.worker.doctranslator.format.pdf.document_il.utils.fontmap import FontMapper
from src.worker.doctranslator.format.pdf.document_il.utils.paragraph_helper import (
    is_cid_paragraph,
)
from src.worker.doctranslator.format.pdf.document_il.utils.paragraph_helper import (
    is_placeholder_only_paragraph,
)
from src.worker.doctranslator.format.pdf.document_il.utils.paragraph_helper import (
    is_pure_numeric_paragraph,
)
from src.worker.doctranslator.format.pdf.translation_config import TranslationConfig
from src.worker.doctranslator.translator.translation_cache import get_translation_cache
from src.worker.doctranslator.translator.translator import BaseTranslator
from src.worker.doctranslator.translator.translator import BatchTranslationResponse
from src.worker.doctranslator.utils.priority_thread_pool_executor import (
    PriorityThreadPoolExecutor,
)

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = Template(
    """$role_block

## Structure Rules
1. Keep **the same number of paragraphs as the input**.
2. Input paragraphs may be **sliced pieces of the same original paragraph**.  
   → You MUST treat each input paragraph **as an independent, fixed unit**.  
   → Do NOT merge paragraphs, split paragraphs, or move content between paragraphs.
3. Inside each paragraph, you may adjust word order for fluency, but:
   - Do NOT change the meaning.
   - Do NOT move placeholders, tags, or code outside their paragraph.
4. Translate ALL human-readable content into $lang_out.

## Do NOT Modify
- Tags (e.g., <style>, <b>, <code>): keep them exactly the same.  
  *Translate tag-internal text except code blocks (<code>…</code>)*.
- Placeholders: `{v1}`, `{name}`, `%s`, `%d`, `[[...]]`, `%%...%%` — keep exactly unchanged.
- JSON keys or structure.

$glossary_usage_rules_block
## Output Format
Return a JSON array of the same length.  
For each item:
- Keep the same "id" and remove other fields like "input" and "layout_label".
- Add "output" with the translated text only.
- No extra text, no ```json blocks.

## Style
- Produce fluent, professional $lang_out.
- Preserve punctuation unless needed for target language fluency.

### Example
Input:
[
    {
    "id": 0,
    "input": "{v1}<style id='2'>hello</style>, world!",
    "layout_label": "text"
    }
]
Output:
[
    {
    "id": 0,
    "output": "{v1}<style id='2'>你好</style>，世界！"
    }
]

$contextual_hints_block

$glossary_tables_block

## Here is the input:

$json_input_str"""
)


class BatchParagraph:
    def __init__(
        self,
        paragraphs: list[PdfParagraph],
        pages: list[Page],
        page_tracker: PageTranslateTracker,
    ):
        self.paragraphs = paragraphs
        self.pages = pages
        self.trackers = [page_tracker.new_paragraph() for _ in paragraphs]


class ILTranslatorLLMOnly:
    stage_name = "Translate Paragraphs"

    def __init__(
        self,
        translate_engine: BaseTranslator,
        translation_config: TranslationConfig,
        tokenizer=None,
    ):
        self.translate_engine = translate_engine
        self.translation_config = translation_config
        self.font_mapper = FontMapper(translation_config)
        self.shared_context_cross_split_part = (
            translation_config.shared_context_cross_split_part
        )

        if tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        else:
            self.tokenizer = tokenizer

        # Cache glossaries at initialization
        self._cached_glossaries = (
            self.shared_context_cross_split_part.get_glossaries_for_translation(
                translation_config.auto_extract_glossary
            )
        )

        self.il_translator = ILTranslator(
            translate_engine=translate_engine,
            translation_config=translation_config,
            tokenizer=self.tokenizer,
        )
        self.il_translator.use_as_fallback = True
        try:
            self.translate_engine.do_llm_translate(None)
        except NotImplementedError as e:
            raise ValueError("LLM translator not supported") from e

        self.ok_count = 0
        self.fallback_count = 0
        self.total_count = 0

    def calc_token_count(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return 0

    def find_title_paragraph(self, docs: Document) -> PdfParagraph | None:
        """Find the first paragraph with layout_label 'title' in the document.

        Args:
            docs: The document to search in

        Returns:
            The first title paragraph found, or None if no title paragraph exists
        """
        for page in docs.page:
            for paragraph in page.pdf_paragraph:
                if paragraph.layout_label == "title":
                    logger.info(f"Found title paragraph: {paragraph.unicode}")
                    return paragraph
        return None

    def _apply_adaptive_batch_plan(self, docs: Document) -> None:
        """Right-size this run's paragraph batches for the worker pool.

        The DOCX pipeline computes its plan inside `_batch_units()` because it
        has the whole unit list up front. PDF batches per-page inside
        `process_page()`, so the document-level token total is only knowable
        here -- computing it once in `translate()` and overriding the config's
        cap gives PDF the same one-wave-per-pool sizing without restructuring
        the per-page batching loop.
        """
        total_payload = 0
        for page in docs.page:
            for paragraph in page.pdf_paragraph:
                if paragraph.debug_id is None or paragraph.unicode is None:
                    continue
                total_payload += self.calc_token_count(paragraph.unicode)

        plan = compute_batch_plan(
            total_payload,
            self.translation_config.pool_max_workers,
            base_max_tokens=self.translation_config.llm_translation_batch_max_tokens,
            base_max_items=self.translation_config.llm_translation_batch_max_paragraphs,
        )
        self.translation_config.llm_translation_batch_max_tokens = plan.max_tokens
        self.translation_config.llm_translation_batch_max_paragraphs = plan.max_items

        # batch_count is not known until process_page() has run, so estimate it
        # from the plan purely for the utilisation/waves figures in the log.
        estimated_batches = (
            -(-total_payload // plan.max_tokens) if plan.max_tokens else 0
        )
        log_batch_plan("PdfTranslateParagraphs", plan, estimated_batches)

    def translate(self, docs: Document) -> None:
        self.il_translator.docs = docs
        tracker = DocumentTranslateTracker()
        self.mid = 0
        self._apply_adaptive_batch_plan(docs)

        if not self.translation_config.shared_context_cross_split_part.first_paragraph:
            # Try to find the first title paragraph
            title_paragraph = self.find_title_paragraph(docs)
            self.translation_config.shared_context_cross_split_part.first_paragraph = (
                copy.deepcopy(title_paragraph)
            )
            self.translation_config.shared_context_cross_split_part.recent_title_paragraph = copy.deepcopy(
                title_paragraph
            )
            if title_paragraph:
                logger.info(f"Found first title paragraph: {title_paragraph.unicode}")

        # count total paragraph
        total = sum(
            [
                len(
                    [
                        p
                        for p in page.pdf_paragraph
                        if p.debug_id is not None and p.unicode is not None
                    ]
                )
                for page in docs.page
            ]
        )
        translated_ids = set()
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total,
        ) as pbar:
            with PriorityThreadPoolExecutor(
                max_workers=self.translation_config.pool_max_workers,
            ) as executor2:
                with PriorityThreadPoolExecutor(
                    max_workers=self.translation_config.pool_max_workers,
                ) as executor:
                    self.process_cross_page_paragraph(
                        docs,
                        executor,
                        pbar,
                        tracker,
                        executor2,
                        translated_ids,
                    )
                    # Cross-column detection per page (after cross-page processing)
                    for page in docs.page:
                        self.process_cross_column_paragraph(
                            page,
                            executor,
                            pbar,
                            tracker,
                            executor2,
                            translated_ids,
                        )
                    for page in docs.page:
                        self.process_page(
                            page,
                            executor,
                            pbar,
                            tracker.new_page(),
                            executor2,
                            translated_ids,
                        )

        path = self.translation_config.get_working_file_path("translate_tracking.json")

        if (
            self.translation_config.debug
            or self.translation_config.working_dir is not None
        ):
            logger.debug(f"save translate tracking to {path}")
            with Path(path).open("w", encoding="utf-8") as f:
                f.write(tracker.to_json())
        cache_stats = get_translation_cache().stats_snapshot()
        logger.info(
            f"Translation completed. Total: {self.total_count}, "
            f"Successful: {self.ok_count}, Fallback: {self.fallback_count}, "
            f"cache_hits_cumulative={cache_stats['cache_hits']} "
            f"cache_misses_cumulative={cache_stats['cache_misses']} "
            f"cache_hit_rate_cumulative={cache_stats['cache_hit_rate']}"
        )

    def _is_body_text_paragraph(self, paragraph: PdfParagraph) -> bool:
        """判断正文段落（当前仅 layout_label == 'text'）。

        Args:
            paragraph: PDF paragraph to check

        Returns:
            True if this is a body text paragraph, False otherwise
        """
        return paragraph.layout_label in (
            "text",
            "plain text",
            "paragraph_hybrid",
        )

    def _should_translate_paragraph(
        self,
        paragraph: PdfParagraph,
        translated_ids: set[int] | None = None,
        require_body_text: bool = False,
    ) -> bool:
        """Check if a paragraph should be translated based on common filtering criteria.

        Args:
            paragraph: PDF paragraph to check
            translated_ids: Set of already translated paragraph IDs
            require_body_text: Whether to additionally check if paragraph is body text

        Returns:
            True if paragraph should be translated, False otherwise
        """
        # Basic validation checks
        if paragraph.debug_id is None or paragraph.unicode is None:
            return False

        # Check if already translated
        if translated_ids is not None and id(paragraph) in translated_ids:
            return False

        # CID paragraph check
        if is_cid_paragraph(paragraph):
            return False

        # Minimum length check
        if len(paragraph.unicode) < self.translation_config.min_text_length:
            return False

        # Body text check if requested
        if require_body_text and not self._is_body_text_paragraph(paragraph):
            return False

        return True

    def _filter_paragraphs(
        self,
        page: Page,
        translated_ids: set[int] | None = None,
        require_body_text: bool = False,
    ) -> list[PdfParagraph]:
        """Get list of paragraphs that should be translated from a page.

        Args:
            page: Page to get paragraphs from
            translated_ids: Set of already translated paragraph IDs
            require_body_text: Whether to filter for body text paragraphs only

        Returns:
            List of paragraphs that should be translated
        """
        return [
            paragraph
            for paragraph in page.pdf_paragraph
            if self._should_translate_paragraph(
                paragraph, translated_ids, require_body_text
            )
        ]

    def _build_font_maps(
        self, page: Page
    ) -> tuple[dict[str, PdfFont], dict[int, dict[str, PdfFont]]]:
        """Build font maps for a page.

        Args:
            page: The page to build font maps for

        Returns:
            Tuple of (page_font_map, page_xobj_font_map)
        """
        page_font_map = {}
        for font in page.pdf_font:
            page_font_map[font.font_id] = font

        page_xobj_font_map = {}
        for xobj in page.pdf_xobject:
            page_xobj_font_map[xobj.xobj_id] = page_font_map.copy()
            for font in xobj.pdf_font:
                page_xobj_font_map[xobj.xobj_id][font.font_id] = font

        return page_font_map, page_xobj_font_map

    def process_cross_page_paragraph(
        self,
        docs: Document,
        executor: PriorityThreadPoolExecutor,
        pbar: tqdm | None = None,
        tracker: DocumentTranslateTracker | None = None,
        executor2: PriorityThreadPoolExecutor | None = None,
        translated_ids: set[int] | None = None,
    ):
        """Process cross-page paragraphs by combining last body text paragraph of current page
        with first body text paragraph of next page.

        Args:
            docs: Document containing pages to process
            executor: Thread pool executor for translation tasks
            pbar: Progress bar for tracking translation progress
            tracker: Page translation tracker
            executor2: Secondary executor for fallback translation
            translated_ids: Set of already translated paragraph IDs
        """
        self.translation_config.raise_if_cancelled()

        if tracker is None:
            tracker = DocumentTranslateTracker()

        if translated_ids is None:
            translated_ids = set()

        # Process adjacent page pairs
        for i in range(len(docs.page) - 1):
            page_curr = docs.page[i]
            page_next = docs.page[i + 1]

            # Find body text paragraphs in current page
            curr_body_paragraphs = self._filter_paragraphs(
                page_curr, translated_ids, require_body_text=True
            )

            # Find body text paragraphs in next page
            next_body_paragraphs = self._filter_paragraphs(
                page_next, translated_ids, require_body_text=True
            )

            # Get last paragraph from current page and first paragraph from next page
            if not curr_body_paragraphs or not next_body_paragraphs:
                continue

            last_curr_paragraph = curr_body_paragraphs[-1]
            first_next_paragraph = next_body_paragraphs[0]

            # Skip if either paragraph is already translated
            if (
                id(last_curr_paragraph) in translated_ids
                or id(first_next_paragraph) in translated_ids
            ):
                continue

            # Build font maps for both pages
            curr_font_map, curr_xobj_font_map = self._build_font_maps(page_curr)
            next_font_map, next_xobj_font_map = self._build_font_maps(page_next)

            # Merge font maps
            merged_font_map = {**curr_font_map, **next_font_map}
            merged_xobj_font_map = {**curr_xobj_font_map, **next_xobj_font_map}

            # Calculate total token count
            total_token_count = self.calc_token_count(
                last_curr_paragraph.unicode
            ) + self.calc_token_count(first_next_paragraph.unicode)

            # Create batch with both paragraphs
            cross_page_paragraphs = [last_curr_paragraph, first_next_paragraph]
            cross_page_pages = [page_curr, page_next]
            batch_paragraph = BatchParagraph(
                cross_page_paragraphs, cross_page_pages, tracker.new_cross_page()
            )

            self.mid += 1
            # Submit translation task (force submit regardless of token count)
            executor.submit(
                self.translate_paragraph,
                batch_paragraph,
                pbar,
                merged_font_map,
                merged_xobj_font_map,
                self.translation_config.shared_context_cross_split_part.first_paragraph,
                self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
                executor2,
                priority=1048576 - total_token_count,
                paragraph_token_count=total_token_count,
                mp_id=self.mid,
            )

            # Mark paragraphs as translated
            translated_ids.add(id(last_curr_paragraph))
            translated_ids.add(id(first_next_paragraph))

    def process_cross_column_paragraph(
        self,
        page: Page,
        executor: PriorityThreadPoolExecutor,
        pbar: tqdm | None = None,
        tracker: DocumentTranslateTracker | None = None,
        executor2: PriorityThreadPoolExecutor | None = None,
        translated_ids: set[int] | None = None,
    ):
        """Process cross-column paragraphs within the same page.

        If two adjacent body-text paragraphs have a gap in their y2 coordinate
        greater than 20 units, they are considered split across columns and
        will be translated together.
        """
        self.translation_config.raise_if_cancelled()

        if tracker is None:
            tracker = DocumentTranslateTracker()
        if translated_ids is None:
            translated_ids = set()

        # Filter body-text paragraphs maintaining original order
        body_paragraphs = self._filter_paragraphs(
            page, translated_ids, require_body_text=True
        )
        if len(body_paragraphs) < 2:
            return

        # Build font maps once for the whole page
        page_font_map, page_xobj_font_map = self._build_font_maps(page)

        for idx in range(len(body_paragraphs) - 1):
            p1 = body_paragraphs[idx]
            p2 = body_paragraphs[idx + 1]

            # Skip already translated
            if id(p1) in translated_ids or id(p2) in translated_ids:
                continue

            # Safety checks for box information
            if not (
                p1.box and p2.box and p1.box.y2 is not None and p2.box.y2 is not None
            ):
                continue

            if p2.box.y2 - p1.box.y2 <= 20:
                continue

            total_token_count = self.calc_token_count(
                p1.unicode
            ) + self.calc_token_count(p2.unicode)

            batch = BatchParagraph([p1, p2], [page, page], tracker.new_cross_column())
            self.mid += 1
            executor.submit(
                self.translate_paragraph,
                batch,
                pbar,
                page_font_map,
                page_xobj_font_map,
                self.translation_config.shared_context_cross_split_part.first_paragraph,
                self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
                executor2,
                priority=1048576 - total_token_count,
                paragraph_token_count=total_token_count,
                mp_id=self.mid,
            )

            translated_ids.add(id(p1))
            translated_ids.add(id(p2))

    def _is_paragraph_skippable(
        self,
        paragraph: PdfParagraph,
        translated_ids: set,
        pbar,
    ) -> bool:
        """Return True if the paragraph should be skipped (and advance pbar if needed)."""
        if id(paragraph) in translated_ids:
            return True
        if paragraph.debug_id is None or paragraph.unicode is None:
            return True
        if is_cid_paragraph(paragraph):
            if pbar:
                pbar.advance(1)
            return True
        if len(paragraph.unicode) < self.translation_config.min_text_length:
            if pbar:
                pbar.advance(1)
            return True
        if is_pure_numeric_paragraph(paragraph):
            if pbar:
                pbar.advance(1)
            return True
        if is_placeholder_only_paragraph(paragraph):
            if pbar:
                pbar.advance(1)
            return True
        return False

    def _handle_skipped_paragraph(self, paragraph: PdfParagraph, pbar):
        """Handle a skipped paragraph by advancing progress bar."""
        if (
            pbar
            and not is_cid_paragraph(paragraph)
            and len(paragraph.unicode) >= self.translation_config.min_text_length
            and not is_pure_numeric_paragraph(paragraph)
            and not is_placeholder_only_paragraph(paragraph)
        ):
            return
        if pbar:
            pbar.advance(1)

    def _submit_batch(
        self,
        paragraphs: list,
        page: Page,
        tracker,
        pbar,
        page_font_map: dict,
        page_xobj_font_map: dict,
        executor,
        executor2,
        total_token_count: int,
    ):
        """Submit a batch of paragraphs for translation."""
        self.mid += 1
        executor.submit(
            self.translate_paragraph,
            BatchParagraph(paragraphs, [page] * len(paragraphs), tracker),
            pbar,
            page_font_map,
            page_xobj_font_map,
            self.translation_config.shared_context_cross_split_part.first_paragraph,
            self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
            executor2,
            priority=1048576 - total_token_count,
            paragraph_token_count=total_token_count,
            mp_id=self.mid,
        )

    def process_page(
        self,
        page: Page,
        executor: PriorityThreadPoolExecutor,
        pbar: tqdm | None = None,
        tracker: PageTranslateTracker = None,
        executor2: PriorityThreadPoolExecutor | None = None,
        translated_ids: set | None = None,
    ):
        self.translation_config.raise_if_cancelled()
        page_font_map, page_xobj_font_map = self._build_font_maps(page)

        paragraphs = []
        total_token_count = 0
        max_tokens = self.translation_config.llm_translation_batch_max_tokens
        max_paragraphs = self.translation_config.llm_translation_batch_max_paragraphs

        for paragraph in page.pdf_paragraph:
            if self._is_paragraph_skippable(paragraph, translated_ids, pbar):
                continue

            total_token_count += self.calc_token_count(paragraph.unicode)
            paragraphs.append(paragraph)
            translated_ids.add(id(paragraph))
            if paragraph.layout_label == "title":
                self.shared_context_cross_split_part.recent_title_paragraph = (
                    copy.deepcopy(paragraph)
                )

            if total_token_count > max_tokens or len(paragraphs) > max_paragraphs:
                self._submit_batch(
                    paragraphs,
                    page,
                    tracker,
                    pbar,
                    page_font_map,
                    page_xobj_font_map,
                    executor,
                    executor2,
                    total_token_count,
                )
                paragraphs = []
                total_token_count = 0

        if paragraphs:
            self._submit_batch(
                paragraphs,
                page,
                tracker,
                pbar,
                page_font_map,
                page_xobj_font_map,
                executor,
                executor2,
                total_token_count,
            )

    def _build_batch_inputs(
        self,
        batch_paragraph: "BatchParagraph",
        pbar,
        page_font_map: dict,
        xobj_font_map: dict,
        mp_id: int,
    ) -> tuple[list, list, list]:
        """Pre-translate each paragraph and build LLM input list.

        Returns (inputs, llm_translate_trackers, paragraph_unicodes).
        """
        inputs = []
        llm_translate_trackers = []
        paragraph_unicodes = []
        for i, paragraph in enumerate(batch_paragraph.paragraphs):
            tracker = batch_paragraph.trackers[i]
            text, translate_input = self.il_translator.pre_translate_paragraph(
                paragraph, tracker, page_font_map, xobj_font_map
            )
            if text is None:
                pbar.advance(1)
                continue
            tracker.record_multi_paragraph_id(mp_id)
            llm_translate_tracker = tracker.new_llm_translate_tracker()
            llm_translate_trackers.append(llm_translate_tracker)
            inputs.append(
                (
                    text,
                    translate_input,
                    paragraph,
                    tracker,
                    llm_translate_tracker,
                    paragraph_unicodes,
                )
            )
            paragraph_unicodes.append(paragraph.unicode)
        return inputs, llm_translate_trackers, paragraph_unicodes

    @staticmethod
    def _unwrap_dict_output(parsed_output: dict, n_inputs: int) -> list:
        """Extract a list from a dict-shaped LLM output."""
        for key in ("translations", "results", "items", "data", "outputs"):
            value = parsed_output.get(key)
            if isinstance(value, list):
                return value
        if parsed_output.get("output", parsed_output.get("input", False)):
            return [parsed_output]
        for key in ("translation", "translated_text", "text"):
            value = parsed_output.get(key)
            if isinstance(value, str):
                if n_inputs == 1:
                    return [{"id": 0, "output": value}]
                raise ValueError(
                    "LLM output contains a single translation string while multiple inputs were provided"
                )
        raise ValueError(
            "LLM output JSON object is missing expected translation fields"
        )

    @staticmethod
    def _unwrap_llm_output(parsed_output: object, n_inputs: int) -> list:
        """Normalise varied LLM JSON output shapes into a plain list of translation items."""
        if isinstance(parsed_output, dict):
            return ILTranslatorLLMOnly._unwrap_dict_output(parsed_output, n_inputs)
        if isinstance(parsed_output, str):
            if n_inputs == 1:
                return [{"id": 0, "output": parsed_output}]
            raise ValueError(
                "LLM output JSON is a string while multiple inputs were provided"
            )
        if not isinstance(parsed_output, list):
            raise ValueError(
                f"Unexpected LLM output JSON type: {type(parsed_output).__name__}"
            )
        return parsed_output

    def _validate_translation_quality(
        self,
        input_unicode: str,
        output_unicode: str,
        llm_translate_tracker,
    ) -> bool:
        """Check translation quality heuristics. Returns True if should fallback."""
        trimed_input = re.sub(r"[. 。…，]{20,}", ".", input_unicode)
        input_token_count = self.calc_token_count(trimed_input)
        output_token_count = self.calc_token_count(output_unicode)
        same_as_input = trimed_input == output_unicode

        if (
            same_as_input
            and input_token_count > 10
            and not self.translation_config.disable_same_text_fallback
        ):
            llm_translate_tracker.set_error_message(
                "Translation result is the same as input, fallback."
            )
            llm_translate_tracker.set_placeholder_full_match()
            logger.warning("Translation result is the same as input, fallback.")
            return True

        if not (0.3 < output_token_count / input_token_count < 3):
            llm_translate_tracker.set_error_message(
                f"Translation result is too long or too short. Input: {input_token_count}, Output: {output_token_count}"
            )
            logger.warning(
                f"Translation result is too long or too short. Input: {input_token_count}, Output: {output_token_count}"
            )
            llm_translate_tracker.set_placeholder_full_match()
            return True

        if not self.translation_config.disable_same_text_fallback:
            edit_distance = Levenshtein.distance(input_unicode, output_unicode)
            if edit_distance < 5 and input_token_count > 20:
                llm_translate_tracker.set_error_message(
                    f"Translation result edit distance is too small. distance: {edit_distance}, input: {input_unicode}, output: {output_unicode}"
                )
                logger.warning(
                    f"Translation result edit distance is too small. distance: {edit_distance}, input: {input_unicode}, output: {output_unicode}"
                )
                llm_translate_tracker.set_placeholder_full_match()
                return True

        return False

    def _submit_fallback_translation(
        self,
        id_: int,
        inputs: list,
        batch_paragraph: "BatchParagraph",
        pbar,
        page_font_map: dict,
        xobj_font_map: dict,
        title_paragraph,
        local_title_paragraph,
        executor,
    ) -> None:
        """Submit a fallback single-paragraph translation task."""
        self.fallback_count += 1
        inputs[id_][4].set_fallback_to_translate()
        logger.warning(
            f"Fallback to simple translation. paragraph id: {inputs[id_][2].debug_id}"
        )
        para_token_count = self.calc_token_count(inputs[id_][2].unicode)
        paragraph_unicodes = inputs[id_][5]
        inputs[id_][2].unicode = paragraph_unicodes[id_]
        executor.submit(
            self.il_translator.translate_paragraph,
            inputs[id_][2],
            batch_paragraph.pages[id_],
            pbar,
            inputs[id_][3],
            page_font_map,
            xobj_font_map,
            priority=1048576 - para_token_count,
            paragraph_token_count=para_token_count,
            title_paragraph=title_paragraph,
            local_title_paragraph=local_title_paragraph,
        )

    def _apply_single_translation(
        self,
        id_: int,
        output: object,
        inputs: list,
        batch_paragraph: "BatchParagraph",
        llm_translate_trackers: list,
        pbar,
        page_font_map: dict,
        xobj_font_map: dict,
        title_paragraph,
        local_title_paragraph,
        executor,
    ) -> bool:
        """Validate and apply a single translation result. Returns True if a fallback was used."""
        should_fallback = True
        try:
            if not isinstance(output, str):
                logger.warning(f"Translation result is not a string. Output: {output}")
                return should_fallback
            id_ = int(id_)
            if id_ >= len(inputs):
                logger.warning(f"Invalid id {id_}, skipping")
                return should_fallback
            translated_text = re.sub(r"[. 。…，]{20,}", ".", output)
            translate_input = inputs[id_][1]
            llm_translate_tracker = inputs[id_][4]
            input_unicode = inputs[id_][0]
            output_unicode = translated_text

            if self._validate_translation_quality(
                input_unicode, output_unicode, llm_translate_tracker
            ):
                return should_fallback

            self.il_translator.post_translate_paragraph(
                inputs[id_][2],
                inputs[id_][3],
                translate_input,
                translated_text,
            )
            should_fallback = False
            if pbar:
                pbar.advance(1)
        except (ValueError, TypeError, KeyError) as e:
            error_message = f"Error translating paragraph. Error: {e}."
            logger.exception(error_message)
            for tracker in llm_translate_trackers:
                tracker.set_error_message(error_message)
        finally:
            self.total_count += 1
            if should_fallback:
                self._submit_fallback_translation(
                    id_,
                    inputs,
                    batch_paragraph,
                    pbar,
                    page_font_map,
                    xobj_font_map,
                    title_paragraph,
                    local_title_paragraph,
                    executor,
                )
            else:
                self.ok_count += 1
        return should_fallback

    def _parse_translation_results(self, llm_output: str, n_inputs: int) -> dict:
        """Parse and validate LLM JSON output into a {id: output} dict."""
        cleaned = self._clean_json_output(llm_output.strip())
        parsed = json.loads(cleaned)
        parsed = self._unwrap_llm_output(parsed, n_inputs)
        results = {}
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid translation item type: {type(item).__name__}"
                )
            if "id" not in item:
                raise ValueError("Translation item missing id field")
            results[item["id"]] = item.get("output", item.get("input"))
        if len(results) != n_inputs:
            raise ValueError(
                f"Translation results length mismatch. Expected: {n_inputs}, Got: {len(results)}"
            )
        return results

    def _handle_translate_paragraph_error(
        self,
        e: Exception,
        llm_translate_trackers: list,
        inputs: list,
        should_translate_paragraph: list,
        batch_paragraph: "BatchParagraph",
        executor,
        page_font_map: dict,
        xobj_font_map: dict,
        title_paragraph,
        local_title_paragraph,
    ) -> None:
        """Handle a translation-level error by logging and submitting fallback tasks."""
        error_message = f"Error {e} during translation. try fallback"
        logger.warning(error_message)
        for tracker in llm_translate_trackers:
            tracker.set_error_message(error_message)
            tracker.set_fallback_to_translate()
        self.total_count += len(llm_translate_trackers)
        self.fallback_count += len(llm_translate_trackers)
        for i, input_ in enumerate(inputs):
            input_[2].unicode = input_[5][i]
        if not should_translate_paragraph:
            should_translate_paragraph = list(range(len(batch_paragraph.paragraphs)))
        for i in should_translate_paragraph:
            paragraph = batch_paragraph.paragraphs[i]
            tracker = batch_paragraph.trackers[i]
            if paragraph.debug_id is None:
                continue
            paragraph_token_count = self.calc_token_count(paragraph.unicode)
            executor.submit(
                self.il_translator.translate_paragraph,
                paragraph,
                batch_paragraph.pages[i],
                None,
                tracker,
                page_font_map,
                xobj_font_map,
                priority=1048576 - paragraph_token_count,
                paragraph_token_count=paragraph_token_count,
                title_paragraph=title_paragraph,
                local_title_paragraph=local_title_paragraph,
            )

    def _build_json_format_input(self, inputs: list) -> tuple[list, list]:
        """Build the JSON input list for the LLM and the list of ids to translate.

        Returns (json_format_input, should_translate_paragraph).
        """
        json_format_input = []
        should_translate_paragraph = []
        for id_, input_text in enumerate(inputs):
            ti = input_text[1]
            tracker = input_text[3]
            tracker.record_multi_paragraph_index(id_)
            placeholders_hint = ti.get_placeholders_hint()
            obj = {
                "id": id_,
                "input": input_text[0],
                "layout_label": input_text[2].layout_label,
            }
            if placeholders_hint and self.translation_config.add_formula_placehold_hint:
                obj["formula_placeholders_hint"] = placeholders_hint
            json_format_input.append(obj)
            should_translate_paragraph.append(id_)
        return json_format_input, should_translate_paragraph

    def translate_paragraph(
        self,
        batch_paragraph: "BatchParagraph",
        pbar: "tqdm | None" = None,
        page_font_map: "dict[str, PdfFont]" = None,
        xobj_font_map: "dict[int, dict[str, PdfFont]]" = None,
        title_paragraph: "PdfParagraph | None" = None,
        local_title_paragraph: "PdfParagraph | None" = None,
        executor: "PriorityThreadPoolExecutor | None" = None,
        paragraph_token_count: int = 0,
        mp_id: int = 0,
    ):
        """Translate a paragraph using pre and post processing functions."""
        self.translation_config.raise_if_cancelled()
        should_translate_paragraph = []
        try:
            inputs, llm_translate_trackers, paragraph_unicodes = (
                self._build_batch_inputs(
                    batch_paragraph, pbar, page_font_map, xobj_font_map, mp_id
                )
            )
            if not inputs:
                return

            json_format_input, should_translate_paragraph = (
                self._build_json_format_input(inputs)
            )
            # orjson is materially faster than stdlib json for this hot path
            # (runs once per translation batch, i.e. many times per document).
            json_format_input_str = orjson.dumps(
                json_format_input, option=orjson.OPT_INDENT_2
            ).decode()

            batch_text_for_glossary_matching = "\n".join(
                item.get("input", "") for item in json_format_input
            )
            final_input = self._build_llm_prompt(
                json_input_str=json_format_input_str,
                title_paragraph=title_paragraph,
                local_title_paragraph=local_title_paragraph,
                batch_text_for_glossary_matching=batch_text_for_glossary_matching,
            )
            for llm_translate_tracker in llm_translate_trackers:
                llm_translate_tracker.set_input(final_input)
            llm_output = self.translate_engine.llm_translate(
                final_input,
                rate_limit_params={"paragraph_token_count": paragraph_token_count},
                response_schema=BatchTranslationResponse,
            )
            for llm_translate_tracker in llm_translate_trackers:
                llm_translate_tracker.set_output(llm_output)
            translation_results = self._parse_translation_results(
                llm_output, len(inputs)
            )
            for id_, output in translation_results.items():
                self._apply_single_translation(
                    id_,
                    output,
                    inputs,
                    batch_paragraph,
                    llm_translate_trackers,
                    pbar,
                    page_font_map,
                    xobj_font_map,
                    title_paragraph,
                    local_title_paragraph,
                    executor,
                )
        except (ValueError, TypeError, KeyError, RuntimeError) as e:
            self._handle_translate_paragraph_error(
                e,
                llm_translate_trackers,
                inputs,
                should_translate_paragraph,
                batch_paragraph,
                executor,
                page_font_map,
                xobj_font_map,
                title_paragraph,
                local_title_paragraph,
            )

    def _build_llm_role_block(self) -> str:
        """Build the role/system block for the LLM prompt, including domain guidance."""
        return get_domain_role_block(
            getattr(self.translation_config, "domain", None),
            self.translation_config.lang_out,
            getattr(self.translation_config, "custom_system_prompt", None),
        )

    def _build_llm_contextual_hints_block(
        self,
        title_paragraph: PdfParagraph | None,
        local_title_paragraph: PdfParagraph | None,
    ) -> str:
        """Build the contextual hints block for the LLM prompt."""
        contextual_lines: list[str] = []
        hint_idx = 1
        if title_paragraph:
            contextual_lines.append(
                f"{hint_idx}. First title in full text: {title_paragraph.unicode}"
            )
            hint_idx += 1

        if local_title_paragraph and not (
            title_paragraph
            and local_title_paragraph.debug_id == title_paragraph.debug_id
        ):
            contextual_lines.append(
                f"{hint_idx}. The most recent title is: {local_title_paragraph.unicode}"
            )

        if contextual_lines:
            return (
                "## Contextual Hints for Better Translation\n"
                + "\n".join(contextual_lines)
                + "\n"
            )
        return ""

    def _build_llm_glossary_blocks(
        self, batch_text_for_glossary_matching: str
    ) -> tuple[str, str]:
        """Build glossary usage rules and glossary table blocks for the LLM prompt.

        Returns (glossary_usage_rules_block, glossary_tables_block).
        """
        glossary_entries_per_glossary: dict[str, list[tuple[str, str]]] = {}
        if self._cached_glossaries:
            for glossary in self._cached_glossaries:
                active_entries = glossary.get_active_entries_for_text(
                    batch_text_for_glossary_matching
                )
                if active_entries:
                    glossary_entries_per_glossary[glossary.name] = sorted(
                        active_entries
                    )

        if not glossary_entries_per_glossary:
            return "", ""

        glossary_usage_rules_block = (
            "## Glossary\n"
            "If a glossary is provided:\n"
            "- Always use the exact target term.\n"
            "- Apply glossary items even inside tags or when broken by hyphens/line breaks.\n"
            "- If glossary does NOT include a term, translate it naturally.\n\n"
        )
        glossary_table_lines: list[str] = ["## Glossary Tables", ""]
        for glossary_name, entries in glossary_entries_per_glossary.items():
            glossary_table_lines.append(f"### Glossary: {glossary_name}")
            glossary_table_lines.append("")
            glossary_table_lines.append(
                "| Source Term | Target Term |\n|-------------|-------------|"
            )
            for original_source, target_text in entries:
                glossary_table_lines.append(f"| {original_source} | {target_text} |")
            glossary_table_lines.append("")
        glossary_tables_block = "\n".join(glossary_table_lines)
        return glossary_usage_rules_block, glossary_tables_block

    def _build_llm_prompt(
        self,
        json_input_str: str,
        title_paragraph: PdfParagraph | None,
        local_title_paragraph: PdfParagraph | None,
        batch_text_for_glossary_matching: str,
    ) -> str:
        """Build LLM prompt using a single template for easier maintenance."""
        role_block = self._build_llm_role_block()
        contextual_hints_block = self._build_llm_contextual_hints_block(
            title_paragraph, local_title_paragraph
        )
        glossary_usage_rules_block, glossary_tables_block = (
            self._build_llm_glossary_blocks(batch_text_for_glossary_matching)
        )

        return PROMPT_TEMPLATE.substitute(
            role_block=role_block,
            glossary_usage_rules_block=glossary_usage_rules_block,
            contextual_hints_block=contextual_hints_block,
            json_input_str=json_input_str,
            glossary_tables_block=glossary_tables_block,
            lang_out=self.translation_config.lang_out,
        )

    def _clean_json_output(self, llm_output: str) -> str:
        # Clean up JSON output by removing common wrapper tags
        llm_output = llm_output.strip()
        if llm_output.startswith("<json>"):
            llm_output = llm_output[6:]
        if llm_output.endswith("</json>"):
            llm_output = llm_output[:-7]
        if llm_output.startswith("```json"):
            llm_output = llm_output[7:]
        if llm_output.startswith("```"):
            llm_output = llm_output[3:]
        if llm_output.endswith("```"):
            llm_output = llm_output[:-3]
        return llm_output.strip()
