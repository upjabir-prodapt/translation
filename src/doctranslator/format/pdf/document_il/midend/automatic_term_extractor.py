from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import tiktoken
from tqdm import tqdm

from src.doctranslator.format.pdf.document_il import (
    Document as ILDocument,  # Renamed to avoid conflict
)
from src.doctranslator.format.pdf.document_il import (
    PdfParagraph,  # Renamed to avoid conflict
)
from src.doctranslator.format.pdf.document_il.midend.il_translator import Page
from src.doctranslator.format.pdf.document_il.utils.paragraph_helper import (
    is_cid_paragraph,
)
from src.doctranslator.format.pdf.document_il.utils.paragraph_helper import (
    is_placeholder_only_paragraph,
)
from src.doctranslator.format.pdf.document_il.utils.paragraph_helper import (
    is_pure_numeric_paragraph,
)
from src.doctranslator.utils.priority_thread_pool_executor import (
    PriorityThreadPoolExecutor,
)

if TYPE_CHECKING:
    from src.doctranslator.format.pdf.translation_config import TranslationConfig
    from src.doctranslator.translator.translator import BaseTranslator

logger = logging.getLogger(__name__)

LLM_PROMPT_TEMPLATE: str = """
You are an expert multilingual terminologist. Extract key terms from the text and translate them into {target_language}.

### Extraction Rules
1. Include only: named entities (people, orgs, locations, theorem/algorithm names, dates) and domain-specific nouns/noun phrases essential to meaning.
2. No full sentences. Ignore function words.
3. Use minimal noun phrases (≤5 words unless a named entity). No generic academic nouns (e.g., model, case, property) unless part of a standard term.
4. No mathematical items: variables (X1, a, ε), symbols (=, +, →, ⊥⊥, ∈), subscripts/superscripts, formula fragments, mappings (T: H1→H2), etc. Keep only natural-language concepts.
5. Extract each term once. Keep order of first appearance.

### Translation Rules
1. Translate each term into {target_language}.
2. If in the reference glossary, use its translation exactly.
3. Keep proper names in original language unless a well-known translation exists.
4. Ensure consistent translations.

{reference_glossary_section}

### Output Format
- Return ONLY a valid JSON array.
- Each element: {{"src": "...", "tgt": "..."}}.
- No comments, no backticks, no extra text.
- If no terms: [].

### Example
For terms “LLM”, “GPT”:
{example_output}

Input Text:
```
{text_to_process}
```

Return JSON ONLY. NO OTHER TEXT.
Result:
"""


class BatchParagraph:
    def __init__(
        self,
        paragraphs: list[PdfParagraph],
        page_tracker: PageTermExtractTracker,
    ):
        self.paragraphs = paragraphs
        self.tracker = page_tracker.new_paragraph()


class DocumentTermExtractTracker:
    def __init__(self):
        self.page = []

    def new_page(self):
        page = PageTermExtractTracker()
        self.page.append(page)
        return page

    def to_json(self):
        pages = []
        for page in self.page:
            paragraphs = []
            for para in page.paragraph:
                o_str = getattr(para, "output", None)
                i_str = getattr(para, "input", None)
                pdf_unicodes = getattr(para, "pdf_unicodes", None)
                if not pdf_unicodes:
                    continue
                paragraphs.append(
                    {
                        "pdf_unicodes": pdf_unicodes,
                        "output": o_str,
                        "input": i_str,
                    },
                )
            pages.append({"paragraph": paragraphs})
        return json.dumps({"page": pages}, ensure_ascii=False, indent=2)


class PageTermExtractTracker:
    def __init__(self):
        self.paragraph = []

    def new_paragraph(self):
        paragraph = ParagraphTermExtractTracker()
        self.paragraph.append(paragraph)
        return paragraph


class ParagraphTermExtractTracker:
    def __init__(self):
        self.pdf_unicodes = []

    def append_paragraph_unicode(self, unicode: str):
        self.pdf_unicodes.append(unicode)

    def set_output(self, output: str):
        self.output = output

    def set_input(self, _input: str):
        self.input = _input


class AutomaticTermExtractor:
    stage_name = "Automatic Term Extraction"

    def __init__(
        self,
        translate_engine: BaseTranslator,
        translation_config: TranslationConfig,
    ):
        self.translate_engine = translate_engine
        self.translation_config = translation_config
        self.shared_context = translation_config.shared_context_cross_split_part
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

        # Check if the translate_engine has llm_translate capability
        if not hasattr(self.translate_engine, "llm_translate") or not callable(
            self.translate_engine.llm_translate
        ):
            raise ValueError(
                "The provided translate_engine does not support LLM-based translation, which is required for AutomaticTermExtractor."
            )

    def calc_token_count(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return 0

    def _snapshot_token_usage(self) -> tuple[int, int, int, int]:
        if not self.translate_engine:
            return 0, 0, 0, 0
        token_counter = getattr(self.translate_engine, "token_count", None)
        prompt_counter = getattr(self.translate_engine, "prompt_token_count", None)
        completion_counter = getattr(
            self.translate_engine, "completion_token_count", None
        )
        cache_hit_prompt_counter = getattr(
            self.translate_engine, "cache_hit_prompt_token_count", None
        )
        total_tokens = token_counter.value if token_counter else 0
        prompt_tokens = prompt_counter.value if prompt_counter else 0
        completion_tokens = completion_counter.value if completion_counter else 0
        cache_hit_prompt_tokens = (
            cache_hit_prompt_counter.value if cache_hit_prompt_counter else 0
        )
        return total_tokens, prompt_tokens, completion_tokens, cache_hit_prompt_tokens

    def _clean_json_output(self, llm_output: str) -> str:
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

    def _decode_json_safely(self, cleaned_text: str, request_id: str):
        """Parse JSON string and return decoded object, or None on failure."""
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            preview = cleaned_text[:300].replace("\n", "\\n")
            logger.warning(
                f"Request ID {request_id}: Term extraction JSON parse error: {e}. "
                f"response_head={preview!r}",
            )
        except Exception as e:
            logger.warning(
                f"Request ID {request_id}: Unexpected error parsing term extraction output: {e}",
            )
        return None

    def _normalize_extracted_data_to_list(self, extracted_data) -> list | None:
        """Coerce a parsed JSON object into a list of term dicts, or return None."""
        if isinstance(extracted_data, list):
            return extracted_data
        if isinstance(extracted_data, dict):
            for key in ("terms", "items", "results", "data"):
                value = extracted_data.get(key)
                if isinstance(value, list):
                    return value
            if "src" in extracted_data and "tgt" in extracted_data:
                return [extracted_data]
        return None

    def _parse_terms_json(self, llm_response_text: str, request_id: str) -> list[dict]:
        cleaned_response_text = self._clean_json_output(llm_response_text or "")
        if not cleaned_response_text:
            logger.warning(
                f"Request ID {request_id}: Empty term extraction LLM output after cleaning",
            )
            return []

        extracted_data = self._decode_json_safely(cleaned_response_text, request_id)
        if extracted_data is None:
            return []

        result = self._normalize_extracted_data_to_list(extracted_data)
        if result is None:
            logger.warning(
                f"Request ID {request_id}: Term extraction output is not a JSON list "
                f"(got {type(extracted_data).__name__})",
            )
            return []
        return result

    def _process_llm_response(self, llm_response_text: str, request_id: str):
        extracted_data = self._parse_terms_json(llm_response_text, request_id)
        for item in extracted_data:
            if isinstance(item, dict) and "src" in item and "tgt" in item:
                src_term = str(item["src"]).strip()
                tgt_term = str(item["tgt"]).strip()
                if src_term and tgt_term and len(src_term) < 100:
                    self.shared_context.add_raw_extracted_term_pair(
                        src_term,
                        tgt_term,
                    )
            else:
                logger.warning(
                    f"Request ID {request_id}: Skipping malformed term item: {item}",
                )

    def _should_skip_paragraph(self, paragraph, pbar) -> bool:
        """Return True and advance pbar if this paragraph should be excluded from extraction."""
        if paragraph.debug_id is None or paragraph.unicode is None:
            pbar.advance(1)
            return True
        if is_cid_paragraph(paragraph):
            pbar.advance(1)
            return True
        if is_pure_numeric_paragraph(paragraph):
            pbar.advance(1)
            return True
        if is_placeholder_only_paragraph(paragraph):
            pbar.advance(1)
            return True
        return False

    def _submit_batch(
        self,
        paragraphs: list,
        tracker: PageTermExtractTracker,
        executor: PriorityThreadPoolExecutor,
        pbar,
        total_token_count: int,
    ) -> None:
        """Submit a batch of paragraphs to the executor for term extraction."""
        executor.submit(
            self.extract_terms_from_paragraphs,
            BatchParagraph(paragraphs, tracker),
            pbar,
            total_token_count,
            priority=1048576 - total_token_count,
        )

    def _is_batch_full(self, total_token_count: int, paragraphs: list) -> bool:
        """Return True when the current batch exceeds token or paragraph limits."""
        max_tokens = self.translation_config.llm_term_extraction_batch_max_tokens
        max_paragraphs = (
            self.translation_config.llm_term_extraction_batch_max_paragraphs
        )
        return total_token_count > max_tokens or len(paragraphs) > max_paragraphs

    def process_page(
        self,
        page: Page,
        executor: PriorityThreadPoolExecutor,
        pbar: tqdm | None = None,
        tracker: PageTermExtractTracker = None,
    ):
        self.translation_config.raise_if_cancelled()
        paragraphs = []
        total_token_count = 0
        for paragraph in page.pdf_paragraph:
            if self._should_skip_paragraph(paragraph, pbar):
                continue
            total_token_count += self.calc_token_count(paragraph.unicode)
            paragraphs.append(paragraph)
            if self._is_batch_full(total_token_count, paragraphs):
                self._submit_batch(
                    paragraphs, tracker, executor, pbar, total_token_count
                )
                paragraphs = []
                total_token_count = 0

        if paragraphs:
            self._submit_batch(paragraphs, tracker, executor, pbar, total_token_count)

    def _build_reference_glossary_section(self, inputs: list[str]) -> str:
        """Build the reference glossary section string for the LLM prompt."""
        user_glossaries = self.shared_context.user_glossaries
        if not user_glossaries:
            return ""

        text_for_glossary = "\n\n".join(inputs)
        glossary_entries = {}
        for glossary in user_glossaries:
            active_entries = glossary.get_active_entries_for_text(text_for_glossary)
            if active_entries:
                glossary_entries[glossary.name] = active_entries

        if not glossary_entries:
            return ""

        section = "Reference Glossaries (for consistency and quality):\n"
        for glossary_name, entries in glossary_entries.items():
            section += f"\n{glossary_name}:\n"
            for src, tgt in sorted(set(entries)):
                section += f"- {src} → {tgt}\n"
        section += (
            "\nPlease consider these existing translations for consistency when extracting "
            "new terms. IMPORTANT: You should also extract terms that appear in the reference "
            "glossaries above if they are found in the input text - don't skip them just "
            "because they already exist in the reference."
        )
        return section

    def _store_valid_term(self, term: dict, request_id: str) -> bool:
        """Validate and store a single extracted term pair. Returns True if stored."""
        if not (isinstance(term, dict) and "src" in term and "tgt" in term):
            logger.warning(
                f"Request ID {request_id}: Skipping malformed term item: {term}",
            )
            return False
        src_term = str(term["src"]).strip()
        tgt_term = str(term["tgt"]).strip()
        if src_term == tgt_term and len(src_term) < 3:
            return False
        if src_term and tgt_term and len(src_term) < 100:
            self.shared_context.add_raw_extracted_term_pair(src_term, tgt_term)
            return True
        return False

    def extract_terms_from_paragraphs(
        self,
        paragraphs: BatchParagraph,
        pbar: tqdm | None = None,
        paragraph_token_count: int = 0,
    ):
        self.translation_config.raise_if_cancelled()
        try:
            inputs = [p.unicode for p in paragraphs.paragraphs if p.unicode]
            tracker = paragraphs.tracker
            for u in inputs:
                tracker.append_paragraph_unicode(u)
            if not inputs:
                return
            request_id = (
                f"ate-{id(paragraphs)}-"
                f"{paragraphs.paragraphs[0].debug_id if paragraphs.paragraphs else 'na'}"
            )

            reference_glossary_section = self._build_reference_glossary_section(inputs)

            prompt = LLM_PROMPT_TEMPLATE.format(
                target_language=self.translation_config.lang_out,
                text_to_process="\n\n".join(inputs),
                reference_glossary_section=reference_glossary_section,
                example_output="""[
  {"src": "LLM", "tgt": "大语言模型"},
  {"src": "GPT", "tgt": "GPT"}
]""",
            )
            tracker.set_input(prompt)
            output = self.translate_engine.llm_translate(
                prompt,
                rate_limit_params={
                    "paragraph_token_count": paragraph_token_count,
                    "request_json_mode": True,
                },
            )
            tracker.set_output(output)
            response = self._parse_terms_json(output, request_id)
            if not response:
                logger.warning(
                    f"Request ID {request_id}: No valid extracted terms parsed from LLM output",
                )
                return

            valid_terms = sum(
                1 for term in response if self._store_valid_term(term, request_id)
            )
            logger.debug(
                f"Request ID {request_id}: Parsed {valid_terms} valid term pairs "
                f"from {len(response)} extracted items",
            )

        except Exception as e:
            logger.warning(
                f"Error during automatic terms extract: {e}. "
                f"paragraph_count={len(paragraphs.paragraphs)} token_count={paragraph_token_count}",
            )
            return
        finally:
            if pbar:
                pbar.advance(len(paragraphs.paragraphs))

    def procress(self, doc_il: ILDocument):
        logger.info(f"{self.stage_name}: Starting term extraction for document.")
        start_total, start_prompt, start_completion, start_cache_hit_prompt = (
            self._snapshot_token_usage()
        )
        tracker = DocumentTermExtractTracker()
        total = sum(len(page.pdf_paragraph) for page in doc_il.page)
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total,
        ) as pbar:
            max_workers = self.translation_config.term_pool_max_workers
            logger.info(
                f"Using {max_workers} worker threads for automatic term extraction."
            )
            with PriorityThreadPoolExecutor(
                max_workers=max_workers,
            ) as executor:
                for page in doc_il.page:
                    self.process_page(page, executor, pbar, tracker.new_page())

        self.shared_context.finalize_auto_extracted_glossary()
        end_total, end_prompt, end_completion, end_cache_hit_prompt = (
            self._snapshot_token_usage()
        )
        self.translation_config.record_term_extraction_usage(
            end_total - start_total,
            end_prompt - start_prompt,
            end_completion - start_completion,
            end_cache_hit_prompt - start_cache_hit_prompt,
        )

        if (
            self.translation_config.debug
            or self.translation_config.working_dir is not None
        ):
            path = self.translation_config.get_working_file_path(
                "term_extractor_tracking.json"
            )
            logger.debug(f"save translate tracking to {path}")
            with Path(path).open("w", encoding="utf-8") as f:
                f.write(tracker.to_json())

            path = self.translation_config.get_working_file_path(
                "term_extractor_freq.json"
            )
            logger.debug(f"save term frequency to {path}")
            with Path(path).open("w", encoding="utf-8") as f:
                json.dump(
                    self.shared_context.raw_extracted_terms,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            path = self.translation_config.get_working_file_path(
                "auto_extractor_glossary.csv"
            )
            logger.debug(f"save auto extracted glossary to {path}")
            with Path(path).open("w", encoding="utf-8") as f:
                auto_extracted_glossary = self.shared_context.auto_extracted_glossary
                if auto_extracted_glossary:
                    f.write(auto_extracted_glossary.to_csv())
