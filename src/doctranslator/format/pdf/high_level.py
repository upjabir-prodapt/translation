import asyncio
import concurrent.futures
import copy
import hashlib
import io
import logging
import pathlib
import re
import shutil
import threading
import time
from asyncio import CancelledError
from pathlib import Path
from typing import Any
from typing import BinaryIO

import pymupdf
from pymupdf import Document
from pymupdf import Font

import src.doctranslator.asynchronize as asynchronize
from src.api.services.dlp_service import DlpService
from src.config.constants import settings
from src.doctranslator.doctranslator_exception.DocTranslatorException import (
    ExtractTextError,
)
from src.doctranslator.doctranslator_exception.DocTranslatorException import (
    InputFileGeneratedByDocTranslatorError,
)
from src.doctranslator.format.pdf.converter import TranslateConverter
from src.doctranslator.format.pdf.dlp_adapter import apply_dlp_to_document
from src.doctranslator.format.pdf.dlp_adapter import unmask_document_with_tokens
from src.doctranslator.format.pdf.document_il import il_version_1
from src.doctranslator.format.pdf.document_il.backend.pdf_creater import (
    SAVE_PDF_STAGE_NAME,
)
from src.doctranslator.format.pdf.document_il.backend.pdf_creater import (
    SUBSET_FONT_STAGE_NAME,
)
from src.doctranslator.format.pdf.document_il.backend.pdf_creater import PDFCreater
from src.doctranslator.format.pdf.document_il.backend.pdf_creater import reproduce_cmap
from src.doctranslator.format.pdf.document_il.frontend.il_creater import ILCreater
from src.doctranslator.format.pdf.document_il.midend.add_debug_information import (
    AddDebugInformation,
)
from src.doctranslator.format.pdf.document_il.midend.automatic_term_extractor import (
    AutomaticTermExtractor,
)
from src.doctranslator.format.pdf.document_il.midend.detect_scanned_file import (
    DetectScannedFile,
)
from src.doctranslator.format.pdf.document_il.midend.il_translator import ILTranslator
from src.doctranslator.format.pdf.document_il.midend.il_translator_llm_only import (
    ILTranslatorLLMOnly,
)
from src.doctranslator.format.pdf.document_il.midend.layout_parser import LayoutParser
from src.doctranslator.format.pdf.document_il.midend.paragraph_finder import (
    ParagraphFinder,
)
from src.doctranslator.format.pdf.document_il.midend.styles_and_formulas import (
    StylesAndFormulas,
)
from src.doctranslator.format.pdf.document_il.midend.table_parser import TableParser
from src.doctranslator.format.pdf.document_il.midend.typesetting import Typesetting
from src.doctranslator.format.pdf.document_il.utils.fontmap import FontMapper
from src.doctranslator.format.pdf.document_il.xml_converter import XMLConverter
from src.doctranslator.format.pdf.pdfinterp import PDFPageInterpreterEx
from src.doctranslator.format.pdf.result_merger import ResultMerger
from src.doctranslator.format.pdf.split_manager import SplitManager
from src.doctranslator.format.pdf.translation_config import TranslateResult
from src.doctranslator.format.pdf.translation_config import TranslationConfig
from src.doctranslator.format.pdf.translation_config import WatermarkOutputMode
from src.doctranslator.pdfminer.pdfdocument import PDFDocument
from src.doctranslator.pdfminer.pdfinterp import PDFResourceManager
from src.doctranslator.pdfminer.pdfpage import PDFPage
from src.doctranslator.pdfminer.pdfparser import PDFParser
from src.doctranslator.progress_monitor import ProgressMonitor
from src.doctranslator.utils import memory
from src.doctranslator.utils.common import close_process_pool

logger = logging.getLogger(__name__)

TRANSLATE_STAGES = [
    (ILCreater.stage_name, 14.12),  # Parse PDF and Create IR
    (DetectScannedFile.stage_name, 2.45),  # DetectScannedFile
    (LayoutParser.stage_name, 14.03),  # Parse Page Layout
    (TableParser.stage_name, 1.0),  # Parse Table
    (ParagraphFinder.stage_name, 6.26),  # Parse Paragraphs
    (StylesAndFormulas.stage_name, 1.66),  # Parse Formulas and Styles
    # (RemoveDescent.stage_name, 0.15),  # Remove Char Descent
    (AutomaticTermExtractor.stage_name, 30.0),  # Extract Terms
    (ILTranslator.stage_name, 46.96),  # Translate Paragraphs
    (Typesetting.stage_name, 4.71),  # Typesetting
    (FontMapper.stage_name, 0.61),  # Add Fonts
    (PDFCreater.stage_name, 1.96),  # Generate drawing instructions
    (SUBSET_FONT_STAGE_NAME, 0.92),  # Subset font
    (SAVE_PDF_STAGE_NAME, 6.34),  # Save PDF
]

resfont_map = {
    "zh-cn": "china-ss",
    "zh-tw": "china-ts",
    "zh-hans": "china-ss",
    "zh-hant": "china-ts",
    "zh": "china-ss",
    "ja": "japan-s",
    "ko": "korea-s",
}

AUTO_FIX_FAILED_MSG = "auto fix failed, please check the pdf file"


def safe_save(doc, *args, **kwargs):
    try:
        # first try, saving without options
        doc.save(*args, **kwargs)
    except Exception:
        # second try, saving with 'garbage=3' for object missing
        doc.ez_save(*args, **kwargs)


def check_metadata(pdf: Document):
    meta = pdf.metadata
    if not meta:
        return
    producer = meta.get("producer", None)
    disclaimer = "Translation_generated_by_AI,please_carefully_discern"
    if (
        producer
        and disclaimer in producer
        and ("BabelDOC" in producer or "DocTranslator" in producer)
    ):
        raise InputFileGeneratedByDocTranslatorError(
            "Input file was already translated by DocTranslator or legacy BabelDOC; "
            "cannot translate again."
        )


def _build_creator_field(
    existing_creator: str | None, producer: str | None
) -> str | None:
    """Combine existing creator and producer into a single creator string."""
    if not producer:
        return existing_creator
    if not existing_creator:
        return producer
    return f"{existing_creator}, {producer}"


def _build_translated_by_string(translate_config: TranslationConfig) -> str:
    """Build the producer string that marks the file as translated by DocTranslator."""
    translated_by = (
        f"DocTranslator{settings.WATERMARK_VERSION}_{time.time()}"
        "_Translation_generated_by_AI,please_carefully_discern"
    )
    if translate_config.metadata_extra_data:
        translated_by += f"_{translate_config.metadata_extra_data}"
    return translated_by


def _sanitize_metadata_values(meta: dict) -> dict:
    """Remove surrogate characters from all string metadata values."""
    for k, v in meta.items():
        if v:
            meta[k] = re.sub(r"[\uD800-\uDFFF]", "", v)
    return meta


def add_metadata(
    translate_result: TranslateResult, translate_config: TranslationConfig
):
    processed = []
    for attr in (
        "mono_pdf_path",
        "dual_pdf_path",
        "no_watermark_mono_pdf_path",
        "no_watermark_dual_pdf_path",
    ):
        path = getattr(translate_result, attr)
        if not path or path in processed:
            continue
        processed.append(path)

        temp_path = translate_config.get_working_file_path(f"{path.stem}.cmap.pdf")
        pdf = pymupdf.open(path)
        meta = pdf.metadata or {}
        meta["creator"] = _build_creator_field(
            meta.get("creator"), meta.get("producer")
        )
        meta["producer"] = _build_translated_by_string(translate_config)
        meta = _sanitize_metadata_values(meta)
        pdf.set_metadata(meta)
        safe_save(pdf, temp_path)
        shutil.move(temp_path, path)


def fix_cmap(translate_result: TranslateResult, translate_config: TranslationConfig):
    processed = []
    for attr in (
        "mono_pdf_path",
        "dual_pdf_path",
        "no_watermark_mono_pdf_path",
        "no_watermark_dual_pdf_path",
    ):
        path = getattr(translate_result, attr)
        if not path or path in processed:
            continue
        processed.append(path)

        temp_path = translate_config.get_working_file_path(f"{path.stem}.cmap.pdf")
        pdf = pymupdf.open(path)
        reproduce_cmap(pdf)
        safe_save(pdf, temp_path)
        shutil.move(temp_path, path)


def verify_file_hash(file_path: str, expected_hash: str) -> bool:
    """Verify the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        # Read the file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash


def translator_supports_llm(translator) -> bool:
    if not translator or not hasattr(translator, "do_llm_translate"):
        return False
    try:
        translator.do_llm_translate(None)
        return True
    except NotImplementedError:
        return False
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug(f"translator {translator} failed llm detection: {exc}")
        return False


class ParseILOptions:
    """Optional low-level parameters for :func:`start_parse_il`.

    Grouping these here keeps the function signature within the 13-parameter limit.
    """

    def __init__(
        self,
        vfont: str = "",
        vchar: str = "",
        thread: int = 0,
        lang_in: str = "",
        lang_out: str = "",
        service: str = "",
        noto: Font = None,
        cancellation_event: asyncio.Event = None,
        envs: dict | None = None,
        prompt: list | None = None,
    ):
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.service = service
        self.noto = noto
        self.cancellation_event = cancellation_event
        self.envs = envs or {}
        self.prompt = prompt or []


def start_parse_il(
    inf: BinaryIO,
    pages: list[int] | None = None,
    doc_zh: Document = None,
    resfont: str = "",
    il_creater: ILCreater = None,
    translation_config: TranslationConfig = None,
    options: ParseILOptions | None = None,
    **kwarg: Any,
) -> None:
    opts = options or ParseILOptions(
        vfont=kwarg.get("vfont", ""),
        vchar=kwarg.get("vchar", ""),
        thread=kwarg.get("thread", 0),
        lang_in=kwarg.get("lang_in", ""),
        lang_out=kwarg.get("lang_out", ""),
        service=kwarg.get("service", ""),
        noto=kwarg.get("noto", None),
        cancellation_event=kwarg.get("cancellation_event", None),
        envs=kwarg.get("envs", {}),
        prompt=kwarg.get("prompt", []),
    )
    rsrcmgr = PDFResourceManager()
    layout = {}
    device = TranslateConverter(
        rsrcmgr,
        opts.vfont,
        opts.vchar,
        opts.thread,
        layout,
        opts.lang_in,
        opts.lang_out,
        opts.service,
        resfont,
        opts.noto,
        opts.envs,
        opts.prompt,
        il_creater=il_creater,
    )

    if il_creater is None:
        raise AssertionError
    if translation_config is None:
        raise AssertionError
    obj_patch = {}
    interpreter = PDFPageInterpreterEx(rsrcmgr, device, obj_patch, il_creater)
    if pages:
        total_pages = len(pages)
    else:
        total_pages = doc_zh.page_count

    il_creater.on_total_pages(total_pages)

    parser = PDFParser(inf)
    doc = PDFDocument(parser)

    for pageno, page in enumerate(PDFPage.create_pages(doc)):
        if opts.cancellation_event and opts.cancellation_event.is_set():
            raise CancelledError("task cancelled")
        if pages and (pageno not in pages):
            continue
        page.pageno = pageno

        if not translation_config.should_translate_page(pageno + 1):
            continue

        height, width = (
            page.cropbox[3] - page.cropbox[1],
            page.cropbox[2] - page.cropbox[0],
        )
        if height > 1200 or width > 2000:
            logger.warning(f"page {pageno + 1} is too large, maybe unable to translate")
            # continue

        translation_config.raise_if_cancelled()
        # The current program no longer relies on
        # the following layout recognition results,
        # but in order to facilitate the migration of pdf2zh,
        # the relevant code is temporarily retained.
        # pix = doc_zh[page.pageno].get_pixmap()
        # image = np.frombuffer(pix.samples, np.uint8).reshape(
        #     pix.height, pix.width, 3
        # )[:, :, ::-1]
        # page_layout = model.predict(
        #     image, imgsz=int(pix.height / 32) * 32)[0]
        # # kdtree 是不可能 kdtree 的，不如直接渲染成图片，用空间换时间
        # box = np.ones((pix.height, pix.width))
        # h, w = box.shape
        # vcls = ["abandon", "figure", "table",
        #         "isolate_formula", "formula_caption"]
        # for i, d in enumerate(page_layout.boxes):
        #     if page_layout.names[int(d.cls)] not in vcls:
        #         x0, y0, x1, y1 = d.xyxy.squeeze()
        #         x0, y0, x1, y1 = (
        #             np.clip(int(x0 - 1), 0, w - 1),
        #             np.clip(int(h - y1 - 1), 0, h - 1),
        #             np.clip(int(x1 + 1), 0, w - 1),
        #             np.clip(int(h - y0 + 1), 0, h - 1),
        #         )
        #         box[y0:y1, x0:x1] = i + 2
        # for i, d in enumerate(page_layout.boxes):
        #     if page_layout.names[int(d.cls)] in vcls:
        #         x0, y0, x1, y1 = d.xyxy.squeeze()
        #         x0, y0, x1, y1 = (
        #             np.clip(int(x0 - 1), 0, w - 1),
        #             np.clip(int(h - y1 - 1), 0, h - 1),
        #             np.clip(int(x1 + 1), 0, w - 1),
        #             np.clip(int(h - y0 + 1), 0, h - 1),
        #         )
        #         box[y0:y1, x0:x1] = 0
        # layout[page.pageno] = box
        # 新建一个 xref 存放新指令流
        # page.page_xref = doc_zh.get_new_xref()  # hack 插入页面的新 xref
        # doc_zh.update_object(page.page_xref, "<<>>")
        # doc_zh.update_stream(page.page_xref, b"")
        # doc_zh[page.pageno].set_contents(page.page_xref)
        ops_base = interpreter.process_page(page)
        il_creater.on_page_base_operation(ops_base)
        il_creater.on_page_end()
    il_creater.on_finish()
    device.close()


def translate(translation_config: TranslationConfig) -> TranslateResult:
    with ProgressMonitor(get_translation_stage(translation_config)) as pm:
        return do_translate(pm, translation_config)


def get_translation_stage(
    translation_config: TranslationConfig,
) -> list[tuple[str, float]]:
    result = copy.deepcopy(TRANSLATE_STAGES)
    should_remove = []

    # If only parsing and generating PDF, skip all translation-related stages
    if translation_config.only_parse_generate_pdf:
        should_remove.extend(
            [
                DetectScannedFile.stage_name,
                LayoutParser.stage_name,
                TableParser.stage_name,
                ParagraphFinder.stage_name,
                StylesAndFormulas.stage_name,
                AutomaticTermExtractor.stage_name,
                ILTranslator.stage_name,
                Typesetting.stage_name,
            ]
        )
    else:
        # Original logic for selective removal
        if not translation_config.table_model:
            should_remove.append(TableParser.stage_name)
        if translation_config.skip_scanned_detection:
            should_remove.append(DetectScannedFile.stage_name)
        if not translation_config.auto_extract_glossary:
            should_remove.append(AutomaticTermExtractor.stage_name)
        if translation_config.skip_translation:
            should_remove.append(ILTranslator.stage_name)

    result = [x for x in result if x[0] not in should_remove]
    return result


def _apply_dlp_if_enabled(
    docs: il_version_1.Document,
    translation_config: TranslationConfig,
    *,
    stage_label: str,
    allow_repeat: bool = False,
) -> None:
    """Apply IL-level DLP masking before translation stages."""
    if not translation_config.enable_dlp:
        return
    if translation_config.dlp_applied_pre_translation and not allow_repeat:
        return

    job_id = (translation_config.dlp_job_id or "").strip()
    if not job_id:
        logger.warning("DLP enabled but no dlp_job_id configured, skipping DLP masking")
        return

    source_language = (
        translation_config.dlp_source_language or translation_config.lang_in or ""
    ).strip()
    if not source_language:
        logger.warning(
            "DLP enabled but no source language configured, skipping DLP masking",
        )
        return

    dlp_result = apply_dlp_to_document(
        docs=docs,
        dlp_service=DlpService(),
        job_id=job_id,
        source_language=source_language,
        token_counter_start=translation_config.dlp_token_counter,
    )
    translation_config.dlp_provider = dlp_result.dlp_provider
    if not dlp_result.applied:
        logger.debug(f"DLP found no eligible IL text during stage '{stage_label}'")
        return
    if allow_repeat:
        if dlp_result.token_rows:
            translation_config.dlp_token_rows.extend(dlp_result.token_rows)
            translation_config.dlp_token_counter += len(dlp_result.token_rows)
        logger.info(
            f"Applied optional post-translation DLP during stage '{stage_label}' on {dlp_result.chunk_count} chunks"
        )
        return

    translation_config.dlp_token_rows = dlp_result.token_rows
    translation_config.dlp_chunk_mode = "il_paragraph"
    translation_config.dlp_applied_pre_translation = True
    translation_config.dlp_chunk_count = dlp_result.chunk_count
    translation_config.dlp_token_counter += len(dlp_result.token_rows)
    logger.info(
        f"Applied IL DLP during stage '{stage_label}' on {dlp_result.chunk_count} chunks with {len(dlp_result.token_rows)} tokens"
    )


def _handle_async_translate_interrupt(
    exc: BaseException, cancel_event: threading.Event
) -> None:
    """Set the cancellation event and log if the exception is a KeyboardInterrupt."""
    if isinstance(exc, KeyboardInterrupt):
        logger.info(
            "Translation cancelled by user through keyboard interrupt",
        )
    cancel_event.set()


def _unmask_before_pdf_if_enabled(
    docs: il_version_1.Document, translation_config: TranslationConfig
) -> None:
    """Restore masked tokens before typesetting/PDF write."""
    if not translation_config.enable_dlp:
        return
    if not translation_config.dlp_unmask_before_pdf:
        return
    if not translation_config.dlp_token_rows:
        return
    replaced_count = unmask_document_with_tokens(
        docs=docs,
        token_rows=translation_config.dlp_token_rows,
    )
    logger.info(f"Restored {replaced_count} masked values before PDF generation")


async def async_translate(translation_config: TranslationConfig):
    """Asynchronously translate a PDF file with real-time progress reporting.

    This function yields progress events that can be used to update progress bars
    or other UI elements. The events are dictionaries with the following structure:

    - progress_start: {
        "type": "progress_start",
        "stage": str,              # Stage name
        "stage_progress": float,   # Always 0.0
        "stage_current": int,      # Current count (0)
        "stage_total": int         # Total items in stage
    }
    - progress_update: {
        "type": "progress_update",
        "stage": str,              # Stage name
        "stage_progress": float,   # Stage progress (0-100)
        "stage_current": int,      # Current items processed
        "stage_total": int,        # Total items in stage
        "overall_progress": float  # Overall progress (0-100)
    }
    - progress_end: {
        "type": "progress_end",
        "stage": str,              # Stage name
        "stage_progress": float,   # Always 100.0
        "stage_current": int,      # Equal to stage_total
        "stage_total": int,        # Total items processed
        "overall_progress": float  # Overall progress (0-100)
    }
    - finish: {
        "type": "finish",
        "translate_result": TranslateResult
    }
    - error: {
        "type": "error",
        "error": str
    }

    Args:
        translation_config: Configuration for the translation process

    Yields:
        dict: Progress events during translation

    Raises:
        CancelledError: If the translation is cancelled
        Exception: Any other errors during translation
    """
    loop = asyncio.get_running_loop()
    callback = asynchronize.AsyncCallback()

    finish_event = asyncio.Event()
    cancel_event = threading.Event()
    with ProgressMonitor(
        get_translation_stage(translation_config),
        progress_change_callback=callback.step_callback,
        finish_callback=callback.finished_callback,
        finish_event=finish_event,
        cancel_event=cancel_event,
        loop=loop,
        report_interval=translation_config.report_interval,
    ) as pm:
        future = loop.run_in_executor(None, do_translate, pm, translation_config)
        try:
            async for event in callback:
                event = event.kwargs
                yield event
                if event["type"] == "error":
                    break
        except (CancelledError, KeyboardInterrupt) as exc:
            _handle_async_translate_interrupt(exc, cancel_event)
    if cancel_event.is_set():
        future.cancel()
    logger.info("Waiting for translation to finish...")
    await finish_event.wait()


class MemoryMonitor:
    """Monitor memory usage of current process and all child processes."""

    def __init__(self, interval=0.1):
        """Initialize memory monitor.

        Args:
            interval: Monitoring interval in seconds, defaults to 0.1s (100ms)
        """
        self.interval = interval
        self.peak_memory_usage = 0
        self.monitor_thread = None
        self.stop_event = None
        self.last_pss_check_time = None

    def __enter__(self):
        """Start memory monitoring."""
        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory_usage, daemon=True
        )
        self.monitor_thread.start()
        logger.debug("Memory monitoring started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and log peak memory usage."""
        if not self.monitor_thread:
            return

        self.stop_event.set()
        self.monitor_thread.join(timeout=2.0)
        logger.info(f"Peak memory usage: {self.peak_memory_usage:.2f} MB")

    def _monitor_memory_usage(self):
        """Background thread that periodically checks memory usage."""
        while not self.stop_event.is_set():
            try:
                # Use throttled memory check with 2-second PSS throttle
                total_memory, self.last_pss_check_time = (
                    memory.get_memory_usage_with_throttle(
                        include_children=True,
                        prefer_pss=True,
                        last_pss_check_time=self.last_pss_check_time,
                        pss_throttle_seconds=2.0,
                    )
                )

                # Convert to MB for better readability
                total_memory_mb = total_memory / (1024 * 1024)
                if total_memory_mb > self.peak_memory_usage:
                    self.peak_memory_usage = total_memory_mb
            except Exception as e:
                logger.warning(f"Error monitoring memory: {e}")

            time.sleep(self.interval)

    def get_peek_memory_psutil(self):
        """Get peak memory usage using psutil (for backwards compatibility)."""
        return memory.get_memory_usage_bytes(include_children=True, prefer_pss=True)


def fix_null_page_content(doc: Document) -> list[int]:
    invalid_page = []
    for x in range(len(doc)):
        xref = doc[x].xref
        if doc.xref_object(xref) == "null":
            invalid_page.append(x)
    for x in invalid_page:
        doc.delete_page(x)
        doc.insert_page(x)
    return invalid_page


def _fix_single_xref_entry(doc: Document, i: int) -> None:
    """Apply null-xref fixes for a single xref index."""
    obj = doc.xref_object(i)
    if obj == "null":
        doc.update_object(i, "[]")
    elif obj and (
        "/ASCII85Decode" in obj or "/LZWDecode" in obj
    ):  # make pdfminer happy
        data = doc.xref_stream(i)
        doc.update_stream(i, data)
    elif obj and "/Annots" in obj:
        doc.xref_set_key(i, "Annots", "null")


def fix_null_xref(doc: Document) -> None:
    """Fix null xref in PDF file by replacing them with empty arrays.

    Args:
        doc: PyMuPDF Document object to fix
    """
    for i in range(1, doc.xref_length()):
        try:
            _fix_single_xref_entry(doc, i)
        except Exception:
            doc.update_object(i, "[]")


def _resolve_xref_filters(doc) -> None:
    """Rewrite any xref-filtered page content streams so they are inline."""
    page_contents = []
    for page in doc:
        page_contents.extend(page.get_contents())
    for page_piece in page_contents:
        f = doc.xref_get_key(page_piece, "Filter")
        if f[0] == "xref":
            data = doc.xref_stream(page_piece)
            doc.update_stream(page_piece, data)


def _merge_multi_content_streams(doc) -> None:
    """Merge pages that have more than one content stream into a single stream."""
    for page in doc:
        contents = page.get_contents()
        if len(contents) > 1:
            page_streams = [doc.xref_stream(i) for i in contents]
            r = doc.get_new_xref()
            doc.update_object(r, "<<>>")
            doc.update_stream(r, b" ".join(page_streams))
            doc.xref_set_key(page.xref, "Contents", f"{r} 0 R")


def fix_filter(doc):
    _resolve_xref_filters(doc)
    _merge_multi_content_streams(doc)


def update_page_bbox(doc, page, box, key):
    if doc.xref_get_key(page.xref, key)[0] == "array":
        doc.xref_set_key(page.xref, key, f"[{box.x0} {box.y0} {box.x1} {box.y1}]")


def _build_part_config(
    i: int,
    split_point,
    translation_config: TranslationConfig,
    original_doc: Document,
    results: dict,
) -> TranslationConfig | None:
    """Create a per-part config and extract the sub-PDF; returns None when the part is skipped."""
    part_config = copy.copy(translation_config)
    part_config.skip_clean = True
    should_translate_pages = [
        page - split_point.start_page + 1
        for page in range(split_point.start_page, split_point.end_page + 1)
        if translation_config.should_translate_page(page + 1)
    ]
    part_config.pages = None
    part_config.page_ranges = [(x, x) for x in should_translate_pages]
    if translation_config.only_include_translated_page and not should_translate_pages:
        results[i] = None
        return None

    if i > 0:
        part_config.skip_scanned_detection = True

    part_config.working_dir = translation_config.get_part_working_dir(i)
    part_config.output_dir = translation_config.get_part_output_dir(i)

    part_temp_input_path = part_config.get_working_file_path(f"input.part{i}.pdf")
    part_config.input_file = part_temp_input_path

    temp_doc = Document()
    for x in range(split_point.start_page, split_point.end_page + 1):
        xref = original_doc[x].xref
        if original_doc.xref_get_key(xref, "Annots")[0] != "null":
            original_doc.xref_set_key(xref, "Annots", "null")
    temp_doc.insert_pdf(
        original_doc, from_page=split_point.start_page, to_page=split_point.end_page
    )
    safe_save(temp_doc, part_temp_input_path)
    if temp_doc.page_count != split_point.end_page - split_point.start_page + 1:
        raise AssertionError

    if i > 0:
        part_config.watermark_output_mode = WatermarkOutputMode.NoWatermark
    return part_config


def _merge_part_dlp_into_parent(
    part_config: TranslationConfig,
    translation_config: TranslationConfig,
    dlp_merge_lock: threading.Lock,
) -> None:
    """Thread-safe merge of DLP token data from a completed part into the parent config."""
    if not (part_config.enable_dlp and part_config.dlp_token_rows):
        return
    with dlp_merge_lock:
        chunk_offset = translation_config.dlp_chunk_count
        for row in part_config.dlp_token_rows:
            merged_row = dict(row)
            merged_row["chunk_index"] = int(merged_row.get("chunk_index", 0)) + int(
                chunk_offset
            )
            translation_config.dlp_token_rows.append(merged_row)
        translation_config.dlp_chunk_count += int(part_config.dlp_chunk_count)
        translation_config.dlp_token_counter = max(
            int(translation_config.dlp_token_counter),
            int(part_config.dlp_token_counter),
        )
        translation_config.dlp_provider = (
            part_config.dlp_provider or translation_config.dlp_provider
        )
        translation_config.dlp_chunk_mode = "il_paragraph"


def _run_split_translation(
    pm: ProgressMonitor,
    translation_config: TranslationConfig,
    original_pdf_path,
    split_points: list,
) -> TranslateResult:
    """Execute translation for each split part in parallel then merge."""
    pm.total_parts = len(split_points)
    _mw = max(1, int(settings.SPLIT_PART_MAX_CONCURRENT))
    logger.info(
        f"[pdf_translate] Parallel part workers: {_mw} (SPLIT_PART_MAX_CONCURRENT)"
    )
    results: dict[int, TranslateResult | None] = {}
    original_watermark_mode = translation_config.watermark_output_mode
    original_doc = Document(original_pdf_path)
    part_configs: dict[int, TranslationConfig] = {}
    dlp_merge_lock = threading.Lock()

    for i, split_point in enumerate(split_points):
        part_config = _build_part_config(
            i, split_point, translation_config, original_doc, results
        )
        if part_config is not None:
            part_configs[i] = part_config

    def run_part(part_idx: int, part_cfg: TranslationConfig):
        part_monitor = pm.create_part_monitor(part_idx, len(split_points))
        return _do_translate_single(part_monitor, part_cfg)

    max_part_workers = max(1, int(settings.SPLIT_PART_MAX_CONCURRENT))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_part_workers
    ) as executor:
        futures = {
            executor.submit(run_part, idx, cfg): idx
            for idx, cfg in part_configs.items()
        }
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            part_config = part_configs[i]
            try:
                part_result = future.result()
                results[i] = part_result
                _wd = getattr(part_config, "working_dir", "")
                logger.info(
                    f"[pdf_translate] Split part done: "
                    f"idx={i + 1}/{len(split_points)} working_dir={_wd}"
                )
                _merge_part_dlp_into_parent(
                    part_config, translation_config, dlp_merge_lock
                )
            except Exception as e:
                logger.error(f"Error in part {i}: {e}")
                pm.translate_error(e)
                raise
            finally:
                translation_config.cleanup_part_working_dir(i)

    translation_config.watermark_output_mode = original_watermark_mode
    _check_translation_continuity(results, split_points)
    merger = ResultMerger(translation_config)
    logger.info("start merge results")
    result = merger.merge_results(results, split_points=split_points)
    logger.info("finish merge results")
    return result


def _dispatch_translation(
    pm: ProgressMonitor,
    translation_config: TranslationConfig,
    original_pdf_path,
) -> TranslateResult:
    """Route to single-pass or split translation depending on config."""
    if not translation_config.split_strategy:
        logger.info(
            f"[pdf_translate] Single-pass translation (no split): {original_pdf_path}"
        )
        return _do_translate_single(pm, translation_config)

    split_manager = SplitManager(translation_config)
    split_points = split_manager.determine_split_points(translation_config)

    if not split_points:
        logger.warning("No split points determined, falling back to single translation")
        return _do_translate_single(pm, translation_config)

    logger.info(
        f"[pdf_translate] Split translation: {len(split_points)} parts for {original_pdf_path}"
    )
    if len(split_points) == 1:
        logger.info("[pdf_translate] Single logical part after split — using one pass")
        return _do_translate_single(pm, translation_config)

    return _run_split_translation(
        pm, translation_config, original_pdf_path, split_points
    )


def _populate_result_statistics(
    result: TranslateResult, translation_config: TranslationConfig
) -> None:
    """Copy aggregate valid-text statistics from the shared context into *result*."""
    try:
        sc = translation_config.shared_context_cross_split_part
        result.total_valid_character_count = getattr(sc, "valid_char_count_total", 0)
        token_total = getattr(sc, "total_valid_text_token_count", None)
        result.total_valid_text_token_count = (
            token_total if isinstance(token_total, int) else 0
        )
    except Exception as e:
        logger.warning(f"Failed to populate valid text statistics: {e}")
        try:
            result.total_valid_character_count = 0
            result.total_valid_text_token_count = 0
        except Exception:
            pass


def _check_input_metadata(original_pdf_path) -> None:
    """Validate that the input PDF was not already translated by DocTranslator."""
    try:
        check_metadata(Document(original_pdf_path))
    except InputFileGeneratedByDocTranslatorError as e:
        logger.error(
            f"input file {original_pdf_path} was already translated "
            "(DocTranslator or legacy BabelDOC); cannot translate again."
        )
        raise e
    except Exception as e:
        logger.warning(f"Error in check metadata, continue: {e}")


def _try_migrate_toc(
    translation_config: TranslationConfig, result: TranslateResult
) -> None:
    """Attempt TOC migration; log errors without raising."""
    try:
        migrate_toc(translation_config, result)
    except Exception as e:
        logger.error(f"Failed to migrate TOC from {translation_config.input_file}: {e}")


def _finalize_translate_result(
    pm: ProgressMonitor,
    translation_config: TranslationConfig,
    original_pdf_path,
) -> TranslateResult:
    """Run the full translation pipeline and populate result metadata."""
    start_time = time.time()
    peak_memory_usage = 0
    with MemoryMonitor() as memory_monitor:
        result = _dispatch_translation(pm, translation_config, original_pdf_path)
        peak_memory_usage = memory_monitor.peak_memory_usage

    finish_time = time.time()
    result.total_seconds = finish_time - start_time
    logger.info(
        f"finish translate: {original_pdf_path}, cost: {finish_time - start_time} s"
    )

    _populate_result_statistics(result, translation_config)
    result.original_pdf_path = translation_config.input_file
    result.peak_memory_usage = peak_memory_usage

    fix_cmap(result, translation_config)
    add_metadata(result, translation_config)
    _try_migrate_toc(translation_config, result)
    pm.translate_done(result)
    return result


def _log_translate_error(e: Exception, translation_config: TranslationConfig) -> None:
    """Log a translation error at the appropriate level."""
    if translation_config.debug:
        logger.exception("translate error:")
    else:
        logger.error(f"translate error: {e}")


def do_translate(
    pm: ProgressMonitor, translation_config: TranslationConfig
) -> TranslateResult:
    try:
        translation_config.progress_monitor = pm
        original_pdf_path = translation_config.input_file
        logger.info(f"start to translate: {original_pdf_path}")
        _check_input_metadata(original_pdf_path)
        return _finalize_translate_result(pm, translation_config, original_pdf_path)

    except Exception as e:
        _log_translate_error(e, translation_config)
        pm.disable = False
        pm.translate_error(e)
        raise
    finally:
        logger.debug("do_translate finally")
        pm.on_finish()
        translation_config.cleanup_temp_files()


def migrate_toc(
    translation_config: TranslationConfig, translate_result: TranslateResult
):
    if translation_config.use_alternating_pages_dual:
        logger.info(
            'skipping TOC migration for "use_alternating_pages_dual" mode',
        )
        return
    old_doc = Document(translation_config.input_file)
    if not old_doc:
        return
    try:
        fix_filter(old_doc)
        fix_null_xref(old_doc)
    except Exception:
        logger.exception(AUTO_FIX_FAILED_MSG)

    toc_data = old_doc.get_toc()

    if not toc_data:
        logger.info("No TOC found in the original PDF, skipping migration.")
        return

    files = {
        translate_result.dual_pdf_path,
        # translate_result.mono_pdf_path,
        translate_result.no_watermark_dual_pdf_path,
        # translate_result.no_watermark_mono_pdf_path
    }

    for f in files:
        if not f:
            continue
        mig_toc_temp_input = translation_config.get_working_file_path(
            "mig_toc_temp.pdf"
        )
        shutil.copy(f, mig_toc_temp_input)
        new_doc = Document(mig_toc_temp_input.as_posix())
        if not new_doc:
            continue

        new_doc.set_toc(toc_data)
        PDFCreater.save_pdf_with_timeout(
            new_doc,
            f.as_posix(),
            translation_config=translation_config,
            clean=not translation_config.skip_clean,
            tag="mig_toc",
        )


def _fix_xref_media_box(doc: Document, xref: int) -> dict:
    """Fix the MediaBox and subsidiary boxes for a single page/pages xref.

    Returns a dict mapping box key → original value for any box that was altered.
    """
    box_set = {}
    mediabox = doc.xref_get_key(xref, "MediaBox")
    if mediabox[0] == "array":
        try:
            _, _, x1, y1 = mediabox[1].replace("[", "").replace("]", "").split(" ")
            doc.xref_set_key(xref, "MediaBox", f"[0 0 {x1} {y1}]")
            box_set["MediaBox"] = mediabox[1]
        except Exception:
            logger.warning(
                "Attempt to fix media box failed; "
                "some pages may not have been processed correctly.",
            )
    for k in ["CropBox", "BleedBox", "TrimBox", "ArtBox"]:
        box = doc.xref_get_key(xref, k)
        if box[0] != "null":
            box_set[k] = box[1]
            doc.xref_set_key(xref, k, "null")
    return box_set


# mediabox -> '[0 nul 792]'
def fix_media_box(doc: Document) -> dict:
    mediabox_data = {}
    for x in range(1, doc.xref_length()):
        t = doc.xref_get_key(x, "Type")
        if t[1] in ["/Pages", "/Page"]:
            box_set = _fix_xref_media_box(doc, x)
            if box_set:
                mediabox_data[x] = box_set
    return mediabox_data


def check_cid_char(il: il_version_1.Document):
    chars = []
    for page in il.page:
        chars.extend(page.pdf_character)

    cid_count = 0
    for char in chars:
        if re.match(r"^\(cid:\d+\)$", char.char_unicode):
            cid_count += 1

    return cid_count > len(chars) * 0.8


def _extract_last_sentence(doc) -> str:
    """Return a representative trailing sentence from the last page of *doc*."""
    last_page_text = doc[-1].get_text().strip() if doc.page_count else ""
    if "." in last_page_text:
        return last_page_text.split(".")[-2].strip()
    return last_page_text[-100:]


def _extract_first_sentence(doc, overlap_pages: int) -> str:
    """Return a representative leading sentence from *doc* after the overlap region."""
    first_content_page = min(overlap_pages, doc.page_count - 1)
    first_page_text = (
        doc[first_content_page].get_text().strip() if doc.page_count else ""
    )
    if "." in first_page_text:
        return first_page_text.split(".")[0].strip()
    return first_page_text[:100]


def _log_boundary_continuity(
    last_sentence: str, first_sentence: str, idx_a: int, idx_b: int
) -> None:
    """Emit a warning when the boundary sentence lacks terminal punctuation."""
    if last_sentence and last_sentence[-1] not in ".!?。！？":
        logger.warning(
            f"Continuity check: chunk {idx_a} may end mid-sentence at boundary "
            f"with chunk {idx_b}. Last text: {last_sentence!r}"
        )
    else:
        logger.debug(
            f"Continuity check passed at boundary {idx_a}→{idx_b}. "
            f"Boundary text: {last_sentence!r} / {first_sentence!r}"
        )


def _check_translation_continuity(
    results: dict,
    split_points: list,
) -> None:
    """Log a warning when translated text at chunk boundaries appears discontinuous.

    For each consecutive pair of chunks the function extracts the last sentence
    of chunk N and the first sentence of chunk N+1 (after skipping its overlap
    pages) and checks whether either boundary sentence ends mid-word — a simple
    heuristic that catches the most common continuity breakages without requiring
    an LLM call.
    """
    from pymupdf import Document

    sorted_indices = sorted(results.keys())
    for i in range(len(sorted_indices) - 1):
        idx_a = sorted_indices[i]
        idx_b = sorted_indices[i + 1]
        result_a = results.get(idx_a)
        result_b = results.get(idx_b)

        if result_a is None or result_b is None:
            continue

        pdf_path_a = result_a.mono_pdf_path or result_a.no_watermark_mono_pdf_path
        pdf_path_b = result_b.mono_pdf_path or result_b.no_watermark_mono_pdf_path
        if not pdf_path_a or not pdf_path_b:
            continue

        try:
            doc_a = Document(str(pdf_path_a))
            last_sentence = _extract_last_sentence(doc_a)

            overlap_b = (
                split_points[idx_b].overlap_pages if idx_b < len(split_points) else 0
            )
            doc_b = Document(str(pdf_path_b))
            first_sentence = _extract_first_sentence(doc_b, overlap_b)

            _log_boundary_continuity(last_sentence, first_sentence, idx_a, idx_b)
        except Exception as e:
            logger.debug(f"Continuity check skipped for boundary {idx_a}→{idx_b}: {e}")


def _save_debug_decompressed_pdf(
    original_pdf_path, translation_config: TranslationConfig
) -> None:
    """In debug mode, save a decompressed copy of the input PDF for inspection."""
    doc_input = Document(original_pdf_path)
    logger.debug("debug mode, save decompressed input pdf")
    output_path = translation_config.get_working_file_path("input.decompressed.pdf")
    try:
        fix_null_page_content(doc_input)
        fix_filter(doc_input)
        fix_null_xref(doc_input)
    except Exception:
        logger.exception(AUTO_FIX_FAILED_MSG)
    safe_save(doc_input, output_path, expand=True, pretty=True)


def _prepare_working_pdf(
    original_pdf_path, translation_config: TranslationConfig
) -> tuple:
    """Copy and sanitize the input PDF into the working directory.

    Returns ``(doc_pdf2zh, temp_pdf_path, mediabox_data)``.
    """
    temp_pdf_path = translation_config.get_working_file_path("input.pdf")
    doc_pdf2zh = Document(original_pdf_path)
    safe_save(doc_pdf2zh, temp_pdf_path)
    try:
        fix_null_page_content(doc_pdf2zh)
        fix_filter(doc_pdf2zh)
        fix_null_xref(doc_pdf2zh)
    except Exception:
        logger.exception(AUTO_FIX_FAILED_MSG)
    mediabox_data = fix_media_box(doc_pdf2zh)
    safe_save(doc_pdf2zh, temp_pdf_path)
    return doc_pdf2zh, temp_pdf_path, mediabox_data


def _build_il_document(
    doc_pdf2zh, temp_pdf_path, translation_config: TranslationConfig
):
    """Parse the PDF stream and build the intermediate-language document."""
    il_creater = ILCreater(translation_config)
    il_creater.mupdf = doc_pdf2zh
    logger.info("[pdf_translate] Phase: parse PDF stream → IR (PDF interpreter)")
    with Path(temp_pdf_path).open("rb") as f:
        start_parse_il(
            f,
            doc_zh=doc_pdf2zh,
            resfont=None,
            il_creater=il_creater,
            translation_config=translation_config,
        )
    logger.info("[pdf_translate] Phase: PDF interpreter finished; building IR document")
    docs = il_creater.create_il()
    _page_count = len(docs.page) if getattr(docs, "page", None) else 0
    logger.info(f"[pdf_translate] Phase: IR document ready (pages={_page_count})")
    del il_creater
    return docs


def _run_layout_and_structure_phases(
    docs,
    doc_pdf2zh,
    temp_pdf_path,
    mediabox_data,
    translation_config: TranslationConfig,
    xml_converter,
):
    """Run scanned detection, layout, table, paragraph, and style phases."""
    if translation_config.skip_scanned_detection:
        logger.info("[pdf_translate] Phase: skipped scanned-file detection")
    else:
        logger.info("[pdf_translate] Phase: scanned-file detection")
        DetectScannedFile(translation_config).process(
            docs, temp_pdf_path, mediabox_data
        )
        logger.info("[pdf_translate] Phase: scanned-file detection finished")
        if translation_config.debug:
            xml_converter.write_json(
                docs,
                translation_config.get_working_file_path("detect_scanned_file.json"),
            )

    logger.info("[pdf_translate] Phase: layout parsing ( ONNX / DocLayout )")
    docs = LayoutParser(translation_config).process(docs, doc_pdf2zh)
    logger.info("[pdf_translate] Phase: layout parsing finished")
    close_process_pool()
    if translation_config.debug:
        xml_converter.write_json(
            docs, translation_config.get_working_file_path("layout_generator.json")
        )

    if translation_config.table_model:
        logger.info("[pdf_translate] Phase: table parsing")
        docs = TableParser(translation_config).process(docs, doc_pdf2zh)
        logger.info("[pdf_translate] Phase: table parsing finished")
        if translation_config.debug:
            xml_converter.write_json(
                docs, translation_config.get_working_file_path("table_parser.json")
            )

    logger.info("[pdf_translate] Phase: paragraph finding")
    ParagraphFinder(translation_config).process(docs)
    logger.info("[pdf_translate] Phase: paragraph finding finished")
    if translation_config.debug:
        xml_converter.write_json(
            docs, translation_config.get_working_file_path("paragraph_finder.json")
        )

    logger.info("[pdf_translate] Phase: formulas and styles")
    StylesAndFormulas(translation_config).process(docs)
    logger.info("[pdf_translate] Phase: formulas and styles finished")
    if translation_config.debug:
        xml_converter.write_json(
            docs, translation_config.get_working_file_path("styles_and_formulas.json")
        )
    return docs


def _run_translation_phase(
    docs, translation_config: TranslationConfig, xml_converter
) -> None:
    """Run glossary extraction and paragraph translation."""
    translate_engine = translation_config.translator
    term_extraction_engine = translation_config.get_term_extraction_translator()
    support_llm_translate = translator_supports_llm(translate_engine)
    support_llm_term_extraction = translator_supports_llm(term_extraction_engine)

    if support_llm_term_extraction and translation_config.auto_extract_glossary:
        logger.info("[pdf_translate] Phase: automatic term / glossary extraction (LLM)")
        AutomaticTermExtractor(term_extraction_engine, translation_config).procress(
            docs
        )
        logger.info("[pdf_translate] Phase: glossary extraction finished")

    if not translation_config.skip_translation:
        translator_kind = (
            "ILTranslatorLLMOnly" if support_llm_translate else "ILTranslator"
        )
        logger.info(f"[pdf_translate] Phase: paragraph translation ({translator_kind})")
        il_translator = (
            ILTranslatorLLMOnly(translate_engine, translation_config)
            if support_llm_translate
            else ILTranslator(translate_engine, translation_config)
        )
        il_translator.translate(docs)
        del il_translator
        logger.info("[pdf_translate] Phase: paragraph translation finished")
    else:
        logger.info("[pdf_translate] Phase: skipped paragraph translation")

    if translation_config.enable_dlp and translation_config.dlp_post_translation:
        _apply_dlp_if_enabled(
            docs, translation_config, stage_label="post_translation", allow_repeat=True
        )
    _unmask_before_pdf_if_enabled(docs, translation_config)

    if translation_config.debug:
        xml_converter.write_json(
            docs, translation_config.get_working_file_path("il_translated.json")
        )
        AddDebugInformation(translation_config).process(docs)
        xml_converter.write_json(
            docs,
            translation_config.get_working_file_path("add_debug_information.json"),
        )


def _try_generate_watermark_bytes(
    doc_pdf2zh, translation_config: TranslationConfig, docs, mediabox_data
) -> tuple:
    """Generate per-page watermark bytes; returns (mono_bytes, dual_bytes) or (None, None)."""
    if translation_config.watermark_output_mode != WatermarkOutputMode.Both:
        return None, None
    try:
        return generate_first_page_with_watermark(
            doc_pdf2zh, translation_config, docs, mediabox_data
        )
    except Exception:
        logger.warning(
            "Failed to generate watermark for first page, using no watermark"
        )
        translation_config.watermark_output_mode = WatermarkOutputMode.NoWatermark
        return None, None


def _merge_watermarks_into_result(
    result,
    mono_watermark_bytes,
    dual_watermark_bytes,
    translation_config: TranslationConfig,
) -> None:
    """Overlay the watermark first-page bytes into *result* in-place."""
    try:
        if mono_watermark_bytes:
            result.mono_pdf_path = merge_watermark_doc(
                result.mono_pdf_path, mono_watermark_bytes, translation_config
            )
    except Exception:
        result.mono_pdf_path = result.no_watermark_mono_pdf_path

    try:
        if dual_watermark_bytes:
            result.dual_pdf_path = merge_watermark_doc(
                result.dual_pdf_path, dual_watermark_bytes, translation_config
            )
    except Exception:
        result.dual_pdf_path = result.no_watermark_dual_pdf_path


def _do_translate_single(
    pm: ProgressMonitor,
    translation_config: TranslationConfig,
) -> TranslateResult | None:
    """Original translation logic for a single document or part"""
    translation_config.progress_monitor = pm

    if translation_config.shared_context_cross_split_part.auto_enabled_ocr_workaround:
        translation_config.ocr_workaround = True
        translation_config.skip_scanned_detection = True

    original_pdf_path = translation_config.input_file
    if translation_config.debug:
        _save_debug_decompressed_pdf(original_pdf_path, translation_config)

    doc_pdf2zh, temp_pdf_path, mediabox_data = _prepare_working_pdf(
        original_pdf_path, translation_config
    )

    xml_converter = XMLConverter()
    docs = _build_il_document(doc_pdf2zh, temp_pdf_path, translation_config)
    _apply_dlp_if_enabled(docs, translation_config, stage_label="post_create_il")

    if translation_config.only_include_translated_page and not docs.page:
        return None

    if translation_config.debug:
        xml_converter.write_json(
            docs, translation_config.get_working_file_path("create_il.debug.json")
        )

    if check_cid_char(docs):
        raise ExtractTextError("The document contains too many CID chars.")

    if translation_config.only_parse_generate_pdf:
        logger.info(
            "[pdf_translate] Mode: parse-only / generate PDF — skipping translation phases"
        )
        pdf_creater = PDFCreater(temp_pdf_path, docs, translation_config, mediabox_data)
        result = pdf_creater.write(translation_config)
        result.original_pdf_path = translation_config.input_file
        return result

    docs = _run_layout_and_structure_phases(
        docs,
        doc_pdf2zh,
        temp_pdf_path,
        mediabox_data,
        translation_config,
        xml_converter,
    )
    _apply_dlp_if_enabled(docs, translation_config, stage_label="pre_translation")
    _run_translation_phase(docs, translation_config, xml_converter)

    mono_watermark_bytes, dual_watermark_bytes = _try_generate_watermark_bytes(
        doc_pdf2zh, translation_config, docs, mediabox_data
    )

    logger.info("[pdf_translate] Phase: typesetting")
    Typesetting(translation_config).typesetting_document(docs)
    logger.info("[pdf_translate] Phase: typesetting finished")
    if translation_config.debug:
        xml_converter.write_json(
            docs, translation_config.get_working_file_path("typsetting.json")
        )

    pdf_creater = PDFCreater(temp_pdf_path, docs, translation_config, mediabox_data)
    logger.info("[pdf_translate] Phase: PDF generation (draw ops, fonts, subset, save)")
    result = pdf_creater.write(translation_config)
    logger.info("[pdf_translate] Phase: PDF generation finished")

    _merge_watermarks_into_result(
        result, mono_watermark_bytes, dual_watermark_bytes, translation_config
    )
    result.original_pdf_path = translation_config.input_file
    return result


def generate_first_page_with_watermark(
    mupdf: Document,
    translation_config: TranslationConfig,
    doc_il: il_version_1.Document,
    mediabox_data: dict[int, Any] | None = None,
) -> tuple[io.BytesIO, io.BytesIO]:
    first_page_doc = Document()
    first_page_doc.insert_pdf(mupdf, from_page=0, to_page=0)

    il_only_first_page_doc = il_version_1.Document()
    il_only_first_page_doc.total_pages = 1
    il_only_first_page_doc.page = [copy.deepcopy(doc_il.page[0])]

    watermarked_config = copy.copy(translation_config)
    watermarked_config.watermark_output_mode = WatermarkOutputMode.Watermarked
    try:
        watermarked_config.progress_monitor.disable = True
        watermarked_temp_pdf_path = watermarked_config.get_working_file_path(
            "watermarked_temp_input.pdf"
        )
        safe_save(first_page_doc, watermarked_temp_pdf_path)

        Typesetting(watermarked_config).typsetting_document(il_only_first_page_doc)
        pdf_creater = PDFCreater(
            watermarked_temp_pdf_path.as_posix(),
            il_only_first_page_doc,
            watermarked_config,
            mediabox_data,
        )
        result = pdf_creater.write(watermarked_config)
        mono_pdf_bytes = None
        dual_pdf_bytes = None
        if result.mono_pdf_path:
            mono_pdf_bytes = io.BytesIO()
            with Path(result.mono_pdf_path).open("rb") as f:
                mono_pdf_bytes.write(f.read())
            result.mono_pdf_path.unlink()
            mono_pdf_bytes.seek(0)

        if result.dual_pdf_path:
            dual_pdf_bytes = io.BytesIO()
            with Path(result.dual_pdf_path).open("rb") as f:
                dual_pdf_bytes.write(f.read())
            result.dual_pdf_path.unlink()
            dual_pdf_bytes.seek(0)

        return mono_pdf_bytes, dual_pdf_bytes
    finally:
        watermarked_config.progress_monitor.disable = False


def merge_watermark_doc(
    no_watermark_pdf_path: pathlib.PosixPath,
    watermark_first_page_pdf_bytes: io.BytesIO,
    translation_config: TranslationConfig,
) -> pathlib.PosixPath:
    if not no_watermark_pdf_path.exists():
        raise FileNotFoundError(
            f"no_watermark_pdf_path not found: {no_watermark_pdf_path}"
        )
    if not watermark_first_page_pdf_bytes:
        raise FileNotFoundError(
            f"watermark_first_page_pdf_bytes not found: {watermark_first_page_pdf_bytes}"
        )

    no_watermark_pdf = Document(no_watermark_pdf_path.as_posix())
    no_watermark_pdf.delete_page(0)

    watermark_first_page_pdf = Document("pdf", watermark_first_page_pdf_bytes)
    no_watermark_pdf.insert_pdf(
        watermark_first_page_pdf, from_page=0, to_page=0, start_at=0
    )

    new_save_path = no_watermark_pdf_path.with_name(
        no_watermark_pdf_path.name.replace(".no_watermark", "")
    )

    PDFCreater.save_pdf_with_timeout(
        no_watermark_pdf,
        new_save_path.as_posix(),
        translation_config=translation_config,
        clean=not translation_config.skip_clean,
    )
    return new_save_path
