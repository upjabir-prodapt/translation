import logging
from pathlib import Path

from pymupdf import Document

from src.worker.doctranslator.format.pdf.document_il.backend.pdf_creater import (
    PDFCreater,
)
from src.worker.doctranslator.format.pdf.split_manager import SplitPoint
from src.worker.doctranslator.format.pdf.translation_config import TranslateResult
from src.worker.doctranslator.format.pdf.translation_config import TranslationConfig

logger = logging.getLogger(__name__)


class ResultMerger:
    """Handles merging of split translation results"""

    def __init__(self, translation_config: TranslationConfig):
        self.config = translation_config

    def _build_overlap_pages_list(
        self,
        sorted_results: dict,
        split_points: list[SplitPoint] | None,
    ) -> list[int]:
        """Build the per-part overlap page counts for merge skipping."""
        overlap_pages_list: list[int] = []
        for part_idx in sorted_results:
            if split_points is not None and part_idx < len(split_points):
                overlap_pages_list.append(split_points[part_idx].overlap_pages)
            else:
                overlap_pages_list.append(0)
        return overlap_pages_list

    def _try_merge_mono(
        self,
        results: dict,
        sorted_results: dict,
        mono_file_name: str,
        overlap_pages_list: list[int],
    ) -> Path | None:
        """Attempt to merge monolingual PDFs; returns path or None on failure."""
        try:
            if (
                any(r.mono_pdf_path for r in results.values())
                and not self.config.no_mono
            ):
                return self._merge_pdfs(
                    [
                        r.mono_pdf_path
                        for r in sorted_results.values()
                        if r.mono_pdf_path
                    ],
                    mono_file_name,
                    tag="merged_mono",
                    overlap_pages_list=overlap_pages_list,
                )
        except Exception as e:
            logger.error(f"Error merging monolingual PDFs: {e}")
        return None

    def _try_merge_dual(
        self,
        results: dict,
        sorted_results: dict,
        dual_file_name: str,
        overlap_pages_list: list[int],
    ) -> Path | None:
        """Attempt to merge dual-language PDFs; returns path or None on failure."""
        try:
            if (
                any(r.dual_pdf_path for r in results.values())
                and not self.config.no_dual
            ):
                return self._merge_pdfs(
                    [
                        r.dual_pdf_path
                        for r in sorted_results.values()
                        if r.dual_pdf_path
                    ],
                    dual_file_name,
                    tag="merged_dual",
                    overlap_pages_list=overlap_pages_list,
                )
        except Exception as e:
            logger.error(f"Error merging dual-language PDFs: {e}")
        return None

    def _try_merge_no_watermark_pdfs(
        self,
        results: dict,
        sorted_results: dict,
        mono_file_name_no_watermark: str,
        overlap_pages_list: list[int],
    ) -> tuple[Path | None, Path | None]:
        """Attempt to merge no-watermark PDFs; returns (mono_path, dual_path)."""
        has_watermark_variants = any(
            r.dual_pdf_path != r.no_watermark_dual_pdf_path
            or r.mono_pdf_path != r.no_watermark_mono_pdf_path
            for r in results.values()
        )
        if not has_watermark_variants:
            return None, None

        merged_no_watermark_mono_path = None
        merged_no_watermark_dual_path = None

        try:
            if (
                any(r.no_watermark_mono_pdf_path for r in results.values())
                and not self.config.no_mono
            ):
                merged_no_watermark_mono_path = self._merge_pdfs(
                    [
                        r.no_watermark_mono_pdf_path
                        for r in sorted_results.values()
                        if r.no_watermark_mono_pdf_path
                    ],
                    mono_file_name_no_watermark,
                    tag="merged_no_watermark_mono",
                    overlap_pages_list=overlap_pages_list,
                )
        except Exception as e:
            logger.error(f"Error merging no-watermark PDFs: {e}")

        try:
            if (
                any(r.no_watermark_dual_pdf_path for r in results.values())
                and not self.config.no_dual
            ):
                merged_no_watermark_dual_path = self._merge_pdfs(
                    [
                        r.no_watermark_dual_pdf_path
                        for r in sorted_results.values()
                        if r.no_watermark_dual_pdf_path
                    ],
                    "merged_no_watermark_dual.pdf",
                    tag="merged_no_watermark_dual",
                    overlap_pages_list=overlap_pages_list,
                )
        except Exception as e:
            logger.error(f"Error merging no-watermark PDFs: {e}")

        return merged_no_watermark_mono_path, merged_no_watermark_dual_path

    def _save_auto_extracted_glossary(
        self, basename: str, debug_suffix: str
    ) -> Path | None:
        """Save the auto-extracted glossary if configured; returns the path or None."""
        if not (
            self.config.save_auto_extracted_glossary
            and self.config.shared_context_cross_split_part.auto_extracted_glossary
        ):
            return None
        auto_extracted_glossary_path = self.config.get_output_file_path(
            f"{basename}{debug_suffix}.{self.config.lang_out}.glossary.csv"
        )
        with auto_extracted_glossary_path.open("w", encoding="utf-8") as f:
            logger.info(
                f"save auto extracted glossary to {auto_extracted_glossary_path}"
            )
            f.write(
                self.config.shared_context_cross_split_part.auto_extracted_glossary.to_csv()
            )
        return auto_extracted_glossary_path

    def _build_output_file_names(self, basename: str) -> tuple[str, str, str, str]:
        """Return (mono_name, dual_name, no_wm_mono_name, no_wm_debug_suffix)."""
        debug_suffix = ".debug" if self.config.debug else ""
        mono_file_name = f"{basename}{debug_suffix}.{self.config.lang_out}.mono.pdf"
        dual_file_name = f"{basename}{debug_suffix}.{self.config.lang_out}.dual.pdf"
        no_wm_suffix = debug_suffix + ".no_watermark"
        mono_file_name_no_watermark = (
            f"{basename}{no_wm_suffix}.{self.config.lang_out}.mono.pdf"
        )
        return mono_file_name, dual_file_name, mono_file_name_no_watermark, no_wm_suffix

    @staticmethod
    def _reconcile_watermark_paths(
        merged_result: TranslateResult,
        merged_mono_path,
        merged_dual_path,
        merged_no_watermark_mono_path,
        merged_no_watermark_dual_path,
    ) -> None:
        """Cross-assign mono/dual watermark paths so neither side is left None."""
        merged_result.no_watermark_mono_pdf_path = merged_no_watermark_mono_path
        merged_result.no_watermark_dual_pdf_path = merged_no_watermark_dual_path

        if merged_result.no_watermark_mono_pdf_path is None:
            merged_result.no_watermark_mono_pdf_path = merged_mono_path
        elif merged_result.mono_pdf_path is None:
            merged_result.mono_pdf_path = merged_no_watermark_mono_path

        if merged_result.no_watermark_dual_pdf_path is None:
            merged_result.no_watermark_dual_pdf_path = merged_dual_path
        elif merged_result.dual_pdf_path is None:
            merged_result.dual_pdf_path = merged_no_watermark_dual_path

    def merge_results(
        self,
        results: dict[int, TranslateResult | None],
        split_points: list[SplitPoint] | None = None,
    ) -> TranslateResult:
        """Merge multiple translation results into one"""
        if not results:
            raise ValueError("No results to merge")

        basename = Path(self.config.input_file).stem
        (
            mono_file_name,
            dual_file_name,
            mono_file_name_no_watermark,
            no_wm_suffix,
        ) = self._build_output_file_names(basename)

        results = {k: v for k, v in results.items() if v is not None}
        sorted_results = dict(sorted(results.items()))
        overlap_pages_list = self._build_overlap_pages_list(
            sorted_results, split_points
        )

        merged_mono_path = self._try_merge_mono(
            results, sorted_results, mono_file_name, overlap_pages_list
        )
        merged_dual_path = self._try_merge_dual(
            results, sorted_results, dual_file_name, overlap_pages_list
        )
        merged_no_watermark_mono_path, merged_no_watermark_dual_path = (
            self._try_merge_no_watermark_pdfs(
                results, sorted_results, mono_file_name_no_watermark, overlap_pages_list
            )
        )
        auto_extracted_glossary_path = self._save_auto_extracted_glossary(
            basename, no_wm_suffix
        )

        merged_result = TranslateResult(
            mono_pdf_path=merged_mono_path,
            dual_pdf_path=merged_dual_path,
            auto_extracted_glossary_path=auto_extracted_glossary_path,
        )
        self._reconcile_watermark_paths(
            merged_result,
            merged_mono_path,
            merged_dual_path,
            merged_no_watermark_mono_path,
            merged_no_watermark_dual_path,
        )
        total_time = sum(
            r.total_seconds for r in results.values() if hasattr(r, "total_seconds")
        )
        merged_result.total_seconds = total_time
        return merged_result

    def _merge_pdfs(
        self,
        pdf_paths: list[str | Path],
        output_name: str,
        tag: str,
        overlap_pages_list: list[int] | None = None,
    ) -> Path | None:
        """Merge multiple PDFs into one, skipping overlap context pages per chunk."""
        if not pdf_paths:
            return None

        output_path = self.config.get_output_file_path(output_name)
        merged_doc = Document()

        for idx, pdf_path in enumerate(pdf_paths):
            doc = Document(str(pdf_path))
            from_page = (
                overlap_pages_list[idx]
                if overlap_pages_list and idx < len(overlap_pages_list)
                else 0
            )
            if from_page > 0:
                logger.debug(
                    f"Skipping {from_page} overlap page(s) from chunk {idx} "
                    f"during merge."
                )
            merged_doc.insert_pdf(doc, from_page=from_page)

        merged_doc = PDFCreater.subset_fonts_in_subprocess(
            merged_doc, self.config, tag=tag
        )
        PDFCreater.save_pdf_with_timeout(
            merged_doc, str(output_path), translation_config=self.config
        )

        return output_path
