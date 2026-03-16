import asyncio
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import sys
from pathlib import Path
from typing import Any

import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn

import babeldoc.format.pdf.high_level
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode
from babeldoc.glossary import Glossary
from babeldoc.translator.translator import OpenAITranslator
from babeldoc.translator.translator import set_translate_rate_limiter
from babeldoc.utils.common import enable_process_pool

logger = logging.getLogger(__name__)
__version__ = "0.5.23"


def _candidate_config_paths() -> list[Path]:
    paths: list[Path] = []

    env_config = os.environ.get("BABELDOC_CONFIG")
    if env_config:
        paths.append(Path(env_config).expanduser())

    paths.append(Path.cwd() / "config" / "config.json")
    paths.append(Path(__file__).resolve().parent.parent / "config" / "config.json")
    return paths


def _find_config_file() -> Path | None:
    """Find first available config file."""
    for config_path in _candidate_config_paths():
        if config_path.is_file():
            return config_path
    return None


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = _find_config_file()
        if config_path is None:
            logger.error("No config file found. Please provide config.json")
            sys.exit(1)

    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    if config_path.suffix != ".json":
        logger.error(
            f"Unsupported config format: {config_path.suffix}. Only .json is supported"
        )
        sys.exit(1)

    logger.info(f"Loading config from: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    # Extract babeldoc section if it exists
    if "babeldoc" in data:
        config = data["babeldoc"]
    else:
        config = data

    # Convert kebab-case to underscore for Python compatibility
    normalized_config = {}
    for key, value in config.items():
        normalized_key = key.replace("-", "_")
        normalized_config[normalized_key] = value

    return normalized_config


def get_default_config() -> dict[str, Any]:
    """Return default configuration values."""
    return {
        "openai": False,
        "debug": False,
        "warmup": False,
        "lang_in": "en",
        "lang_out": "zh",
        "qps": 4,
        "min_text_length": 5,
        "report_interval": 0.1,
        "no_dual": False,
        "no_mono": False,
        "ignore_cache": False,
        "split_short_lines": False,
        "short_line_split_factor": 0.8,
        "skip_clean": False,
        "dual_translate_first": False,
        "disable_rich_text_translate": False,
        "enhance_compatibility": False,
        "use_alternating_pages_dual": False,
        "watermark_output_mode": "watermarked",
        "translate_table_text": False,
        "show_char_box": False,
        "skip_scanned_detection": False,
        "ocr_workaround": False,
        "add_formula_placehold_hint": False,
        "disable_same_text_fallback": False,
        "auto_extract_glossary": True,
        "auto_enable_ocr_workaround": False,
        "only_include_translated_page": False,
        "save_auto_extracted_glossary": False,
        "disable_graphic_element_process": False,
        "merge_alternating_line_numbers": True,
        "skip_translation": False,
        "skip_form_render": False,
        "skip_curve_render": False,
        "only_parse_generate_pdf": False,
        "remove_non_formula_lines": False,
        "non_formula_line_iou_threshold": 0.9,
        "figure_table_protection_threshold": 0.9,
        "skip_formula_offset_calculation": False,
        "openai_model": "gpt-4o-mini",
        "enable_json_mode_if_requested": False,
        "send_dashscope_header": False,
        "no_send_temperature": False,
    }


class Config:
    """Configuration class to hold all settings."""

    # Core settings
    openai: bool
    debug: bool
    warmup: bool
    lang_in: str
    lang_out: str
    qps: int
    min_text_length: int
    report_interval: float

    # Translation behavior
    no_dual: bool
    no_mono: bool
    ignore_cache: bool
    split_short_lines: bool
    short_line_split_factor: float
    skip_clean: bool
    dual_translate_first: bool
    disable_rich_text_translate: bool
    enhance_compatibility: bool
    use_alternating_pages_dual: bool
    watermark_output_mode: str
    translate_table_text: bool
    show_char_box: bool
    skip_scanned_detection: bool
    ocr_workaround: bool
    add_formula_placehold_hint: bool
    disable_same_text_fallback: bool
    auto_extract_glossary: bool
    auto_enable_ocr_workaround: bool
    only_include_translated_page: bool
    save_auto_extracted_glossary: bool
    disable_graphic_element_process: bool
    merge_alternating_line_numbers: bool
    skip_translation: bool
    skip_form_render: bool
    skip_curve_render: bool
    only_parse_generate_pdf: bool
    remove_non_formula_lines: bool
    non_formula_line_iou_threshold: float
    figure_table_protection_threshold: float
    skip_formula_offset_calculation: bool

    # OpenAI settings
    openai_model: str
    enable_json_mode_if_requested: bool
    send_dashscope_header: bool
    no_send_temperature: bool

    def __init__(self, config_dict: dict[str, Any]):
        defaults = get_default_config()
        defaults.update(config_dict)

        for key, value in defaults.items():
            setattr(self, key, value)


async def main():
    # Load configuration
    config_path_arg = None
    if len(sys.argv) > 1 and sys.argv[1] not in ["-h", "--help", "--version"]:
        # Check if user provided a config path as first argument
        if not sys.argv[1].startswith("-"):
            config_path_arg = sys.argv[1]

    config_dict = load_config(config_path_arg)
    config = Config(config_dict)

    if getattr(config, "debug", False):
        logging.getLogger().setLevel(logging.DEBUG)

    if getattr(config, "warmup", False):
        from loaders import warmup

        warmup()
        logger.info("Warmup completed, exiting...")
        return

    # 验证翻译服务选择
    if not getattr(config, "openai", False):
        logger.error("必须选择一个翻译服务：设置 openai = true")
        sys.exit(1)

    # 验证 OpenAI 参数
    if config.openai and not getattr(config, "openai_api_key", None):
        logger.error("使用 OpenAI 服务时必须提供 API key")
        sys.exit(1)

    if getattr(config, "enable_process_pool", False):
        enable_process_pool()

    # 实例化翻译器
    if config.openai:
        translator_kwargs: dict[str, Any] = {}
        if getattr(config, "openai_reasoning", None) is not None:
            translator_kwargs["reasoning"] = config.openai_reasoning
        translator = OpenAITranslator(
            lang_in=getattr(config, "lang_in", "en"),
            lang_out=getattr(config, "lang_out", "zh"),
            model=getattr(config, "openai_model", "gpt-4o-mini"),
            base_url=getattr(config, "openai_base_url", None),
            api_key=getattr(config, "openai_api_key", None),
            ignore_cache=getattr(config, "ignore_cache", False),
            enable_json_mode_if_requested=getattr(
                config, "enable_json_mode_if_requested", False
            ),
            send_dashscope_header=getattr(config, "send_dashscope_header", False),
            send_temperature=not getattr(config, "no_send_temperature", False),
            **translator_kwargs,
        )
        term_extraction_translator = translator
        if (
            getattr(config, "openai_term_extraction_model", None)
            or getattr(config, "openai_term_extraction_base_url", None)
            or getattr(config, "openai_term_extraction_api_key", None)
        ):
            term_translator_kwargs: dict[str, Any] = {}
            if getattr(config, "openai_term_extraction_reasoning", None) is not None:
                term_translator_kwargs["reasoning"] = (
                    config.openai_term_extraction_reasoning
                )
            term_extraction_translator = OpenAITranslator(
                lang_in=config.lang_in,
                lang_out=config.lang_out,
                model=getattr(config, "openai_term_extraction_model", None)
                or config.openai_model,
                base_url=getattr(config, "openai_term_extraction_base_url", None)
                or getattr(config, "openai_base_url", None),
                api_key=getattr(config, "openai_term_extraction_api_key", None)
                or config.openai_api_key,
                ignore_cache=config.ignore_cache,
                enable_json_mode_if_requested=config.enable_json_mode_if_requested,
                send_dashscope_header=config.send_dashscope_header,
                send_temperature=not config.no_send_temperature,
                **term_translator_kwargs,
            )
    else:
        raise ValueError("Invalid translator type")

    # 设置翻译速率限制
    set_translate_rate_limiter(getattr(config, "qps", 4))
    # 初始化文档布局模型
    if getattr(config, "rpc_doclayout", None):
        from babeldoc.docvision.rpc_doclayout import RpcDocLayoutModel

        doc_layout_model = RpcDocLayoutModel(host=config.rpc_doclayout)
    elif getattr(config, "rpc_doclayout2", None):
        from babeldoc.docvision.rpc_doclayout2 import RpcDocLayoutModel

        doc_layout_model = RpcDocLayoutModel(host=config.rpc_doclayout2)
    elif getattr(config, "rpc_doclayout3", None):
        from babeldoc.docvision.rpc_doclayout3 import RpcDocLayoutModel

        doc_layout_model = RpcDocLayoutModel(host=config.rpc_doclayout3)
    elif getattr(config, "rpc_doclayout4", None):
        from babeldoc.docvision.rpc_doclayout4 import RpcDocLayoutModel

        doc_layout_model = RpcDocLayoutModel(host=config.rpc_doclayout4)
    elif getattr(config, "rpc_doclayout5", None):
        from babeldoc.docvision.rpc_doclayout5 import RpcDocLayoutModel

        doc_layout_model = RpcDocLayoutModel(host=config.rpc_doclayout5)
    elif getattr(config, "rpc_doclayout6", None):
        from babeldoc.docvision.rpc_doclayout6 import RpcDocLayoutModel

        doc_layout_model = RpcDocLayoutModel(host=config.rpc_doclayout6)
    elif getattr(config, "rpc_doclayout7", None):
        from babeldoc.docvision.rpc_doclayout7 import RpcDocLayoutModel

        doc_layout_model = RpcDocLayoutModel(host=config.rpc_doclayout7)
    else:
        from babeldoc.docvision.doclayout import DocLayoutModel

        doc_layout_model = DocLayoutModel.load_onnx()

    if getattr(config, "translate_table_text", False):
        from babeldoc.docvision.table_detection.rapidocr import RapidOCRModel

        table_model = RapidOCRModel()
    else:
        table_model = None

    # Load glossaries
    loaded_glossaries: list[Glossary] = []
    glossary_files = getattr(config, "glossary_files", None)
    if glossary_files:
        if isinstance(glossary_files, str):
            paths_str = glossary_files.split(",")
        else:
            paths_str = glossary_files
        for p_str in paths_str:
            file_path = Path(str(p_str).strip())
            if not file_path.exists():
                logger.error(f"Glossary file not found: {file_path}")
                continue
            if not file_path.is_file():
                logger.error(f"Glossary path is not a file: {file_path}")
                continue
            try:
                glossary_obj = Glossary.from_csv(file_path, config.lang_out)
                if glossary_obj.entries:
                    loaded_glossaries.append(glossary_obj)
                    logger.info(
                        f"Loaded glossary '{glossary_obj.name}' with {len(glossary_obj.entries)} entries."
                    )
                else:
                    logger.info(
                        f"Glossary '{file_path.stem}' loaded with no applicable entries for lang_out '{config.lang_out}'."
                    )
            except Exception as e:
                logger.error(f"Failed to load glossary from {file_path}: {e}")

    files = getattr(config, "files", None)
    if not files:
        logger.error("No input files specified. Please provide 'files' in config.")
        sys.exit(1)

    pending_files = []
    if isinstance(files, str):
        files = [files]
    for file in files:
        # 清理文件路径，去除两端的引号
        if isinstance(file, str) and file.startswith("--files="):
            file = file[len("--files=") :]
        file = str(file).lstrip("-").strip("\"'")
        if not Path(file).exists():
            logger.error(f"文件不存在：{file}")
            exit(1)
        if not file.lower().endswith(".pdf"):
            logger.error(f"文件不是 PDF 文件：{file}")
            exit(1)
        pending_files.append(file)

    output = getattr(config, "output", None)
    if output:
        if not Path(output).exists():
            logger.info(f"输出目录不存在，创建：{output}")
            try:
                Path(output).mkdir(parents=True, exist_ok=True)
            except OSError:
                logger.critical(
                    f"Failed to create output folder at {output}",
                    exc_info=True,
                )
                exit(1)
    else:
        output = None

    working_dir_path = getattr(config, "working_dir", None)
    if working_dir_path:
        working_dir = Path(working_dir_path)
        if not working_dir.exists():
            logger.info(f"工作目录不存在，创建：{working_dir}")
            try:
                working_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                logger.critical(
                    f"Failed to create working directory at {working_dir}",
                    exc_info=True,
                )
                exit(1)
    else:
        working_dir = None

    watermark_output_mode = WatermarkOutputMode.Watermarked
    if getattr(config, "no_watermark", False):
        watermark_output_mode = WatermarkOutputMode.NoWatermark
    else:
        wm_mode = getattr(config, "watermark_output_mode", "watermarked")
        if wm_mode == "both":
            watermark_output_mode = WatermarkOutputMode.Both
        elif wm_mode == "watermarked":
            watermark_output_mode = WatermarkOutputMode.Watermarked
        elif wm_mode == "no_watermark":
            watermark_output_mode = WatermarkOutputMode.NoWatermark

    split_strategy = None
    max_pages_per_part = getattr(config, "max_pages_per_part", None)
    if max_pages_per_part:
        split_strategy = TranslationConfig.create_max_pages_per_part_split_strategy(
            max_pages_per_part
        )

    total_term_extraction_total_tokens = 0
    total_term_extraction_prompt_tokens = 0
    total_term_extraction_completion_tokens = 0
    total_term_extraction_cache_hit_prompt_tokens = 0

    for file in pending_files:
        # 清理文件路径，去除两端的引号
        file = file.strip("\"'")
        # 创建配置对象
        translation_config = TranslationConfig(
            input_file=file,
            font=None,
            pages=getattr(config, "pages", None),
            output_dir=output,
            translator=translator,
            term_extraction_translator=term_extraction_translator,
            debug=getattr(config, "debug", False),
            lang_in=config.lang_in,
            lang_out=config.lang_out,
            no_dual=getattr(config, "no_dual", False),
            no_mono=getattr(config, "no_mono", False),
            qps=getattr(config, "qps", 4),
            formular_font_pattern=getattr(config, "formular_font_pattern", None),
            formular_char_pattern=getattr(config, "formular_char_pattern", None),
            split_short_lines=getattr(config, "split_short_lines", False),
            short_line_split_factor=getattr(config, "short_line_split_factor", 0.8),
            doc_layout_model=doc_layout_model,
            skip_clean=getattr(config, "skip_clean", False),
            dual_translate_first=getattr(config, "dual_translate_first", False),
            disable_rich_text_translate=getattr(
                config, "disable_rich_text_translate", False
            ),
            enhance_compatibility=getattr(config, "enhance_compatibility", False),
            use_alternating_pages_dual=getattr(
                config, "use_alternating_pages_dual", False
            ),
            report_interval=getattr(config, "report_interval", 0.1),
            min_text_length=getattr(config, "min_text_length", 5),
            watermark_output_mode=watermark_output_mode,
            split_strategy=split_strategy,
            table_model=table_model,
            show_char_box=getattr(config, "show_char_box", False),
            skip_scanned_detection=getattr(config, "skip_scanned_detection", False),
            ocr_workaround=getattr(config, "ocr_workaround", False),
            custom_system_prompt=getattr(config, "custom_system_prompt", None),
            working_dir=working_dir,
            add_formula_placehold_hint=getattr(
                config, "add_formula_placehold_hint", False
            ),
            disable_same_text_fallback=getattr(
                config, "disable_same_text_fallback", False
            ),
            glossaries=loaded_glossaries,
            pool_max_workers=getattr(config, "pool_max_workers", None),
            auto_extract_glossary=getattr(config, "auto_extract_glossary", True),
            auto_enable_ocr_workaround=getattr(
                config, "auto_enable_ocr_workaround", False
            ),
            primary_font_family=getattr(config, "primary_font_family", None),
            only_include_translated_page=getattr(
                config, "only_include_translated_page", False
            ),
            save_auto_extracted_glossary=getattr(
                config, "save_auto_extracted_glossary", False
            ),
            enable_graphic_element_process=not getattr(
                config, "disable_graphic_element_process", False
            ),
            merge_alternating_line_numbers=getattr(
                config, "merge_alternating_line_numbers", True
            ),
            skip_translation=getattr(config, "skip_translation", False),
            skip_form_render=getattr(config, "skip_form_render", False),
            skip_curve_render=getattr(config, "skip_curve_render", False),
            only_parse_generate_pdf=getattr(config, "only_parse_generate_pdf", False),
            remove_non_formula_lines=getattr(config, "remove_non_formula_lines", False),
            non_formula_line_iou_threshold=getattr(
                config, "non_formula_line_iou_threshold", 0.9
            ),
            figure_table_protection_threshold=getattr(
                config, "figure_table_protection_threshold", 0.9
            ),
            skip_formula_offset_calculation=getattr(
                config, "skip_formula_offset_calculation", False
            ),
            metadata_extra_data=getattr(config, "metadata_extra_data", None),
            term_pool_max_workers=getattr(config, "term_pool_max_workers", None),
        )

        def nop(_x):
            pass

        getattr(doc_layout_model, "init_font_mapper", nop)(translation_config)
        # Create progress handler
        progress_context, progress_handler = create_progress_handler(
            translation_config, show_log=False
        )

        # 开始翻译
        with progress_context:
            async for event in babeldoc.format.pdf.high_level.async_translate(
                translation_config
            ):
                progress_handler(event)
                if translation_config.debug:
                    logger.debug(event)
                if event["type"] == "error":
                    logger.error(f"Error: {event['error']}")
                    break
                if event["type"] == "finish":
                    result = event["translate_result"]
                    logger.info(str(result))
                    break
        usage = translation_config.term_extraction_token_usage
        total_term_extraction_total_tokens += usage["total_tokens"]
        total_term_extraction_prompt_tokens += usage["prompt_tokens"]
        total_term_extraction_completion_tokens += usage["completion_tokens"]
        total_term_extraction_cache_hit_prompt_tokens += usage[
            "cache_hit_prompt_tokens"
        ]
    logger.info(f"Total tokens: {translator.token_count.value}")
    logger.info(f"Prompt tokens: {translator.prompt_token_count.value}")
    logger.info(f"Completion tokens: {translator.completion_token_count.value}")
    logger.info(
        f"Cache hit prompt tokens: {translator.cache_hit_prompt_token_count.value}"
    )
    logger.info(
        "Term extraction tokens: total=%s prompt=%s completion=%s cache_hit_prompt=%s",
        total_term_extraction_total_tokens,
        total_term_extraction_prompt_tokens,
        total_term_extraction_completion_tokens,
        total_term_extraction_cache_hit_prompt_tokens,
    )
    if term_extraction_translator is not translator:
        logger.info(
            "Term extraction translator raw tokens: total=%s prompt=%s completion=%s cache_hit_prompt=%s",
            term_extraction_translator.token_count.value,
            term_extraction_translator.prompt_token_count.value,
            term_extraction_translator.completion_token_count.value,
            term_extraction_translator.cache_hit_prompt_token_count.value,
        )


def create_progress_handler(
    translation_config: TranslationConfig, show_log: bool = False
):
    """Create a progress handler function based on the configuration.

    Args:
        translation_config: The translation configuration.

    Returns:
        A tuple of (progress_context, progress_handler), where progress_context is a context
        manager that should be used to wrap the translation process, and progress_handler
        is a function that will be called with progress events.
    """
    if translation_config.use_rich_pbar:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        translate_task_id = progress.add_task("translate", total=100)
        stage_tasks = {}

        def progress_handler(event):
            if show_log and random.random() <= 0.1:  # noqa: S311
                logger.info(event)
            if event["type"] == "progress_start":
                if event["stage"] not in stage_tasks:
                    stage_tasks[event["stage"]] = progress.add_task(
                        f"{event['stage']} ({event['part_index']}/{event['total_parts']})",
                        total=event.get("stage_total", 100),
                    )
            elif event["type"] == "progress_update":
                stage = event["stage"]
                if stage in stage_tasks:
                    progress.update(
                        stage_tasks[stage],
                        completed=event["stage_current"],
                        total=event["stage_total"],
                        description=f"{event['stage']} ({event['part_index']}/{event['total_parts']})",
                        refresh=True,
                    )
                progress.update(
                    translate_task_id,
                    completed=event["overall_progress"],
                    refresh=True,
                )
            elif event["type"] == "progress_end":
                stage = event["stage"]
                if stage in stage_tasks:
                    progress.update(
                        stage_tasks[stage],
                        completed=event["stage_total"],
                        total=event["stage_total"],
                        description=f"{event['stage']} ({event['part_index']}/{event['total_parts']})",
                        refresh=True,
                    )
                    progress.update(
                        translate_task_id,
                        completed=event["overall_progress"],
                        refresh=True,
                    )
                progress.refresh()

        return progress, progress_handler
    else:
        pbar = tqdm.tqdm(total=100, desc="translate")

        def progress_handler(event):
            if event["type"] == "progress_update":
                pbar.update(event["overall_progress"] - pbar.n)
                pbar.set_description(
                    f"{event['stage']} ({event['stage_current']}/{event['stage_total']})",
                )
            elif event["type"] == "progress_end":
                pbar.set_description(f"{event['stage']} (Complete)")
                pbar.refresh()

        return pbar, progress_handler


# for backward compatibility
def create_cache_folder():
    return babeldoc.format.pdf.high_level.create_cache_folder()


# for backward compatibility
def download_font_assets():
    return babeldoc.format.pdf.high_level.download_font_assets()


class EvictQueue(queue.Queue):
    def __init__(self, maxsize):
        self.discarded = 0
        super().__init__(maxsize)

    def put(self, item, block=False, timeout=None):
        while True:
            try:
                super().put(item, block=False)
                break
            except queue.Full:
                try:
                    self.get_nowait()
                    self.discarded += 1
                except queue.Empty:
                    pass


def speed_up_logs():
    import logging.handlers

    root_logger = logging.getLogger()
    log_que = EvictQueue(1000)
    queue_handler = logging.handlers.QueueHandler(log_que)
    queue_listener = logging.handlers.QueueListener(log_que, *root_logger.handlers)
    queue_listener.start()
    root_logger.handlers = [queue_handler]


def cli():
    """Command line interface entry point."""
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

    logging.getLogger("httpx").setLevel("CRITICAL")
    logging.getLogger("httpx").propagate = False
    logging.getLogger("openai").setLevel("CRITICAL")
    logging.getLogger("openai").propagate = False
    logging.getLogger("httpcore").setLevel("CRITICAL")
    logging.getLogger("httpcore").propagate = False
    logging.getLogger("http11").setLevel("CRITICAL")
    logging.getLogger("http11").propagate = False
    for v in logging.Logger.manager.loggerDict.values():
        if getattr(v, "name", None) is None:
            continue
        if (
            v.name.startswith("pdfminer")
            or v.name.startswith("httpx")
            or "http11" in v.name
            or "openai" in v.name
            or "pdfminer" in v.name
        ):
            v.disabled = True
            v.propagate = False

    speed_up_logs()
    babeldoc.format.pdf.high_level.init()
    asyncio.run(main())


if __name__ == "__main__":
    if sys.platform == "darwin" or sys.platform == "win32":
        mp.set_start_method("spawn")
    else:
        mp.set_start_method("forkserver")
    cli()
