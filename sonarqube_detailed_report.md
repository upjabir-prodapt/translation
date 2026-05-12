# SonarQube Detailed Report: Translation Project
**Date:** 2026-05-08

## Summary Metrics
- **Total Issues:** 386
### Issues by Type
- **CODE_SMELL:** 372
- **BUG:** 9
- **VULNERABILITY:** 5
### Issues by Severity
- **CRITICAL:** 141
- **MAJOR:** 137
- **MINOR:** 85
- **BLOCKER:** 23

## Top Issues (First 20)
| Severity | Type | Component | Message |
|---|---|---|---|
| CRITICAL | CODE_SMELL | src/doctranslator/format/pdf/document_il/midend/automatic_term_extractor.py | Refactor this function to reduce its Cognitive Complexity from 16 to the 15 allowed. |
| CRITICAL | CODE_SMELL | src/doctranslator/format/pdf/document_il/midend/automatic_term_extractor.py | Refactor this function to reduce its Cognitive Complexity from 34 to the 15 allowed. |
| CRITICAL | CODE_SMELL | src/doctranslator/format/pdf/document_il/midend/il_translator_llm_only.py | Refactor this function to reduce its Cognitive Complexity from 82 to the 15 allowed. |
| CRITICAL | CODE_SMELL | src/doctranslator/format/pdf/high_level.py | Refactor this function to reduce its Cognitive Complexity from 41 to the 15 allowed. |
| MAJOR | CODE_SMELL | src/doctranslator/progress_monitor.py | Add argument(s) corresponding to the message's replacement field(s). |
| CRITICAL | CODE_SMELL | src/doctranslator/format/pdf/document_il/midend/il_translator_llm_only.py | Refactor this function to reduce its Cognitive Complexity from 36 to the 15 allowed. |
| CRITICAL | CODE_SMELL | src/doctranslator/format/pdf/document_il/midend/typesetting.py | Refactor this function to reduce its Cognitive Complexity from 47 to the 15 allowed. |
| CRITICAL | CODE_SMELL | src/doctranslator/format/pdf/high_level.py | Refactor this function to reduce its Cognitive Complexity from 84 to the 15 allowed. |
| CRITICAL | CODE_SMELL | src/doctranslator/format/pdf/translation_config.py | Refactor this function to reduce its Cognitive Complexity from 25 to the 15 allowed. |
| MAJOR | CODE_SMELL | src/doctranslator/format/pdf/translation_config.py | Method "__init__" has 62 parameters, which is greater than the 13 authorized. |
| CRITICAL | CODE_SMELL | src/doctranslator/format/pdf/document_il/utils/fontmap.py | Refactor this function to reduce its Cognitive Complexity from 16 to the 15 allowed. |
| CRITICAL | CODE_SMELL | src/doctranslator/utils/priority_thread_pool_executor.py | Refactor this function to reduce its Cognitive Complexity from 18 to the 15 allowed. |
| MAJOR | CODE_SMELL | src/doctranslator/asynchronize/__init__.py | Rename field "args" |
| MAJOR | CODE_SMELL | src/doctranslator/docvision/doclayout.py | Remove the unused function parameter "kwargs". |
| MAJOR | CODE_SMELL | src/doctranslator/docvision/doclayout.py | Remove the unused function parameter "imgsz". |
| MINOR | BUG | src/doctranslator/docvision/doclayout.py | Introduce a new variable or use its initial value before reassigning 'batch_size'. |
| MINOR | CODE_SMELL | src/doctranslator/docvision/doclayout.py | Remove the unused local variable "max_height". |
| MAJOR | CODE_SMELL | src/doctranslator/docvision/doclayout.py | Remove this commented out code. |
| MAJOR | CODE_SMELL | src/doctranslator/docvision/table_detection/rapidocr.py | Remove the unused function parameter "imgsz". |
| MAJOR | CODE_SMELL | src/doctranslator/docvision/table_detection/rapidocr.py | Remove the unused function parameter "kwargs". |

## File Coverage Breakdown (Top 20 Files by Size)
| File | Lines of Code | Coverage | Complexity |
|---|---|---|---|