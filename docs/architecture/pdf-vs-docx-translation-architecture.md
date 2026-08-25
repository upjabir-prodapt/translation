# Translation Pipeline Architecture: PDF vs DOCX

This document explains the end-to-end architecture for both the **existing PDF
translation pipeline** and the **planned native DOCX translation pipeline**,
component by component, with worked examples for each format. It is written
to answer: *"how does translation actually happen, and how is DOCX different
from PDF?"*

---

## 1. High-Level Comparison

| Aspect | PDF Pipeline (existing) | DOCX Pipeline (planned, native) |
|---|---|---|
| Input parsing | Custom `pdfminer`-based PDF interpreter → Intermediate Language (IL) tree | `python-docx` OOXML object model (paragraphs, runs, tables, headers/footers) |
| Structural unit | "Page" (PDF page) containing paragraphs/characters/curves | No pages exist structurally; document is one continuous flow of paragraphs + tables + headers/footers |
| Layout detection | ONNX vision model (DocLayout) rasterizes each page and detects text/figure/table regions | **Not needed** — OOXML already has explicit, addressable structure (`<w:p>`, `<w:tbl>`, `<w:tr>`, `<w:tc>`) |
| OCR / scanned-page detection | Yes (`DetectScannedFile`) | **Not applicable** — DOCX is never a scanned image |
| Table detection | ONNX/RapidOCR table-region detection on rendered pixels | **Not needed** — tables are native XML elements (`document.tables`, walked directly) |
| Translation batching unit | Paragraphs within a PDF page, batched by token/paragraph-count limits; cross-page/cross-column paragraph merging for split sentences | Paragraphs in **document reading order** (body → tables → headers/footers), batched by the *same* token/paragraph-count limits; no cross-page merging needed (paragraphs are already whole XML units) |
| Very large documents | Pre-split into ~10-page "chunks", each chunk translated independently in parallel, then reassembled | No chunk/part infra needed; same batch-size caps apply directly across the whole paragraph stream |
| Formatting reconstruction | Full PDF content-stream redraw: font subsetting, glyph positioning, draw operators | In-place text replacement inside existing `<w:r>` (run) XML elements — original styles/layout/tables/images untouched by construction |
| Images | Re-drawn as part of PDF content stream reconstruction | **Untouched** — DOCX image relationships are never touched since only run text is modified |
| Output format | New PDF file (font-subsetted, redrawn) | Same `.docx` file, only text content mutated |
| LLM call pattern | Batch JSON `{"id":N,"input":...}` → `{"id":N,"output":...}`, same schema for all batches | **Identical** batch JSON schema and batching config — full code/infra reuse |
| Glossary / DLP / Quality Judge / Translation cache / Rate limiter | Shared services, PDF-IL-specific glue code | **Same shared services**, DOCX-paragraph-specific glue code |

**Bottom line:** DOCX translation is architecturally a *lighter* version of
the PDF pipeline — it skips every PDF-only concern (layout vision model, OCR,
font subsetting, PDF content-stream drawing) because OOXML already gives us
structured, addressable text. Both pipelines converge on the exact same
LLM-calling machinery (translator classes, cache, rate limiter, glossary,
DLP, quality judge) — only the "extract text from the document" and "write
translated text back into the document" steps differ.

---

## 2. PDF Translation — End-to-End Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              PDF TRANSLATION PIPELINE                             │
└──────────────────────────────────────────────────────────────────────────────────┘

 [1] Prepare Working PDF
     • Copy input PDF into job working dir
     • Fix malformed PDF structure (null pages, bad xrefs, filters, mediabox)
     • pymupdf.Document opened for downstream stages
        │
        ▼
 [2] Parse PDF Stream → Intermediate Language (IL) Tree      (ILCreater)
     • Custom pdfminer-based PDFPageInterpreterEx walks every PDF content
       stream operator (text-show, font-select, curve-draw, xobject, etc.)
     • Builds a structured tree: Document → Page → PdfParagraph →
       PdfParagraphComposition → PdfSameStyleCharacters/Formula
     • 100% LANGUAGE-INDEPENDENT — pure source-document structural parse
        │
        ▼
 [3] Detect Scanned File                                     (DetectScannedFile)
     • Heuristics on the IL tree + rendered pixels to flag OCR-needed pages
     • LANGUAGE-INDEPENDENT
        │
        ▼
 [4] Layout Parsing                                           (LayoutParser)
     • Rasterize each page → ONNX DocLayout model → bounding boxes for
       text/title/figure/table/list regions
     • LANGUAGE-INDEPENDENT (this is why it's a "parse once" candidate for
       multi-target-language jobs — see shared_document_prep.py)
        │
        ▼
 [5] Table Parsing                                             (TableParser)
     • RapidOCR/ONNX table-region detection + cell reconstruction
     • LANGUAGE-INDEPENDENT
        │
        ▼
 [6] Paragraph Finding                                     (ParagraphFinder)
     • Groups raw characters into paragraph objects using layout regions
     • LANGUAGE-INDEPENDENT
        │
        ▼
 [7] Styles & Formulas                                (StylesAndFormulas)
     • Detects inline math/formula runs, style spans (bold/italic/color)
     • LANGUAGE-INDEPENDENT
        │
        ▼
 [8] DLP Masking (pre-translation, optional)              (DlpService)
     • Extract all paragraph.unicode strings → mask_chunks() → PII replaced
       with __DLP_TOKEN_NNNN__ placeholders (Google Cloud DLP API, regex
       fallback) → written back into paragraph.unicode
        │
        ▼
 [9] Automatic Term/Glossary Extraction (optional)   (AutomaticTermExtractor)
     • Batches paragraphs → LLM call asking for {"src":.., "tgt":..} term
       pairs relative to target language
     • Builds an in-memory auto-extracted Glossary for this job
     • [NEW] After job succeeds: merge genuinely-new terms into the
       domain's GCS glossary JSON (see Section 4)
        │
        ▼
 [10] Paragraph Translation                    (ILTranslatorLLMOnly / ILTranslator)
      • Walks pages; for each page, batches paragraphs by
        LLM_TRANSLATION_BATCH_MAX_TOKENS / MAX_PARAGRAPHS
      • Cross-page & cross-column paragraph merging: paragraphs whose
        sentence continues across a page/column boundary are batched
        together for translation coherence
      • Each batch → one LLM call:
            Prompt = role + structure rules + glossary tables (domain +
                     per-job + auto-extracted) + JSON array of paragraphs
            Response = JSON array of translated paragraphs
      • BaseTranslator.llm_translate() → Redis translation cache
        (Memorystore via PSC) checked first → cache miss → real LLM
        call (Gemini/Claude) → cache populated with a 7-day TTL
      • Per-batch validation: same-text fallback, length-ratio fallback,
        edit-distance fallback → single-paragraph re-translate on failure
        │
        ▼
 [11] DLP Unmasking (post-translation)
      • Replace __DLP_TOKEN_NNNN__ placeholders back with original values
      • Any leaked tokens stripped + logged critical
        │
        ▼
 [12] Typesetting                                              (Typesetting)
      • Re-flows translated text into original paragraph bounding boxes
      • Computes new font scale factors per paragraph to fit
        │
        ▼
 [13] Font Mapping & Subsetting                    (FontMapper, PDFCreater)
      • Maps translated Unicode text to appropriate font files (CJK fonts,
        Latin fonts) per language
      • Subsets fonts to only the glyphs actually used (keeps file small)
        │
        ▼
 [14] PDF Generation                                          (PDFCreater)
      • Draws translated text + original graphics as new PDF content
        streams
      • Cover page (job metadata) prepended if enabled
        │
        ▼
 [15] Save PDF
      • Final translated PDF written to working dir → uploaded to GCS
        │
        ▼
 [16] Quality Judge                                    (GoogleADKJudgeAgent)
      • Concatenated source+translated text → one LLM call scoring
        alignment/omission/hallucination → composite final_score
        │
        ▼
 [17] Model-Attempt Loop                          (ModelAttemptOrchestrator)
      • If judge score < QUALITY_THRESHOLD, repeat steps 1–16 with next
        model in the model_chain (up to MAX_MODEL_ATTEMPTS)
      • Best-scoring attempt's output is kept; losing attempts' working
        dirs are cleaned up immediately
```

### Example: Translating a PDF ("Contract.pdf", English → French)

1. Input: `Contract.pdf`, 12 pages, 3 tables, a few bold/italic clauses.
2. Steps 1–7 build an IL tree: `Document.page[0..11]`, each page has
   `pdf_paragraph` list (e.g. page 0 has 8 paragraphs: title, 6 body
   paragraphs, 1 table-derived paragraph).
3. Step 8 (DLP): a clause containing `john.doe@example.com` and a phone
   number gets replaced with `__DLP_TOKEN_0001__` and `__DLP_TOKEN_0002__`
   before any LLM ever sees the real values.
4. Step 9 (term extraction): the LLM notices `"Effective Date"`,
   `"Force Majeure"` are recurring legal terms not yet in
   `legal.json` for French → extracted as candidate terms.
5. Step 10 (translation): page 0's 8 paragraphs (small) get sent as ONE
   batch:
   ```json
   [
     {"id": 0, "input": "AGREEMENT", "layout_label": "title"},
     {"id": 1, "input": "This Agreement is entered into as of the Effective Date...", "layout_label": "text"},
     ...
   ]
   ```
   → LLM response:
   ```json
   [
     {"id": 0, "output": "ACCORD"},
     {"id": 1, "output": "Le présent accord est conclu à la Date d'Effet..."},
     ...
   ]
   ```
   Glossary table injected into the prompt ensures `"Effective Date"` →
   `"Date d'Effet"` consistently, using the domain glossary entry (or the
   just-extracted candidate if not yet in the domain glossary).
6. Step 11: `__DLP_TOKEN_0001__`/`0002__` replaced back with the real email
   and phone number in the translated text.
7. Steps 12–15: translated French text (usually ~15-20% longer than
   English) is re-flowed into the original paragraph boxes, French-capable
   font subset embedded, new PDF pages drawn and saved.
8. Step 16: judge scores the whole document 0.91 → passes threshold →
   attempt accepted, no retry needed.
9. New terms `"Effective Date"`→`"Date d'Effet"`,
   `"Force Majeure"`→`"Force Majeure"` (kept as-is) are merged into
   `legal.json`'s French entries once the job is marked `completed`.

---

## 3. DOCX Translation — End-to-End Architecture (Planned)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             DOCX TRANSLATION PIPELINE                             │
└──────────────────────────────────────────────────────────────────────────────────┘

 [1] Open DOCX Document                                    (python-docx Document)
     • Load .docx directly — no PDF conversion, no LibreOffice subprocess
     • 100% LANGUAGE-INDEPENDENT
        │
        ▼
 [2] Extract Translatable Units                    (docx_translator.extract_units)
     • Walk in document reading order:
         a. Body paragraphs (document.paragraphs)
         b. Tables — recursively, including nested tables
            (table.rows[i].cells[j].paragraphs, including any tables
             embedded inside a cell)
         c. Headers & footers for every section
            (section.header.paragraphs, section.footer.paragraphs,
             plus first-page/even-page variants if defined)
     • Each unit = one paragraph object with a stable id + reference back
       to its python-docx Paragraph (so we can write the translation back
       into the exact same object later)
     • Skip empty/whitespace-only paragraphs (advance an id but no LLM call)
     • LANGUAGE-INDEPENDENT — identical output regardless of target language
        │
        ▼
 [3] DLP Masking (pre-translation, optional)                (DlpService — REUSED)
     • Same DlpService.mask_chunks() as PDF — operates on plain strings,
       no PDF-specific coupling. Extract text of every unit → mask → write
       masked text back into each Paragraph's first run
        │
        ▼
 [4] Automatic Term/Glossary Extraction (optional)  (shared, ported from PDF)
     • Same batching + same LLM prompt style as AutomaticTermExtractor,
       operating on DOCX paragraph text instead of PdfParagraph.unicode
     • Builds in-memory auto-extracted Glossary for this job
     • [NEW] After job succeeds: merge genuinely-new terms into the
       domain's GCS glossary JSON (shared logic — see Section 4; same
       mechanism used by PDF)
        │
        ▼
 [5] Paragraph Translation                        (docx_translator — NEW, mirrors
                                                     ILTranslatorLLMOnly's batching)
     • Batches units (body + tables + headers/footers, in document order)
       by the SAME LLM_TRANSLATION_BATCH_MAX_TOKENS / MAX_PARAGRAPHS caps
       used by PDF
     • No cross-page merging needed — every paragraph is already a
       complete XML unit, never split across a page boundary
     • Each batch → one LLM call using the IDENTICAL JSON schema as PDF:
            {"id":0,"input":"...", "layout_label":"text"} →
            {"id":0,"output":"..."}
       (layout_label distinguishes body/table-cell/header/footer text so
        the prompt can apply appropriate register, e.g. footers are often
        short boilerplate)
     • BaseTranslator.llm_translate() → same Redis translation cache
       (Memorystore via PSC), same per-provider rate limiter, same
       Gemini/Claude translator classes as PDF — ZERO new LLM-calling
       code, full reuse
     • Same validation fallbacks (same-text / length-ratio / edit-distance)
       as PDF, single-paragraph re-translate on failure
        │
        ▼
 [6] DLP Unmasking (post-translation)                        (shared logic)
     • Replace __DLP_TOKEN_NNNN__ back with original values in each
       Paragraph's text
        │
        ▼
 [7] Write-Back Into DOCX                          (docx_translator.write_back)
     • For each translated paragraph: keep the paragraph's FIRST run's
       formatting (font, bold, italic, size, color, language tag); set its
       text to the translated string; delete any additional runs in that
       paragraph
     • Table cell paragraphs, header/footer paragraphs updated the same way
     • Images, embedded objects, styles.xml, numbering.xml, section
       properties — UNTOUCHED (we never touch anything except run.text)
        │
        ▼
 [8] Save DOCX
     • document.save() → translated .docx written to working dir →
       uploaded to GCS (no format conversion, output is .docx)
        │
        ▼
 [9] Quality Judge                                  (GoogleADKJudgeAgent — REUSED)
     • Concatenated source+translated text → same LLM judge call as PDF
        │
        ▼
 [10] Model-Attempt Loop                    (DOCX-specific, lighter-weight version
                                              of ModelAttemptOrchestrator)
      • If judge score < QUALITY_THRESHOLD, repeat steps 3–9 with next
        model in the model_chain
      • Much cheaper per-attempt than PDF: no re-parsing, no ONNX
        inference, no font subsetting — just re-run translation batches
        against the same in-memory paragraph list
```

### Example: Translating a DOCX ("Report.docx", English → German)

1. Input: `Report.docx` — 1 title page paragraph, a 2-column table with 5
   rows, a header with the company name on every page, a footer with page
   numbers and a confidentiality notice.
2. Step 2 extracts units in order:
   - Header paragraph: `"Acme Corp — Internal"` (id 0)
   - Body paragraph: `"Quarterly Report"` (id 1, title style)
   - Body paragraph: `"This report summarizes Q3 performance..."` (id 2)
   - Table row 1 cell 0: `"Region"` (id 3), cell 1: `"Revenue"` (id 4)
   - Table row 2 cell 0: `"EMEA"` (id 5), cell 1: `"$4.2M"` (id 6)
   - ... (rows 3–5 similarly)
   - Footer paragraph: `"Confidential — do not distribute"` (id 11)
3. Step 3 (DLP): no PII detected in this example — DLP pass is a no-op.
4. Step 4 (term extraction): `"Q3"`, `"EMEA"` flagged as domain terms not
   yet in `commercial.json` for German.
5. Step 5 (translation): all 12 units are small, so they fit in ONE batch:
   ```json
   [
     {"id": 0, "input": "Acme Corp — Internal", "layout_label": "header"},
     {"id": 1, "input": "Quarterly Report", "layout_label": "title"},
     {"id": 2, "input": "This report summarizes Q3 performance...", "layout_label": "text"},
     {"id": 3, "input": "Region", "layout_label": "table_cell"},
     {"id": 4, "input": "Revenue", "layout_label": "table_cell"},
     ...
     {"id": 11, "input": "Confidential — do not distribute", "layout_label": "footer"}
   ]
   ```
   → translated German response with the same ids.
   Numeric cells like `"$4.2M"` are still sent through the batch (so the
   LLM can decide currency/number localization) but typically returned
   unchanged or minimally formatted per target-locale conventions.
6. Step 6: no DLP tokens to unmask in this example.
7. Step 7: each paragraph's first run gets its text replaced
   (`"Quarterly Report"` → `"Quartalsbericht"`), formatting (bold title
   style, table cell borders, header/footer boilerplate style) is
   preserved because we only touched `run.text`, never the paragraph's
   style or the table/section XML.
8. Step 8: `Report.docx` saved with translated text, original images (e.g.
   a company logo in the header) completely untouched.
9. Step 9: judge scores 0.94 → passes → attempt accepted.
10. `"Q3"`/`"EMEA"` German translations merged into `commercial.json`
    after job completion.

---

## 4. Shared Component: Auto-Extracted Glossary Write-Back to GCS

Both PDF and DOCX pipelines can discover new domain terms during
translation. Previously, these were only used transiently for that one job
(kept in-memory in `SharedContextCrossSplitPart.auto_extracted_glossary`)
and thrown away afterward. The new shared behavior:

```
                     ┌─────────────────────────────┐
                     │   Job completes successfully │
                     └───────────────┬─────────────┘
                                     ▼
                  Collect auto-extracted term pairs found
                  during this job (src, tgt) for
                  (domain, target_language)
                                     │
                                     ▼
                  Filter out terms already present in the
                  currently-cached domain glossary
                  (Glossary.normalized_lookup)
                                     │
                                     ▼
              ┌──────────────────────────────────────────┐
              │ Read-Modify-Write loop against GCS:       │
              │  1. Re-download gs://.../{domain}.json    │
              │     (get current blob generation number)  │
              │  2. Merge new terms into the JSON's        │
              │     "translations" map for this language   │
              │     (skip any term added by a concurrent   │
              │     job in the meantime)                    │
              │  3. Upload with if_generation_match=<gen>  │
              │     → GCS rejects the write if another job │
              │       updated the file first                │
              │  4. On rejection: retry from step 1         │
              │     (bounded retry count)                    │
              └──────────────────────────────────────────┘
                                     │
                                     ▼
                Refresh local per-worker glossary cache file
                so subsequent jobs on this instance see the
                update immediately (not waiting for TTL expiry)
```

**Why "only after successful completion"**: if a job fails validation, is
cancelled, or produces a low-quality translation, its candidate term
extractions are discarded — the shared domain glossary only ever learns
from jobs that were judged good enough to ship. This prevents a bad
model attempt from polluting a glossary that every future job (any
language pair, any user) will read from.

**Why optimistic concurrency (generation match) instead of a lock**: GCS
has no native distributed lock; using `if_generation_match` is the
standard "compare-and-swap" pattern for GCS objects — it's safe under
many concurrent workers/jobs writing to the same domain file (e.g. two
sibling jobs in a multi-target-language batch discovering different new
terms at the same time) without needing an external lock service.

---

## 5. Shared Component Reuse Matrix

| Component | Used by PDF | Used by DOCX | Notes |
|---|---|---|---|
| `BaseTranslator` / `GeminiVertexAITranslator` / `ClaudeVertexAITranslator` | ✅ | ✅ | Zero changes needed — operates on plain prompt strings |
| Redis translation cache (Memorystore via PSC) (`translation_cache.py`) | ✅ | ✅ | Cache key includes provider/model/lang pair/text — format-agnostic; 7-day TTL enforced natively by Redis |
| Per-provider rate limiter (`rate_limiter.py`) | ✅ | ✅ | No changes needed |
| `DlpService` | ✅ | ✅ | Operates on `list[str]`, no PDF coupling |
| `GlossaryService` (domain glossary load + new write-back) | ✅ | ✅ | Write-back method shared; both pipelines call it identically |
| `GoogleADKJudgeAgent` (quality judge) | ✅ | ✅ | Takes `source_text`/`translated_text` strings only |
| `LanguageDetectionService` | ✅ (pymupdf-based) | ⚠️ needs small DOCX variant (extract text via python-docx instead of pymupdf) |
| ONNX DocLayout model, RapidOCR table detection | ✅ | ❌ not needed | OOXML structure replaces the need for visual layout detection |
| `TranslationConfig`, `ModelAttemptOrchestrator`, `TranslationAttemptRunner` | ✅ (PDF-IL-coupled) | ❌ new lighter DOCX-specific equivalents | PDF versions carry PDF-only fields (doc_layout_model, split_strategy, watermark mode, working_dir/IL-tree paths) that don't apply to DOCX |
| `TempWorkspaceService`, `AssemblyService`, BigQuery/GCS repositories, Cloud Tasks orchestration | ✅ | ✅ | Fully format-agnostic already |

---

## 6. Why DOCX Doesn't Need "Page-by-Page" Translation

PDF pages are a real structural/rendering boundary — text can be visually
split mid-sentence across a page, and the layout-detection step only knows
about one page's pixels at a time, which is *why* PDF needs cross-page
paragraph merging logic. DOCX has no equivalent problem: a `<w:p>`
paragraph element is always a complete unit in the XML regardless of where
LibreOffice/Word later decides to visually break pages when rendering — the
underlying document model has no per-page boundaries to reconcile. This is
precisely why the DOCX pipeline can batch "in document order" without any
special-casing, and why it requires less pipeline machinery than PDF.
