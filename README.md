# PDF Table Extractor

Extract **only tables** from mixed-content PDF documents (digital, scanned, or mixed) using a 6-phase LLM-powered pipeline built with **LangChain** and **OpenAI GPT-4o**.

Narrative text, headers, footers, page numbers, and decorative elements are automatically ignored.

## Pipeline Overview

```
PDF Input
   |
   v
[1] Page Classification ---- digital (has text layer) or scanned (needs OCR)
   |
   v
[2] OCR (if needed) -------- pytesseract + pdf2image (skipped for digital pages)
   |
   v
[3] Table Detection -------- pdfplumber.find_tables() -> GPT-4o vision fallback
   |
   v
[4] Structured Extraction --- LangChain + ChatOpenAI + StructuredOutputParser
   |
   v
[5] Multi-Page Merge -------- match column headers across pages -> stitch rows
   |
   v
[6] LLM Validation ---------- verify row/col counts, fix OCR noise, confidence
   |
   v
Export: JSON + CSV + Excel
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/pdf-table-extractor.git
cd pdf-table-extractor

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
copy .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-your-key-here
```

### 3. Run

```bash
# Process all PDFs in ./input
python main.py --input ./input --output ./output

# Process a single file
python main.py --file doc_01.pdf

# Use a cheaper/faster model
python main.py --model gpt-4o-mini

# Skip LLM validation step
python main.py --no-validate
```

## Output Format

For each input PDF, the pipeline produces three output files:

### JSON (`<doc_id>_tables.json`)

```json
{
  "source_file": "doc_01.pdf",
  "total_pages": 2,
  "tables": [
    {
      "table_id": "table_1",
      "title": "Transaction Summary",
      "columns": ["Date", "Description", "Debit", "Credit", "Balance"],
      "rows": [
        ["2025-01-05", "Opening Balance", "", "", "5,000.00"],
        ["2025-01-06", "ATM Withdrawal", "200.00", "", "4,800.00"]
      ],
      "page_start": 1,
      "page_end": 1
    }
  ]
}
```

### CSV (`<doc_id>_table_1.csv`)

One CSV file per table with headers as the first row.

### Excel (`<doc_id>_tables.xlsx`)

One workbook per document, one sheet per table.

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--input` | `./input` | Directory containing PDF files |
| `--output` | `./output` | Directory for output files |
| `--file` | — | Process a single PDF file |
| `--model` | `gpt-4o` | OpenAI model to use |
| `--no-validate` | off | Skip the LLM validation phase |
| `--max-retries` | `2` | Max retries if validation fails |

## Project Structure

```
.
├── main.py                  # CLI entry point
├── requirements.txt
├── .env.example
├── input/                   # Place PDF files here
├── output/                  # Results are written here
└── src/
    ├── schemas.py           # Pydantic models (ExtractedTable, DocumentTables)
    ├── classifier.py        # Digital vs scanned page detection (pdfplumber)
    ├── ocr.py               # Tesseract OCR + page-to-image helpers
    ├── detector.py          # Table detection (pdfplumber + GPT-4o vision)
    ├── extractor.py         # LangChain extraction chains (text + vision)
    ├── merger.py            # Multi-page table stitching by column matching
    ├── validator.py         # LLM-based validation and cleanup
    ├── exporter.py          # JSON / CSV / Excel export
    └── pipeline.py          # Full 6-phase orchestrator
```

## Approach and Trade-offs

### Detection Strategy: pdfplumber-first, LLM-fallback

- **pdfplumber** is used as the primary detector — it's fast, free, and very accurate on bordered tables with clear cell boundaries.
- When pdfplumber finds no tables (common with borderless/whitespace-aligned tables), the pipeline falls back to **GPT-4o text extraction** or **GPT-4o vision** (for scanned pages).
- This hybrid approach keeps API costs low while still handling edge cases.

### OCR: Conditional, not blanket

Pages are classified before processing. OCR (Tesseract) only runs on pages that lack an extractable text layer. This avoids degrading quality on digital PDFs and saves processing time.

### Multi-page tables

The merger compares column headers across consecutive pages. If headers match (exact or >= 70% overlap), rows are stitched into a single logical table with `page_start` and `page_end` tracking the span.

### Validation

An optional LLM validation pass checks for:
- Column count mismatches across rows
- Narrative text leaking into cells
- OCR artifacts (garbled characters, misplaced decimals)
- Empty/orphan rows

If issues are found, the extraction is retried (up to `--max-retries` times).

## System Requirements

- **Python** 3.9+
- **OpenAI API key** (GPT-4o or GPT-4o-mini)
- **Tesseract OCR** — only needed for scanned PDFs ([install guide](https://github.com/tesseract-ocr/tesseract))
- **Poppler** — only needed for scanned PDFs, required by `pdf2image` ([install guide](https://github.com/Belval/pdf2image#how-to-install))

## Assumptions

1. Input PDFs are well-formed and not password-protected.
2. Tables have visually distinct structure (borders, alignment, or header rows) — the pipeline does not extract data from free-form prose formatted to look tabular.
3. For scanned PDFs, Tesseract and Poppler must be installed and on the system PATH.
4. OpenAI API usage incurs costs — `gpt-4o-mini` can be used as a lower-cost alternative with `--model gpt-4o-mini`.

## Improvements with More Time

- **Layout detection model** (e.g., DETR/LayoutLMv3) for table region detection without LLM calls.
- **Confidence scores** per cell/row from the LLM validation pass.
- **Bounding box output** for each detected table region.
- **Parallel processing** of multiple PDFs using async/multiprocessing.
- **Evaluation script** to compare extracted tables against ground truth using cell-level F1.
- **Caching** of LLM responses to avoid redundant API calls during retries.
