import os

from langchain_openai import ChatOpenAI
from openai import OpenAI

from .classifier import classify_pages
from .detector import detect_tables_pdfplumber, detect_tables_vision
from .extractor import extract_table_from_text, extract_table_from_image
from .merger import merge_tables
from .ocr import ocr_page, page_to_image, image_to_base64, OCR_AVAILABLE
from .validator import validate_table
from .exporter import export_json, export_csv, export_excel
from .schemas import ExtractedTable, DocumentTables


def process_pdf(
    pdf_path: str,
    output_dir: str,
    model: str = "gpt-4o",
    validate: bool = True,
    max_retries: int = 2,
) -> DocumentTables:
    """Full pipeline: classify → detect → extract → merge → validate → export."""
    filename = os.path.basename(pdf_path)
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    llm = ChatOpenAI(model=model, temperature=0, max_tokens=4096)
    openai_client = OpenAI()

    # Phase 1: Classify pages
    print("\n[1/6] Classifying pages...")
    pages = classify_pages(pdf_path)
    total_pages = len(pages)
    digital_count = sum(1 for p in pages if p["is_digital"])
    scanned_count = total_pages - digital_count
    print(f"  Total pages: {total_pages} ({digital_count} digital, {scanned_count} scanned)")

    # Phase 2: OCR scanned pages
    if scanned_count > 0:
        print("\n[2/6] Running OCR on scanned pages...")
        if not OCR_AVAILABLE:
            print("  WARNING: OCR dependencies not installed. Falling back to vision extraction.")
        else:
            for page in pages:
                if not page["is_digital"]:
                    print(f"  OCR page {page['page_num']}...")
                    page["text"] = ocr_page(pdf_path, page["page_num"])
                    page["is_digital"] = True
    else:
        print("\n[2/6] No scanned pages detected, skipping OCR.")

    # Phase 3: Detect tables
    print("\n[3/6] Detecting tables...")
    raw_tables = []
    table_counter = 0

    for page in pages:
        page_num = page["page_num"]
        page_text = page["text"]

        # Try pdfplumber first
        plumber_tables = detect_tables_pdfplumber(pdf_path, page_num)

        if plumber_tables:
            print(f"  Page {page_num}: {len(plumber_tables)} table(s) found via pdfplumber")
            for pt in plumber_tables:
                table_counter += 1
                headers = pt["rows"][0] if pt["rows"] else []
                rows = pt["rows"][1:] if len(pt["rows"]) > 1 else []
                headers = [str(h) if h else "" for h in headers]
                rows = [[str(c) if c else "" for c in row] for row in rows]

                raw_tables.append(ExtractedTable(
                    table_id=f"table_{table_counter}",
                    title="",
                    columns=headers,
                    rows=rows,
                    page_start=page_num,
                    page_end=page_num,
                ))
        elif page_text.strip():
            # Fall back to LLM text extraction
            print(f"  Page {page_num}: No pdfplumber tables, trying LLM text extraction...")
            table_counter += 1
            extracted = extract_table_from_text(
                llm, page_text, page_num, table_id=f"table_{table_counter}"
            )
            if extracted and extracted.columns:
                raw_tables.append(extracted)
                print(f"  Page {page_num}: LLM extracted table with {len(extracted.columns)} columns, {len(extracted.rows)} rows")
            else:
                table_counter -= 1
                print(f"  Page {page_num}: No tables found")
        else:
            # No text — try vision
            print(f"  Page {page_num}: No text, trying vision extraction...")
            try:
                img = page_to_image(pdf_path, page_num)
                if img:
                    img_b64 = image_to_base64(img)
                    table_counter += 1
                    extracted = extract_table_from_image(
                        llm, img_b64, page_num, table_id=f"table_{table_counter}"
                    )
                    if extracted and extracted.columns:
                        raw_tables.append(extracted)
                        print(f"  Page {page_num}: Vision extracted table with {len(extracted.columns)} columns")
                    else:
                        table_counter -= 1
                        print(f"  Page {page_num}: No tables found via vision")
            except Exception as e:
                print(f"  Page {page_num}: Vision fallback failed: {e}")

    print(f"\n  Raw tables found: {len(raw_tables)}")

    # Phase 4: Merge multi-page tables
    print("\n[4/6] Merging multi-page tables...")
    merged_tables = merge_tables(raw_tables)
    print(f"  Tables after merge: {len(merged_tables)}")
    for t in merged_tables:
        span = f"pages {t.page_start}-{t.page_end}" if t.page_start != t.page_end else f"page {t.page_start}"
        print(f"  - {t.table_id}: {len(t.columns)} cols, {len(t.rows)} rows ({span})")

    # Phase 5: Validate
    if validate and merged_tables:
        print("\n[5/6] Validating extracted tables...")
        validated_tables = []
        for table in merged_tables:
            for attempt in range(max_retries + 1):
                cleaned, report = validate_table(llm, table)
                conf = report["confidence"]
                if report["is_valid"]:
                    print(f"  {table.table_id}: VALID (confidence: {conf:.0%})")
                    validated_tables.append(cleaned)
                    break
                elif attempt < max_retries:
                    print(f"  {table.table_id}: Issues found, retrying ({attempt + 1}/{max_retries})...")
                    print(f"    Issues: {', '.join(report['issues'][:3])}")
                    table = cleaned
                else:
                    print(f"  {table.table_id}: Accepting with issues (confidence: {conf:.0%})")
                    print(f"    Issues: {', '.join(report['issues'][:3])}")
                    validated_tables.append(cleaned)
        merged_tables = validated_tables
    else:
        print("\n[5/6] Skipping validation.")

    # Phase 6: Export
    print("\n[6/6] Exporting results...")
    doc_tables = DocumentTables(
        source_file=filename,
        total_pages=total_pages,
        tables=merged_tables,
    )

    os.makedirs(output_dir, exist_ok=True)

    json_path = export_json(doc_tables, output_dir)
    print(f"  JSON: {json_path}")

    csv_paths = export_csv(doc_tables, output_dir)
    for p in csv_paths:
        print(f"  CSV:  {p}")

    excel_path = export_excel(doc_tables, output_dir)
    print(f"  XLSX: {excel_path}")

    print(f"\nDone: {filename} → {len(merged_tables)} table(s) exported.")
    return doc_tables
