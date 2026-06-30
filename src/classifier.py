from typing import Dict, List

import pdfplumber


def classify_pages(pdf_path: str) -> List[Dict]:
    """Classify each page as digital (has text layer) or scanned (needs OCR)."""
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            has_text = len(text.strip()) > 20
            results.append({
                "page_num": i + 1,
                "is_digital": has_text,
                "text": text if has_text else "",
                "width": page.width,
                "height": page.height,
            })
    return results
