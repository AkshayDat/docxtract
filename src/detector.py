from typing import Dict, List

import pdfplumber
from openai import OpenAI


def detect_tables_pdfplumber(pdf_path: str, page_num: int) -> List[Dict]:
    """Detect table regions on a page using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        tables = page.find_tables()
        results = []
        for i, table in enumerate(tables):
            extracted = table.extract()
            if extracted and len(extracted) > 1:
                results.append({
                    "table_index": i,
                    "bbox": table.bbox,
                    "rows": extracted,
                    "row_count": len(extracted),
                })
        return results


def detect_tables_vision(client: OpenAI, image_b64: str, page_num: int) -> dict:
    """Use OpenAI GPT-4o vision to detect and describe tables on a page image."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document analysis expert. Analyze the page image and identify "
                    "all tables present. For each table, describe its position, column headers, "
                    "and approximate row count. Respond ONLY with a JSON object: "
                    '{"tables_found": true/false, "count": N, "descriptions": [{"title": "...", "columns": [...], "approx_rows": N}]}'
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze page {page_num} for tables:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ],
        max_tokens=1000,
        temperature=0,
    )
    import json
    try:
        text = response.choices[0].message.content
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {"tables_found": False, "count": 0, "descriptions": []}
