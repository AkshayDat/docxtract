import json
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from .schemas import ExtractedTable


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise document table extractor. You receive raw text from a PDF page "
        "that may contain tables mixed with narrative text, headers, footers, and notes.\n\n"
        "Your job:\n"
        "1. Extract ONLY the tabular data — ignore all narrative paragraphs, section titles, "
        "page headers/footers, and instructional text like 'ignore this'.\n"
        "2. Identify column headers from the table structure.\n"
        "3. Extract every data row with values aligned to the correct columns.\n"
        "4. If a cell is empty, use an empty string.\n"
        "5. Preserve exact numeric values — do not round or reformat.\n\n"
        "Respond with a JSON object matching this schema:\n"
        '{{"table_id": "table_N", "title": "...", "columns": [...], '
        '"rows": [[...], ...], "page_start": N, "page_end": N}}\n\n'
        "If NO table is found on the page, respond with:\n"
        '{{"table_id": "none", "title": "", "columns": [], "rows": [], '
        '"page_start": {page_num}, "page_end": {page_num}}}'
    ),
    (
        "human",
        "Page {page_num} content:\n\n{page_text}"
    ),
])


VISION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise document table extractor. You receive an image of a PDF page.\n\n"
        "Your job:\n"
        "1. Extract ONLY the tabular data from the image.\n"
        "2. Identify column headers from the table structure.\n"
        "3. Extract every data row with values aligned to the correct columns.\n"
        "4. If a cell is empty, use an empty string.\n"
        "5. Preserve exact numeric values.\n\n"
        "Respond with a JSON object matching this schema:\n"
        '{{"table_id": "table_N", "title": "...", "columns": [...], '
        '"rows": [[...], ...], "page_start": N, "page_end": N}}\n\n'
        "If NO table is found, respond with:\n"
        '{{"table_id": "none", "title": "", "columns": [], "rows": [], '
        '"page_start": {page_num}, "page_end": {page_num}}}'
    ),
    (
        "human",
        [
            {"type": "text", "text": "Extract tables from page {page_num}:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image_b64}"}},
        ],
    ),
])


def extract_table_from_text(
    llm: ChatOpenAI,
    page_text: str,
    page_num: int,
    table_id: str = "table_1",
) -> Optional[ExtractedTable]:
    """Extract a table from raw page text using LangChain + OpenAI."""
    parser = JsonOutputParser(pydantic_object=ExtractedTable)
    chain = EXTRACTION_PROMPT | llm | parser

    result = chain.invoke({
        "page_text": page_text,
        "page_num": page_num,
    })

    if result.get("table_id") == "none":
        return None

    result["table_id"] = table_id
    return ExtractedTable(**result)


def extract_table_from_image(
    llm: ChatOpenAI,
    image_b64: str,
    page_num: int,
    table_id: str = "table_1",
) -> Optional[ExtractedTable]:
    """Extract a table from a page image using GPT-4o vision."""
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document table extractor. Extract ONLY the tabular data "
                    "from the image. Respond with a JSON object: "
                    '{"table_id": "table_N", "title": "...", "columns": [...], '
                    '"rows": [[...], ...], "page_start": N, "page_end": N}'
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Extract tables from page {page_num}:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ],
        max_tokens=4096,
        temperature=0,
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        data = json.loads(text)
        if data.get("table_id") == "none":
            return None
        data["table_id"] = table_id
        return ExtractedTable(**data)
    except (json.JSONDecodeError, Exception):
        return None
