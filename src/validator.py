import json
from typing import Dict, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .schemas import ExtractedTable


VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a data quality validator for extracted tables. You receive a table as JSON "
        "and must check for these issues:\n"
        "1. Column count mismatches — does every row have the same number of cells as there are columns?\n"
        "2. Narrative text leaking into cells — are any cells actually sentences/paragraphs rather than data?\n"
        "3. OCR artifacts — garbled characters, misplaced decimals, obvious typos in numbers.\n"
        "4. Empty/missing data — rows that are entirely empty.\n\n"
        "Respond with a JSON object:\n"
        '{{"is_valid": true/false, "issues": ["issue 1", ...], '
        '"cleaned_rows": [[...], ...], "confidence": 0.0-1.0}}\n\n'
        "If the table is valid, return the rows unchanged in cleaned_rows. "
        "If you find fixable issues (OCR noise, extra whitespace), fix them in cleaned_rows. "
        "If unfixable, set is_valid to false."
    ),
    (
        "human",
        "Validate this extracted table:\n\n"
        "Columns: {columns}\n\n"
        "Rows ({row_count} total):\n{rows_json}"
    ),
])


def validate_table(
    llm: ChatOpenAI,
    table: ExtractedTable,
) -> Tuple[ExtractedTable, Dict]:
    """Validate and clean an extracted table. Returns (cleaned_table, validation_report)."""
    parser = JsonOutputParser()
    chain = VALIDATION_PROMPT | llm | parser

    rows_json = json.dumps(table.rows[:50], indent=2)

    result = chain.invoke({
        "columns": json.dumps(table.columns),
        "row_count": len(table.rows),
        "rows_json": rows_json,
    })

    report = {
        "is_valid": result.get("is_valid", False),
        "issues": result.get("issues", []),
        "confidence": result.get("confidence", 0.0),
    }

    if result.get("cleaned_rows"):
        cleaned = table.model_copy(update={"rows": result["cleaned_rows"]})
    else:
        cleaned = table

    return cleaned, report
