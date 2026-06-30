from typing import List

from .schemas import ExtractedTable


def normalize_columns(columns: List[str]) -> List[str]:
    """Normalize column names for comparison."""
    return [c.strip().lower().replace(" ", "_") for c in columns]


def tables_are_continuation(table_a: ExtractedTable, table_b: ExtractedTable) -> bool:
    """Check if table_b is a continuation of table_a (same columns, consecutive pages)."""
    if table_b.page_start != table_a.page_end + 1:
        return False
    norm_a = normalize_columns(table_a.columns)
    norm_b = normalize_columns(table_b.columns)
    if norm_a == norm_b:
        return True
    if len(norm_a) == len(norm_b):
        matches = sum(1 for a, b in zip(norm_a, norm_b) if a == b)
        return matches / len(norm_a) >= 0.7
    return False


def merge_tables(tables: List[ExtractedTable]) -> List[ExtractedTable]:
    """Merge consecutive tables that span multiple pages into single logical tables."""
    if not tables:
        return []

    merged = [tables[0]]

    for table in tables[1:]:
        last = merged[-1]
        if tables_are_continuation(last, table):
            merged[-1] = ExtractedTable(
                table_id=last.table_id,
                title=last.title or table.title,
                columns=last.columns,
                rows=last.rows + table.rows,
                page_start=last.page_start,
                page_end=table.page_end,
            )
        else:
            merged.append(table)

    for i, table in enumerate(merged):
        merged[i] = table.model_copy(update={"table_id": f"table_{i + 1}"})

    return merged
