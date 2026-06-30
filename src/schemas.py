from typing import List

from pydantic import BaseModel, Field


class ExtractedTable(BaseModel):
    table_id: str = Field(description="Unique identifier like 'table_1', 'table_2'")
    title: str = Field(default="", description="Table title/caption if found")
    columns: List[str] = Field(description="List of column header names")
    rows: List[List[str]] = Field(description="List of rows, each row is a list of cell values")
    page_start: int = Field(description="First page where this table appears (1-indexed)")
    page_end: int = Field(description="Last page where this table appears (1-indexed)")


class DocumentTables(BaseModel):
    source_file: str = Field(description="Original PDF filename")
    total_pages: int = Field(description="Total pages in the PDF")
    tables: List[ExtractedTable] = Field(description="All tables extracted from the document")
