"""Output data schemas and document models (Section 21, 22)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SizeSpecRow:
    specification: str
    values: list[Any]  # Strings like "1½", list for stacked values, or None
    parsed_values: list[float | None] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"specification": self.specification, "values": self.values}
        if self.parsed_values is not None:
            d["parsed_values"] = self.parsed_values
        return d


@dataclass
class SizeSpecTable:
    unit: str = "inch"
    columns: list[str] = field(default_factory=list)
    rows: list[SizeSpecRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "columns": self.columns,
            "rows": [r.to_dict() for r in self.rows],
        }


@dataclass
class GarmentSpecDocument:
    category: str = ""
    style_code: str = ""
    name: str = ""
    size_spec_table: SizeSpecTable = field(default_factory=SizeSpecTable)
    fabric_colour_code_table: list[dict[str, Any]] = field(default_factory=list)
    remarks: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "style_code": self.style_code,
            "name": self.name,
            "size_spec_table": self.size_spec_table.to_dict(),
            "fabric_colour_code_table": self.fabric_colour_code_table,
            "remarks": self.remarks,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class CellCoordinateMetadata:
    """Intermediate coordinate metadata for debugging and traceability (Section 22)."""

    row: int
    column: int
    bbox: list[int]  # [x1, y1, x2, y2]
    ocr: dict[str, Any]  # {"text": "2½", "confidence": 0.94}
    status: str = "accepted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "column": self.column,
            "bbox": self.bbox,
            "ocr": self.ocr,
            "status": self.status,
        }


@dataclass
class ExtractionResult:
    document: GarmentSpecDocument
    metadata: list[CellCoordinateMetadata] = field(default_factory=list)
    table_bbox: tuple[int, int, int, int] | None = None
    validation: dict[str, Any] | None = None

    def to_dict(self, include_metadata: bool = True) -> dict[str, Any]:
        res = self.document.to_dict()
        if include_metadata and self.metadata:
            res["_debug_coordinates"] = [m.to_dict() for m in self.metadata]
        if self.validation:
            res["_validation"] = self.validation
        return res

    def to_json(self, indent: int = 2, include_metadata: bool = False) -> str:
        return json.dumps(
            self.to_dict(include_metadata=include_metadata),
            indent=indent,
            ensure_ascii=False,
        )
