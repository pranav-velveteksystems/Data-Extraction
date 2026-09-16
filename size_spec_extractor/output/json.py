"""JSON formatting and export for extraction results (Section 21, 22)."""

import os

from ..reconstruction.schema import ExtractionResult


def to_json(
    result: ExtractionResult, indent: int = 2, include_metadata: bool = False
) -> str:
    """Serialize extraction result to JSON string."""
    return result.to_json(indent=indent, include_metadata=include_metadata)


def save_json(
    result: ExtractionResult,
    file_path: str,
    indent: int = 2,
    include_metadata: bool = False,
) -> str:
    """Save extraction result to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    json_str = to_json(result, indent=indent, include_metadata=include_metadata)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    return file_path
