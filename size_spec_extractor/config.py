"""Configuration models for the size specification extraction pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PreprocessingConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    denoise_strength: int = 5
    adaptive_block_size: int = 31
    adaptive_c: int = 11
    preserve_resolution: bool = True
    enable_perspective_correction: bool = True
    min_quad_area_ratio: float = 0.20


@dataclass
class TableDetectionConfig:
    mode: str = "auto"  # "auto", "relative", "manual"
    relative_roi: tuple[float, float, float, float] = (
        0.38,
        0.01,
        0.98,
        0.555,
    )  # (ymin, xmin, ymax, xmax) Section 5
    manual_bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)
    min_table_area_ratio: float = 0.05


@dataclass
class LineDetectionConfig:
    h_kernel_length: int = 40
    v_kernel_length: int = 30
    h_kernel_ratio: float = 0.04
    v_kernel_ratio: float = 0.04
    peak_threshold_ratio: float = 0.12
    line_merge_distance: int = 6
    min_cell_height: int = 14
    min_cell_width: int = 14


@dataclass
class CellProcessingConfig:
    margin_ratio_x: float = 0.10
    margin_ratio_y: float = 0.08
    min_ink_ratio: float = 0.005
    upscale_factor: int = 3
    detect_stacked_values: bool = True
    stacked_gap_threshold: int = 4


@dataclass
class OCRConfig:
    engine: str = "tesseract"  # "tesseract", "paddle", "easy", "mock"
    header_psm: int = 7
    numeric_psm: int = 7
    whitelist_numeric: str = "0123456789½¼¾.-/"
    whitelist_header: str = (
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-/ "
    )
    whitelist_size_header: str = "0123456789MYXSL- /"
    min_confidence: float = 0.60
    high_confidence: float = 0.90
    multi_pass: bool = True


@dataclass
class ReconstructionConfig:
    header_row_index: int = 0
    spec_col_index: int = 0
    unit: str = "inch"
    category: str = ""
    style_code: str = ""
    name: str = ""


@dataclass
class ValidationConfig:
    strict: bool = False
    max_measurement_value: float = 120.0
    min_measurement_value: float = 0.0


@dataclass
class HeaderBoxConfig:
    enabled: bool = True
    mode: str = "auto"  # "auto", "metadata", "full", "manual"
    manual_bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)
    filename: str = "header_box.png"
    metadata_filename: str = "metadata_box.png"
    attach_to_reconstructed: bool = True
    attach_to_reconstructed_image: bool = False
    divider_thickness: int = 0
    divider_color: tuple[int, int, int] = (0, 0, 0)


@dataclass
class LLMConfig:
    enabled: bool = True
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    result_filename: str = "result.json"
    prompt: str | None = None


@dataclass
class ExtractorConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    table_detection: TableDetectionConfig = field(default_factory=TableDetectionConfig)
    header_box: HeaderBoxConfig = field(default_factory=HeaderBoxConfig)
    line_detection: LineDetectionConfig = field(default_factory=LineDetectionConfig)
    cell_processing: CellProcessingConfig = field(default_factory=CellProcessingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    debug: bool = False
    debug_dir: str | None = None
    include_coordinates: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtractorConfig:
        hdr_data = dict(d.get("header_box", {}))
        if "manual_bbox" in hdr_data and hdr_data["manual_bbox"] is not None:
            hdr_data["manual_bbox"] = tuple(hdr_data["manual_bbox"])
        if "divider_color" in hdr_data and hdr_data["divider_color"] is not None:
            hdr_data["divider_color"] = tuple(hdr_data["divider_color"])

        tbl_data = dict(d.get("table_detection", {}))
        if "manual_bbox" in tbl_data and tbl_data["manual_bbox"] is not None:
            tbl_data["manual_bbox"] = tuple(tbl_data["manual_bbox"])

        llm_data = d.get("llm", {})
        if isinstance(llm_data, LLMConfig):
            llm_cfg = llm_data
        elif isinstance(llm_data, dict):
            llm_cfg = LLMConfig(**llm_data)
        else:
            llm_cfg = LLMConfig()

        return cls(
            preprocessing=PreprocessingConfig(**d.get("preprocessing", {})),
            table_detection=TableDetectionConfig(**tbl_data),
            header_box=HeaderBoxConfig(**hdr_data),
            line_detection=LineDetectionConfig(**d.get("line_detection", {})),
            cell_processing=CellProcessingConfig(**d.get("cell_processing", {})),
            ocr=OCRConfig(**d.get("ocr", {})),
            reconstruction=ReconstructionConfig(**d.get("reconstruction", {})),
            validation=ValidationConfig(**d.get("validation", {})),
            llm=llm_cfg,
            debug=d.get("debug", False),
            debug_dir=d.get("debug_dir"),
            include_coordinates=d.get("include_coordinates", True),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ExtractorConfig:
        return cls.from_dict(json.loads(json_str))
