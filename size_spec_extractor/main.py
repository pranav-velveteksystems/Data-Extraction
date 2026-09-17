"""Command line interface for Garment Size Specification Table Extractor.

Extracts the size specification table from a garment specification sheet image and saves it as an image.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import ExtractorConfig
from .extractor import (
    SizeSpecExtractor,
    extract_table_segments,
    save_table_image,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    if args is None:
        args = sys.argv[1:]

    # Normalize /boost, -boost into --boost
    normalized_args: list[str] = []
    for a in args:
        if a in ("/boost", "-boost"):
            normalized_args.append("--boost")
        else:
            normalized_args.append(a)

    parser = argparse.ArgumentParser(
        description="Garment Size Specification Table Extractor - Detects and segments the size spec table into box images."
    )
    # Positional or optional input image
    parser.add_argument(
        "image_pos",
        nargs="?",
        default=None,
        metavar="IMAGE",
        help="Path to garment specification sheet image (positional argument)",
    )
    parser.add_argument(
        "--boost",
        "-boost",
        dest="boost",
        action="store_true",
        default=True,
        help="Enable detecting internal cell values, centering them in each block, and reconstructing new table image (default: enabled).",
    )
    parser.add_argument(
        "--no-boost",
        dest="boost",
        action="store_false",
        help="Disable cell value centering and reconstructed table image generation.",
    )
    parser.add_argument(
        "--image",
        "-i",
        dest="image_flag",
        help="Path to garment specification sheet image (-i/--image)",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        help="Path to output folder (or table image file). "
        "Defaults to '<image_stem>_output/'.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Explicit directory path to save extracted table image and segmented box images.",
    )
    parser.add_argument(
        "--output-image",
        dest="output_image",
        help="Optional path to save standalone table image file (e.g. table.png).",
    )
    parser.add_argument(
        "--table-filename",
        dest="table_filename",
        default="table.png",
        help="Filename for the main table image inside output directory (default: 'table.png').",
    )
    parser.add_argument(
        "--reconstructed-table-filename",
        "--reconstructed-filename",
        dest="reconstructed_table_filename",
        default="reconstructed_table.png",
        help="Filename for the reconstructed table image inside output directory (default: 'reconstructed_table.png').",
    )
    parser.add_argument(
        "--output-reconstructed-image",
        "--output-reconstructed",
        dest="output_reconstructed_image",
        help="Optional path to save standalone reconstructed table image file.",
    )
    parser.add_argument(
        "--header-box-filename",
        dest="header_box_filename",
        default="header_box.png",
        help="Filename for the extracted top metadata box inside output directory (default: 'header_box.png').",
    )
    parser.add_argument(
        "--output-header-box-image",
        "--output-header-box",
        dest="output_header_box_image",
        help="Optional path to save standalone extracted top metadata box image file.",
    )
    parser.add_argument(
        "--header-box",
        "--metadata-box",
        dest="header_box_enabled",
        action="store_true",
        default=True,
        help="Enable extraction of the top metadata rectangle (default: enabled).",
    )
    parser.add_argument(
        "--no-header-box",
        "--no-metadata-box",
        dest="header_box_enabled",
        action="store_false",
        help="Disable extraction of the top metadata rectangle.",
    )
    parser.add_argument(
        "--header-box-mode",
        dest="header_box_mode",
        choices=["auto", "metadata", "full", "manual"],
        default="auto",
        help="Mode for top rectangle extraction ('auto', 'metadata', 'full', 'manual').",
    )
    parser.add_argument(
        "--manual-header-bbox",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Manual top rectangle bounding box: x1 y1 x2 y2",
    )
    parser.add_argument(
        "--cell-format",
        dest="cell_format",
        default="png",
        help="Image format/extension for cropped cell box images (default: 'png').",
    )
    parser.add_argument(
        "--llm",
        dest="llm_enabled",
        action="store_true",
        default=True,
        help="Enable LLM extraction with reconstructed table image using OpenAI SDK (default: enabled).",
    )
    parser.add_argument(
        "--no-llm",
        dest="llm_enabled",
        action="store_false",
        help="Disable LLM extraction.",
    )
    parser.add_argument(
        "--llm-model",
        dest="llm_model",
        default=None,
        help="OpenAI model for LLM extraction (default: OPENAI_MODEL from .env or 'gemma-4-31b-it').",
    )
    parser.add_argument(
        "--llm-base-url",
        dest="llm_base_url",
        default=None,
        help="OpenAI API base URL (default: OPENAI_BASE_URL from .env).",
    )
    parser.add_argument(
        "--llm-api-key",
        dest="llm_api_key",
        default=None,
        help="OpenAI API key (default: OPENAI_API_KEY from .env).",
    )
    parser.add_argument(
        "--result-filename",
        "--llm-result-filename",
        dest="result_filename",
        default="result.json",
        help="Filename for LLM extraction result JSON inside output directory (default: 'result.json').",
    )

    parser.add_argument(
        "--config",
        "-c",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--debug-dir",
        "-d",
        help="Directory to save visual debugging artifacts",
    )
    parser.add_argument(
        "--no-perspective",
        action="store_true",
        help="Disable automatic perspective/skew correction",
    )
    parser.add_argument(
        "--manual-bbox",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Manual table bounding box: x1 y1 x2 y2",
    )

    # Optional backwards compatibility flags for full OCR extraction
    parser.add_argument("--output-json", "-j", help=argparse.SUPPRESS)
    parser.add_argument("--output-html", "-H", help=argparse.SUPPRESS)
    parser.add_argument("--ocr-engine", help=argparse.SUPPRESS)
    parser.add_argument("--unit", "-u", default="inch", help=argparse.SUPPRESS)
    parser.add_argument("--category", help=argparse.SUPPRESS)
    parser.add_argument("--style-code", help=argparse.SUPPRESS)
    parser.add_argument("--name", help=argparse.SUPPRESS)
    parser.add_argument(
        "--include-coordinates",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )

    parsed = parser.parse_args(normalized_args)
    image_path = parsed.image_pos or parsed.image_flag
    if not image_path:
        parser.error(
            "An input image must be specified (e.g. 'python main.py <path_to_image>' or '--image <path>')."
        )
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    image_path = args.image_pos or args.image_flag

    if not os.path.exists(image_path):
        print(f"[!] Error: Image file not found: {image_path}", file=sys.stderr)
        return 1

    # Load or initialize configuration
    if args.config:
        if not os.path.exists(args.config):
            print(f"[!] Error: Config file not found: {args.config}", file=sys.stderr)
            return 1
        with open(args.config, "r", encoding="utf-8") as f:
            config = ExtractorConfig.from_json(f.read())
    else:
        config = ExtractorConfig()

    # Apply CLI overrides
    if args.debug_dir:
        config.debug = True
        config.debug_dir = args.debug_dir

    if args.no_perspective:
        config.preprocessing.enable_perspective_correction = False

    if args.manual_bbox:
        config.table_detection.mode = "manual"
        config.table_detection.manual_bbox = tuple(args.manual_bbox)

    if hasattr(args, "header_box_enabled"):
        config.header_box.enabled = args.header_box_enabled
    if hasattr(args, "header_box_mode") and args.header_box_mode:
        config.header_box.mode = args.header_box_mode
    if hasattr(args, "manual_header_bbox") and args.manual_header_bbox:
        config.header_box.mode = "manual"
        config.header_box.manual_bbox = tuple(args.manual_header_bbox)
    if hasattr(args, "header_box_filename") and args.header_box_filename:
        config.header_box.filename = args.header_box_filename

    if hasattr(args, "llm_enabled") and args.llm_enabled is not None:
        config.llm.enabled = args.llm_enabled
    if getattr(args, "llm_model", None):
        config.llm.model = args.llm_model
    if getattr(args, "llm_base_url", None):
        config.llm.base_url = args.llm_base_url
    if getattr(args, "llm_api_key", None):
        config.llm.api_key = args.llm_api_key
    if getattr(args, "result_filename", None):
        config.llm.result_filename = args.result_filename

    # Determine output folder and standalone image path

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    output_dir: str
    standalone_image_path: str | None = None

    if args.output_dir:
        output_dir = args.output_dir
        if args.output_image:
            standalone_image_path = args.output_image
        elif args.output and Path(args.output).suffix.lower() in image_exts:
            standalone_image_path = args.output
    elif args.output_image and not args.output:
        standalone_image_path = args.output_image
        out_p = Path(args.output_image)
        output_dir = (
            str(out_p.parent / out_p.stem) if str(out_p.parent) != "." else out_p.stem
        )
    elif args.output:
        out_p = Path(args.output)
        if out_p.suffix.lower() in image_exts:
            standalone_image_path = args.output
            if (
                out_p.name.lower() in ("table.png", "table.jpg", "table.jpeg")
                and str(out_p.parent) != "."
            ):
                output_dir = str(out_p.parent)
            else:
                output_dir = (
                    str(out_p.parent / out_p.stem)
                    if str(out_p.parent) != "."
                    else out_p.stem
                )
        else:
            output_dir = args.output
            if args.output_image:
                standalone_image_path = args.output_image
    else:
        stem = Path(image_path).stem
        output_dir = f"{stem}_output"
        standalone_image_path = f"{stem}_table.png"

    # Handle full OCR pipeline only if explicitly requested via legacy flags
    if getattr(args, "output_json", None) or getattr(args, "output_html", None):
        if args.unit:
            config.reconstruction.unit = args.unit
        if args.category:
            config.reconstruction.category = args.category
        if args.style_code:
            config.reconstruction.style_code = args.style_code
        if args.name:
            config.reconstruction.name = args.name
        if args.ocr_engine:
            config.ocr.engine = args.ocr_engine

        extractor = SizeSpecExtractor(config=config)
        result = extractor.extract(
            image_path,
            output_table_image=standalone_image_path,
            output_dir=output_dir,
        )
        print(f"[+] Table output folder saved to: {os.path.abspath(output_dir)}")
        if standalone_image_path:
            print(f"[+] Table image saved to: {os.path.abspath(standalone_image_path)}")

        from .output.html import save_html
        from .output.json import save_json

        if args.output_json:
            saved_json = save_json(
                result, args.output_json, include_metadata=config.include_coordinates
            )
            print(f"[+] JSON output saved to: {saved_json}")
        if args.output_html:
            saved_html = save_html(result, args.output_html)
            print(f"[+] HTML output saved to: {saved_html}")
        if getattr(result, "llm_result_path", None):
            print(f"[+] LLM JSON output saved to: {result.llm_result_path}")
        return 0

    # Default action: Detect size spec table and segment cells into output folder
    try:
        print(f"[*] Processing image: {image_path}")
        table_fname = args.table_filename or "table.png"
        is_boost = getattr(args, "boost", True)
        if is_boost:
            recon_fname = (
                getattr(args, "reconstructed_table_filename", "reconstructed_table.png")
                or "reconstructed_table.png"
            )
        else:
            recon_fname = None
        cell_fmt = (args.cell_format or "png").lstrip(".")
        hdr_fname = (
            getattr(args, "header_box_filename", "header_box.png") or "header_box.png"
        )
        result = extract_table_segments(
            image_input=image_path,
            output_dir=output_dir,
            config=config,
            table_filename=table_fname,
            cell_format=cell_fmt,
            reconstructed_table_filename=recon_fname,
            header_box_filename=hdr_fname,
        )

        # Save standalone image file if needed (e.g. for CLI compatibility)
        if standalone_image_path:
            abs_standalone = os.path.abspath(standalone_image_path)
            abs_table = os.path.abspath(result.table_image_path or "")
            if abs_standalone != abs_table:
                save_table_image(result.table_image, standalone_image_path)

        # Save standalone reconstructed image file if requested
        standalone_recon = getattr(args, "output_reconstructed_image", None)
        if standalone_recon and result.reconstructed_table_with_header is not None:
            save_table_image(result.reconstructed_table_with_header, standalone_recon)

        # Save standalone header box image file if requested
        standalone_hdr = getattr(args, "output_header_box_image", None)
        if standalone_hdr and result.header_box_image is not None:
            save_table_image(result.header_box_image, standalone_hdr)

        grid = result.grid
        roi = result.table_roi
        total_cells = grid.num_rows * grid.num_cols
        saved_abs = os.path.abspath(output_dir)

        print("[+] Size specification table extracted successfully:")
        print(f"    - Input image: {image_path}")
        print(
            f"    - Table Bounding Box: {roi.bbox} "
            f"(x1={roi.x1}, y1={roi.y1}, x2={roi.x2}, y2={roi.y2})"
        )
        print(f"    - Table Dimensions: {roi.width} x {roi.height} px")
        if result.header_box_path:
            print(f"    - Top Metadata Box Image: {result.header_box_path}")
        if result.header_box_bbox:
            print(f"    - Top Box Bounding Box: {result.header_box_bbox}")
        print(
            f"    - Grid Dimensions: {grid.num_rows} rows x {grid.num_cols} columns ({total_cells} cells)"
        )
        print(f"    - Output Folder: {saved_abs}")
        print(f"    - Main Table Image: {os.path.join(saved_abs, table_fname)}")
        if result.reconstructed_table_path:
            print(f"    - Reconstructed Table Image: {result.reconstructed_table_path}")
        if result.llm_result_path:
            print(f"    - LLM JSON Result: {result.llm_result_path}")
        if standalone_image_path:
            print(
                f"    - Saved table image to: {os.path.abspath(standalone_image_path)}"
            )
        if standalone_recon:
            print(
                f"    - Saved reconstructed table image to: {os.path.abspath(standalone_recon)}"
            )
        if standalone_hdr:
            print(
                f"    - Saved top metadata box image to: {os.path.abspath(standalone_hdr)}"
            )
        print(
            f"    - Segmented Cell Images: {total_cells} box images saved as [0][0].{cell_fmt} ... [{grid.num_rows - 1}][{grid.num_cols - 1}].{cell_fmt}"
        )

        return 0
    except Exception as e:
        print(f"[!] Extraction failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
