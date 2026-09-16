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
        "--cell-format",
        dest="cell_format",
        default="png",
        help="Image format/extension for cropped cell box images (default: 'png').",
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

    parsed = parser.parse_args(args)
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
        return 0

    # Default action: Detect size spec table and segment cells into output folder
    try:
        print(f"[*] Processing image: {image_path}")
        table_fname = args.table_filename or "table.png"
        cell_fmt = (args.cell_format or "png").lstrip(".")
        result = extract_table_segments(
            image_input=image_path,
            output_dir=output_dir,
            config=config,
            table_filename=table_fname,
            cell_format=cell_fmt,
        )

        # Save standalone image file if needed (e.g. for CLI compatibility)
        if standalone_image_path:
            abs_standalone = os.path.abspath(standalone_image_path)
            abs_table = os.path.abspath(result.table_image_path or "")
            if abs_standalone != abs_table:
                save_table_image(result.table_image, standalone_image_path)

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
        print(
            f"    - Grid Dimensions: {grid.num_rows} rows x {grid.num_cols} columns ({total_cells} cells)"
        )
        print(f"    - Output Folder: {saved_abs}")
        print(f"    - Main Table Image: {os.path.join(saved_abs, table_fname)}")
        if standalone_image_path:
            print(
                f"    - Saved table image to: {os.path.abspath(standalone_image_path)}"
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
