"""Web application and REST API for garment size specification extraction.

Features:
- Single-page frontend visualization (HTML table rendering + metadata cards)
- POST /api/extract: uploads designsheet, processes through pipeline, returns JSON data only
- Automatically cleans up all intermediate files, table images, and temporary folders
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import sys
import tempfile

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from size_spec_extractor.config import ExtractorConfig
from size_spec_extractor.extractor import extract_table_segments
from size_spec_extractor.llm import (
    extract_remarks_with_llm,
    extract_with_llm,
    load_llm_env,
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"}

app = Flask(__name__, template_folder="templates")


def is_allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    """Render single-page frontend application."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint and LLM configuration status."""
    env_vars = load_llm_env()
    has_key = bool(env_vars.get("OPENAI_API_KEY", "").strip())
    model = env_vars.get("OPENAI_MODEL", "gpt-4o")
    return jsonify(
        {
            "status": "healthy",
            "openai_configured": has_key,
            "model": model,
        }
    )


@app.route("/api/extract", methods=["POST"])
def extract_api():
    """API endpoint to extract size specification data from designsheet images.

    Accepts:
        Multipart form with:
            - 'front_image' / 'front' / 'image' / 'file': Front designsheet (measurements & table).
            - 'back_image' / 'back': (Optional) Back designsheet (remarks & notes).

    Returns:
        JSON object containing:
            - item_name
            - category
            - style_code
            - size_spec_table (HTML <table> string)
            - remarks (extracted from back image if provided)

    Note:
        All intermediate folders and images (front, back, table.png, reconstructed_table.png,
        header_box.png, cell images, etc.) are deleted immediately after processing.
    """
    # Check uploaded front file
    front_file = None
    for k in ["front_image", "front", "image", "file"]:
        if k in request.files and request.files[k].filename:
            front_file = request.files[k]
            break

    if front_file is None or not front_file.filename:
        return (
            jsonify(
                {
                    "error": "No image file provided in request (use 'front_image', 'front', or 'image')."
                }
            ),
            400,
        )

    front_filename = secure_filename(front_file.filename)
    if not is_allowed_file(front_filename):
        return (
            jsonify(
                {
                    "error": f"Unsupported file extension. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                }
            ),
            400,
        )

    # Check optional back file
    back_file = None
    for k in ["back_image", "back"]:
        if k in request.files and request.files[k].filename:
            back_file = request.files[k]
            break

    if back_file is not None and back_file.filename:
        back_filename = secure_filename(back_file.filename)
        if not is_allowed_file(back_filename):
            return (
                jsonify(
                    {
                        "error": f"Unsupported back file extension. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                    }
                ),
                400,
            )

    # Check if OPENAI_API_KEY is configured
    env_vars = load_llm_env()
    api_key = env_vars.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return (
            jsonify(
                {
                    "error": "OPENAI_API_KEY is not configured in .env file. Please add your OpenAI API key to .env to enable extraction."
                }
            ),
            400,
        )

    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp(prefix="size_spec_api_")
    try:
        # Save front file
        front_ext = (
            front_filename.rsplit(".", 1)[1].lower() if "." in front_filename else "png"
        )
        temp_front_path = os.path.join(temp_dir, f"front_designsheet.{front_ext}")
        front_file.save(temp_front_path)

        # Save back file if provided
        temp_back_path = None
        if back_file is not None and back_file.filename:
            back_ext = (
                back_filename.rsplit(".", 1)[1].lower()
                if "." in back_filename
                else "png"
            )
            temp_back_path = os.path.join(temp_dir, f"back_designsheet.{back_ext}")
            back_file.save(temp_back_path)

        # Run CV table extraction & reconstruction on front image (run_llm=False)
        config = ExtractorConfig()
        result = extract_table_segments(
            image_input=temp_front_path,
            output_dir=temp_dir,
            config=config,
            run_llm=False,
        )

        recon_target = (
            result.reconstructed_table_path
            if (
                result.reconstructed_table_path
                and os.path.exists(result.reconstructed_table_path)
            )
            else (
                result.table_image_path
                if (result.table_image_path and os.path.exists(result.table_image_path))
                else (
                    result.reconstructed_table_image
                    if result.reconstructed_table_image is not None
                    else result.table_image
                )
            )
        )

        # If back image provided, call both LLM requests at the same time concurrently
        if temp_back_path is not None:
            print(
                "[*] Calling both LLM requests (front table + back remarks) at the same time in parallel..."
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                front_future = executor.submit(
                    extract_with_llm,
                    recon_target,
                    output_dir=temp_dir,
                    config=config,
                )
                back_future = executor.submit(
                    extract_remarks_with_llm,
                    temp_back_path,
                    config=config,
                )
                front_result = front_future.result()
                remarks = back_future.result()
                llm_data = front_result[0] if front_result else None
        else:
            # Only front image provided: execute front LLM request
            front_result = extract_with_llm(
                recon_target,
                output_dir=temp_dir,
                config=config,
            )
            llm_data = front_result[0] if front_result else None
            remarks = ""

        if not llm_data:
            return (
                jsonify(
                    {
                        "error": "Extraction pipeline completed, but LLM failed to return valid JSON data. Please verify your OpenAI credentials and image format."
                    }
                ),
                500,
            )

        llm_data["remarks"] = remarks or llm_data.get("remarks", "")

        # Return clean JSON payload directly
        return jsonify(llm_data), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process design sheet: {e!s}"}), 500

    finally:
        # Explicitly delete temporary folder and all generated table/segment images
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as err:
                print(
                    f"[!] Warning: failed to clean up temporary directory {temp_dir}: {err}",
                    file=sys.stderr,
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Garment Size Spec Extraction Web App & API"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "5050")),
        help="Port to bind (default: 5000)",
    )
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        f"[*] Starting Size Spec Extractor Web Server on http://{args.host}:{args.port}"
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
