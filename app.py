"""Web application and REST API for garment size specification extraction.

Features:
- Single-page frontend visualization (HTML table rendering + metadata cards)
- POST /api/extract: uploads designsheet, processes through pipeline, returns JSON data only
- Automatically cleans up all intermediate files, table images, and temporary folders
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from size_spec_extractor.config import ExtractorConfig
from size_spec_extractor.extractor import extract_table_segments
from size_spec_extractor.llm import load_llm_env

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
    """API endpoint to extract size specification data from designsheet image.

    Accepts:
        Multipart form with file field 'image' or 'file'.

    Returns:
        JSON object containing:
            - item_name
            - category
            - style_code
            - size_spec_table (HTML <table> string)

    Note:
        All intermediate folders and images (table.png, reconstructed_table.png,
        header_box.png, cell images, etc.) are deleted immediately after processing.
    """
    # Check uploaded file
    file = None
    if "image" in request.files:
        file = request.files["image"]
    elif "file" in request.files:
        file = request.files["file"]

    if file is None or not file.filename:
        return jsonify(
            {
                "error": "No image file provided in request (use 'image' or 'file' field)."
            }
        ), 400

    filename = secure_filename(file.filename)
    if not is_allowed_file(filename):
        return (
            jsonify(
                {
                    "error": f"Unsupported file extension. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
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
        # Save uploaded file
        ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "png"
        temp_input_path = os.path.join(temp_dir, f"input_designsheet.{ext}")
        file.save(temp_input_path)

        # Run extraction pipeline
        config = ExtractorConfig()
        result = extract_table_segments(
            image_input=temp_input_path,
            output_dir=temp_dir,
            config=config,
            run_llm=True,
        )

        llm_data = result.llm_result
        if not llm_data:
            return (
                jsonify(
                    {
                        "error": "Extraction pipeline completed, but LLM failed to return valid JSON data. Please verify your OpenAI credentials and image format."
                    }
                ),
                500,
            )

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
