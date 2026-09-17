"""LLM integration using OpenAI SDK for size specification table extraction."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    from dotenv import dotenv_values, find_dotenv, load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore
    find_dotenv = None  # type: ignore
    dotenv_values = None  # type: ignore

DEFAULT_LLM_PROMPT = """You are an expert system for analyzing garment specification sheets and size measurement tables.

Analyze the provided reconstructed table image, which contains:
1. Top metadata box (containing Category, Style Code, and/or Item Name / Garment Title if available).
2. Size specification table (containing measurement attributes and size columns with numerical measurements).

Extract the information and return ONLY a valid JSON object with the following fields:
{
  "item_name": "<garment item name or title, e.g. 'Ladies Blouse', or empty string if not found>",
  "name": "<same as item_name>",
  "category": "<category name, e.g. 'Denim Pants', or empty string if not found>",
  "style_code": "<style code or product number, e.g. 'ST-1049', or empty string if not found>",
  "size_spec_table": "<table>...</table>"
}

Requirements for "size_spec_table":
- Must be a single valid HTML table string starting with <table> and ending with </table>.
- Include <thead> with <tr> and <th> tags for header rows (such as column headers: Spec / Measurement Description, and all size labels like S, M, L, XL, etc.).
- Include <tbody> with <tr> and <td> tags for all measurement rows and their corresponding values.
- Retain all measurement fractions, decimals, units, and empty cells faithfully.
- Ensure the HTML table in "size_spec_table" is escaped properly within the JSON string and does not contain unescaped raw control characters.
- value must be in number format. means convert 2 1/2 to 2.5

Do NOT wrap the JSON in markdown code blocks if using json_object mode, or if you do, ensure it is strictly parseable JSON.
Return ONLY valid JSON.
"""


def load_llm_env(env_path: Path | str | None = None) -> dict[str, str]:
    """Load OpenAI configuration environment variables from .env file or environment.

    Args:
        env_path: Optional explicit path to .env file.

    Returns:
        Dictionary containing OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL.
    """
    env_data: dict[str, str] = {}
    target_env_file: Path | None = None

    if env_path is not None and os.path.exists(str(env_path)):
        target_env_file = Path(env_path)
    else:
        found = find_dotenv(usecwd=True) if find_dotenv is not None else ""
        if found and os.path.exists(found):
            target_env_file = Path(found)
        else:
            proj_root_env = Path(__file__).resolve().parent.parent / ".env"
            if proj_root_env.exists():
                target_env_file = proj_root_env
            elif (Path.cwd() / ".env").exists():
                target_env_file = Path.cwd() / ".env"

    if target_env_file and target_env_file.is_file():
        if dotenv_values is not None:
            with contextlib.suppress(OSError, ValueError):
                raw_dict = dotenv_values(str(target_env_file))
                for k, v in raw_dict.items():
                    if k and v is not None:
                        env_data[k] = str(v).strip().strip("'\"")
        else:
            with (
                contextlib.suppress(OSError, ValueError),
                open(target_env_file, "r", encoding="utf-8") as f,
            ):
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k:
                        env_data[k] = v

        if load_dotenv is not None:
            with contextlib.suppress(OSError, ValueError):
                load_dotenv(dotenv_path=str(target_env_file), override=bool(env_path))

    # Ensure os.environ is populated from env_data if os.environ is empty or unset
    for k, v in env_data.items():
        if v and not os.environ.get(k, "").strip():
            os.environ[k] = v

    def resolve_val(key: str, default: str = "") -> str:
        env_val = os.getenv(key, "").strip()
        if env_val:
            return env_val
        file_val = env_data.get(key, "").strip()
        if file_val:
            return file_val
        return default

    return {
        "OPENAI_API_KEY": resolve_val("OPENAI_API_KEY"),
        "OPENAI_BASE_URL": resolve_val("OPENAI_BASE_URL"),
        "OPENAI_MODEL": resolve_val("OPENAI_MODEL", "gemma-4-31b-it"),
    }


def encode_image_to_base64(
    image_input: str | Path | np.ndarray | Image.Image,
) -> str:
    """Encode an image file path, numpy array, or PIL Image into a base64 string.

    Args:
        image_input: File path, numpy array (BGR), or PIL Image.

    Returns:
        Base64-encoded string of PNG image bytes.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {image_input}")
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Cannot encode empty image array.")
        success, buffer = cv2.imencode(".png", image_input)
        if not success:
            raise ValueError("Failed to encode image array to PNG.")
        return base64.b64encode(buffer).decode("utf-8")
    elif isinstance(image_input, Image.Image):
        buf = BytesIO()
        image_input.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")


def _clean_html_table_string(raw_table: Any) -> str:
    """Clean and normalize HTML table output, converting list/dict structures if needed."""
    if raw_table is None:
        return ""

    if isinstance(raw_table, list):
        if not raw_table:
            return "<table></table>"
        if isinstance(raw_table[0], dict):
            headers = list(raw_table[0].keys())
            th_cells = "".join(f"<th>{h}</th>" for h in headers)
            rows = []
            for item in raw_table:
                if isinstance(item, dict):
                    td_cells = "".join(f"<td>{item.get(h, '')}</td>" for h in headers)
                    rows.append(f"<tr>{td_cells}</tr>")
            return f"<table><thead><tr>{th_cells}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        elif isinstance(raw_table[0], (list, tuple)):
            header = raw_table[0]
            th_cells = "".join(f"<th>{h}</th>" for h in header)
            rows = []
            for row in raw_table[1:]:
                td_cells = "".join(f"<td>{cell}</td>" for cell in row)
                rows.append(f"<tr>{td_cells}</tr>")
            return f"<table><thead><tr>{th_cells}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    s = str(raw_table).strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    lower = s.lower()
    start_idx = lower.find("<table")
    end_idx = lower.rfind("</table>")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        s = s[start_idx : end_idx + len("</table>")]

    return s.strip()


def parse_llm_json_response(content: str) -> dict[str, Any]:
    """Parse and normalize JSON response string from LLM.

    Args:
        content: Raw text content from LLM response.

    Returns:
        Dictionary containing normalized fields: item_name, name, category, style_code, size_spec_table.
    """
    import re

    text = content.strip()

    # Strip markdown code blocks if present (```json ... ``` or ``` ...)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    data: Any = None
    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        # Fallback 1: clean trailing commas
        cleaned = re.sub(r",\s*([\]}])", r"\1", text)
        try:
            data = json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            # Fallback 2: find outermost { and }
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                sub = cleaned[first_brace : last_brace + 1]
                try:
                    data = json.loads(sub, strict=False)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Could not parse valid JSON from LLM response:\n{content}"
                    )
            else:
                raise ValueError(
                    f"Could not parse valid JSON from LLM response:\n{content}"
                )

    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object (dict), got {type(data).__name__}")

    # Extract & normalize field names
    item_name = (
        data.get("item_name")
        or data.get("name")
        or data.get("item name")
        or data.get("title")
        or data.get("garment_title")
        or ""
    )
    category = (
        data.get("category")
        or data.get("catogory")
        or data.get("garment_category")
        or ""
    )
    style_code = (
        data.get("style_code")
        or data.get("style code")
        or data.get("style")
        or data.get("style_no")
        or data.get("style_num")
        or ""
    )
    raw_size_table = (
        data.get("size_spec_table")
        or data.get("size spec table")
        or data.get("table")
        or data.get("size_spec")
        or data.get("html_table")
        or ""
    )
    size_spec_table = _clean_html_table_string(raw_size_table)

    result = dict(data)
    str_item_name = str(item_name).strip()
    str_category = str(category).strip()
    str_style_code = str(style_code).strip()

    result["item_name"] = str_item_name
    result["name"] = str_item_name
    result["item name"] = str_item_name
    result["category"] = str_category
    result["catogory"] = str_category
    result["style_code"] = str_style_code
    result["style code"] = str_style_code
    result["size_spec_table"] = size_spec_table
    result["size spec table"] = size_spec_table

    return result


def call_openai_vision(
    image_b64: str,
    prompt: str = DEFAULT_LLM_PROMPT,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gemma-4-31b-it",
) -> str:
    """Call OpenAI Chat Completions API with vision input.

    Args:
        image_b64: Base64-encoded image string or data URL.
        prompt: Prompt instructing LLM on desired output.
        api_key: OpenAI API key.
        base_url: Optional OpenAI base URL.
        model: OpenAI model name.

    Returns:
        Response message content string from the LLM.
    """
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url and base_url.strip():
        client_kwargs["base_url"] = base_url.strip()

    client = OpenAI(**client_kwargs)

    if image_b64.startswith("data:"):
        image_url = image_b64
    else:
        # Detect mime type from base64 magic bytes
        if image_b64.startswith("/9j/"):
            mime = "image/jpeg"
        elif image_b64.startswith("iVBORw0KGgo"):
            mime = "image/png"
        elif image_b64.startswith("UklGR"):
            mime = "image/webp"
        elif image_b64.startswith("R0lGOD"):
            mime = "image/gif"
        else:
            mime = "image/png"
        image_url = f"data:{mime};base64,{image_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        err_str = str(e).lower()
        if (
            "response_format" in err_str
            or "json_object" in err_str
            or "400" in err_str
            or "bad request" in err_str
        ):
            # Fallback for models/endpoints that don't support response_format
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
        else:
            raise

    choice = response.choices[0]
    return choice.message.content or ""


def extract_with_llm(
    image_input: str | Path | np.ndarray | Image.Image,
    output_dir: str | Path | None = None,
    config: Any = None,
    env_file: str | Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract garment metadata and HTML size spec table from reconstructed table image using OpenAI LLM.

    Args:
        image_input: File path, numpy array, or PIL Image of reconstructed table.
        output_dir: Output directory where result.json will be saved.
        config: LLMConfig or ExtractorConfig instance.
        env_file: Optional path to .env file.

    Returns:
        tuple of (parsed_json_dict, saved_result_json_path). If skipped or failed, returns (None, None).
    """
    # Load .env variables
    env_vars = load_llm_env(env_file)

    # Check if config provides LLM settings
    llm_cfg = getattr(config, "llm", config)

    enabled = getattr(llm_cfg, "enabled", True)
    if not enabled:
        return None, None

    if getattr(llm_cfg, "api_key", None) is not None:
        api_key = (llm_cfg.api_key or "").strip()
    else:
        api_key = (env_vars.get("OPENAI_API_KEY", "") or "").strip()

    if not api_key:
        print("[*] Skipping LLM extraction: OPENAI_API_KEY not configured in .env")
        return None, None

    if getattr(llm_cfg, "base_url", None) is not None:
        base_url = (llm_cfg.base_url or "").strip() or None
    else:
        base_url = (env_vars.get("OPENAI_BASE_URL", "") or "").strip() or None

    if getattr(llm_cfg, "model", None) is not None and str(llm_cfg.model).strip():
        model = str(llm_cfg.model).strip()
    else:
        model = (env_vars.get("OPENAI_MODEL", "") or "").strip() or "gpt-4o"
    prompt = getattr(llm_cfg, "prompt", None) or DEFAULT_LLM_PROMPT
    result_filename = (
        getattr(llm_cfg, "result_filename", "result.json") or "result.json"
    )

    try:
        print(f"[*] Calling LLM ({model}) for size specification extraction...")
        image_b64 = encode_image_to_base64(image_input)
        raw_content = call_openai_vision(
            image_b64=image_b64,
            prompt=prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        parsed_json = parse_llm_json_response(raw_content)

        saved_path = None
        if output_dir is not None:
            out_dir_path = Path(output_dir)
            out_dir_path.mkdir(parents=True, exist_ok=True)
            saved_path_obj = out_dir_path / result_filename
            with open(saved_path_obj, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            saved_path = str(saved_path_obj.resolve())
            print(f"[+] LLM extraction result saved to: {saved_path}")

        return parsed_json, saved_path
    except Exception as e:
        print(f"[!] Warning: LLM extraction failed: {e}", file=sys.stderr)
        return None, None
