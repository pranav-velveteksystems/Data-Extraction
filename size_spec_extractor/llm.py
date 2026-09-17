"""LLM integration using direct HTTP API calls for size specification table extraction."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import urllib.error
import urllib.request
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

DEFAULT_REMARKS_PROMPT = """You are an expert system analyzing garment specification sheets.
Analyze the provided image of the garment specification sheet (back side / notes section).
Extract all remarks, notes, sewing/workmanship instructions, packaging requirements, washing instructions, or special comments.

Return ONLY a valid JSON object with the following field:
{
  "remarks": "<full extracted remarks and notes text, or empty string if no remarks found>"
}

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
        # First priority: Always use value from .env file if defined
        if key in env_data and env_data[key] is not None and env_data[key].strip():
            return env_data[key].strip()
        # Fallback: os.environ only if not defined in .env file
        env_val = os.getenv(key, "").strip()
        if env_val:
            return env_val
        return default

    # Sync os.environ with .env so any library or subprocess gets exact .env values
    for k in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]:
        if k in env_data and env_data[k].strip():
            os.environ[k] = env_data[k].strip()

    return {
        "OPENAI_API_KEY": resolve_val("OPENAI_API_KEY", ""),
        "OPENAI_BASE_URL": resolve_val("OPENAI_BASE_URL", ""),
        "OPENAI_MODEL": resolve_val("OPENAI_MODEL", "gpt-4o"),
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

    remarks = (
        data.get("remarks")
        or data.get("notes")
        or data.get("comments")
        or data.get("comment")
        or ""
    )
    result["remarks"] = str(remarks).strip()

    return result


def get_chat_completions_endpoint(base_url: str | None = None) -> str:
    """Build the Chat Completions HTTP endpoint from base_url."""
    if not base_url or not base_url.strip():
        return "https://api.openai.com/v1/chat/completions"

    clean_url = base_url.strip().rstrip("/")
    if clean_url.endswith("/chat/completions"):
        return clean_url
    if clean_url.endswith("/v1"):
        return f"{clean_url}/chat/completions"
    return f"{clean_url}/v1/chat/completions"


def make_http_chat_request(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Send HTTP POST request to Chat Completions endpoint."""
    body_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body_bytes,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_bytes = resp.read()
        return json.loads(content_bytes.decode("utf-8"))


def call_openai_vision(
    image_b64: str,
    prompt: str = DEFAULT_LLM_PROMPT,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-4o",
    timeout: float = 120.0,
) -> str:
    """Call OpenAI-compatible Chat Completions HTTP API directly with vision input.

    Args:
        image_b64: Base64-encoded image string or data URL.
        prompt: Prompt instructing LLM on desired output.
        api_key: API key for authorization.
        base_url: Optional base URL (defaults to https://api.openai.com/v1).
        model: Model name to use.
        timeout: Request timeout in seconds.

    Returns:
        Response message content string from the LLM.
    """
    endpoint = get_chat_completions_endpoint(base_url)

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

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "SizeSpecExtractor/1.0",
    }
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }

    try:
        data = make_http_chat_request(endpoint, headers, payload, timeout=timeout)
    except urllib.error.HTTPError as err:
        err_body = ""
        with contextlib.suppress(Exception):
            err_body = err.read().decode("utf-8", errors="replace")

        err_lower = f"{err} {err_body}".lower()
        if (
            "response_format" in err_lower
            or "json_object" in err_lower
            or err.code == 400
        ) and "response_format" in payload:
            # Fallback for models/endpoints that don't support response_format
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format", None)
            try:
                data = make_http_chat_request(
                    endpoint, headers, fallback_payload, timeout=timeout
                )
            except urllib.error.HTTPError as fallback_err:
                fallback_body = ""
                with contextlib.suppress(Exception):
                    fallback_body = fallback_err.read().decode(
                        "utf-8", errors="replace"
                    )
                raise RuntimeError(
                    f"HTTP {fallback_err.code} from LLM API ({endpoint}): {fallback_body or fallback_err.reason}"
                ) from fallback_err
        else:
            raise RuntimeError(
                f"HTTP {err.code} from LLM API ({endpoint}): {err_body or err.reason}"
            ) from err
    except urllib.error.URLError as err:
        raise RuntimeError(
            f"Network error connecting to LLM API ({endpoint}): {err.reason}"
        ) from err

    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        raise ValueError(
            f"Invalid response from LLM API: missing 'choices' list in {data}"
        )

    first_choice = choices[0]
    message = first_choice.get("message", {})
    if isinstance(message, dict) and "content" in message:
        return message.get("content", "") or ""
    if "text" in first_choice:
        return first_choice.get("text", "") or ""
    return ""


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


def extract_remarks_with_llm(
    image_input: str | Path | np.ndarray | Image.Image,
    config: Any = None,
    env_file: str | Path | None = None,
    prompt: str = DEFAULT_REMARKS_PROMPT,
) -> str:
    """Extract remarks and notes directly from the back image of a spec sheet using LLM.

    Args:
        image_input: File path, numpy array, or PIL Image of back image.
        config: Optional LLMConfig or ExtractorConfig instance.
        env_file: Optional path to .env file.
        prompt: Optional prompt to use for remarks extraction.

    Returns:
        Extracted remarks string (empty string if failed or not found).
    """
    env_vars = load_llm_env(env_file)
    llm_cfg = getattr(config, "llm", config)

    enabled = getattr(llm_cfg, "enabled", True)
    if not enabled:
        return ""

    if getattr(llm_cfg, "api_key", None) is not None:
        api_key = (llm_cfg.api_key or "").strip()
    else:
        api_key = (env_vars.get("OPENAI_API_KEY", "") or "").strip()

    if not api_key:
        print("[*] Skipping remarks extraction: OPENAI_API_KEY not configured in .env")
        return ""

    if getattr(llm_cfg, "base_url", None) is not None:
        base_url = (llm_cfg.base_url or "").strip() or None
    else:
        base_url = (env_vars.get("OPENAI_BASE_URL", "") or "").strip() or None

    if getattr(llm_cfg, "model", None) is not None and str(llm_cfg.model).strip():
        model = str(llm_cfg.model).strip()
    else:
        model = (env_vars.get("OPENAI_MODEL", "") or "").strip() or "gpt-4o"

    try:
        print(f"[*] Calling LLM ({model}) for remarks extraction on back image...")
        image_b64 = encode_image_to_base64(image_input)
        raw_content = call_openai_vision(
            image_b64=image_b64,
            prompt=prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        parsed = parse_llm_json_response(raw_content)
        remarks = parsed.get("remarks", "")
        if not remarks and isinstance(parsed, dict):
            for alt_k in [
                "notes",
                "comment",
                "comments",
                "instruction",
                "instructions",
            ]:
                if parsed.get(alt_k):
                    remarks = parsed[alt_k]
                    break
        return str(remarks or "").strip()
    except Exception as e:
        print(f"[!] Warning: Remarks extraction failed: {e}", file=sys.stderr)
        return ""
