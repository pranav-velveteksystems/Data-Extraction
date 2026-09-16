"""Output package for JSON and HTML generation."""

from .html import generate_html_table, save_html
from .json import save_json, to_json

__all__ = ["generate_html_table", "save_html", "save_json", "to_json"]
