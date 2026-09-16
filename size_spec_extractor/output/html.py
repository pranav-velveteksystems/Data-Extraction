"""HTML table generation and export."""

import html
import os

from ..reconstruction.schema import ExtractionResult


def generate_html_table(result: ExtractionResult) -> str:
    """Generate modern, responsive HTML page with size spec table and metadata."""
    doc = result.document
    tbl = doc.size_spec_table

    headers_html = "".join([f"<th>{html.escape(col)}</th>" for col in tbl.columns])

    rows_html = []
    for row in tbl.rows:
        cells_html = [f"<td class='spec-col'>{html.escape(row.specification)}</td>"]
        for val in row.values:
            if val is None:
                cells_html.append("<td class='cell-null'>—</td>")
            elif isinstance(val, list):
                # Vertically stacked values
                stacked_inner = "".join(
                    [
                        f"<div class='stacked-val'>{html.escape(str(v))}</div>"
                        for v in val
                    ]
                )
                cells_html.append(f"<td class='cell-stacked'>{stacked_inner}</td>")
            else:
                cells_html.append(f"<td class='cell-val'>{html.escape(str(val))}</td>")
        rows_html.append(f"<tr>{''.join(cells_html)}</tr>")

    table_body = "\n".join(rows_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Size Spec Table - {html.escape(doc.style_code or "Extracted")}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #f8fafc;
      color: #1e293b;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 8px;
      box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
      padding: 24px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 16px;
      margin-bottom: 20px;
    }}
    .meta-group {{
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
    }}
    .meta-item {{
      background: #f1f5f9;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 14px;
    }}
    .meta-label {{
      font-weight: 600;
      color: #64748b;
    }}
    .meta-value {{
      font-weight: bold;
      color: #0f172a;
    }}
    .table-container {{
      overflow-x: auto;
      margin-top: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: center;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #cbd5e1;
      padding: 10px 8px;
      vertical-align: middle;
    }}
    th {{
      background-color: #1e293b;
      color: #f8fafc;
      font-weight: 600;
      letter-spacing: 0.5px;
    }}
    th:first-child {{
      text-align: left;
    }}
    .spec-col {{
      text-align: left;
      font-weight: 600;
      background-color: #f8fafc;
      width: 25%;
    }}
    .cell-val {{
      font-weight: bold;
      color: #0369a1;
      background-color: #f0f9ff;
    }}
    .cell-null {{
      color: #94a3b8;
      background-color: #ffffff;
    }}
    .cell-stacked {{
      background-color: #fefce8;
      padding: 4px;
    }}
    .stacked-val {{
      font-weight: bold;
      color: #854d0e;
      padding: 2px 0;
    }}
    .stacked-val:not(:last-child) {{
      border-bottom: 1px dashed #fde047;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>Garment Size Specification Sheet</h2>
      <div class="meta-group">
        <div class="meta-item"><span class="meta-label">Category:</span> <span class="meta-value">{html.escape(doc.category or "N/A")}</span></div>
        <div class="meta-item"><span class="meta-label">Style Code:</span> <span class="meta-value">{html.escape(doc.style_code or "N/A")}</span></div>
        <div class="meta-item"><span class="meta-label">Unit:</span> <span class="meta-value">{html.escape(tbl.unit or "inch")}</span></div>
      </div>
    </div>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Specification</th>
            {headers_html}
          </tr>
        </thead>
        <tbody>
          {table_body}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
    return html_content


def save_html(result: ExtractionResult, file_path: str) -> str:
    """Save extraction result as HTML page."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    content = generate_html_table(result)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path
