# Garment Size Spec & Remarks Extraction API Documentation

REST API documentation for the Garment Size Specification and Remarks Extraction service.

---

## 1. Overview & Base URL

The API provides endpoints to analyze garment specification sheets (front size measurement table and optional back remarks sheet), extract structured garment metadata and HTML measurement tables using computer vision and multimodal LLMs, and return a clean JSON payload.

- **Default Server Base URL**: `http://localhost:5000`
- **Content Types**:
  - `GET /api/health` &rarr; `application/json`
  - `POST /api/extract` &rarr; `multipart/form-data` (Response: `application/json`)

### Environment Configuration (`.env`)
The extraction pipeline reads its credentials directly from `.env`:
```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # Or custom OpenAI-compatible endpoint
OPENAI_MODEL=gpt-4o                        # Vision-capable model name
```

---

## 2. API Endpoints

### Endpoint 1: Health Check & Configuration Status

#### `GET /api/health`

Checks if the server is healthy and whether the LLM API credentials are configured.

#### Request
- **Method**: `GET`
- **Path**: `/api/health`
- **Headers**: None required
- **Query Parameters**: None

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "openai_configured": true,
  "model": "gpt-4o"
}
```

#### Field Descriptions
| Field | Type | Description |
|---|---|---|
| `status` | string | Health status of the API server (`"healthy"`). |
| `openai_configured` | boolean | `true` if `OPENAI_API_KEY` is present and non-empty in `.env`, `false` otherwise. |
| `model` | string | Active LLM model configured in `.env`. |

#### Example `curl`
```bash
curl -X GET http://localhost:5000/api/health
```

---

### Endpoint 2: Extract Size Specification & Remarks

#### `POST /api/extract`

Extracts garment measurements, size table, and back-sheet remarks from design sheets.
- **Front Image**: Processed through OpenCV table detection, cell segmentation, value centering, and LLM table extraction.
- **Back Image** *(Optional)*: Dispatched directly to the LLM for notes and remarks extraction.
- **Parallel Dispatch**: When both images are supplied, both LLM requests execute **at the exact same time in parallel** using background threads.
- **Zero Disk Residue**: All uploaded files, cropped table segments, reconstructed tables, and intermediate outputs are deleted immediately after the response is generated.

#### Request
- **Method**: `POST`
- **Path**: `/api/extract`
- **Content-Type**: `multipart/form-data`

#### Form-Data Fields
| Parameter | Type | Required | Description |
|---|---|---|---|
| `front_image` | File (Binary) | **Yes** | Front design sheet containing the measurement table. *(Aliases: `front`, `image`, `file`)* |
| `back_image` | File (Binary) | **No** | Back design sheet containing remarks, sewing notes, and special instructions. *(Alias: `back`)* |

> **Supported Image Formats**: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`

---

#### Response: `200 OK` (Successful Extraction)

```json
{
  "item_name": "LADIES WOVEN BLOUSE",
  "name": "LADIES WOVEN BLOUSE",
  "category": "WOVEN TOPS",
  "style_code": "ST-1049-A",
  "size_spec_table": "<table><thead><tr><th>Measurement Description</th><th>S</th><th>M</th><th>L</th><th>XL</th></tr></thead><tbody><tr><td>Total Length from HPS</td><td>24.5</td><td>25</td><td>25.5</td><td>26</td></tr><tr><td>Chest Width (1\" below armhole)</td><td>19</td><td>20</td><td>21</td><td>22.5</td></tr><tr><td>Waist Width</td><td>17.5</td><td>18.5</td><td>19.5</td><td>21</td></tr><tr><td>Sweep / Bottom Opening</td><td>20</td><td>21</td><td>22</td><td>23.5</td></tr><tr><td>Sleeve Length from Shoulder</td><td>23</td><td>23.5</td><td>24</td><td>24.5</td></tr></tbody></table>",
  "remarks": "1. 100% Cotton Poplin pre-shrunk.\n2. Double needle stitching on all seams with 1/4\" spacing.\n3. Spare button attached to inner care label.\n4. Machine wash cold with like colors, tumble dry low."
}
```

#### Field Descriptions
| Field | Type | Description |
|---|---|---|
| `item_name` | string | Extracted garment name / product title. |
| `name` | string | Alias for `item_name`. |
| `category` | string | Garment category (e.g. `"WOVEN TOPS"`, `"DENIM PANTS"`). |
| `style_code` | string | Product / style code number (e.g. `"ST-1049-A"`). |
| `size_spec_table` | string | Full HTML `<table>...</table>` string with `<thead>` (measurement labels and sizes) and `<tbody>` (numerical measurement values). Numerical fractions are converted to standard decimals (e.g., `2 1/2` &rarr; `2.5`). |
| `remarks` | string | Extracted instructions, sewing details, washing instructions, and comments from the back image. If no back image was provided or no remarks exist, this is an empty string `""`. |

---

#### Error Responses

##### `400 Bad Request` — Missing File
```json
{
  "error": "No image file provided in request (use 'front_image', 'front', or 'image')."
}
```

##### `400 Bad Request` — Unsupported Extension
```json
{
  "error": "Unsupported file extension. Allowed: bmp, jpeg, jpg, png, tif, tiff, webp"
}
```

##### `400 Bad Request` — Missing API Key in `.env`
```json
{
  "error": "OPENAI_API_KEY is not configured in .env file. Please add your OpenAI API key to .env to enable extraction."
}
```

##### `500 Internal Server Error` — Extraction or LLM Failure
```json
{
  "error": "Extraction pipeline completed, but LLM failed to return valid JSON data. Please verify your OpenAI credentials and image format."
}
```

---

## 3. Code Examples

### 3.1 `curl` Examples

#### With Both Front and Back Images (Parallel Execution)
```bash
curl -X POST http://localhost:5000/api/extract \
  -F "front_image=@/path/to/designsheet_front.png" \
  -F "back_image=@/path/to/designsheet_back.png"
```

#### With Front Image Only
```bash
curl -X POST http://localhost:5000/api/extract \
  -F "front_image=@/path/to/designsheet_front.png"
```

---

### 3.2 Python (`requests`)

```python
import requests

url = "http://localhost:5000/api/extract"

# Both front and back images
files = {
    "front_image": open("front.png", "rb"),
    "back_image": open("back.png", "rb"),
}

response = requests.post(url, files=files)
data = response.json()

print(f"Item Name: {data.get('item_name')}")
print(f"Category: {data.get('category')}")
print(f"Style Code: {data.get('style_code')}")
print(f"Remarks: {data.get('remarks')}")
print(f"HTML Table:\n{data.get('size_spec_table')}")
```

---

### 3.3 JavaScript (`fetch`)

```javascript
const formData = new FormData();
formData.append("front_image", frontFileInput.files[0]);
if (backFileInput.files[0]) {
  formData.append("back_image", backFileInput.files[0]);
}

const response = await fetch("http://localhost:5000/api/extract", {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log("Extracted Data:", result);

// Render HTML table directly
document.getElementById("table-container").innerHTML = result.size_spec_table;
```

---

## 4. Running the Server

Start the Flask server on port `5000` (or configure via `--port` / `--host`):

```bash
# Default port 5000
python3 app.py

# Custom port and host
python3 app.py --host 0.0.0.0 --port 8080
```
