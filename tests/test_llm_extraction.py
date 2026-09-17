"""Tests for LLM size specification extraction using OpenAI SDK."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from size_spec_extractor.config import ExtractorConfig, LLMConfig
from size_spec_extractor.extractor import extract_table_segments
from size_spec_extractor.llm import (
    call_openai_vision,
    encode_image_to_base64,
    extract_with_llm,
    get_chat_completions_endpoint,
    load_llm_env,
    parse_llm_json_response,
)
from size_spec_extractor.main import main, parse_args


class TestLLMExtraction(unittest.TestCase):
    def setUp(self):
        self._orig_env = dict(os.environ)
        self.test_dir = tempfile.mkdtemp()
        self.sample_img_path = os.path.join(self.test_dir, "sample.png")
        # Create a small dummy table image
        dummy_img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        cv2.putText(
            dummy_img,
            "Category: Dress",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        cv2.rectangle(dummy_img, (10, 50), (290, 190), (0, 0, 0), 2)
        cv2.imwrite(self.sample_img_path, dummy_img)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_llm_env_from_custom_file(self):
        env_file = os.path.join(self.test_dir, ".env.test")
        with open(env_file, "w", encoding="utf-8") as f:
            f.write("OPENAI_API_KEY=test_key_123\n")
            f.write("OPENAI_BASE_URL=https://custom.api.com/v1\n")
            f.write("OPENAI_MODEL=ag/gemini-3.7-flash-low(low)-mini\n")

        env = load_llm_env(env_path=env_file)
        self.assertEqual(env["OPENAI_API_KEY"], "test_key_123")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://custom.api.com/v1")
        self.assertEqual(env["OPENAI_MODEL"], "ag/gemini-3.7-flash-low(low)-mini")

    def test_encode_image_to_base64(self):
        # File path
        b64_file = encode_image_to_base64(self.sample_img_path)
        self.assertIsInstance(b64_file, str)
        self.assertGreater(len(b64_file), 10)

        # Numpy array
        arr = np.ones((50, 50, 3), dtype=np.uint8) * 128
        b64_arr = encode_image_to_base64(arr)
        self.assertIsInstance(b64_arr, str)
        self.assertGreater(len(b64_arr), 10)

        # PIL Image
        pil_img = Image.new("RGB", (30, 30), color="blue")
        b64_pil = encode_image_to_base64(pil_img)
        self.assertIsInstance(b64_pil, str)
        self.assertGreater(len(b64_pil), 10)

        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            encode_image_to_base64(os.path.join(self.test_dir, "nonexistent.png"))

    def test_parse_llm_json_response_clean(self):
        raw = json.dumps(
            {
                "item_name": "Summer Frock",
                "category": "Kids Wear",
                "style_code": "SF-2024",
                "size_spec_table": "<table><thead><tr><th>Spec</th><th>S</th><th>M</th></tr></thead><tbody><tr><td>Chest</td><td>18</td><td>20</td></tr></tbody></table>",
            }
        )
        parsed = parse_llm_json_response(raw)
        self.assertEqual(parsed["item_name"], "Summer Frock")
        self.assertEqual(parsed["name"], "Summer Frock")
        self.assertEqual(parsed["category"], "Kids Wear")
        self.assertEqual(parsed["style_code"], "SF-2024")
        self.assertIn("<table>", parsed["size_spec_table"])
        self.assertIn("</table>", parsed["size_spec_table"])

    def test_parse_llm_json_response_markdown_wrapped(self):
        raw = """```json
{
  "name": "Men Polo T-Shirt",
  "catogory": "T-Shirts",
  "style code": "POLO-99",
  "table": "<table><tr><td>Length</td><td>28</td></tr></table>"
}
```"""
        parsed = parse_llm_json_response(raw)
        self.assertEqual(parsed["item_name"], "Men Polo T-Shirt")
        self.assertEqual(parsed["name"], "Men Polo T-Shirt")
        self.assertEqual(parsed["category"], "T-Shirts")
        self.assertEqual(parsed["style_code"], "POLO-99")
        self.assertEqual(
            parsed["size_spec_table"],
            "<table><tr><td>Length</td><td>28</td></tr></table>",
        )

    def test_skip_llm_when_api_key_empty(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            cfg = LLMConfig(enabled=True, api_key="")
            out_dir = os.path.join(self.test_dir, "out_skip")
            result, path = extract_with_llm(
                self.sample_img_path, output_dir=out_dir, config=cfg
            )
            self.assertIsNone(result)
            self.assertIsNone(path)
            self.assertFalse(os.path.exists(os.path.join(out_dir, "result.json")))

    def test_skip_llm_when_disabled(self):
        cfg = LLMConfig(enabled=False, api_key="sk-test")
        out_dir = os.path.join(self.test_dir, "out_disabled")
        result, path = extract_with_llm(
            self.sample_img_path, output_dir=out_dir, config=cfg
        )
        self.assertIsNone(result)
        self.assertIsNone(path)

    @patch("size_spec_extractor.llm.call_openai_vision")
    def test_extract_with_llm_mocked_success(self, mock_call):
        mock_response = json.dumps(
            {
                "item_name": "Denim Jacket",
                "category": "Outerwear",
                "style_code": "DJ-8801",
                "size_spec_table": "<table><thead><tr><th>Size</th><th>Chest</th></tr></thead><tbody><tr><td>L</td><td>42</td></tr></tbody></table>",
            }
        )
        mock_call.return_value = mock_response

        out_dir = os.path.join(self.test_dir, "out_success")
        cfg = LLMConfig(
            enabled=True,
            api_key="sk-mock-key",
            model="ag/gemini-3.7-flash-low(low)",
            result_filename="result.json",
        )

        parsed, path = extract_with_llm(
            self.sample_img_path, output_dir=out_dir, config=cfg
        )

        self.assertIsNotNone(parsed)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.basename(path), "result.json")

        with open(path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["item_name"], "Denim Jacket")
        self.assertEqual(saved_data["name"], "Denim Jacket")
        self.assertEqual(saved_data["category"], "Outerwear")
        self.assertEqual(saved_data["style_code"], "DJ-8801")
        self.assertIn("<table>", saved_data["size_spec_table"])
        self.assertIn("42", saved_data["size_spec_table"])

    @patch("size_spec_extractor.llm.call_openai_vision")
    def test_extract_table_segments_integration_with_llm(self, mock_call):
        mock_response = json.dumps(
            {
                "item_name": "Sample Dress",
                "category": "Dresses",
                "style_code": "SD-100",
                "size_spec_table": "<table><tr><td>Length</td><td>36</td></tr></table>",
            }
        )
        mock_call.return_value = mock_response

        out_dir = os.path.join(self.test_dir, "segments_llm_out")
        config = ExtractorConfig()
        config.llm.enabled = True
        config.llm.api_key = "sk-mock-key-for-test"

        result = extract_table_segments(
            image_input=self.sample_img_path,
            output_dir=out_dir,
            config=config,
        )

        self.assertIsNotNone(result.llm_result)
        self.assertIsNotNone(result.llm_result_path)
        self.assertEqual(result["llm_result"]["style_code"], "SD-100")
        self.assertTrue(os.path.exists(result.llm_result_path))

        # Check result.json inside output folder
        res_json_file = os.path.join(out_dir, "result.json")
        self.assertTrue(os.path.exists(res_json_file))
        with open(res_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["item_name"], "Sample Dress")

    def test_cli_parser_llm_options(self):
        args = parse_args(
            [
                self.sample_img_path,
                "--no-llm",
                "--llm-model",
                "ag/gemini-3.7-flash-low(low)-mini",
                "--llm-base-url",
                "https://api.openai.com/v1",
                "--llm-api-key",
                "test-cli-key",
                "--result-filename",
                "custom_result.json",
            ]
        )
        self.assertFalse(args.llm_enabled)
        self.assertEqual(args.llm_model, "ag/gemini-3.7-flash-low(low)-mini")
        self.assertEqual(args.llm_base_url, "https://api.openai.com/v1")
        self.assertEqual(args.llm_api_key, "test-cli-key")
        self.assertEqual(args.result_filename, "custom_result.json")

    @patch("size_spec_extractor.llm.call_openai_vision")
    def test_cli_main_with_llm(self, mock_call):
        mock_response = json.dumps(
            {
                "item_name": "CLI Garment",
                "category": "Shirts",
                "style_code": "SH-001",
                "size_spec_table": "<table><tr><td>Chest</td><td>21</td></tr></table>",
            }
        )
        mock_call.return_value = mock_response

        out_dir = os.path.join(self.test_dir, "cli_test_out")
        exit_code = main(
            [
                self.sample_img_path,
                "-o",
                out_dir,
                "--llm-api-key",
                "sk-cli-test-key",
                "--llm-model",
                "ag/gemini-3.7-flash-low(low)",
            ]
        )
        self.assertEqual(exit_code, 0)
        res_path = os.path.join(out_dir, "result.json")
        self.assertTrue(os.path.exists(res_path))
        with open(res_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d["item_name"], "CLI Garment")
        self.assertEqual(d["style_code"], "SH-001")

    @patch("size_spec_extractor.llm.call_openai_vision")
    def test_cli_main_with_no_llm_flag(self, mock_call):
        out_dir = os.path.join(self.test_dir, "cli_no_llm_out")
        exit_code = main(
            [
                self.sample_img_path,
                "-o",
                out_dir,
                "--no-llm",
                "--llm-api-key",
                "sk-cli-test-key",
            ]
        )
        self.assertEqual(exit_code, 0)
        mock_call.assert_not_called()
        res_path = os.path.join(out_dir, "result.json")
        self.assertFalse(os.path.exists(res_path))

    @patch("size_spec_extractor.llm.call_openai_vision")
    def test_llm_api_failure_handled_gracefully(self, mock_call):
        mock_call.side_effect = RuntimeError("OpenAI rate limit exceeded")
        out_dir = os.path.join(self.test_dir, "out_err")
        cfg = LLMConfig(enabled=True, api_key="sk-test-key")

        res, path = extract_with_llm(
            self.sample_img_path, output_dir=out_dir, config=cfg
        )
        self.assertIsNone(res)
        self.assertIsNone(path)

    def test_parse_llm_json_response_with_conversational_text(self):
        text = 'Sure! Here is the extracted JSON for you:\n{"item_name": "Silk Scarf", "category": "Accessories", "style_code": "SC-10", "size_spec_table": "<table><tr><td>Length</td><td>180</td></tr></table>"}\nHope this helps!'
        parsed = parse_llm_json_response(text)
        self.assertEqual(parsed["item_name"], "Silk Scarf")
        self.assertEqual(parsed["style_code"], "SC-10")

    def test_parse_llm_json_response_invalid(self):
        with self.assertRaises(ValueError):
            parse_llm_json_response("This is not JSON at all.")
        with self.assertRaises(TypeError):
            parse_llm_json_response("[1, 2, 3]")

    @patch("size_spec_extractor.llm.make_http_chat_request")
    def test_call_openai_vision_response_format_fallback(self, mock_http_request):
        import io
        import urllib.error

        mock_success_response = {
            "choices": [{"message": {"content": '{"item_name": "Fallback Item"}'}}]
        }

        def side_effect(endpoint, headers, payload, timeout=120.0):
            if "response_format" in payload:
                raise urllib.error.HTTPError(
                    url=endpoint,
                    code=400,
                    msg="response_format is not supported on this model",
                    hdrs={},
                    fp=io.BytesIO(b'{"error": "response_format is not supported"}'),
                )
            return mock_success_response

        mock_http_request.side_effect = side_effect

        content = call_openai_vision(
            image_b64="dGVzdA==",
            api_key="sk-test",
            base_url="https://proxy.example.com",
            model="custom-model",
        )
        self.assertIn("Fallback Item", content)
        self.assertEqual(mock_http_request.call_count, 2)

    def test_parse_llm_json_trailing_commas_and_control_chars(self):
        raw = """{
  "item name": "Cotton Cardigan",
  "catogory": "Sweaters",
  "style code": "CC-901",
  "size spec table": "<table>
    <thead>
      <tr><th>Spec</th><th>S</th><th>M</th></tr>
    </thead>
    <tbody>
      <tr><td>Chest</td><td>20</td><td>22</td></tr>
    </tbody>
  </table>",
}"""
        parsed = parse_llm_json_response(raw)
        self.assertEqual(parsed["item_name"], "Cotton Cardigan")
        self.assertEqual(parsed["name"], "Cotton Cardigan")
        self.assertEqual(parsed["item name"], "Cotton Cardigan")
        self.assertEqual(parsed["category"], "Sweaters")
        self.assertEqual(parsed["catogory"], "Sweaters")
        self.assertEqual(parsed["style_code"], "CC-901")
        self.assertEqual(parsed["style code"], "CC-901")
        self.assertIn("<table>", parsed["size_spec_table"])
        self.assertIn("Chest", parsed["size_spec_table"])
        self.assertEqual(parsed["size_spec_table"], parsed["size spec table"])

    def test_parse_llm_json_html_codeblock_inside_table(self):
        raw = json.dumps(
            {
                "item_name": "Leather Belt",
                "category": "Accessories",
                "style_code": "LB-01",
                "size_spec_table": "```html\n<table><tr><td>Size</td><td>32</td></tr></table>\n```",
            }
        )
        parsed = parse_llm_json_response(raw)
        self.assertEqual(
            parsed["size_spec_table"],
            "<table><tr><td>Size</td><td>32</td></tr></table>",
        )
        self.assertFalse(parsed["size_spec_table"].startswith("```"))

    def test_parse_llm_json_list_of_rows_to_html(self):
        raw = json.dumps(
            {
                "item_name": "Denim Shorts",
                "category": "Shorts",
                "style_code": "DS-22",
                "size_spec_table": [
                    {"Measurement": "Waist", "28": "29", "30": "31"},
                    {"Measurement": "Hip", "28": "37", "30": "39"},
                ],
            }
        )
        parsed = parse_llm_json_response(raw)
        self.assertIn("<table><thead><tr>", parsed["size_spec_table"])
        self.assertIn("<th>Measurement</th>", parsed["size_spec_table"])
        self.assertIn("<td>Waist</td>", parsed["size_spec_table"])

    @patch("size_spec_extractor.llm.make_http_chat_request")
    def test_call_openai_vision_detects_jpeg_mime(self, mock_http_request):
        mock_http_request.return_value = {
            "choices": [{"message": {"content": '{"item_name": "Test"}'}}]
        }

        # Base64 starting with /9j/ is JPEG
        call_openai_vision(
            image_b64="/9j/4AAQSkZJRg==",
            api_key="sk-test",
        )
        call_args = mock_http_request.call_args
        payload = call_args[0][2]
        messages = payload["messages"]
        image_url = messages[0]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))

    def test_get_chat_completions_endpoint(self):
        self.assertEqual(
            get_chat_completions_endpoint(),
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(
            get_chat_completions_endpoint("https://cloud.olakrutrim.com/v1"),
            "https://cloud.olakrutrim.com/v1/chat/completions",
        )
        self.assertEqual(
            get_chat_completions_endpoint(
                "https://cloud.olakrutrim.com/v1/chat/completions"
            ),
            "https://cloud.olakrutrim.com/v1/chat/completions",
        )
        self.assertEqual(
            get_chat_completions_endpoint("https://api.custom.com"),
            "https://api.custom.com/v1/chat/completions",
        )

    @patch("size_spec_extractor.llm.call_openai_vision")
    def test_extractor_extract_end_to_end_final_step(self, mock_call):
        from size_spec_extractor.extractor import SizeSpecExtractor

        mock_call.return_value = json.dumps(
            {
                "item_name": "End To End Jacket",
                "category": "Jackets",
                "style_code": "JK-99",
                "size_spec_table": "<table><tr><td>Length</td><td>30</td></tr></table>",
            }
        )
        out_dir = os.path.join(self.test_dir, "e2e_out")
        extractor = SizeSpecExtractor()
        extractor.config.llm.api_key = "sk-mock-key"
        res = extractor.extract(self.sample_img_path, output_dir=out_dir)

        self.assertIsNotNone(res.llm_result)
        self.assertEqual(res.llm_result["item_name"], "End To End Jacket")
        self.assertIsNotNone(res.llm_result_path)
        self.assertTrue(os.path.exists(res.llm_result_path))

        # Test subscripting on ExtractionResult
        self.assertEqual(
            res["item_name"] if "item_name" in res else res["llm_result"]["item_name"],
            "End To End Jacket",
        )
        self.assertEqual(res["result_json"]["style_code"], "JK-99")
        self.assertEqual(res["result.json"], res.llm_result_path)


if __name__ == "__main__":
    unittest.main()
