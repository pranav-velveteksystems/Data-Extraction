"""Tests for web app and /api/extract REST API."""

import io
import os
import unittest
from unittest.mock import MagicMock, patch

from app import app, is_allowed_file


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    def test_allowed_extensions(self):
        self.assertTrue(is_allowed_file("sample.png"))
        self.assertTrue(is_allowed_file("sample.jpg"))
        self.assertTrue(is_allowed_file("sample.JPEG"))
        self.assertTrue(is_allowed_file("sample.webp"))
        self.assertFalse(is_allowed_file("sample.pdf"))
        self.assertFalse(is_allowed_file("sample.txt"))
        self.assertFalse(is_allowed_file("noextension"))

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Garment Size Spec Extractor", response.data)
        self.assertIn(b"size_spec_table", response.data)

    def test_health_route(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("openai_configured", data)

    def test_extract_no_file(self):
        response = self.client.post("/api/extract")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("No image file provided", data["error"])

    def test_extract_invalid_extension(self):
        data = {"image": (io.BytesIO(b"dummy pdf content"), "doc.pdf")}
        response = self.client.post(
            "/api/extract", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn("Unsupported file extension", json_data["error"])

    @patch("app.load_llm_env")
    def test_extract_missing_api_key(self, mock_load_env):
        mock_load_env.return_value = {"OPENAI_API_KEY": ""}
        data = {"image": (io.BytesIO(b"fake image data"), "sample.png")}
        response = self.client.post(
            "/api/extract", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn("OPENAI_API_KEY is not configured", json_data["error"])

    @patch("app.extract_with_llm")
    @patch("app.load_llm_env")
    @patch("app.extract_table_segments")
    def test_extract_success_and_cleanup(
        self, mock_extract, mock_load_env, mock_extract_llm
    ):
        mock_load_env.return_value = {
            "OPENAI_API_KEY": "sk-test-mock-key",
            "OPENAI_MODEL": "gpt-4o",
            "OPENAI_BASE_URL": "",
        }

        mock_result = MagicMock()
        mock_result.reconstructed_table_path = None
        mock_result.table_image_path = None
        mock_result.reconstructed_table_image = "dummy_img"
        mock_extract_llm.return_value = (
            {
                "item_name": "Test Blouse",
                "category": "Tops",
                "style_code": "ST-999",
                "size_spec_table": "<table><thead><tr><th>Spec</th><th>S</th><th>M</th></tr></thead><tbody><tr><td>Chest</td><td>36</td><td>38</td></tr></tbody></table>",
            },
            "result.json",
        )

        temp_dir_captured = []

        def fake_extract(*args, **kwargs):
            out_dir = kwargs.get("output_dir")
            temp_dir_captured.append(out_dir)
            # Simulate creating intermediate images in temp_dir
            if out_dir and os.path.exists(out_dir):
                table_path = os.path.join(out_dir, "table.png")
                recon_path = os.path.join(out_dir, "reconstructed_table.png")
                with open(table_path, "wb") as f:
                    f.write(b"fake table png")
                with open(recon_path, "wb") as f:
                    f.write(b"fake recon table png")
                mock_result.reconstructed_table_path = recon_path
            return mock_result

        mock_extract.side_effect = fake_extract

        # Provide a small valid png
        img_bytes = io.BytesIO(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        data = {"image": (img_bytes, "test_designsheet.png")}

        response = self.client.post(
            "/api/extract", data=data, content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertEqual(res_json["item_name"], "Test Blouse")
        self.assertEqual(res_json["category"], "Tops")
        self.assertEqual(res_json["style_code"], "ST-999")
        self.assertIn("<table>", res_json["size_spec_table"])

        # Check that the temporary output directory and all intermediate files were deleted!
        self.assertEqual(len(temp_dir_captured), 1)
        used_temp_dir = temp_dir_captured[0]
        self.assertFalse(
            os.path.exists(used_temp_dir),
            f"Temporary directory {used_temp_dir} was NOT cleaned up!",
        )

    @patch("app.extract_with_llm")
    @patch("app.extract_remarks_with_llm")
    @patch("app.load_llm_env")
    @patch("app.extract_table_segments")
    def test_extract_with_front_and_back_images(
        self, mock_extract, mock_load_env, mock_extract_remarks, mock_extract_llm
    ):
        mock_load_env.return_value = {
            "OPENAI_API_KEY": "sk-test-mock-key",
            "OPENAI_MODEL": "gpt-4o",
            "OPENAI_BASE_URL": "",
        }

        mock_result = MagicMock()
        mock_result.reconstructed_table_path = "recon.png"
        mock_extract.return_value = mock_result
        mock_extract_llm.return_value = (
            {
                "item_name": "Two Piece Suit",
                "category": "Suits",
                "style_code": "ST-555",
                "size_spec_table": "<table><tr><td>Length</td><td>40</td></tr></table>",
            },
            "result.json",
        )
        mock_extract_remarks.return_value = "Dry clean only. 100% pure wool."

        front_bytes = io.BytesIO(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        back_bytes = io.BytesIO(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        data = {
            "front_image": (front_bytes, "front.png"),
            "back_image": (back_bytes, "back.png"),
        }

        response = self.client.post(
            "/api/extract", data=data, content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertEqual(res_json["item_name"], "Two Piece Suit")
        self.assertEqual(res_json["remarks"], "Dry clean only. 100% pure wool.")
        mock_extract_llm.assert_called_once()
        mock_extract_remarks.assert_called_once()


if __name__ == "__main__":
    unittest.main()
