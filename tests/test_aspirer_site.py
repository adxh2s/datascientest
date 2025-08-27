# tests/test_aspirer_site.py
import unittest
from unittest.mock import patch, MagicMock
import argparse

from extracteur_web.aspirer_site import choisir_format, aspirer_site

class TestAspirerSite(unittest.TestCase):
    def test_choisir_format_json(self):
        result = choisir_format("url", "<html>test</html>", "json")
        self.assertIn("url", result)
        self.assertIn("contenu", result)

    @patch("extracteur_web.aspirer_site.requests.get")
    def test_aspirer_site(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html>test</html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = aspirer_site("https://example.com")
        self.assertEqual(result, "<html>test</html>")

if __name__ == "__main__":
    unittest.main()
