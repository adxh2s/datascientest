# tests/test_formats.py
import unittest
from unittest.mock import patch
from formats.formats import format_json, format_texte, format_html

class TestFormats(unittest.TestCase):
    def test_format_json(self):
        url = "https://example.com"
        html = "<html><body>test</body></html>"
        result = format_json(url, html)
        self.assertIn("url", result)
        self.assertIn("contenu", result)

    def test_format_texte(self):
        html = "<html><body>test</body></html>"
        result = format_texte(html)
        self.assertEqual(result.strip(), "test")

    def test_format_html(self):
        html = "<html><body>test</body></html>"
        result = format_html(html)
        self.assertEqual(result, html)

    # Test sur format_db nécessite un mock des connexions DB (exemple simplifié)
    @patch("formats.formats.psycopg2.connect")
    def test_format_db_postgresql(self, mock_connect):
        from formats.formats import format_db_postgresql
        mock_cursor = mock_connect.return_value.cursor.return_value
        result = format_db_postgresql("url", "<html>test</html>", "host", "port", "db", "user", "pass")
        self.assertEqual(result, "Contenu enregistré dans la base de données PostgreSQL.")

    # Idem pour MySQL, Oracle si besoin

if __name__ == "__main__":
    unittest.main()
