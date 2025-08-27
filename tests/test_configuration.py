# tests/test_configuration.py
import unittest
import tempfile
import json
from pathlib import Path
from utils.configuration import ConfigModel, lire_config

class TestConfiguration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file_json = Path(self.temp_dir.name) / "config.json"
        self.config_data = {
            "url": "https://example.com",
            "format": "json",
            "output": "resultat.json",
        }
        with open(self.temp_file_json, "w") as f:
            json.dump(self.config_data, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ConfigModel(self):
        model = ConfigModel(**self.config_data)
        self.assertEqual(model.url, "https://example.com")
        self.assertEqual(model.format, "json")

    def test_lire_config_json(self):
        config = lire_config(str(self.temp_file_json))
        self.assertEqual(config.url, "https://example.com")

if __name__ == "__main__":
    unittest.main()
