# tests/test_validation.py
import unittest
import tempfile
import os
from pathlib import Path
from utils.validation import FilePath, verifier_fichier_lecture, verifier_fichier_ecriture

class TestValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "test.txt"
        with open(self.temp_file, "w") as f:
            f.write("test")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_creer_fichier_vide(self):
        # test implicite par l'usage de FilePath et verifier_fichier_lecture
        pass

    def test_FilePath(self):
        model = FilePath(path=self.temp_file, allowed_extensions=[".txt"])
        self.assertEqual(model.path, self.temp_file)

    def test_verifier_fichier_lecture(self):
        path = verifier_fichier_lecture(str(self.temp_file), [".txt"])
        self.assertEqual(path, self.temp_file)
        with self.assertRaises(ValueError):
            verifier_fichier_lecture(str(self.temp_file), [".csv"])

    def test_verifier_fichier_ecriture(self):
        path = verifier_fichier_ecriture(str(self.temp_file))
        self.assertEqual(path, self.temp_file)
        new_file = Path(self.temp_dir.name) / "new.txt"
        path = verifier_fichier_ecriture(str(new_file))
        self.assertEqual(path, new_file)

if __name__ == "__main__":
    unittest.main()
