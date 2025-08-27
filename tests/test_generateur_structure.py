#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test pytest pour le script de génération de structures de projet.
- Vérifie la création des dossiers et fichiers attendus.
- Détecte l’OS (Linux/Unix/Windows) pour d’éventuelles adaptations.
"""

import platform
import tempfile
import shutil
from pathlib import Path

# Import du script à tester (à adapter selon votre structure)
from extracteur_web.aspirer_site import creer_structure, creer_fichier_vide  # Exemple, à remplacer par l'import du vrai script de génération

import pytest

@pytest.fixture
def temp_project():
    """Crée un dossier temporaire pour le projet de test."""
    temp_dir = tempfile.TemporaryDirectory()
    base_path = Path(temp_dir.name)
    project_name = "mon_projet_test"
    project_path = base_path / project_name
    yield base_path, project_name, project_path
    temp_dir.cleanup()

def test_creer_structure(temp_project):
    base_path, project_name, project_path = temp_project
    creer_structure(base_path, project_name)

    # Vérification des fichiers et dossiers attendus
    assert (project_path / "pyproject.toml").exists()
    assert (project_path / "README.md").exists()
    assert (project_path / "requirements.txt").exists()
    assert (project_path / ".gitignore").exists()

    package = project_path / project_name
    assert package.exists()
    assert (package / "__init__.py").exists()
    assert (package / "main.py").exists()

    for subpackage in ["formats", "utils"]:
        sub_dir = package / subpackage
        assert sub_dir.exists()
        assert (sub_dir / "__init__.py").exists()

    tests = project_path / "tests"
    assert tests.exists()
    assert (tests / "__init__.py").exists()

def test_creer_fichier_vide(temp_project):
    base_path, _, _ = temp_project
    test_file = base_path / "test_vide.txt"
    creer_fichier_vide(test_file)
    assert test_file.exists()

def test_detection_os():
    os_type = platform.system().lower()
    print(f"OS détecté : {os_type}")
    if os_type in ["linux", "darwin"]:
        print("Test sur Linux/Unix")
    elif os_type == "windows":
        print("Test sur Windows")
    else:
        print("Autre OS")
    # Ce bloc sert à illustrer la détection, vous pouvez l’utiliser pour adapter le comportement des tests
