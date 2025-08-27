import os
import argparse
import subprocess
from pathlib import Path

def creer_fichier_vide(chemin):
    with open(chemin, 'w', encoding='utf-8'):
        pass

def creer_structure(base_path, project_name):
    base = Path(base_path) / project_name
    base.mkdir(parents=True, exist_ok=True)

    # Fichiers racine
    (base / "pyproject.toml").touch()
    (base / "README.md").touch()
    (base / "requirements.txt").touch()
    (base / ".gitignore").touch()

    # Package principal
    package = base / project_name
    package.mkdir(exist_ok=True)
    creer_fichier_vide(package / "__init__.py")
    
    # Fichier principal
    with open(package / "main.py", "w") as f:
        f.write("print('Hello World!')\n")

    # Sous-packages
    for subpackage in ["formats", "utils"]:
        sub_dir = package / subpackage
        sub_dir.mkdir(exist_ok=True)
        creer_fichier_vide(sub_dir / "__init__.py")

    # Répertoire tests
    tests = base / "tests"
    tests.mkdir(exist_ok=True)
    creer_fichier_vide(tests / "__init__.py")

    return base

def initialiser_et_publier(base_path, project_name, visibility="public", description=""):
    try:
        # Initialisation Git
        subprocess.run(["git", "init"], cwd=base_path, check=True)
        
        # Création du dépôt GitHub
        create_cmd = [
            "gh", "repo", "create", 
            project_name,
            "--"+visibility,
            "--description", description,
            "--source", str(base_path),
            "--push"
        ]
        subprocess.run(create_cmd, check=True)
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la publication sur GitHub: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de projet Python avec publication GitHub")
    parser.add_argument("--base_path", default=".", help="Chemin de base pour le projet")
    parser.add_argument("--project_name", required=True, help="Nom du projet")
    parser.add_argument("--visibility", choices=["public", "private"], default="public", 
                      help="Visibilité du dépôt GitHub")
    parser.add_argument("--description", default="Mon nouveau projet Python", 
                      help="Description du dépôt GitHub")
    
    args = parser.parse_args()

    # Création de la structure
    project_path = creer_structure(args.base_path, args.project_name)
    print(f"✅ Structure créée dans {project_path}")

    # Publication sur GitHub
    print("\n🚀 Publication sur GitHub...")
    if initialiser_et_publier(project_path, args.project_name, args.visibility, args.description):
        print(f"\n🎉 Projet publié avec succès sur https://github.com/{subprocess.getoutput('gh api user --jq .login')}/{args.project_name}")
    else:
        print("❌ Échec de la publication")
