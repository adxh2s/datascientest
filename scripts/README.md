# Extracteur Web

Un outil Python pour aspirer le contenu d’un site web et le transformer dans différents formats de sortie (JSON, texte, HTML, base de données : PostgreSQL, MySQL, Oracle).  
Le projet est conçu pour la modularité et la maintenabilité, avec une organisation claire des utilitaires et des formats de sortie[1][3].

---

## Fonctionnalités

- **Aspiration de contenu web** : récupération du contenu HTML d’une page via son URL.
- **Formats de sortie flexibles** : JSON, texte brut, HTML, ou enregistrement direct en base de données (PostgreSQL, MySQL, Oracle).
- **Gestion de configuration** : lecture depuis un fichier (JSON, texte, XML) ou argument en ligne de commande.
- **Validation robuste** : utilisation de Pydantic et pathlib pour la validation des fichiers et des arguments.
- **Structure modulaire** : utilitaires et formats de sortie centralisés dans des modules dédiés.

---

## Structure du projet

extracteur_web/
│
├── pyproject.toml # Définition du projet, dépendances, etc.
├── README.md # Documentation du projet
│
├── extracteur_web/ # Package principal (Python)
│ ├── init.py # Fichier vide pour que le dossier soit un module
│ ├── aspirer_site.py # Script principal
│ │
│ ├── formats/ # Sous-package pour les formats de sortie
│ │ ├── init.py
│ │ └── formats.py
│ │
│ └── utils/ # Sous-package pour les utilitaires
│ ├── init.py
│ ├── validation.py
│ └── configuration.py
│
├── tests/ # Répertoire pour les tests unitaires
│ ├── init.py
│ ├── test_validation.py
│ └── test_configuration.py
│
└── requirements.txt # Dépendances (optionnel, pour compatibilité)

## Installation

1. **Cloner le projet**

git clone https://github.com/votre_compte/extracteur_web.git
cd extracteur_web

2. **Installer les dépendances**
pip install -r requirements.txt

ou
pip install -e .

---

## Utilisation

python -m extracteur_web.aspirer_site https://example.com --format json -o resultat.json


Voir l’aide complète avec :
python -m extracteur_web.aspirer_site --help


---

## Dépendances

- Python ≥ 3.8
- requests
- beautifulsoup4
- pydantic
- psycopg2-binary (pour PostgreSQL)
- mysql-connector-python (pour MySQL)
- cx-Oracle (pour Oracle)

---

## Licence

MIT

---

## Contribuer

Les contributions sont les bienvenues !  
Ouvrez une issue ou une pull request pour proposer des améliorations.