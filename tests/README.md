# README – Dossier de tests automatisés

Ce dossier contient l’ensemble des tests automatisés pour le projet **extracteur_web**.  
Les tests garantissent la robustesse, la maintenabilité et la compatibilité multiplateforme du code (Linux, Unix, Windows)[1][2].

---

## Structure des tests

tests/
├── init.py # Fichier vide pour que le dossier soit un package Python
├── test_validation.py # Tests pour utils/validation.py
├── test_configuration.py # Tests pour utils/configuration.py
├── test_formats.py # Tests pour formats/formats.py
├── test_aspirer_site.py # Tests pour aspirer_site.py
└── test_generateur_structure.py # Tests pour le script de génération de structure

text

---

## Types de tests

- **Tests unitaires** : Chaque fichier vérifie le comportement d’un module ou d’une fonction spécifique.
- **Mock des dépendances externes** : Les appels réseau et base de données sont simulés pour garantir l’isolation des tests.
- **Compatibilité multiplateforme** : Les tests sont écrits pour fonctionner sur Linux, Unix et Windows[1].

---

## Prérequis

- **Python** (version 3.8 ou supérieure recommandée)
- **pytest** (pour exécuter les tests de façon avancée)
- **Modules du projet** : Les tests nécessitent que les modules du projet soient accessibles (installation en mode développement recommandée : `pip install -e .`)

---

## Installation

1. **Installez pytest** (si ce n’est pas déjà fait) :

pip install pytest

text

2. **Installez le projet en mode développement** (si les modules ne sont pas déjà accessibles) :

pip install -e .

text

---

## Exécution des tests

### 1. **Exécuter tous les tests**

Depuis la racine du projet :

pytest tests/

text

### 2. **Exécuter un test spécifique**

Par exemple, pour tester uniquement le générateur de structure :

pytest tests/test_generateur_structure.py

text

### 3. **Options avancées**

- **Afficher la sortie détaillée** :

pytest tests/ -v

text

- **Afficher les print dans la console** :

pytest tests/ -s

text

---

## Détail des fichiers de test

- **test_validation.py**  
- Teste les fonctions de validation de fichiers et de chemins.
- **test_configuration.py**  
- Teste la lecture et la validation des fichiers de configuration.
- **test_formats.py**  
- Teste les fonctions de transformation du contenu (JSON, texte, HTML, base de données).
- **test_aspirer_site.py**  
- Teste le script principal (aspiration web, choix du format).
- **test_generateur_structure.py**  
- Teste le script de génération de structure, avec détection de l’OS pour d’éventuelles adaptations.

---

## Bonnes pratiques

- **Ajoutez un test pour chaque nouvelle fonctionnalité.**
- **Utilisez des mocks pour simuler les dépendances externes (web, base de données).**
- **Exécutez les tests régulièrement pour détecter rapidement les régressions.**

---

## Documentation complémentaire

Pour plus d’informations sur les tests unitaires en Python et pytest, consultez la documentation officielle :  
https://docs.pytest.org/en/latest/

---

Ce README fournit toutes les instructions nécessaires pour installer, configurer et exécuter les tests automatisés du projet, en toute compatibilité multiplateforme[1][2].