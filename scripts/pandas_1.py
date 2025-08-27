#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
##############################################################################
# APPLICATION : Aspiration et transformation de contenu web
# DESCRIPTION : Aspire le contenu d’une page web, le transforme dans le format
#               choisi (JSON, texte brut ou HTML) et écrit le résultat dans un
#               fichier, à l’écran ou dans une base de données.
#               Prend en charge un fichier de configuration (JSON, texte ou XML)
#               pour définir toutes les options.
#
# PARAMETRES :
#   - url (obligatoire, peut être dans le fichier de configuration)
#   - --format (optionnel) : format de sortie ('json', 'texte', 'html', 'db')
#                           valeur par défaut : 'json'
#   - -o, --output (optionnel) : chemin du fichier de sortie (ignoré si 'db')
#   - --db-connection (optionnel) : chaîne de connexion à la base de données
#   - --db-name (optionnel) : nom de la base de données
#   - --db-user (optionnel) : code utilisateur de la base de données
#   - --db-password (optionnel) : mot de passe de la base de données
#   - -c, --config (optionnel) : chemin du fichier de configuration (JSON, texte ou XML)
#
# EXEMPLE DE LIGNE DE COMMANDE :
#   python aspirer_site.py https://example.com --format texte -o resultat.txt
#   python aspirer_site.py https://example.com --format db --config config.json
#   python aspirer_site.py --config config.json
#
# VERSION : 1.2
# AUTEUR : [Votre nom ou pseudonyme]
# DATE DE CREATION : 2025-06-12
#
# LIBELLE LIBRE :
#   Ce script est conçu pour l’extraction et la transformation de données web
#   dans différents formats, facilitant l’analyse et l’intégration dans des
#   workflows de traitement automatisé[1][2].
#
# DATE DE MODIFICATION : [Date de la dernière modification]
##############################################################################
"""

import json
import argparse
import requests
from bs4 import BeautifulSoup
import os

def lire_config_json(fichier):
    """Lit un fichier de configuration JSON."""
    with open(fichier, 'r', encoding='utf-8') as f:
        return json.load(f)

def lire_config_texte(fichier):
    """Lit un fichier de configuration texte (clé=valeur)."""
    config = {}
    with open(fichier, 'r', encoding='utf-8') as f:
        for ligne in f:
            if '=' in ligne:
                cle, val = ligne.strip().split('=', 1)
                config[cle.strip()] = val.strip()
    return config

def lire_config_xml(fichier):
    """Lit un fichier de configuration XML (simple exemple)."""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(fichier)
        root = tree.getroot()
        config = {}
        for elem in root:
            config[elem.tag] = elem.text
        return config
    except ImportError:
        raise Exception("Module xml.etree.ElementTree non disponible.")
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture du fichier XML : {str(e)}")

def lire_config(fichier):
    """Lit un fichier de configuration selon son extension."""
    ext = os.path.splitext(fichier)[1].lower()
    if ext == '.json':
        return lire_config_json(fichier)
    elif ext == '.txt':
        return lire_config_texte(fichier)
    elif ext == '.xml':
        return lire_config_xml(fichier)
    else:
        raise ValueError("Format de configuration non supporté (JSON, texte ou XML uniquement).")

def aspirer_site(url):
    """Récupère le contenu HTML brut d'une page web."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise Exception(f"Erreur lors de l'aspiration : {str(e)}")

def format_json(url, html):
    """Formate le contenu dans un objet JSON."""
    soup = BeautifulSoup(html, 'html.parser')
    return json.dumps({'url': url, 'contenu': soup.prettify()}, ensure_ascii=False)

def format_texte(html):
    """Extrait le texte brut du HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

def format_html(html):
    """Retourne le contenu HTML brut."""
    return html

def format_db(url, html, connection, db_name, db_user, db_password):
    """Enregistre le contenu dans une base de données PostgreSQL."""
    try:
        import psycopg2
        conn_str = f"{connection}/{db_name}?user={db_user}&password={db_password}"
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_content (
                id SERIAL PRIMARY KEY,
                url TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        soup = BeautifulSoup(html, 'html.parser')
        cursor.execute(
            "INSERT INTO web_content (url, content) VALUES (%s, %s)",
            (url, soup.prettify())
        )
        conn.commit()
        conn.close()
        return "Contenu enregistré dans la base de données."
    except Exception as e:
        raise Exception(f"Erreur lors de l'insertion en base : {str(e)}")

def choisir_format(url, html, format_sortie='json', db_connection=None, db_name=None, db_user=None, db_password=None):
    """Sélectionne le format de sortie à appliquer."""
    if format_sortie == 'json':
        return format_json(url, html)
    elif format_sortie == 'texte':
        return format_texte(html)
    elif format_sortie == 'html':
        return format_html(html)
    elif format_sortie == 'db':
        if not all([db_connection, db_name, db_user, db_password]):
            raise ValueError("Tous les paramètres de base de données doivent être fournis.")
        return format_db(url, html, db_connection, db_name, db_user, db_password)
    else:
        raise ValueError("Format de sortie non reconnu. Choisissez 'json', 'texte', 'html' ou 'db'.")

def fusionner_config_et_args(config, args):
    """Fusionne configuration et arguments de ligne de commande."""
    parametres = {}
    # Priorité à la ligne de commande si l'argument est présent
    parametres['url'] = args.url if args.url else config.get('url')
    parametres['format'] = args.format if args.format != 'json' else config.get('format', 'json')
    parametres['output'] = args.output if args.output else config.get('output')
    parametres['db_connection'] = args.db_connection if args.db_connection else config.get('db_connection')
    parametres['db_name'] = args.db_name if args.db_name else config.get('db_name')
    parametres['db_user'] = args.db_user if args.db_user else config.get('db_user')
    parametres['db_password'] = args.db_password if args.db_password else config.get('db_password')
    return parametres

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aspire le contenu d'une page web et retourne le résultat dans le format choisi (JSON par défaut)."
    )
    parser.add_argument('url', nargs='?', help="URL de la page à aspirer (optionnel si dans le fichier de configuration)")
    parser.add_argument(
        '--format',
        choices=['json', 'texte', 'html', 'db'],
        default='json',
        help="Format de sortie : json, texte, html ou db (défaut: json)"
    )
    parser.add_argument(
        '-o', '--output',
        help="Chemin du fichier de sortie où écrire le résultat (ignoré si 'db')"
    )
    parser.add_argument(
        '--db-connection',
        help="Chaîne de connexion à la base de données"
    )
    parser.add_argument(
        '--db-name',
        help="Nom de la base de données"
    )
    parser.add_argument(
        '--db-user',
        help="Code utilisateur de la base de données"
    )
    parser.add_argument(
        '--db-password',
        help="Mot de passe de la base de données"
    )
    parser.add_argument(
        '-c', '--config',
        help="Chemin du fichier de configuration (JSON, texte ou XML)"
    )
    args = parser.parse_args()

    config = {}
    if args.config:
        try:
            config = lire_config(args.config)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier de configuration : {str(e)}")
            exit(1)

    parametres = fusionner_config_et_args(config, args)
    if not parametres.get('url'):
        print("URL manquante : veuillez la fournir en argument ou dans le fichier de configuration.")
        exit(1)

    try:
        html = aspirer_site(parametres['url'])
        resultat = choisir_format(
            parametres['url'],
            html,
            parametres.get('format', 'json'),
            parametres.get('db_connection'),
            parametres.get('db_name'),
            parametres.get('db_user'),
            parametres.get('db_password')
        )
        print(resultat)
        if parametres.get('output') and parametres.get('format') != 'db':
            with open(parametres['output'], 'w', encoding='utf-8') as f:
                f.write(resultat)
    except Exception as e:
        erreur = json.dumps({'erreur': str(e)}, ensure_ascii=False)
        print(erreur)
        if parametres.get('output') and parametres.get('format') != 'db':
            with open(parametres['output'], 'w', encoding='utf-8') as f:
                f.write(erreur)
