#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASPIRATION ET TRANSFORMATION DE CONTENU WEB
- Utilise Pydantic et pathlib pour la validation et la gestion des fichiers.
- Les fonctions utilitaires sont centralisées dans utils/validation et utils/configuration.
- Les fonctions de format sont centralisées dans formats/formats.
"""

import json
import argparse
import requests
from pathlib import Path
from utils.validation import verifier_fichier_ecriture
from utils.configuration import ConfigModel, lire_config, fusionner_config_et_args
from formats.formats import format_json, format_texte, format_html, format_db

def aspirer_site(url: str) -> str:
    """Récupère le contenu HTML brut d'une page web."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise Exception(f"Erreur lors de l'aspiration : {str(e)}")

def choisir_format(
    url: str,
    html: str,
    format_sortie: str = 'json',
    db_type: Optional[str] = None,
    db_host: Optional[str] = None,
    db_port: Optional[str] = None,
    db_name: Optional[str] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None
) -> str:
    """Sélectionne le format de sortie à appliquer."""
    if format_sortie == 'json':
        return format_json(url, html)
    elif format_sortie == 'texte':
        return format_texte(html)
    elif format_sortie == 'html':
        return format_html(html)
    elif format_sortie == 'db':
        if not all([db_type, db_host, db_port, db_name, db_user, db_password]):
            raise ValueError("Tous les paramètres de base de données doivent être fournis.")
        return format_db(url, html, db_type, db_host, db_port, db_name, db_user, db_password)
    else:
        raise ValueError("Format de sortie non reconnu. Choisissez 'json', 'texte', 'html' ou 'db'.")

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
        '--db-type',
        help="Type de base de données (postgresql, mysql, oracle)"
    )
    parser.add_argument(
        '--db-host',
        help="Hôte de la base de données"
    )
    parser.add_argument(
        '--db-port',
        help="Port de the base de données"
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
    parser.add_argument(
        '--force-cmd',
        action='store_true',
        help="Force l'utilisation exclusive de la ligne de commande"
    )
    parser.add_argument(
        '--force-config',
        action='store_true',
        help="Force l'utilisation exclusive du fichier de configuration"
    )
    args = parser.parse_args()

    if args.force_cmd and args.force_config:
        print("Erreur : --force-cmd et --force-config ne peuvent pas être utilisés ensemble.")
        exit(1)

    config = ConfigModel(url="", format="json")  # Valeur par défaut
    if args.config and not args.force_cmd:
        try:
            config = lire_config(args.config)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier de configuration : {str(e)}")
            exit(1)

    parametres = fusionner_config_et_args(config, args, args.force_cmd, args.force_config)
    if not parametres.url:
        print("URL manquante : veuillez la fournir en argument ou dans le fichier de configuration.")
        exit(1)

    try:
        html = aspirer_site(parametres.url)
        resultat = choisir_format(
            parametres.url,
            html,
            parametres.format,
            parametres.db_type,
            parametres.db_host,
            parametres.db_port,
            parametres.db_name,
            parametres.db_user,
            parametres.db_password
        )
        print(resultat)
        if parametres.output and parametres.format != 'db':
            verifier_fichier_ecriture(parametres.output)
            with open(parametres.output, 'w', encoding='utf-8') as f:
                f.write(resultat)
    except Exception as e:
        erreur = json.dumps({'erreur': str(e)}, ensure_ascii=False)
        print(erreur)
        if parametres.output and parametres.format != 'db':
            try:
                verifier_fichier_ecriture(parametres.output)
                with open(parametres.output, 'w', encoding='utf-8') as f:
                    f.write(erreur)
            except Exception as e2:
                print(f"Erreur lors de l'écriture du fichier de sortie : {str(e2)}")
