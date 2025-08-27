# utils/configuration.py
import json
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, ValidationError
from utils.validation import verifier_fichier_lecture

class ConfigModel(BaseModel):
    url: str
    format: str = "json"
    output: Optional[str] = None
    db_type: Optional[str] = None
    db_host: Optional[str] = None
    db_port: Optional[str] = None
    db_name: Optional[str] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None

def lire_config(fichier: str) -> ConfigModel:
    """Lit un fichier de configuration (JSON, texte ou XML) et le valide avec Pydantic."""
    extensions = ['.json', '.txt', '.xml']
    fichier_valide = verifier_fichier_lecture(fichier, extensions)
    ext = fichier_valide.suffix.lower()
    if ext == '.json':
        with open(fichier_valide, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif ext == '.txt':
        data = {}
        with open(fichier_valide, 'r', encoding='utf-8') as f:
            for ligne in f:
                if '=' in ligne:
                    cle, val = ligne.strip().split('=', 1)
                    data[cle.strip()] = val.strip()
    elif ext == '.xml':
        import xml.etree.ElementTree as ET
        tree = ET.parse(fichier_valide)
        root = tree.getroot()
        data = {elem.tag: elem.text for elem in root}
    else:
        raise ValueError("Format de configuration non supporté.")
    try:
        return ConfigModel(**data)
    except ValidationError as e:
        raise ValueError(f"Erreur de validation de la configuration : {str(e)}")

def fusionner_config_et_args(config: ConfigModel, args: argparse.Namespace, force_cmd: bool, force_config: bool) -> ConfigModel:
    """Fusionne configuration et arguments de ligne de commande selon les options."""
    if force_cmd:
        return ConfigModel(
            url=args.url,
            format=args.format,
            output=args.output,
            db_type=args.db_type,
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password
        )
    elif force_config:
        return config
    else:
        return ConfigModel(
            url=args.url if args.url else config.url,
            format=args.format if args.format != 'json' else config.format,
            output=args.output if args.output else config.output,
            db_type=args.db_type if args.db_type else config.db_type,
            db_host=args.db_host if args.db_host else config.db_host,
            db_port=args.db_port if args.db_port else config.db_port,
            db_name=args.db_name if args.db_name else config.db_name,
            db_user=args.db_user if args.db_user else config.db_user,
            db_password=args.db_password if args.db_password else config.db_password
        )
