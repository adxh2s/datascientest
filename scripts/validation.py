import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, validator

class FilePath(BaseModel):
    path: Path
    allowed_extensions: Optional[List[str]] = None

    @validator('path')
    def validate_path(cls, v: Path):
        if not v.exists():
            raise ValueError(f"Le fichier {v} n'existe pas.")
        if not v.is_file():
            raise ValueError(f"{v} n'est pas un fichier.")
        return v

    @validator('path')
    def validate_readable(cls, v: Path):
        if not os.access(v, os.R_OK):
            raise ValueError(f"Le fichier {v} n'est pas lisible.")
        if v.stat().st_size == 0:
            raise ValueError(f"Le fichier {v} est vide.")
        return v

    @validator('path')
    def validate_extension(cls, v: Path, values):
        allowed_extensions = values.get('allowed_extensions')
        if allowed_extensions is not None:
            if not any(str(v).endswith(ext) for ext in allowed_extensions):
                raise ValueError(f"Extension non autorisée. Extensions acceptées : {allowed_extensions}")
        return v

def verifier_fichier_lecture(fichier: str, extensions_acceptees: List[str]) -> Path:
    """Valide l'existence, la lisibilité, la non-vide et l'extension d'un fichier."""
    return FilePath(path=Path(fichier), allowed_extensions=extensions_acceptees).path

def verifier_fichier_ecriture(fichier: str) -> Path:
    """Valide que le fichier de sortie est modifiable ou créable."""
    p = Path(fichier)
    if p.exists():
        if not p.is_file():
            raise ValueError(f"{p} existe mais n'est pas un fichier.")
        if not os.access(p, os.W_OK):
            raise ValueError(f"Le fichier {p} existe mais n'est pas modifiable.")
    else:
        dossier = p.parent or Path.cwd()
        if not os.access(dossier, os.W_OK):
            raise ValueError(f"Le dossier {dossier} n'est pas accessible en écriture.")
    return p
