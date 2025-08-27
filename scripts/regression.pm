def _extract_feature_names(obj, feature_names=None):
    """
    Essaie d'extraire les noms des variables à partir de différentes méthodes
    d'un objet scikit-learn (modèle ou encodeur), selon la priorité recommandée.
    """
    # 1. Tentative via get_feature_names_out
    if hasattr(obj, 'get_feature_names_out'):
        try:
            # Certains objets requièrent un argument d'entrée, d'autres non.
            if feature_names is not None:
                names = obj.get_feature_names_out(feature_names)
            else:
                names = obj.get_feature_names_out()
            return names
        except Exception:
            pass
    # 2. Tentative via feature_names_in_
    if hasattr(obj, 'feature_names_in_'):
        return obj.feature_names_in_
    # 3. Tentative via argument passé explicitement
    if feature_names is not None:
        return feature_names
    # 4. Fallback générique
    return None

def _is_model(obj):
    """Détermine si un objet possède des coefficients de modèle scikit-learn."""
    return hasattr(obj, "coef_") and hasattr(obj, "intercept_")

def _is_encoder(obj):
    """
    Détermine si l'objet est un encodeur ou un transformateur scikit-learn standard,
    c’est-à-dire qu'il expose get_feature_names_out ou feature_names_in_ 
    (mais n'a pas de coef_ ni intercept_).
    """
    has_names = hasattr(obj, 'get_feature_names_out') or hasattr(obj, 'feature_names_in_')
    is_not_model = not (hasattr(obj, "coef_") and hasattr(obj, "intercept_"))
    return has_names and is_not_model


def coef_dataframe(obj, feature_names=None):
    names = _extract_feature_names(obj, feature_names)

    if _is_model(obj):
        coef = obj.coef_
        intercept = obj.intercept_
        if names is None:
            names = [f"feature_{i}" for i in range(len(coef))]
        index = ['intercept'] + list(names)
        valeurs = np.append(intercept, coef)
    elif _is_encoder(obj):
        index = list(names)
        valeurs = [np.nan] * len(index)
    else:
        raise TypeError("Impossible de déterminer les noms ou les coefficients pour cet objet.")

    return pd.DataFrame({'valeur': valeurs}, index=index)


print("Ordonnée à l'origine ou intercept :", model_en.intercept_)
for index, coef in enumerate(model_en.coef_):
    print("variable :", model_en.feature_names_in_[index], "- pente ou coefficient estimé : ", coef)

display(coef_dataframe(model_en))