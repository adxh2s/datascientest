# 📊 Métriques d’évaluation en Machine Learning

## Régression

| Nom                                | Description                                 | Formule mathématique                                                                 | Signification                                             | Interprétation                    |
|-----------------------------------|---------------------------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------|----------------------------------|
| R² (coefficient de détermination) | Proportion de la variance expliquée         | $R^2 = 1 - \frac{\sum_{i} (y_i - \hat{y}_i)^2}{\sum_{i} (y_i - \bar{y})^2}$          | 1 = parfait, 0 = aussi bon que la moyenne, < 0 = plus mauvais | Vers 1 : excellent, <0 : mauvais |
| MAE (Mean Absolute Error)          | Erreur absolue moyenne                        | $\mathrm{MAE} = \frac{1}{n} \sum_{i} |y_i - \hat{y}_i|$                              | ≥ 0 ; 0 = parfait ; plus petit = meilleur                | 0 = parfait, petite valeur = bon |
| MSE (Mean Squared Error)           | Erreur quadratique moyenne                    | $\mathrm{MSE} = \frac{1}{n} \sum_{i} (y_i - \hat{y}_i)^2$                            | ≥ 0; sensible aux outliers                                 | 0 = parfait, petite valeur = bon |
| RMSE (Root Mean Square Error)      | Racine carrée de la MSE (mêmes unités cible) | $\mathrm{RMSE} = \sqrt{\mathrm{MSE}}$                                               | ≥ 0 ; plus proche de 0 = meilleur                          | Idem MAE/MSE                     |
| MAPE (Mean Absolute Percentage Error) | Erreur relative moyenne en %                 | $\mathrm{MAPE} = \frac{100}{n} \sum_{i} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$  | 0 % = parfait ; exprimé en pourcentage                     | < 10% très bon, > 50% à éviter   |

## Classification

| Nom                           | Description                                 | Formule mathématique                                                | Signification                                   | Interprétation                  |
|-------------------------------|---------------------------------------------|--------------------------------------------------------------------|------------------------------------------------|--------------------------------|
| Accuracy (précision globale)  | Proportion de bonnes prédictions             | $\mathrm{Accuracy} = \frac{TP + TN}{N}$                            | 0 à 1 ; 1 = parfait, 0.5 = hasard               | >0.9 excellent, <0.6 faible    |
| Précision (Precision)          | Ratio vrais positifs sur positifs prévus     | $\mathrm{Precision} = \frac{TP}{TP + FP}$                          | 0 à 1 ; 1 = aucune fausse alerte                 | Haut = peu de faux positifs    |
| Rappel (Recall)                | Ratio vrais positifs sur positifs réels      | $\mathrm{Recall} = \frac{TP}{TP + FN}$                             | 0 à 1 ; 1 = tous retrouvés                       | Haut = bonne sensibilité       |
| F1-score                      | Moyenne harmonique precision/rappel          | $F_1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}$ | 0 à 1 ; 1 = parfait équilibre                  | >0.7 bon, 1 parfait            |
| Log loss (cross-entropy)       | Pénalise probabilités erronées                | $-\frac{1}{N} \sum_i \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$ | Plus petit = meilleur ; min = 0                   | 0 parfait, >0.5 mauvais        |
| AUC ROC                      | Aire sous la courbe ROC                        | Calcul numérique                                                | 0.5 = hasard, 1 = parfait                         | >0.8 bon, 0.5 hasard           |
| MCC (Matthews corr. coeff.)    | Score équilibré prenant les 4 cas en compte  | $\frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | 1 parfait, 0 hasard, -1 inverse                  | >0.7 bon, <0 mauvais           |

**Abréviations :**
- $$TP$$ : vrais positifs
- $$TN$$ : vrais négatifs
- $$FP$$ : faux positifs
- $$FN$$ : faux négatifs
- $$y_i$$ : valeur réelle, $$\hat{y}_i$$ : prédiction, $$N$$ : nombre d’échantillons
