"""
Prédiction du SPI à partir des indices satellite (composantes principales PC1-PC8)
-------------------------------------------------------------------------------------
CORRECTIONS APPORTÉES PAR RAPPORT AUX SCRIPTS ORIGINAUX (sylla_code.py, random_with_acp.py,
randomforest_predictions.py, randomforestonlybruit.py, randomizesearchcv.py,
Randomizedsearchcv_withacp.py) :

1) FUITE DE CIBLE (la plus grave) :
   SPI = (TotalPrecip - mu) / sigma  --  vérifié empiriquement : corr(TotalPrecip, SPI) = 1.000000
   'TotalPrecip' EST le SPI (transformation affine). L'inclure comme feature revient à
   donner la cible au modèle. C'est ce qui explique les R² ~0.999 identiques entre RF,
   XGBoost et SVR. => TotalPrecip est retiré des features. Seules les variables satellite
   (PC1-PC8, dérivées de B2,B3,B4,B8, NDVI, NDWI, EVI, MSAVI) sont conservées, ce qui est
   d'ailleurs cohérent avec l'objectif réel : estimer le risque climatique par satellite,
   SANS mesure de précipitation au sol.

2) FUITE PAR NORMALISATION AVANT SPLIT :
   Dans random_with_acp.py / randomizesearchcv.py, MinMaxScaler.fit_transform() était
   appliqué sur X et y AVANT le train_test_split => les statistiques (min/max) du test
   set fuitaient dans l'entraînement. Ici, le scaler est fit UNIQUEMENT sur X_train.

3) 'max_features': 'auto' est déprécié/supprimé dans les versions récentes de scikit-learn
   pour RandomForestRegressor => remplacé par 'sqrt' / 'log2' / 1.0.

4) La validation croisée reste un K-Fold aléatoire classique (pas de time series split) :
   les 200 échantillons sont des points spatiaux indépendants (pas une série temporelle
   séquentielle), donc RandomizedSearchCV avec KFold(shuffle=True) est approprié.

Après cette correction, attends-toi à un R² nettement plus modeste (probablement dans une
fourchette réaliste, ex. 0.3-0.7 selon le pouvoir prédictif réel des indices satellite sur
le SPI) plutôt que 0.999. C'est le résultat honnête de la question scientifique posée.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# ELM est optionnel (package hpelm parfois instable/pas installé) ---------------------
try:
    from hpelm import ELM
    HPELM_AVAILABLE = True
except ImportError:
    HPELM_AVAILABLE = False
    print("[INFO] hpelm non disponible : la partie ELM sera ignorée.")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Dossier de sortie portable : un sous-dossier "outputs" à côté du script,
# créé automatiquement s'il n'existe pas (fonctionne sur Windows/Mac/Linux).
import os
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# 1) CHARGEMENT DES DONNÉES
# ======================================================================================
file_path = 'composantes_principales.xlsx'
data = pd.read_excel(file_path)

# --- FEATURES CORRIGÉES : uniquement les composantes principales satellite -----------
# TotalPrecip est EXCLU : c'est la variable qui définit SPI elle-même (fuite de cible).
FEATURE_COLS = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']
TARGET_COL = 'SPI'

X = data[FEATURE_COLS].copy()
y = data[TARGET_COL].copy()

print(f"Features utilisées : {FEATURE_COLS}")
print(f"(TotalPrecip retiré volontairement : corr(TotalPrecip, SPI) = "
      f"{data['TotalPrecip'].corr(data['SPI']):.6f} -> fuite directe)\n")

# ======================================================================================
# 2) SPLIT TRAIN/TEST (avant tout fit de scaler ou de modèle)
# ======================================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ======================================================================================
# 3) NORMALISATION -- fit UNIQUEMENT sur le train (corrige la fuite #2)
# ======================================================================================
scaler_X = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler_X.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler_X.transform(X_test), columns=FEATURE_COLS, index=X_test.index
)
# Les modèles arbres (RF, XGBoost) n'ont pas besoin de scaling, mais SVR et ELM oui :
# on utilise les versions scalées pour SVR/ELM, et les versions non scalées (ou scalées,
# c'est équivalent pour un arbre) pour RF/XGBoost.

cv_splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ======================================================================================
# 4) RANDOM FOREST
# ======================================================================================
rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 1.0],  # 'auto' déprécié -> corrigé
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE), rf_param_grid,
    n_iter=30, cv=cv_splitter, n_jobs=-1, verbose=1, random_state=RANDOM_STATE,
    scoring='r2'
)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
joblib.dump(rf_best, os.path.join(OUTPUT_DIR, 'random_forest_model.pkl'))
print("Meilleurs paramètres RF :", rf_search.best_params_)

# ======================================================================================
# 5) XGBOOST
# ======================================================================================
xgb_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.7, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10],
}
xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=RANDOM_STATE), xgb_param_grid,
    n_iter=30, cv=cv_splitter, n_jobs=-1, verbose=1, random_state=RANDOM_STATE,
    scoring='r2'
)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
xgb_best.save_model(os.path.join(OUTPUT_DIR, 'xgboost_model.json'))
print("Meilleurs paramètres XGBoost :", xgb_search.best_params_)

# ======================================================================================
# 6) SVR (sur données scalées)
# ======================================================================================
svr_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear'],
    'epsilon': [0.01, 0.05, 0.1, 0.2],
}
svr_search = RandomizedSearchCV(
    SVR(), svr_param_grid, n_iter=30, cv=cv_splitter, n_jobs=-1, verbose=1,
    random_state=RANDOM_STATE, scoring='r2'
)
svr_search.fit(X_train_scaled, y_train)
svr_best = svr_search.best_estimator_
joblib.dump(svr_best, os.path.join(OUTPUT_DIR, 'svr_model.pkl'))
print("Meilleurs paramètres SVR :", svr_search.best_params_)

# ======================================================================================
# 7) ELM (optionnel, sur données scalées)
# ======================================================================================
elm_preds_test = None
if HPELM_AVAILABLE:
    elm = ELM(X_train_scaled.shape[1], 1)
    elm.add_neurons(100, 'sigm')
    elm.train(X_train_scaled.values, y_train.values.reshape(-1, 1))
    elm_preds_test = elm.predict(X_test_scaled.values).flatten()

# ======================================================================================
# 8) ÉVALUATION TRAIN vs TEST (permet de repérer un éventuel surapprentissage)
# ======================================================================================
def evaluate(model, X_tr, y_tr, X_te, y_te, name, predict_fn=None):
    if predict_fn is None:
        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
    else:
        pred_tr, pred_te = predict_fn()
    row = {
        'Model': name,
        'MSE (Train)': mean_squared_error(y_tr, pred_tr),
        'RMSE (Train)': np.sqrt(mean_squared_error(y_tr, pred_tr)),
        'R2 (Train)': r2_score(y_tr, pred_tr),
        'MSE (Test)': mean_squared_error(y_te, pred_te),
        'RMSE (Test)': np.sqrt(mean_squared_error(y_te, pred_te)),
        'R2 (Test)': r2_score(y_te, pred_te),
    }
    return row, pred_te

results = []
row, rf_pred_test = evaluate(rf_best, X_train, y_train, X_test, y_test, 'Random Forest')
results.append(row)

row, xgb_pred_test = evaluate(xgb_best, X_train, y_train, X_test, y_test, 'XGBoost')
results.append(row)

row, svr_pred_test = evaluate(svr_best, X_train_scaled, y_train, X_test_scaled, y_test, 'SVR')
results.append(row)

if HPELM_AVAILABLE:
    elm_pred_train = elm.predict(X_train_scaled.values).flatten()
    results.append({
        'Model': 'ELM',
        'MSE (Train)': mean_squared_error(y_train, elm_pred_train),
        'RMSE (Train)': np.sqrt(mean_squared_error(y_train, elm_pred_train)),
        'R2 (Train)': r2_score(y_train, elm_pred_train),
        'MSE (Test)': mean_squared_error(y_test, elm_preds_test),
        'RMSE (Test)': np.sqrt(mean_squared_error(y_test, elm_preds_test)),
        'R2 (Test)': r2_score(y_test, elm_preds_test),
    })

results_df = pd.DataFrame(results)
print("\n=== Résultats (sans fuite de données) ===")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUTPUT_DIR, 'resultats_spi_corriges.csv'), index=False)

# ======================================================================================
# 9) VISUALISATION DES MÉTRIQUES DE TEST
# ======================================================================================
metrics_to_plot = ['MSE (Test)', 'RMSE (Test)', 'MAE (Test)' if 'MAE (Test)' in results_df else None, 'R2 (Test)']
metrics_to_plot = [m for m in metrics_to_plot if m in results_df.columns]

plt.figure(figsize=(12, 5))
for i, metric in enumerate(metrics_to_plot):
    plt.subplot(1, len(metrics_to_plot), i + 1)
    sns.barplot(x='Model', y=metric, data=results_df, palette="viridis")
    plt.title(metric)
    plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison.png'), dpi=150)
plt.close()

# ======================================================================================
# 10) COURBES PRÉDICTIONS VS VALEURS RÉELLES (test set)
# ======================================================================================
plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label="Valeurs réelles", linestyle='dashed')
plt.plot(rf_pred_test, label="Random Forest", alpha=0.7)
plt.plot(xgb_pred_test, label="XGBoost", alpha=0.7)
plt.plot(svr_pred_test, label="SVR", alpha=0.7)
if HPELM_AVAILABLE:
    plt.plot(elm_preds_test, label="ELM", alpha=0.7)
plt.legend()
plt.xlabel("Échantillons (test set)")
plt.ylabel("SPI")
plt.title("Comparaison des prédictions (modèle corrigé, sans TotalPrecip)")
plt.savefig(os.path.join(OUTPUT_DIR, 'predictions_vs_reel.png'), dpi=150)
plt.close()

# ======================================================================================
# 11) ROBUSTESSE AU BRUIT (bruit ajouté uniquement sur X_test, jamais sur y_test)
# ======================================================================================
def test_robustness(models_dict, X_test_df, X_test_scaled_df, y_test_s, noise_levels):
    robustness_r2 = {name: [] for name in models_dict}
    for noise in noise_levels:
        Xn = X_test_df + np.random.normal(0, noise * X_test_df.std(), size=X_test_df.shape)
        Xn_scaled = X_test_scaled_df + np.random.normal(
            0, noise * X_test_scaled_df.std(), size=X_test_scaled_df.shape
        )
        for name, (model, uses_scaled) in models_dict.items():
            X_in = Xn_scaled if uses_scaled else Xn
            preds = model.predict(X_in.values if hasattr(X_in, 'values') else X_in)
            robustness_r2[name].append(r2_score(y_test_s, preds))
    return robustness_r2

noise_levels = np.arange(0, 0.55, 0.05)
models_for_noise = {
    'Random Forest': (rf_best, False),
    'XGBoost': (xgb_best, False),
    'SVR': (svr_best, True),
}
robustness_results = test_robustness(models_for_noise, X_test, X_test_scaled, y_test, noise_levels)

plt.figure(figsize=(10, 5))
for name, r2_scores in robustness_results.items():
    plt.plot(noise_levels * 100, r2_scores, label=name, marker='o')
plt.xlabel("Niveau de bruit (%)")
plt.ylabel("R² Score")
plt.title("Robustesse des modèles au bruit gaussien (modèle corrigé)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, 'robustesse_bruit.png'), dpi=150)
plt.close()

# ======================================================================================
# 12) IMPORTANCE DES VARIABLES (SHAP) — sur le Random Forest
# ======================================================================================
try:
    import shap
    explainer = shap.Explainer(rf_best, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
except ImportError:
    print("[INFO] shap non disponible : analyse d'importance ignorée.")

print(f"\nTerminé. Fichiers générés dans {OUTPUT_DIR} :")
print(" - resultats_spi_corriges.csv")
print(" - metrics_comparison.png")
print(" - predictions_vs_reel.png")
print(" - robustesse_bruit.png")
print(" - shap_summary.png (si shap installé)")
print(" - random_forest_model.pkl, xgboost_model.json, svr_model.pkl")
