# ===============================================
# OLIST RECOMMENDATION SYSTEM - TRAINING
# Master 2 - SEP
# ===============================================

"""
Script d'entraînement du modèle de recommandation.

Ce script:
1. Effectue le feature engineering
2. Entraîne le modèle RandomForest
3. Évalue les performances
4. Sauvegarde le modèle pour l'API
"""

import argparse
import sys
import joblib
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# On remplace l'ancien pipeline par tes nouveaux outils de scoring
from ml_pipeline.preprocessing.feature_engineering import RecommendationFeatureEngine, load_and_prepare_data
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_absolute_error

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, MLConfig


def check_data_availability(data_dir: Path) -> bool:
    """
    Vérifie que toutes les données nécessaires sont disponibles.

    Args:
        data_dir: Répertoire des données

    Returns:
        True si toutes les données sont présentes
    """
    required_files = [
        'olist_orders_dataset.csv',
        'olist_order_items_dataset.csv',
        'olist_products_dataset.csv',
        'olist_order_payments_dataset.csv',
        'olist_sellers_dataset.csv',
        'olist_order_reviews_dataset.csv'
    ]

    missing_files = []
    for filename in required_files:
        if not (data_dir / filename).exists():
            missing_files.append(filename)

    if missing_files:
        print(f"Fichiers manquants: {missing_files}")
        return False

    print("Toutes les données requises sont présentes")
    return True



def train_recommendation_model(data_dir: Path) -> dict:
    print("🧠 Démarrage de l'entraînement du modèle de scoring...")

    # A. Chargement et Feature Engineering
    sellers, orders, items, payments, reviews, products = load_and_prepare_data(data_dir)
    engine = RecommendationFeatureEngine()
    df_vendeurs = engine.create_seller_features(sellers, orders, items, payments, reviews, products)

    # B. Définition des X (critères) et y (ce qu'on veut prédire)
    features = ['avg_review_score', 'late_rate', 'avg_installments', 'active_months', 'solvability_score']
    X = df_vendeurs[features].fillna(0)
    y = df_vendeurs['total_revenue'] # On veut prédire la capacité de revenus du vendeur

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # C. Modèle Regressor
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # D. Évaluation
    y_pred = model.predict(X_test)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred)
    }

    # E. Sauvegardes
    joblib.dump(model, MLConfig.REVENUE_MODEL_FILE)
    df_vendeurs.to_csv(MLConfig.SELLER_FEATURES_FILE)
    
    return metrics


def display_training_results(metrics: dict):
    """
    Affiche les résultats du Scoring de Crédit (Régression).
    """
    print("\n" + "=" * 50)
    print("📈 RÉSULTATS DU MODÈLE DE SOLVABILITÉ")
    print("=" * 50)

    # Métriques de Régression
    print(f"Précision du modèle (R² Score) : {metrics['r2']:.3f}")
    print(f"Erreur Moyenne (MAE)          : {metrics['mae']:.2f} R$")

    # Interprétation du R² (Coefficient de détermination)
    r2 = metrics['r2']
    if r2 >= 0.8:
        fiabilite = "Excellente (Prédiction très fiable)"
    elif r2 >= 0.6:
        fiabilite = "Bonne (Utilisable pour décision de crédit)"
    elif r2 >= 0.4:
        fiabilite = "Moyenne (À coupler avec d'autres critères)"
    else:
        fiabilite = "Faible (Risque d'erreur élevé)"

    print(f"Fiabilité du Scoring          : {fiabilite}")

    # Section Conseils pour le Scoring
    print("\n" + "=" * 50)
    print("💡 ANALYSE ET CONSEILS BUSINESS")
    print("=" * 50)

    print(f"• L'erreur moyenne est de {metrics['mae']:.2f} R$.")
    print(f"  Cela signifie que vos prédictions de revenus s'écartent de ce montant en moyenne.")
    
    print("\n🔧 Pistes pour améliorer le score de crédit :")
    print("  • Vérifier les 'Outliers' (vendeurs avec un CA anormalement élevé).")
    print("  • Ajouter des données sur le type de produits (certaines catégories sont plus risquées).")
    print("  • Tester un modèle Gradient Boosting (XGBoost ou LightGBM) pour plus de précision.")

    print("\n🧠 Argumentaire pour mardi :")
    print("  • Nous ne prédisons plus un simple 'achat', mais la capacité financière réelle.")
    print("  • Le score de solvabilité pondère la réputation, la logistique et le volume d'affaires.")

def main():
    """Fonction principale d'entraînement du Scoring SolIA."""
    parser = argparse.ArgumentParser(
        description="Entraîne le modèle de scoring de solvabilité vendeur Olist"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(RAW_DATA_DIR),
        help=f'Répertoire des données (défaut: {RAW_DATA_DIR})'
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("\n" + "=" * 60)
    print("🚀 SOLIA - SYSTÈME DE SCORING DE CRÉDIT VENDEUR")
    print("Master 2 - SEP | Analyse de Solvabilité")
    print("=" * 60)

    print(f"📂 Répertoire de données: {data_dir}")

    # 1. Vérification des 6 bases (incluant paiements et vendeurs)
    if not check_data_availability(data_dir):
        print("\n❌ Erreur : Bases de données incomplètes pour le calcul de solvabilité.")
        print("Veuillez vérifier que 'olist_order_payments_dataset.csv' et 'olist_sellers_dataset.csv' sont présents.")
        return 1

    # 2. Entraînement (Régression CA futur)
    try:
        metrics = train_recommendation_model(data_dir)
    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement : {e}")
        return 1

    # 3. Affichage des métriques de régression (R2, MAE)
    display_training_results(metrics)

    # 4. Instructions pour la démo de mardi
    print("\n" + "=" * 50)
    print("🎯 PROCHAINES ÉTAPES POUR VOTRE PRÉSENTATION")
    print("" + "=" * 50)
    print("1. Lancer l'API : uv run uvicorn backend.app.main:app --reload")
    print("2. Lancer le Dashboard : uv run streamlit run frontend/app.py")
    print("3. Démo Live : Sélectionnez un vendeur et montrez son éligibilité au prêt.")
    print("4. Justification : Expliquez comment la MAE sécurise la décision de crédit.")

    print("\n✨ Modèle de solvabilité prêt pour la mise en production !\n")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
