# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - CONFIG
# Master 2 - SEP
# ===============================================

from pathlib import Path

# ==========================================
# 📁 PATHS CONFIGURATION
# ==========================================
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# ==========================================
# 🤖 ML MODEL CONFIGURATION
# ==========================================
class MLConfig:
    # Random Forest parameters
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }

    # Training configuration
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5

    # Model files
    RECOMMENDATION_MODEL_FILE = MODELS_DIR / "recommendation_model.joblib"
    FEATURE_PIPELINE_FILE = MODELS_DIR / "feature_pipeline.joblib"
    CUSTOMER_FEATURES_FILE = PROCESSED_DATA_DIR / "customer_features.csv"
    PRODUCT_FEATURES_FILE = RAW_DATA_DIR / "olist_products_dataset.csv"
    SELLER_FEATURES_FILE = PROCESSED_DATA_DIR / "seller_features.csv"
    REVENUE_MODEL_FILE = MODELS_DIR / "revenue_prediction_model.joblib"

# ==========================================
# 🚀 API CONFIGURATION
# ==========================================
class APIConfig:
    HOST = "127.0.0.1"
    PORT = 8000
    TITLE = "Prêt SolIA"
    VERSION = "1.0.0"
    DESCRIPTION = """
    **Prêt SolIA**

    Cette API fournit des recommandations personnalisées basées sur:
    - L'historique d'achat des clients
    - Les similarités entre produits
    - Les préférences par catégorie

    **Fonctionnalités:**
    - Recommandations personnalisées par client
    - Prédiction de probabilité d'achat
    - Métriques de performance du modèle
    """

# ==========================================
# 🎨 STREAMLIT CONFIGURATION
# ==========================================
class StreamlitConfig:
    PAGE_TITLE = "SolIA"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    SIDEBAR_STATE = "expanded"

# ==========================================
# 📊 DATA CONFIGURATION
# ==========================================
class DataConfig:
    # Olist dataset URLs (pour téléchargement automatique)
    SOLIA_BASE_URL = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"

    DATASETS = {
        'customers': 'olist_customers_dataset.csv',
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv'
    }


    # Nouvelles features pour le scoring vendeur
    SELLER_FEATURES = [
        'total_revenue',
        'avg_review_score',
        'late_rate',
        'avg_installments',
        'active_months',
        'solvability_score' # 👈 Ta nouvelle colonne
    ]


# ==========================================
# 🔧 LOGGING CONFIGURATION
# ==========================================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[{asctime}] {levelname:<8} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "olist_recommendation.log"),
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default", "file"],
    },
}