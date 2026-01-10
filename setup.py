# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - SETUP
# Master 2 - Data Science Industrielle
# ===============================================

"""
Script de setup pour configurer l'environnement du projet.

Usage:
    python setup.py

Ce script va:
1. Créer les répertoires nécessaires
2. Télécharger les données Olist
3. Vérifier les dépendances
4. Initialiser les logs
"""

import os
import sys
import subprocess
import urllib.request
import pandas as pd
from pathlib import Path
from config import DataConfig, RAW_DATA_DIR, LOGS_DIR

def create_directories():
    """Créé les répertoires nécessaires au projet."""
    print("📁 Création des répertoires...")
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "logs",
        "backend/app",
        "frontend",
        "ml_pipeline",
        "tests"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    print("   🎉 Répertoires créés avec succès!\n")

def download_olist_data():
    """Télécharge les données Olist depuis GitHub."""
    print("📥 Téléchargement des données Olist...")

    for name, filename in DataConfig.DATASETS.items():
        url = DataConfig.OLIST_BASE_URL + filename
        local_path = RAW_DATA_DIR / filename

        if local_path.exists():
            print(f"   ⚠️  {filename} existe déjà")
            continue

        try:
            print(f"   📥 Téléchargement {filename}...")
            urllib.request.urlretrieve(url, local_path)

            # Vérifier que le fichier est bien un CSV valide
            df = pd.read_csv(local_path, nrows=5)
            print(f"   ✅ {filename} - {len(df.columns)} colonnes")

        except Exception as e:
            print(f"   ❌ Erreur téléchargement {filename}: {e}")

    print("   🎉 Données téléchargées avec succès!\n")

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    print("🔧 Vérification des dépendances...")

    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'fastapi',
        'streamlit', 'plotly', 'uvicorn', 'duckdb'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} - MANQUANT")

    if missing_packages:
        print(f"\n⚠️  Packages manquants: {', '.join(missing_packages)}")
        print("📦 Pour installer: pip install -r requirements.txt\n")
        return False
    else:
        print("   🎉 Toutes les dépendances sont installées!\n")
        return True

def create_sample_env():
    """Crée un fichier .env d'exemple."""
    print("🔐 Création du fichier .env d'exemple...")

    env_content = """# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - ENVIRONMENT
# ===============================================

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=True

# ML Model Settings
MODEL_VERSION=1.0.0
RETRAIN_THRESHOLD=0.1

# Logging
LOG_LEVEL=INFO

# Demo Mode (utilise des données simplifiées)
DEMO_MODE=True
"""

    env_path = Path(".env.example")
    with open(env_path, "w", encoding='utf-8') as f:
        f.write(env_content)

    print(f"   ✅ Fichier {env_path} créé")
    print("   📝 Copiez-le vers .env et adaptez selon vos besoins\n")

def create_gitignore():
    """Crée un fichier .gitignore approprié."""
    print("📝 Création du fichier .gitignore...")

    gitignore_content = """# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - GITIGNORE
# ===============================================

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data files (sauf exemples)
data/raw/*.csv
data/processed/*.csv
data/models/*.joblib
data/models/*.pkl

# Logs
logs/*.log

# Environment variables
.env

# Streamlit
.streamlit/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
"""

    with open(".gitignore", "w", encoding='utf-8') as f:
        f.write(gitignore_content)

    print("   ✅ Fichier .gitignore créé\n")

def main():
    """Fonction principale de setup."""
    print("🚀 " + "="*50)
    print("🚀 OLIST RECOMMENDATION SYSTEM - SETUP")
    print("🚀 Master 2 - Data Science Industrielle")
    print("🚀 " + "="*50 + "\n")

    # Étapes de setup
    create_directories()
    create_sample_env()
    create_gitignore()

    # Vérifier les dépendances
    if not check_dependencies():
        print("⚠️  Installez d'abord les dépendances avec:")
        print("   pip install -r requirements.txt\n")
        return False

    # Télécharger les données
    download_olist_data()

    print("🎉 " + "="*50)
    print("🎉 SETUP TERMINÉ AVEC SUCCÈS!")
    print("🎉 " + "="*50 + "\n")

    print("📋 PROCHAINES ÉTAPES:")
    print("   1. 📦 pip install -r requirements.txt (si pas encore fait)")
    print("   2. 🔐 cp .env.example .env")
    print("   3. 🤖 python ml_pipeline/train_model.py")
    print("   4. 🚀 uvicorn backend.app.main:app --reload")
    print("   5. 🎨 streamlit run frontend/app.py")
    print("\n✨ Happy coding! ✨")

if __name__ == "__main__":
    main()