# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - MAKEFILE
# Master 2 - Data Science Industrielle
# ===============================================

.DEFAULT_GOAL := help
.PHONY: help install install-dev setup test clean lint format run-api run-frontend train compare-managers demo

# Colors for output
RED = \033[31m
GREEN = \033[32m
YELLOW = \033[33m
BLUE = \033[34m
MAGENTA = \033[35m
CYAN = \033[36m
RESET = \033[0m

# ==========================================
# 🎯 COMMANDES PRINCIPALES
# ==========================================

help: ## 📋 Affiche l'aide
	@echo "$(CYAN)🚀 Olist Recommendation System - Commandes Make$(RESET)"
	@echo "$(CYAN)Master 2 - Data Science Industrielle$(RESET)"
	@echo ""
	@echo "$(YELLOW)📦 Installation & Setup:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"}; /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(install|setup|clean)"
	@echo ""
	@echo "$(YELLOW)🧪 Tests & Qualité:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"}; /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(test|lint|format)"
	@echo ""
	@echo "$(YELLOW)🚀 Exécution:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"}; /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(run|train|demo)"
	@echo ""
	@echo "$(YELLOW)💡 Comparaison:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"}; /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(compare)"
	@echo ""
	@echo "$(MAGENTA)💡 Utilisez 'make <commande>' pour exécuter$(RESET)"

# ==========================================
# 📦 INSTALLATION & SETUP
# ==========================================

install-uv: ## 🌟 Installe UV (gestionnaire de packages ultra-rapide)
	@echo "$(CYAN)🌟 Installation d'UV - Le futur de la gestion Python$(RESET)"
	@if command -v uv >/dev/null 2>&1; then \
		echo "$(GREEN)✅ UV déjà installé: $$(uv --version)$(RESET)"; \
	else \
		echo "$(YELLOW)📥 Installation d'UV en cours...$(RESET)"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "$(GREEN)✅ UV installé avec succès !$(RESET)"; \
	fi

check-uv: ## 🔍 Vérifie qu'UV est installé
	@if command -v uv >/dev/null 2>&1; then \
		echo "$(GREEN)✅ UV disponible: $$(uv --version)$(RESET)"; \
	else \
		echo "$(RED)❌ UV non installé$(RESET)"; \
		echo "$(YELLOW)💡 Utilisez: make install-uv$(RESET)"; \
		exit 1; \
	fi

install: check-uv ## ⚡ Installation rapide avec UV
	@echo "$(CYAN)⚡ Installation des dépendances avec UV$(RESET)"
	uv sync
	@echo "$(GREEN)✅ Installation terminée !$(RESET)"

install-dev: check-uv ## 🔧 Installation développement avec UV
	@echo "$(CYAN)🔧 Installation mode développement$(RESET)"
	uv sync --extra dev --extra test
	@echo "$(GREEN)✅ Installation développement terminée !$(RESET)"

setup: install-uv install-dev ## 🚀 Setup complet du projet
	@echo "$(CYAN)🚀 Setup complet du projet$(RESET)"
	uv run python scripts/generate_demo_data.py
	@echo "$(GREEN)✅ Setup terminé ! Prêt à coder !$(RESET)"

# ==========================================
# 🧪 TESTS & QUALITÉ
# ==========================================

test: check-uv ## 🧪 Lance les tests rapides
	@echo "$(CYAN)🧪 Lancement des tests avec UV$(RESET)"
	uv run pytest tests/ -m "not slow" -v

test-all: check-uv ## 🧪 Lance tous les tests (incluant lents)
	@echo "$(CYAN)🧪 Lancement de tous les tests$(RESET)"
	uv run pytest tests/ -v

test-unit: check-uv ## 🔬 Tests unitaires uniquement
	@echo "$(CYAN)🔬 Tests unitaires$(RESET)"
	uv run pytest tests/unit/ -v

test-integration: check-uv ## 🔗 Tests d'intégration
	@echo "$(CYAN)🔗 Tests d'intégration$(RESET)"
	uv run pytest tests/integration/ -v

test-e2e: check-uv ## 🎪 Tests end-to-end
	@echo "$(CYAN)🎪 Tests end-to-end$(RESET)"
	uv run pytest tests/e2e/ -v

test-coverage: check-uv ## 📊 Tests avec couverture
	@echo "$(CYAN)📊 Tests avec analyse de couverture$(RESET)"
	uv run pytest tests/ --cov=ml_pipeline --cov=backend --cov=frontend --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)📊 Rapport disponible dans htmlcov/index.html$(RESET)"

lint: check-uv ## 🔍 Analyse du code (linting)
	@echo "$(CYAN)🔍 Analyse de la qualité du code$(RESET)"
	uv run ruff check .
	uv run mypy ml_pipeline/ backend/ frontend/

format: check-uv ## ✨ Formatage automatique du code
	@echo "$(CYAN)✨ Formatage automatique avec Black & Ruff$(RESET)"
	uv run black .
	uv run ruff format .
	uv run isort .
	@echo "$(GREEN)✅ Code formaté automatiquement$(RESET)"

# ==========================================
# 🚀 EXÉCUTION DE L'APPLICATION
# ==========================================

train: check-uv ## 🤖 Entraîne le modèle ML
	@echo "$(CYAN)🤖 Entraînement du modèle de recommandation$(RESET)"
	uv run python ml_pipeline/train_model.py --use-demo-data

run-api: check-uv ## 🚀 Lance l'API FastAPI
	@echo "$(CYAN)🚀 Démarrage de l'API FastAPI$(RESET)"
	@echo "$(YELLOW)💡 API disponible sur http://localhost:8000$(RESET)"
	@echo "$(YELLOW)📚 Documentation sur http://localhost:8000/docs$(RESET)"
	uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

run-frontend: check-uv ## 🎨 Lance l'interface Streamlit
	@echo "$(CYAN)🎨 Démarrage de l'interface Streamlit$(RESET)"
	@echo "$(YELLOW)💡 Interface disponible sur http://localhost:8501$(RESET)"
	uv run streamlit run frontend/app.py

demo: check-uv ## 🎪 Démonstration complète
	@echo "$(CYAN)🎪 Démonstration complète du système$(RESET)"
	@echo "$(YELLOW)1️⃣ Génération des données...$(RESET)"
	uv run python scripts/generate_demo_data.py
	@echo "$(YELLOW)2️⃣ Entraînement du modèle...$(RESET)"
	uv run python ml_pipeline/train_model.py --use-demo-data
	@echo "$(YELLOW)3️⃣ Tests de validation...$(RESET)"
	uv run pytest tests/test_demo.py -v
	@echo "$(GREEN)🎉 Démonstration terminée !$(RESET)"
	@echo "$(MAGENTA)💡 Lancez 'make run-api' et 'make run-frontend' pour utiliser l'app$(RESET)"

# ==========================================
# 💡 COMPARAISONS PÉDAGOGIQUES
# ==========================================

compare-managers: ## ⚔️ Compare pip vs uv (démonstration)
	@echo "$(CYAN)⚔️ COMPARAISON PIP vs UV$(RESET)"
	@echo "$(YELLOW)Cette démo montre l'avantage d'UV sur pip$(RESET)"
	python setup_uv.py --compare

show-uv-advantages: ## 💡 Affiche les avantages d'UV
	@echo "$(CYAN)💡 Avantages d'UV pour les data scientists$(RESET)"
	python setup_uv.py --advantages

benchmark-install: ## 📊 Benchmark d'installation
	@echo "$(CYAN)📊 Benchmark des gestionnaires de packages$(RESET)"
	@echo "$(YELLOW)Test 1: Installation avec pip$(RESET)"
	@time pip install --dry-run -r requirements.txt > /dev/null 2>&1 || true
	@echo "$(YELLOW)Test 2: Résolution avec uv$(RESET)"
	@time uv tree > /dev/null 2>&1 || true
	@echo "$(GREEN)💡 UV est généralement 10-100x plus rapide !$(RESET)"

# ==========================================
# 📊 INFORMATIONS & DEBUG
# ==========================================

info: check-uv ## 📊 Informations sur l'environnement
	@echo "$(CYAN)📊 Informations sur l'environnement$(RESET)"
	@echo "$(YELLOW)Python version:$(RESET)"
	@uv run python --version
	@echo "$(YELLOW)UV version:$(RESET)"
	@uv --version
	@echo "$(YELLOW)Paquets installés:$(RESET)"
	@uv pip list | head -10
	@echo "$(YELLOW)Environnement virtuel:$(RESET)"
	@uv venv --python-version

tree: check-uv ## 🌳 Affiche l'arbre des dépendances
	@echo "$(CYAN)🌳 Arbre des dépendances$(RESET)"
	uv tree

outdated: check-uv ## 📦 Vérifie les packages obsolètes
	@echo "$(CYAN)📦 Packages obsolètes$(RESET)"
	uv pip list --outdated

# ==========================================
# 🧹 NETTOYAGE
# ==========================================

clean: ## 🧹 Nettoie les fichiers temporaires
	@echo "$(CYAN)🧹 Nettoyage des fichiers temporaires$(RESET)"
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✅ Nettoyage terminé$(RESET)"

clean-all: clean ## 🧹 Nettoyage complet (inclut venv)
	@echo "$(CYAN)🧹 Nettoyage complet$(RESET)"
	rm -rf .venv/
	rm -rf data/processed/*
	rm -rf data/models/*
	rm -rf logs/*.log
	@echo "$(GREEN)✅ Nettoyage complet terminé$(RESET)"

# ==========================================
# 🎓 COMMANDES PÉDAGOGIQUES
# ==========================================

student-setup: ## 🎓 Setup rapide pour étudiants
	@echo "$(CYAN)🎓 Setup optimisé pour étudiants$(RESET)"
	@echo "$(YELLOW)1️⃣ Installation d'UV...$(RESET)"
	@make install-uv
	@echo "$(YELLOW)2️⃣ Installation des dépendances...$(RESET)"
	@make install
	@echo "$(YELLOW)3️⃣ Génération des données...$(RESET)"
	uv run python scripts/generate_demo_data.py
	@echo "$(YELLOW)4️⃣ Test rapide...$(RESET)"
	uv run pytest tests/test_demo.py::test_demo_basic -v
	@echo "$(GREEN)🎉 Prêt à apprendre ! 📚$(RESET)"

learning-path: ## 📚 Parcours d'apprentissage recommandé
	@echo "$(CYAN)📚 Parcours d'apprentissage UV + ML$(RESET)"
	@echo "$(YELLOW)1. Comprendre UV:$(RESET) make show-uv-advantages"
	@echo "$(YELLOW)2. Comparer avec pip:$(RESET) make compare-managers"
	@echo "$(YELLOW)3. Setup projet:$(RESET) make student-setup"
	@echo "$(YELLOW)4. Entraîner modèle:$(RESET) make train"
	@echo "$(YELLOW)5. Tester l'API:$(RESET) make run-api"
	@echo "$(YELLOW)6. Interface utilisateur:$(RESET) make run-frontend"
	@echo "$(YELLOW)7. Tests qualité:$(RESET) make test-coverage"
	@echo "$(MAGENTA)💡 Suivez ces étapes pour une expérience complète !$(RESET)"

# ==========================================
# 🚀 COMMANDES COMPOSÉES
# ==========================================

dev-setup: install-uv install-dev format lint ## 🔧 Setup développement complet
	@echo "$(GREEN)✅ Environnement de développement prêt !$(RESET)"

quality-check: lint test-coverage ## ✅ Vérification qualité complète
	@echo "$(GREEN)✅ Vérification qualité terminée$(RESET)"

full-demo: clean setup train test demo ## 🎪 Démonstration complète from scratch
	@echo "$(GREEN)🎉 Démonstration complète réussie !$(RESET)"