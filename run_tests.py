# ===============================================
# 🧪 OLIST RECOMMENDATION SYSTEM - TEST RUNNER
# Master 2 - Data Science Industrielle
# ===============================================

"""
Script pour lancer les tests de manière simple et organisée.

Ce script permet aux étudiants de lancer facilement différents types de tests :
- Tests unitaires uniquement
- Tests d'intégration
- Tests end-to-end
- Tous les tests
- Tests avec couverture de code

Usage:
    python run_tests.py                    # Tous les tests rapides
    python run_tests.py --unit             # Tests unitaires uniquement
    python run_tests.py --integration      # Tests d'intégration uniquement
    python run_tests.py --e2e              # Tests end-to-end uniquement
    python run_tests.py --all              # Tous les tests (incluant lents)
    python run_tests.py --coverage         # Tests avec couverture
    python run_tests.py --help             # Aide
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os

def run_command(command, description):
    """
    Exécute une commande et affiche le résultat.

    Args:
        command: Commande à exécuter (liste)
        description: Description de l'action

    Returns:
        True si succès, False si échec
    """
    print(f"\n🚀 {description}")
    print(f"Commande: {' '.join(command)}")
    print("-" * 60)

    try:
        result = subprocess.run(command, check=True, text=True)
        print(f"✅ {description} - RÉUSSI")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - ÉCHEC (code: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"❌ pytest non trouvé. Installez-le avec: pip install pytest")
        return False

def check_pytest_installation():
    """Vérifie que pytest est installé."""
    try:
        subprocess.run(["pytest", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pytest n'est pas installé.")
        print("📦 Installation: pip install pytest pytest-asyncio")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="🧪 Lanceur de tests pour Olist Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python run_tests.py                    # Tests rapides (unit + integration)
  python run_tests.py --unit             # Tests unitaires uniquement
  python run_tests.py --integration      # Tests d'intégration uniquement
  python run_tests.py --e2e              # Tests end-to-end uniquement
  python run_tests.py --all              # Tous les tests (incluant lents)
  python run_tests.py --coverage         # Tests avec couverture de code
  python run_tests.py --fast             # Tests les plus rapides seulement
  python run_tests.py --verbose          # Mode verbeux
        """
    )

    # Arguments principaux
    parser.add_argument("--unit", action="store_true",
                       help="Lancer uniquement les tests unitaires")
    parser.add_argument("--integration", action="store_true",
                       help="Lancer uniquement les tests d'intégration")
    parser.add_argument("--e2e", action="store_true",
                       help="Lancer uniquement les tests end-to-end")
    parser.add_argument("--all", action="store_true",
                       help="Lancer tous les tests (incluant les tests lents)")
    parser.add_argument("--fast", action="store_true",
                       help="Lancer seulement les tests les plus rapides")

    # Options supplémentaires
    parser.add_argument("--coverage", action="store_true",
                       help="Lancer les tests avec rapport de couverture")
    parser.add_argument("--verbose", action="store_true",
                       help="Mode verbeux avec plus de détails")
    parser.add_argument("--parallel", action="store_true",
                       help="Lancer les tests en parallèle (requiert pytest-xdist)")
    parser.add_argument("--report", action="store_true",
                       help="Générer un rapport HTML des résultats")

    args = parser.parse_args()

    # Vérifier que pytest est installé
    if not check_pytest_installation():
        return 1

    # Configuration de base
    base_command = ["pytest"]

    # Déterminer quels tests lancer
    if args.unit:
        base_command.extend(["-m", "unit"])
        test_description = "Tests unitaires"
    elif args.integration:
        base_command.extend(["-m", "integration"])
        test_description = "Tests d'intégration"
    elif args.e2e:
        base_command.extend(["-m", "e2e"])
        test_description = "Tests end-to-end"
    elif args.all:
        # Tous les tests sans restriction
        test_description = "Tous les tests"
    elif args.fast:
        base_command.extend(["-m", "not slow and not e2e"])
        test_description = "Tests rapides"
    else:
        # Par défaut : tests rapides (unit + integration, pas e2e ni slow)
        base_command.extend(["-m", "not slow and not e2e"])
        test_description = "Tests rapides (unit + integration)"

    # Options supplémentaires
    if args.verbose:
        base_command.extend(["-v", "--tb=long"])

    if args.parallel:
        base_command.extend(["-n", "auto"])  # Requiert pytest-xdist

    if args.coverage:
        base_command.extend([
            "--cov=ml_pipeline",
            "--cov=backend",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
        test_description += " avec couverture"

    if args.report:
        base_command.extend(["--html=reports/test_report.html", "--self-contained-html"])
        # Créer le répertoire reports s'il n'existe pas
        Path("reports").mkdir(exist_ok=True)

    # Afficher les informations sur la session de test
    print("🧪 " + "="*60)
    print("🧪 OLIST RECOMMENDATION SYSTEM - TESTS")
    print("🧪 Master 2 - Data Science Industrielle")
    print("🧪 " + "="*60)
    print(f"📋 Type de tests: {test_description}")
    print(f"📂 Répertoire de travail: {Path.cwd()}")
    print(f"🐍 Python: {sys.version.split()[0]}")

    # Vérifier la structure des tests
    test_dirs = ["tests/unit", "tests/integration", "tests/e2e"]
    existing_dirs = [d for d in test_dirs if Path(d).exists()]
    print(f"📁 Répertoires de tests trouvés: {', '.join(existing_dirs)}")

    # Compter les fichiers de test
    test_files = list(Path("tests").glob("**/test_*.py"))
    print(f"📄 Fichiers de test trouvés: {len(test_files)}")

    # Lancer les tests
    success = run_command(base_command, test_description)

    # Résultats finaux
    print("\n" + "="*60)
    if success:
        print("🎉 TOUS LES TESTS ONT RÉUSSI!")
        if args.coverage:
            print("📊 Rapport de couverture généré dans htmlcov/index.html")
        if args.report:
            print("📋 Rapport HTML généré dans reports/test_report.html")
    else:
        print("💥 CERTAINS TESTS ONT ÉCHOUÉ")
        print("🔍 Consultez les détails ci-dessus pour identifier les problèmes")
    print("="*60)

    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)