# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - SETUP UV
# Master 2 - Data Science Industrielle
# ===============================================

"""
Script de setup utilisant UV - le gestionnaire de packages Python ultra-rapide.

Ce script montre aux étudiants les avantages d'uv par rapport à pip :
- 10-100x plus rapide que pip
- Résolution de dépendances améliorée
- Gestion native des environnements virtuels
- Lock files pour la reproductibilité
- Interface moderne et claire

Usage:
    python setup_uv.py                 # Setup complet
    python setup_uv.py --compare       # Comparaison pip vs uv
    python setup_uv.py --install-uv    # Installer uv seulement
    python setup_uv.py --dev           # Installation développement
"""

import argparse
import subprocess
import sys
import time
import platform
import urllib.request
from pathlib import Path
import os

class UVSetup:
    """Gestionnaire de setup avec UV."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.uv_installed = self.check_uv_installed()

    def check_uv_installed(self) -> bool:
        """Vérifie si uv est installé."""
        try:
            result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ UV installé : {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass

        print("❌ UV n'est pas installé")
        return False

    def install_uv(self):
        """Installe UV automatiquement."""
        print("\n🚀 Installation d'UV...")
        print("UV est le gestionnaire de packages Python nouvelle génération !")

        system = platform.system().lower()

        try:
            if system == "windows":
                # Installation via PowerShell pour Windows
                cmd = ['powershell', '-c',
                       'irm https://astral.sh/uv/install.ps1 | iex']
            else:
                # Installation via curl pour Linux/macOS
                cmd = ['curl', '-LsSf', 'https://astral.sh/uv/install.sh', '|', 'sh']

            print(f"🔄 Exécution : {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)

            if result.returncode == 0:
                print("✅ UV installé avec succès !")
                self.uv_installed = True
                return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors de l'installation d'UV : {e}")
            print("\n📦 Installation manuelle :")
            print("Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
            print("Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh")
            return False

    def create_virtual_environment(self):
        """Crée un environnement virtuel avec UV."""
        if not self.uv_installed:
            print("❌ UV non installé, impossible de créer l'environnement")
            return False

        print("\n🌟 Création de l'environnement virtuel avec UV...")

        try:
            # UV peut créer et gérer les venv automatiquement
            cmd = ['uv', 'venv', '.venv', '--python', '3.9']
            print(f"🔄 {' '.join(cmd)}")

            result = subprocess.run(cmd, check=True, cwd=self.project_root)

            if result.returncode == 0:
                print("✅ Environnement virtuel créé dans .venv/")
                print("💡 UV gère automatiquement l'activation !")
                return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur création venv : {e}")
            return False

    def install_dependencies_uv(self, dev=False):
        """Installe les dépendances avec UV."""
        if not self.uv_installed:
            print("❌ UV non installé")
            return False

        print(f"\n⚡ Installation des dépendances avec UV...")

        try:
            # Installation des dépendances principales
            start_time = time.time()

            if dev:
                print("🔧 Installation en mode développement...")
                cmd = ['uv', 'sync', '--extra', 'dev', '--extra', 'test']
            else:
                print("📦 Installation des dépendances de production...")
                cmd = ['uv', 'sync']

            print(f"🔄 {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, cwd=self.project_root)

            duration = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Installation terminée en {duration:.2f} secondes !")
                print("🚀 C'est ça la puissance d'UV !")
                return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur installation : {e}")
            return False

    def install_dependencies_pip(self, dev=False):
        """Installe les dépendances avec pip (pour comparaison)."""
        print(f"\n🐌 Installation des dépendances avec pip...")

        try:
            start_time = time.time()

            if dev:
                cmd = [sys.executable, '-m', 'pip', 'install', '-e', '.[dev,test]']
            else:
                cmd = [sys.executable, '-m', 'pip', 'install', '-e', '.']

            print(f"🔄 {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, cwd=self.project_root)

            duration = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Installation pip terminée en {duration:.2f} secondes")
                return duration

        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur installation pip : {e}")
            return None

    def compare_pip_vs_uv(self):
        """Compare les performances pip vs UV."""
        print("\n" + "="*60)
        print("⚔️  COMPARAISON PIP vs UV")
        print("="*60)

        print("🎯 Cette démo montre pourquoi UV révolutionne l'écosystème Python")

        # Test avec pip
        print("\n1️⃣ TEST AVEC PIP (méthode traditionnelle)")
        pip_duration = self.install_dependencies_pip(dev=True)

        if not self.uv_installed:
            print("\n⚠️ UV non installé, installation en cours...")
            if not self.install_uv():
                print("❌ Impossible de comparer sans UV")
                return

        # Test avec UV
        print("\n2️⃣ TEST AVEC UV (nouvelle génération)")
        uv_start = time.time()
        uv_success = self.install_dependencies_uv(dev=True)
        uv_duration = time.time() - uv_start

        # Comparaison
        print("\n" + "="*60)
        print("📊 RÉSULTATS DE LA COMPARAISON")
        print("="*60)

        if pip_duration and uv_success:
            speedup = pip_duration / uv_duration
            print(f"⏱️  Pip :     {pip_duration:.2f} secondes")
            print(f"⚡ UV :      {uv_duration:.2f} secondes")
            print(f"🚀 Accélération : {speedup:.1f}x plus rapide avec UV !")

            if speedup > 5:
                print("🏆 UV est SIGNIFICATIVEMENT plus rapide !")
            elif speedup > 2:
                print("✨ UV montre un gain de performance notable")
            else:
                print("📈 UV reste plus rapide que pip")

        print(f"\n💡 Autres avantages d'UV :")
        print("   • Résolution de dépendances plus intelligente")
        print("   • Lock files automatiques pour la reproductibilité")
        print("   • Gestion native des environnements virtuels")
        print("   • Interface utilisateur moderne")
        print("   • Compatible avec pip mais en mieux")

    def generate_uv_lock(self):
        """Génère un fichier de lock pour la reproductibilité."""
        if not self.uv_installed:
            return False

        print("\n🔒 Génération du fichier de lock...")

        try:
            cmd = ['uv', 'lock']
            result = subprocess.run(cmd, check=True, cwd=self.project_root)

            if result.returncode == 0:
                print("✅ Fichier uv.lock généré !")
                print("💡 Ce fichier garantit des installations reproductibles")
                return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur génération lock : {e}")
            return False

    def show_uv_advantages(self):
        """Affiche les avantages pédagogiques d'UV."""
        print("\n" + "🌟" * 20)
        print("🎓 POURQUOI UV POUR LES ÉTUDIANTS ?")
        print("🌟" * 20)

        advantages = [
            ("⚡ Vitesse", "10-100x plus rapide que pip"),
            ("🧠 Intelligence", "Résolution de dépendances améliorée"),
            ("🔒 Reproductibilité", "Lock files automatiques"),
            ("🌍 Environnements", "Gestion native des venv"),
            ("🔄 Compatibilité", "Compatible avec pip et PyPI"),
            ("🚀 Modernité", "Écrit en Rust, interface moderne"),
            ("📊 Transparence", "Affichage clair des opérations"),
            ("💾 Efficacité", "Cache intelligent et réutilisation")
        ]

        for title, description in advantages:
            print(f"{title:<20} {description}")

        print(f"\n🎯 En tant que futurs data scientists, vous devez :")
        print("   • Utiliser les outils les plus performants")
        print("   • Optimiser vos workflows de développement")
        print("   • Maîtriser les technologies émergentes")
        print("   • Améliorer la reproductibilité de vos projets")

    def setup_complete_project(self, dev=False):
        """Setup complet du projet avec UV."""
        print("\n🚀 " + "="*50)
        print("🚀 SETUP COMPLET AVEC UV")
        print("🚀 Olist Recommendation System")
        print("🚀 " + "="*50)

        steps = [
            ("Vérification UV", self.check_uv_or_install),
            ("Environnement virtuel", self.create_virtual_environment),
            ("Installation dépendances", lambda: self.install_dependencies_uv(dev)),
            ("Génération données démo", self.generate_demo_data),
            ("Fichier de lock", self.generate_uv_lock),
            ("Validation installation", self.validate_installation)
        ]

        for step_name, step_func in steps:
            print(f"\n📋 Étape : {step_name}")
            if not step_func():
                print(f"❌ Échec de l'étape : {step_name}")
                return False
            print(f"✅ {step_name} terminé")

        print(f"\n🎉 " + "="*50)
        print("🎉 SETUP TERMINÉ AVEC SUCCÈS !")
        print("🎉 " + "="*50)

        self.show_next_steps()
        return True

    def check_uv_or_install(self):
        """Vérifie UV ou l'installe si nécessaire."""
        if not self.uv_installed:
            return self.install_uv()
        return True

    def generate_demo_data(self):
        """Génère les données de démo."""
        try:
            from scripts.generate_demo_data import main as generate_demo
            generate_demo()
            return True
        except Exception as e:
            print(f"⚠️ Erreur génération données démo : {e}")
            print("💡 Vous pourrez les générer manuellement plus tard")
            return True  # Non bloquant

    def validate_installation(self):
        """Valide que l'installation fonctionne."""
        print("🧪 Validation de l'installation...")

        # Test d'import des modules principaux
        test_imports = [
            'pandas', 'numpy', 'sklearn', 'fastapi', 'streamlit'
        ]

        for module in test_imports:
            try:
                __import__(module)
                print(f"   ✅ {module}")
            except ImportError:
                print(f"   ❌ {module} - échec import")
                return False

        return True

    def show_next_steps(self):
        """Affiche les prochaines étapes."""
        print(f"\n📋 PROCHAINES ÉTAPES :")
        print("1. 🤖 Entraîner le modèle :")
        print("   uv run python ml_pipeline/train_model.py --use-demo-data")
        print("\n2. 🚀 Lancer l'API :")
        print("   uv run uvicorn backend.app.main:app --reload")
        print("\n3. 🎨 Lancer le frontend :")
        print("   uv run streamlit run frontend/app.py")
        print("\n4. 🧪 Lancer les tests :")
        print("   uv run pytest tests/ -v")
        print("\n💡 Notez comme 'uv run' simplifie l'exécution !")

def main():
    parser = argparse.ArgumentParser(
        description="🚀 Setup Olist Recommendation System avec UV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python setup_uv.py                    # Setup complet
  python setup_uv.py --compare          # Démo pip vs uv
  python setup_uv.py --install-uv       # Installer uv seulement
  python setup_uv.py --dev              # Mode développement
        """
    )

    parser.add_argument("--compare", action="store_true",
                       help="Compare les performances pip vs uv")
    parser.add_argument("--install-uv", action="store_true",
                       help="Installe UV uniquement")
    parser.add_argument("--dev", action="store_true",
                       help="Installation en mode développement")
    parser.add_argument("--advantages", action="store_true",
                       help="Affiche les avantages d'UV")

    args = parser.parse_args()
    setup = UVSetup()

    if args.advantages:
        setup.show_uv_advantages()
    elif args.install_uv:
        setup.install_uv()
    elif args.compare:
        setup.compare_pip_vs_uv()
    else:
        # Setup complet
        setup.setup_complete_project(dev=args.dev)

if __name__ == "__main__":
    main()