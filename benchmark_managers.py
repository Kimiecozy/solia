# ===============================================
# 🚀 BENCHMARK PIP vs UV - OLIST PROJECT
# Master 2 - Data Science Industrielle
# ===============================================

"""
Script de benchmark pour comparer pip et uv sur des tâches réelles.

Ce script permet aux étudiants de mesurer concrètement les gains
de performance d'UV sur leur propre machine.

Usage:
    python benchmark_managers.py
    python benchmark_managers.py --packages pandas,numpy,sklearn
    python benchmark_managers.py --verbose
"""

import subprocess
import time
import sys
import tempfile
import shutil
from pathlib import Path
import argparse
import json
from typing import List, Dict, Tuple
import statistics

class PackageManagerBenchmark:
    """Classe pour benchmarker pip vs uv."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []

    def log(self, message: str, force: bool = False):
        """Log avec gestion du mode verbose."""
        if self.verbose or force:
            print(message)

    def run_command_with_timing(self, command: List[str], description: str,
                              cwd: Path = None, timeout: int = 300) -> Tuple[float, bool]:
        """
        Exécute une commande en mesurant le temps.

        Returns:
            (duration, success)
        """
        self.log(f"🔄 {description}")
        self.log(f"   Command: {' '.join(command)}")

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            if success:
                self.log(f"   ✅ Completed in {duration:.2f}s")
            else:
                self.log(f"   ❌ Failed after {duration:.2f}s")
                self.log(f"   Error: {result.stderr[:200]}")

            return duration, success

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.log(f"   ⏰ Timeout after {duration:.2f}s")
            return duration, False
        except Exception as e:
            duration = time.time() - start_time
            self.log(f"   💥 Exception: {e}")
            return duration, False

    def check_manager_availability(self) -> Dict[str, bool]:
        """Vérifie la disponibilité de pip et uv."""
        managers = {}

        # Check pip
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'],
                                  capture_output=True, timeout=10)
            managers['pip'] = result.returncode == 0
            if managers['pip']:
                self.log(f"✅ pip available: {result.stdout.decode().strip()}")
        except:
            managers['pip'] = False
            self.log("❌ pip not available")

        # Check uv
        try:
            result = subprocess.run(['uv', '--version'],
                                  capture_output=True, timeout=10)
            managers['uv'] = result.returncode == 0
            if managers['uv']:
                self.log(f"✅ uv available: {result.stdout.decode().strip()}")
        except:
            managers['uv'] = False
            self.log("❌ uv not available")

        return managers

    def benchmark_environment_creation(self, temp_dir: Path) -> Dict[str, Dict]:
        """Benchmark la création d'environnements virtuels."""
        self.log("\n🌍 BENCHMARK: Création d'environnements virtuels", force=True)

        results = {}

        # Test venv (traditionnel)
        venv_dir = temp_dir / "test_venv"
        duration, success = self.run_command_with_timing(
            [sys.executable, '-m', 'venv', str(venv_dir)],
            "Création venv traditionnel"
        )
        results['venv'] = {'duration': duration, 'success': success}

        # Test uv venv
        uv_dir = temp_dir / "test_uv_venv"
        duration, success = self.run_command_with_timing(
            ['uv', 'venv', str(uv_dir)],
            "Création avec uv venv"
        )
        results['uv_venv'] = {'duration': duration, 'success': success}

        return results

    def benchmark_package_installation(self, temp_dir: Path, packages: List[str]) -> Dict[str, Dict]:
        """Benchmark l'installation de packages."""
        self.log(f"\n📦 BENCHMARK: Installation packages {packages}", force=True)

        results = {}

        # Préparer les environnements
        pip_env = temp_dir / "pip_env"
        uv_env = temp_dir / "uv_env"

        # Créer environnement pip
        subprocess.run([sys.executable, '-m', 'venv', str(pip_env)],
                      capture_output=True)

        # Créer environnement uv
        subprocess.run(['uv', 'venv', str(uv_env)],
                      capture_output=True)

        # Test installation avec pip
        pip_python = pip_env / ('Scripts' if sys.platform == 'win32' else 'bin') / 'python'
        pip_cmd = [str(pip_python), '-m', 'pip', 'install'] + packages

        duration, success = self.run_command_with_timing(
            pip_cmd,
            f"Installation pip: {' '.join(packages)}"
        )
        results['pip'] = {'duration': duration, 'success': success}

        # Test installation avec uv
        duration, success = self.run_command_with_timing(
            ['uv', 'pip', 'install'] + packages,
            f"Installation uv: {' '.join(packages)}",
            cwd=uv_env
        )
        results['uv'] = {'duration': duration, 'success': success}

        return results

    def benchmark_dependency_resolution(self, temp_dir: Path) -> Dict[str, Dict]:
        """Benchmark la résolution de dépendances complexes."""
        self.log(f"\n🧩 BENCHMARK: Résolution de dépendances complexes", force=True)

        # Packages avec dépendances complexes
        complex_packages = [
            'pandas', 'scikit-learn', 'tensorflow',
            'fastapi', 'streamlit', 'plotly'
        ]

        results = {}

        # Test avec pip
        pip_env = temp_dir / "pip_complex"
        subprocess.run([sys.executable, '-m', 'venv', str(pip_env)],
                      capture_output=True)

        pip_python = pip_env / ('Scripts' if sys.platform == 'win32' else 'bin') / 'python'
        duration, success = self.run_command_with_timing(
            [str(pip_python), '-m', 'pip', 'install'] + complex_packages,
            "Résolution pip (packages complexes)",
            timeout=600  # 10 minutes max
        )
        results['pip'] = {'duration': duration, 'success': success}

        # Test avec uv
        uv_env = temp_dir / "uv_complex"
        subprocess.run(['uv', 'venv', str(uv_env)], capture_output=True)

        duration, success = self.run_command_with_timing(
            ['uv', 'pip', 'install'] + complex_packages,
            "Résolution uv (packages complexes)",
            cwd=uv_env,
            timeout=600
        )
        results['uv'] = {'duration': duration, 'success': success}

        return results

    def run_comprehensive_benchmark(self, packages: List[str] = None) -> Dict:
        """Lance un benchmark complet."""
        if packages is None:
            packages = ['pandas', 'numpy', 'scikit-learn', 'fastapi']

        print("🚀 " + "="*60)
        print("🚀 BENCHMARK COMPREHENSIVE PIP vs UV")
        print("🚀 Master 2 - Data Science Industrielle")
        print("🚀 " + "="*60)

        # Vérifier disponibilité
        managers = self.check_manager_availability()
        if not managers.get('pip'):
            print("❌ pip non disponible, arrêt du benchmark")
            return {}
        if not managers.get('uv'):
            print("❌ uv non disponible, arrêt du benchmark")
            print("💡 Installation: curl -LsSf https://astral.sh/uv/install.sh | sh")
            return {}

        # Créer répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            all_results = {
                'environment_creation': self.benchmark_environment_creation(temp_path),
                'package_installation': self.benchmark_package_installation(temp_path, packages),
                'dependency_resolution': self.benchmark_dependency_resolution(temp_path)
            }

            # Analyse des résultats
            self.analyze_results(all_results)

            return all_results

    def analyze_results(self, results: Dict):
        """Analyse et affiche les résultats du benchmark."""
        print("\n📊 " + "="*60)
        print("📊 ANALYSE DES RÉSULTATS")
        print("📊 " + "="*60)

        for category, data in results.items():
            print(f"\n🎯 {category.upper().replace('_', ' ')}")
            print("-" * 40)

            # Extraire les durées
            pip_time = None
            uv_time = None

            for manager, metrics in data.items():
                if 'pip' in manager and metrics['success']:
                    pip_time = metrics['duration']
                    print(f"🐌 pip:  {pip_time:.2f}s")
                elif 'uv' in manager and metrics['success']:
                    uv_time = metrics['duration']
                    print(f"⚡ uv:   {uv_time:.2f}s")

            # Calcul du gain
            if pip_time and uv_time and pip_time > 0:
                speedup = pip_time / uv_time
                time_saved = pip_time - uv_time

                print(f"🚀 Gain: {speedup:.1f}x plus rapide")
                print(f"⏰ Économie: {time_saved:.2f}s")

                if speedup > 10:
                    print("🏆 GAIN EXCEPTIONNEL!")
                elif speedup > 5:
                    print("✨ TRÈS BONNE AMÉLIORATION")
                elif speedup > 2:
                    print("👍 AMÉLIORATION NOTABLE")

        # Résumé global
        self.print_global_summary(results)

    def print_global_summary(self, results: Dict):
        """Affiche un résumé global des performances."""
        print(f"\n🌟 " + "="*60)
        print("🌟 RÉSUMÉ GLOBAL")
        print("🌟 " + "="*60)

        total_pip_time = 0
        total_uv_time = 0
        successful_comparisons = 0

        for category, data in results.items():
            pip_time = None
            uv_time = None

            for manager, metrics in data.items():
                if 'pip' in manager and metrics['success']:
                    pip_time = metrics['duration']
                elif 'uv' in manager and metrics['success']:
                    uv_time = metrics['duration']

            if pip_time and uv_time:
                total_pip_time += pip_time
                total_uv_time += uv_time
                successful_comparisons += 1

        if successful_comparisons > 0:
            overall_speedup = total_pip_time / total_uv_time
            time_saved = total_pip_time - total_uv_time

            print(f"⏱️  Temps total pip: {total_pip_time:.2f}s")
            print(f"⚡ Temps total uv:  {total_uv_time:.2f}s")
            print(f"🚀 Accélération globale: {overall_speedup:.1f}x")
            print(f"⏰ Temps économisé: {time_saved:.2f}s")

            # Projection annuelle
            daily_ops = 5  # Opérations par jour
            yearly_days = 250  # Jours de travail par an
            yearly_savings = time_saved * daily_ops * yearly_days / 3600  # en heures

            print(f"\n💰 PROJECTION ANNUELLE:")
            print(f"   📅 {yearly_days} jours × {daily_ops} opérations/jour")
            print(f"   ⏰ Économie: {yearly_savings:.1f} heures par an")
            print(f"   💰 Valeur (25€/h): {yearly_savings * 25:.0f}€")

            print(f"\n🎓 IMPACT PÉDAGOGIQUE:")
            print("   • Plus de temps pour apprendre le ML")
            print("   • Environnements plus reproductibles")
            print("   • Workflows modernes et efficaces")
            print("   • Préparation pour l'industrie")

def main():
    parser = argparse.ArgumentParser(
        description="🚀 Benchmark pip vs uv pour data scientists",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python benchmark_managers.py
  python benchmark_managers.py --packages pandas,numpy,sklearn
  python benchmark_managers.py --verbose
  python benchmark_managers.py --packages fastapi,streamlit --verbose
        """
    )

    parser.add_argument(
        '--packages',
        type=str,
        help='Packages à installer (séparés par virgule)',
        default='pandas,numpy,scikit-learn,fastapi'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mode verbeux avec détails'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Fichier JSON pour sauvegarder les résultats'
    )

    args = parser.parse_args()

    # Préparer les packages
    packages = [pkg.strip() for pkg in args.packages.split(',')]

    # Lancer le benchmark
    benchmarker = PackageManagerBenchmark(verbose=args.verbose)
    results = benchmarker.run_comprehensive_benchmark(packages)

    # Sauvegarder les résultats si demandé
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Résultats sauvés dans {args.output}")

    if results:
        print(f"\n🎉 Benchmark terminé ! UV montre sa supériorité.")
        print(f"💡 Adoptez UV dès maintenant pour vos projets ML !")
    else:
        print(f"\n⚠️ Benchmark incomplet. Vérifiez l'installation d'UV.")

if __name__ == "__main__":
    main()