# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - SCRIPTS POWERSHELL
# Master 2 - Data Science Industrielle
# ===============================================

<#
.SYNOPSIS
Scripts PowerShell pour automatiser les tâches avec UV sur Windows

.DESCRIPTION
Équivalent Windows du Makefile, optimisé pour UV (le gestionnaire Python ultra-rapide)

.EXAMPLE
.\scripts.ps1 Help
.\scripts.ps1 Install
.\scripts.ps1 CompareManagers
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("Help", "InstallUV", "Install", "InstallDev", "Setup", "Test", "TestAll", "TestUnit",
                 "TestIntegration", "TestE2E", "TestCoverage", "Lint", "Format", "Train", "RunAPI",
                 "RunFrontend", "Demo", "CompareManagers", "ShowAdvantages", "Info", "Clean",
                 "StudentSetup", "LearningPath")]
    [string]$Action = "Help"
)

# Couleurs pour l'affichage
$Colors = @{
    Red = "`e[31m"
    Green = "`e[32m"
    Yellow = "`e[33m"
    Blue = "`e[34m"
    Magenta = "`e[35m"
    Cyan = "`e[36m"
    Reset = "`e[0m"
}

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = "Reset"
    )
    Write-Host "$($Colors[$Color])$Message$($Colors.Reset)"
}

function Test-UVInstalled {
    try {
        $uvVersion = uv --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "✅ UV installé : $uvVersion" "Green"
            return $true
        }
    }
    catch {
        Write-ColoredOutput "❌ UV non installé" "Red"
        return $false
    }
    return $false
}

function Install-UV {
    Write-ColoredOutput "🌟 Installation d'UV - Le futur de la gestion Python" "Cyan"

    if (Test-UVInstalled) {
        Write-ColoredOutput "✅ UV déjà installé" "Green"
        return
    }

    Write-ColoredOutput "📥 Installation d'UV en cours..." "Yellow"
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        Write-ColoredOutput "✅ UV installé avec succès !" "Green"
        # Redémarrer la session pour que UV soit dans le PATH
        Write-ColoredOutput "💡 Redémarrez PowerShell pour utiliser UV" "Yellow"
    }
    catch {
        Write-ColoredOutput "❌ Erreur lors de l'installation : $($_.Exception.Message)" "Red"
        Write-ColoredOutput "📦 Installation manuelle : irm https://astral.sh/uv/install.ps1 | iex" "Yellow"
    }
}

function Assert-UVInstalled {
    if (-not (Test-UVInstalled)) {
        Write-ColoredOutput "❌ UV n'est pas installé" "Red"
        Write-ColoredOutput "💡 Utilisez: .\scripts.ps1 InstallUV" "Yellow"
        exit 1
    }
}

function Install-Dependencies {
    Assert-UVInstalled
    Write-ColoredOutput "⚡ Installation des dépendances avec UV" "Cyan"
    uv sync
    if ($LASTEXITCODE -eq 0) {
        Write-ColoredOutput "✅ Installation terminée !" "Green"
    } else {
        Write-ColoredOutput "❌ Erreur lors de l'installation" "Red"
    }
}

function Install-DevDependencies {
    Assert-UVInstalled
    Write-ColoredOutput "🔧 Installation mode développement" "Cyan"
    uv sync --extra dev --extra test
    if ($LASTEXITCODE -eq 0) {
        Write-ColoredOutput "✅ Installation développement terminée !" "Green"
    } else {
        Write-ColoredOutput "❌ Erreur lors de l'installation" "Red"
    }
}

function Setup-Project {
    Write-ColoredOutput "🚀 Setup complet du projet" "Cyan"
    Install-UV
    Install-DevDependencies

    Write-ColoredOutput "📊 Génération des données de démo..." "Yellow"
    uv run python scripts/generate_demo_data.py

    if ($LASTEXITCODE -eq 0) {
        Write-ColoredOutput "✅ Setup terminé ! Prêt à coder !" "Green"
    }
}

function Run-Tests {
    param([string]$TestType = "fast")

    Assert-UVInstalled

    switch ($TestType) {
        "fast" {
            Write-ColoredOutput "🧪 Tests rapides" "Cyan"
            uv run pytest tests/ -m "not slow" -v
        }
        "all" {
            Write-ColoredOutput "🧪 Tous les tests" "Cyan"
            uv run pytest tests/ -v
        }
        "unit" {
            Write-ColoredOutput "🔬 Tests unitaires" "Cyan"
            uv run pytest tests/unit/ -v
        }
        "integration" {
            Write-ColoredOutput "🔗 Tests d'intégration" "Cyan"
            uv run pytest tests/integration/ -v
        }
        "e2e" {
            Write-ColoredOutput "🎪 Tests end-to-end" "Cyan"
            uv run pytest tests/e2e/ -v
        }
        "coverage" {
            Write-ColoredOutput "📊 Tests avec couverture" "Cyan"
            uv run pytest tests/ --cov=ml_pipeline --cov=backend --cov=frontend --cov-report=html --cov-report=term-missing
            Write-ColoredOutput "📊 Rapport disponible dans htmlcov/index.html" "Green"
        }
    }
}

function Format-Code {
    Assert-UVInstalled
    Write-ColoredOutput "✨ Formatage automatique du code" "Cyan"
    uv run black .
    uv run ruff format .
    uv run isort .
    Write-ColoredOutput "✅ Code formaté automatiquement" "Green"
}

function Lint-Code {
    Assert-UVInstalled
    Write-ColoredOutput "🔍 Analyse de la qualité du code" "Cyan"
    uv run ruff check .
    uv run mypy ml_pipeline/ backend/ frontend/
}

function Train-Model {
    Assert-UVInstalled
    Write-ColoredOutput "🤖 Entraînement du modèle de recommandation" "Cyan"
    uv run python ml_pipeline/train_model.py --use-demo-data
}

function Run-API {
    Assert-UVInstalled
    Write-ColoredOutput "🚀 Démarrage de l'API FastAPI" "Cyan"
    Write-ColoredOutput "💡 API disponible sur http://localhost:8000" "Yellow"
    Write-ColoredOutput "📚 Documentation sur http://localhost:8000/docs" "Yellow"
    uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
}

function Run-Frontend {
    Assert-UVInstalled
    Write-ColoredOutput "🎨 Démarrage de l'interface Streamlit" "Cyan"
    Write-ColoredOutput "💡 Interface disponible sur http://localhost:8501" "Yellow"
    uv run streamlit run frontend/app.py
}

function Run-Demo {
    Assert-UVInstalled
    Write-ColoredOutput "🎪 Démonstration complète du système" "Cyan"

    Write-ColoredOutput "1️⃣ Génération des données..." "Yellow"
    uv run python scripts/generate_demo_data.py

    Write-ColoredOutput "2️⃣ Entraînement du modèle..." "Yellow"
    uv run python ml_pipeline/train_model.py --use-demo-data

    Write-ColoredOutput "3️⃣ Tests de validation..." "Yellow"
    uv run pytest tests/test_demo.py -v

    Write-ColoredOutput "🎉 Démonstration terminée !" "Green"
    Write-ColoredOutput "💡 Lancez '.\scripts.ps1 RunAPI' et '.\scripts.ps1 RunFrontend'" "Magenta"
}

function Compare-Managers {
    Write-ColoredOutput "⚔️ COMPARAISON PIP vs UV" "Cyan"
    Write-ColoredOutput "Cette démo montre l'avantage d'UV sur pip" "Yellow"
    python setup_uv.py --compare
}

function Show-UVAdvantages {
    Write-ColoredOutput "💡 Avantages d'UV pour les data scientists" "Cyan"
    python setup_uv.py --advantages
}

function Show-Info {
    Assert-UVInstalled
    Write-ColoredOutput "📊 Informations sur l'environnement" "Cyan"

    Write-ColoredOutput "Python version:" "Yellow"
    uv run python --version

    Write-ColoredOutput "UV version:" "Yellow"
    uv --version

    Write-ColoredOutput "Paquets installés (top 10):" "Yellow"
    uv pip list | Select-Object -First 10
}

function Clean-Project {
    Write-ColoredOutput "🧹 Nettoyage des fichiers temporaires" "Cyan"

    # Supprimer les répertoires de cache
    $dirs = @("__pycache__", ".pytest_cache", "htmlcov", "*.egg-info", "build", "dist")
    foreach ($dir in $dirs) {
        Get-ChildItem -Path . -Recurse -Directory -Name $dir -ErrorAction SilentlyContinue |
            ForEach-Object { Remove-Item $_ -Recurse -Force }
    }

    # Supprimer les fichiers temporaires
    Get-ChildItem -Path . -Recurse -Name "*.pyc" -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item $_ -Force }

    if (Test-Path ".coverage") { Remove-Item ".coverage" -Force }

    Write-ColoredOutput "✅ Nettoyage terminé" "Green"
}

function Setup-Student {
    Write-ColoredOutput "🎓 Setup optimisé pour étudiants" "Cyan"

    Write-ColoredOutput "1️⃣ Installation d'UV..." "Yellow"
    Install-UV

    Write-ColoredOutput "2️⃣ Installation des dépendances..." "Yellow"
    Install-Dependencies

    Write-ColoredOutput "3️⃣ Génération des données..." "Yellow"
    uv run python scripts/generate_demo_data.py

    Write-ColoredOutput "4️⃣ Test rapide..." "Yellow"
    uv run pytest tests/test_demo.py::test_demo_basic -v

    Write-ColoredOutput "🎉 Prêt à apprendre ! 📚" "Green"
}

function Show-LearningPath {
    Write-ColoredOutput "📚 Parcours d'apprentissage UV + ML" "Cyan"
    Write-ColoredOutput "1. Comprendre UV: .\scripts.ps1 ShowAdvantages" "Yellow"
    Write-ColoredOutput "2. Comparer avec pip: .\scripts.ps1 CompareManagers" "Yellow"
    Write-ColoredOutput "3. Setup projet: .\scripts.ps1 StudentSetup" "Yellow"
    Write-ColoredOutput "4. Entraîner modèle: .\scripts.ps1 Train" "Yellow"
    Write-ColoredOutput "5. Tester l'API: .\scripts.ps1 RunAPI" "Yellow"
    Write-ColoredOutput "6. Interface utilisateur: .\scripts.ps1 RunFrontend" "Yellow"
    Write-ColoredOutput "7. Tests qualité: .\scripts.ps1 TestCoverage" "Yellow"
    Write-ColoredOutput "💡 Suivez ces étapes pour une expérience complète !" "Magenta"
}

function Show-Help {
    Write-ColoredOutput "🚀 Olist Recommendation System - Scripts PowerShell" "Cyan"
    Write-ColoredOutput "Master 2 - Data Science Industrielle" "Cyan"
    Write-Output ""

    Write-ColoredOutput "📦 Installation & Setup:" "Yellow"
    Write-Output "  InstallUV          🌟 Installe UV (gestionnaire ultra-rapide)"
    Write-Output "  Install            ⚡ Installation rapide avec UV"
    Write-Output "  InstallDev         🔧 Installation développement"
    Write-Output "  Setup              🚀 Setup complet du projet"
    Write-Output "  Clean              🧹 Nettoie les fichiers temporaires"
    Write-Output ""

    Write-ColoredOutput "🧪 Tests & Qualité:" "Yellow"
    Write-Output "  Test               🧪 Tests rapides"
    Write-Output "  TestAll            🧪 Tous les tests"
    Write-Output "  TestUnit           🔬 Tests unitaires"
    Write-Output "  TestIntegration    🔗 Tests d'intégration"
    Write-Output "  TestE2E            🎪 Tests end-to-end"
    Write-Output "  TestCoverage       📊 Tests avec couverture"
    Write-Output "  Lint               🔍 Analyse du code"
    Write-Output "  Format             ✨ Formatage automatique"
    Write-Output ""

    Write-ColoredOutput "🚀 Exécution:" "Yellow"
    Write-Output "  Train              🤖 Entraîne le modèle"
    Write-Output "  RunAPI             🚀 Lance l'API FastAPI"
    Write-Output "  RunFrontend        🎨 Lance Streamlit"
    Write-Output "  Demo               🎪 Démonstration complète"
    Write-Output ""

    Write-ColoredOutput "💡 Pédagogique:" "Yellow"
    Write-Output "  CompareManagers    ⚔️ Compare pip vs uv"
    Write-Output "  ShowAdvantages     💡 Avantages d'UV"
    Write-Output "  StudentSetup       🎓 Setup étudiant"
    Write-Output "  LearningPath       📚 Parcours d'apprentissage"
    Write-Output "  Info               📊 Informations système"
    Write-Output ""

    Write-ColoredOutput "💡 Utilisation: .\scripts.ps1 <Commande>" "Magenta"
}

# ==========================================
# 🎯 ROUTER PRINCIPAL
# ==========================================

switch ($Action) {
    "Help" { Show-Help }
    "InstallUV" { Install-UV }
    "Install" { Install-Dependencies }
    "InstallDev" { Install-DevDependencies }
    "Setup" { Setup-Project }
    "Test" { Run-Tests "fast" }
    "TestAll" { Run-Tests "all" }
    "TestUnit" { Run-Tests "unit" }
    "TestIntegration" { Run-Tests "integration" }
    "TestE2E" { Run-Tests "e2e" }
    "TestCoverage" { Run-Tests "coverage" }
    "Lint" { Lint-Code }
    "Format" { Format-Code }
    "Train" { Train-Model }
    "RunAPI" { Run-API }
    "RunFrontend" { Run-Frontend }
    "Demo" { Run-Demo }
    "CompareManagers" { Compare-Managers }
    "ShowAdvantages" { Show-UVAdvantages }
    "Info" { Show-Info }
    "Clean" { Clean-Project }
    "StudentSetup" { Setup-Student }
    "LearningPath" { Show-LearningPath }
    default { Show-Help }
}