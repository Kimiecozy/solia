# 🧪 Guide des Tests - Olist Recommendation System

**Guide complet pour comprendre et utiliser les tests du projet**

---

## 🎯 Vue d'ensemble des Tests

Ce projet implémente une **stratégie de test complète** avec trois niveaux :

| Type de Test | Objectif | Portée | Vitesse |
|--------------|----------|---------|---------|
| **🔬 Tests Unitaires** | Tester des composants individuels | Functions, classes | ⚡ Très rapide |
| **🔗 Tests d'Intégration** | Tester l'interaction entre composants | Modules, services | 🚀 Rapide |
| **🎪 Tests End-to-End** | Tester des workflows complets | Application complète | 🐌 Lent |

### 📊 Couverture de Test

```
tests/
├── 📁 unit/                    # Tests unitaires
│   ├── test_feature_engineering.py    # Feature engineering
│   └── test_recommendation_model.py   # Modèle ML
├── 📁 integration/             # Tests d'intégration
│   ├── test_api_integration.py         # API + Services
│   └── test_ml_pipeline_integration.py # Pipeline ML complet
├── 📁 e2e/                     # Tests end-to-end
│   └── test_user_workflows.py          # Workflows utilisateur
├── conftest.py                 # Configuration commune
└── README_TESTS.md            # Ce guide
```

---

## 🚀 Lancer les Tests

### 🎮 Méthodes Rapides

```bash
# 1. Tous les tests rapides (recommandé pour développement)
python run_tests.py

# 2. Tests unitaires uniquement (très rapide)
python run_tests.py --unit

# 3. Tests d'intégration uniquement
python run_tests.py --integration

# 4. Tests end-to-end (lent)
python run_tests.py --e2e

# 5. Tous les tests (incluant les lents)
python run_tests.py --all
```

### ⚙️ Options Avancées

```bash
# Tests avec couverture de code
python run_tests.py --coverage

# Tests en mode verbeux
python run_tests.py --verbose

# Tests en parallèle (si pytest-xdist installé)
python run_tests.py --parallel

# Générer rapport HTML
python run_tests.py --report
```

### 🐍 Commandes pytest Directes

```bash
# Tests par marker
pytest -m unit                 # Tests unitaires
pytest -m integration         # Tests d'intégration
pytest -m "not slow"          # Tests rapides seulement

# Tests par répertoire
pytest tests/unit/            # Répertoire spécifique
pytest tests/integration/test_api_integration.py  # Fichier spécifique

# Tests par fonction
pytest tests/unit/test_feature_engineering.py::test_customer_feature_engineer_basic
```

---

## 🔬 Tests Unitaires

### 📋 Objectif

Les tests unitaires vérifient que **chaque composant fonctionne correctement de manière isolée**.

### 🎯 Ce qui est testé

- **Feature Engineering** : Transformations de données, calculs RFM
- **Modèle ML** : Entraînement, prédictions, métriques
- **Fonctions utilitaires** : Helpers, validations

### 💡 Exemple de Test Unitaire

```python
def test_customer_feature_engineer_basic(sample_customer_features):
    """
    Test: La transformation des features clients fonctionne.
    """
    # ARRANGE
    engineer = CustomerFeatureEngineer()

    # ACT
    result = engineer.fit_transform(sample_customer_features)

    # ASSERT
    assert not result.empty
    assert 'avg_order_value' in result.columns
    assert all(result['avg_order_value'] > 0)
```

### 🎨 Bonnes Pratiques

- **AAA Pattern** : Arrange (préparer), Act (agir), Assert (vérifier)
- **Isolation** : Chaque test est indépendant
- **Mocking** : Utiliser des mocks pour les dépendances externes
- **Données de test** : Utiliser les fixtures `conftest.py`

---

## 🔗 Tests d'Intégration

### 📋 Objectif

Les tests d'intégration vérifient que **plusieurs composants fonctionnent bien ensemble**.

### 🎯 Ce qui est testé

- **API + Services** : Routes FastAPI avec services ML
- **Pipeline ML** : Feature engineering + Modèle + Prédictions
- **Base de données** : Intégration avec les données
- **Cache** : Fonctionnement du système de cache

### 💡 Exemple de Test d'Intégration

```python
@pytest.mark.asyncio
async def test_recommendations_endpoint_full_flow(api_client, initialized_service):
    """
    Test: Le flux complet de génération de recommandations via l'API.

    Teste l'intégration : Route → Validation → Service → Modèle → Réponse
    """
    # ARRANGE
    request_data = {
        "customer_id": "test_customer_001",
        "n_recommendations": 5
    }

    # ACT
    response = api_client.post("/api/v1/recommendations", json=request_data)

    # ASSERT
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 5
```

### 🔧 Technologies Utilisées

- **TestClient FastAPI** : Pour tester l'API sans démarrer un serveur
- **Mocks** : Pour simuler les services externes
- **Fixtures** : Pour préparer les données d'intégration

---

## 🎪 Tests End-to-End

### 📋 Objectif

Les tests E2E vérifient que **l'application complète fonctionne du point de vue utilisateur**.

### 🎯 Ce qui est testé

- **Workflows complets** : De la génération de données aux recommandations
- **Performance** : Temps de réponse, charge
- **Expérience utilisateur** : Scénarios réalistes

### 💡 Exemple de Test E2E

```python
def test_complete_ml_workflow(isolated_environment):
    """
    Test: Workflow complet d'un data scientist.

    Scénario :
    1. Génération des données
    2. Entraînement du modèle
    3. Génération de recommandations
    4. Vérification des résultats
    """
    # 1. Générer des données
    self._generate_test_data(data_dir)

    # 2. Entraîner le modèle
    performance = self._train_model_with_data(data_dir)

    # 3. Tester les prédictions
    recommendations = self._test_model_predictions()

    # ASSERT
    assert performance['auc_score'] > 0.5
    assert len(recommendations) > 0
```

### ⏱️ Considérations

- **Lenteur** : Ces tests prennent plus de temps
- **Environnement isolé** : Utilisent des répertoires temporaires
- **Réalisme** : Simulent de vrais scénarios d'usage

---

## 🛠️ Configuration des Tests

### 📄 Fichiers de Configuration

#### `pytest.ini`
```ini
[tool:pytest]
testpaths = tests
markers =
    unit: Tests unitaires
    integration: Tests d'intégration
    e2e: Tests end-to-end
    slow: Tests lents
```

#### `conftest.py`
```python
# Fixtures communes à tous les tests
@pytest.fixture
def sample_customer_features():
    """Données clients pour les tests."""
    return pd.DataFrame({...})

@pytest.fixture
def api_client():
    """Client FastAPI pour les tests."""
    return TestClient(app)
```

### 🎯 Fixtures Disponibles

| Fixture | Description | Usage |
|---------|-------------|-------|
| `sample_customer_features` | Données clients factices | Tests de feature engineering |
| `sample_products_data` | Données produits factices | Tests de preprocessing |
| `mock_trained_model` | Modèle ML mocké | Tests de prédictions |
| `api_client` | Client API de test | Tests d'intégration API |
| `isolated_environment` | Environnement isolé | Tests E2E |

---

## 📊 Analyse de la Couverture

### 🚀 Lancer avec Couverture

```bash
# Générer le rapport de couverture
python run_tests.py --coverage

# Voir le rapport dans le navigateur
open htmlcov/index.html
```

### 🎯 Objectifs de Couverture

| Composant | Objectif | Status |
|-----------|----------|---------|
| **Feature Engineering** | > 90% | ✅ |
| **Modèle ML** | > 85% | ✅ |
| **API Routes** | > 95% | ✅ |
| **Services** | > 80% | ⚠️ |

### 📈 Interpréter les Résultats

- **Lignes vertes** : Code testé
- **Lignes rouges** : Code non testé
- **Pourcentage** : % de lignes couvertes
- **Branches** : Conditions if/else testées

---

## 🐛 Debugging des Tests

### 🔍 Tests qui Échouent

```bash
# Mode verbeux pour plus d'infos
pytest -v --tb=long

# Arrêter au premier échec
pytest -x

# Débugger interactivement
pytest --pdb

# Logs détaillés
pytest -s --log-cli-level=DEBUG
```

### 💡 Stratégies de Debug

1. **Lire le message d'erreur** : Souvent très informatif
2. **Vérifier les fixtures** : Données de test correctes ?
3. **Isolation** : Le test fonctionne-t-il seul ?
4. **Mocks** : Les mocks sont-ils configurés correctement ?
5. **Environnement** : Variables d'env, paths, dépendances ?

### 🚨 Erreurs Courantes

| Erreur | Cause | Solution |
|--------|-------|----------|
| `ModuleNotFoundError` | PYTHONPATH incorrect | Vérifier les imports |
| `FixtureNotFound` | Fixture non importée | Vérifier `conftest.py` |
| `AsyncioError` | Test async mal configuré | Utiliser `@pytest.mark.asyncio` |
| `FileNotFound` | Chemins de test incorrects | Vérifier les fixtures de données |

---

## 🎓 Pour les Étudiants

### 📚 Concepts Pédagogiques

#### **Test-Driven Development (TDD)**
1. Écrire le test en premier (qui échoue)
2. Écrire le code minimal pour faire passer le test
3. Refactoriser le code
4. Répéter

#### **Patterns de Test**

```python
# 1. AAA Pattern
def test_example():
    # ARRANGE - Préparer les données
    data = create_test_data()

    # ACT - Exécuter la fonction
    result = process_data(data)

    # ASSERT - Vérifier le résultat
    assert result.is_valid()

# 2. Given-When-Then
def test_customer_segmentation():
    # GIVEN un client avec 5 commandes
    customer = Customer(orders=5)

    # WHEN on calcule son segment
    segment = calculate_segment(customer)

    # THEN il devrait être "Frequent"
    assert segment == "Frequent"
```

### 🏋️ Exercices Proposés

#### **Débutant**
1. **Modifier un test existant** : Changer les assertions
2. **Ajouter des cas de test** : Edge cases, données invalides
3. **Debugger un test qui échoue** : Comprendre pourquoi

#### **Intermédiaire**
4. **Créer un nouveau test unitaire** : Pour une nouvelle fonction
5. **Écrire un test d'intégration** : Entre deux composants
6. **Optimiser les fixtures** : Réutilisabilité, performance

#### **Avancé**
7. **Test de performance** : Mesurer temps d'exécution
8. **Test de régression** : Éviter les régressions
9. **Tests paramétrés** : Plusieurs cas avec une fonction

### 📝 Exemple d'Exercice Complet

**Exercice : Tester une nouvelle métrique ML**

```python
# 1. Créer la fonction (dans ml_pipeline/evaluation/metrics.py)
def precision_at_k(recommendations, actual_purchases, k=5):
    """Calcule la précision@K pour les recommandations."""
    top_k_recs = recommendations[:k]
    relevant = len(set(top_k_recs) & set(actual_purchases))
    return relevant / k if k > 0 else 0

# 2. Écrire le test (dans tests/unit/test_metrics.py)
def test_precision_at_k():
    # ARRANGE
    recommendations = ['prod_1', 'prod_2', 'prod_3', 'prod_4', 'prod_5']
    actual_purchases = ['prod_1', 'prod_3', 'prod_6']

    # ACT
    precision = precision_at_k(recommendations, actual_purchases, k=5)

    # ASSERT
    assert precision == 0.4  # 2 relevant sur 5 = 0.4

# 3. Tester les edge cases
def test_precision_at_k_edge_cases():
    assert precision_at_k([], [], k=5) == 0
    assert precision_at_k(['prod_1'], ['prod_1'], k=1) == 1.0
    assert precision_at_k(['prod_1'], ['prod_2'], k=1) == 0.0
```

---

## 📈 Métriques et Monitoring

### 🎯 KPIs de Qualité des Tests

- **Couverture de code** : > 80%
- **Temps d'exécution** : Tests rapides < 30s
- **Taux de réussite** : > 95%
- **Maintenance** : Tests maintenus avec le code

### 📊 Rapports Générés

```bash
# Rapport HTML complet
pytest --html=reports/tests.html --self-contained-html

# Métriques de performance
pytest --durations=10

# Export des résultats
pytest --junitxml=reports/junit.xml
```

---

## 🚀 Prochaines Étapes

### 🔧 Améliorations Possibles

1. **Tests de charge** : pytest-benchmark
2. **Tests de sécurité** : bandit, safety
3. **Tests de mutation** : mutmut
4. **CI/CD Integration** : GitHub Actions
5. **Tests visuels** : Streamlit app testing

### 📚 Ressources pour Approfondir

- **[pytest Documentation](https://docs.pytest.org/)** : Guide officiel
- **[Testing in Python](https://realpython.com/python-testing/)** : Tutoriel complet
- **[Test-Driven Development](https://testdriven.io/)** : Pratiques avancées
- **[FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)** : Tests API

---

## 🎉 Conclusion

Les tests sont **essentiels** pour un projet ML en production :

- **🔒 Fiabilité** : Garantir que le code fonctionne
- **🚀 Confiance** : Déployer sans crainte
- **🔧 Maintenance** : Faciliter les évolutions
- **📊 Documentation** : Montrer comment utiliser le code

**Bonne pratique** : Toujours écrire des tests pour le nouveau code, et maintenir les tests existants !

---

*Dernière mise à jour : Janvier 2025*
*Guide créé pour le Master 2 Data Science Industrielle* 🎓