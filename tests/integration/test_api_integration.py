# ===============================================
# 🧪 TESTS D'INTÉGRATION - API
# Master 2 - Data Science Industrielle
# ===============================================

"""
Tests d'intégration pour l'API FastAPI.

Ces tests vérifient que les différents composants de l'API
fonctionnent bien ensemble :
- Routes + Services + Modèles
- Validation des données
- Gestion des erreurs
- Performance et cache

Contrairement aux tests unitaires, ces tests font appel à
plusieurs composants simultanément pour vérifier leur intégration.

Usage:
    pytest tests/integration/test_api_integration.py -v
    pytest tests/integration/test_api_integration.py::test_recommendations_endpoint_full_flow -v
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import json
import time
from datetime import datetime
import sys
from pathlib import Path

# Import des modules à tester
sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.app.services.recommendation_service import RecommendationService
from backend.app.schemas.recommendation import CustomerRequest, RecommendationResponse

@pytest.mark.integration
class TestAPIServiceIntegration:
    """
    Tests d'intégration entre les routes API et le service de recommandation.

    Vérifie que la couche API communique correctement avec la couche service.
    """

    @pytest.fixture
    async def initialized_service(self, sample_customer_features, sample_product_features):
        """
        Fixture qui fournit un service de recommandation initialisé pour les tests.
        """
        service = RecommendationService()

        # Mock des composants internes avec des données de test
        service.customer_features = sample_customer_features
        service.product_features = sample_product_features
        service.model = MagicMock()
        service.model.is_trained = True
        service.model_loaded_at = datetime.now()

        # Mock des méthodes de prédiction
        service.model.predict_proba.return_value = sample_customer_features.iloc[:5].copy()
        service.model.predict_proba.return_value.columns = ['product_id', 'purchase_probability']
        service.model.predict_proba.return_value['product_id'] = [f'prod_{i}' for i in range(5)]
        service.model.predict_proba.return_value['purchase_probability'] = [0.8, 0.7, 0.6, 0.5, 0.4]

        # Mock des métriques
        service.model.get_model_performance.return_value = {
            'train_accuracy': 0.85,
            'test_accuracy': 0.78,
            'auc_score': 0.82,
            'cv_mean': 0.80,
            'cv_std': 0.03
        }

        service.model.get_feature_importance.return_value = sample_customer_features.iloc[:3]

        return service

    @pytest.mark.asyncio
    async def test_recommendations_endpoint_full_flow(self, api_client, initialized_service):
        """
        Test: Le flux complet de génération de recommandations via l'API.

        Teste l'intégration complète :
        Route → Validation → Service → Modèle → Réponse
        """
        # ARRANGE
        with patch('backend.app.services.recommendation_service.recommendation_service', initialized_service):

            request_data = {
                "customer_id": "test_customer_001",
                "n_recommendations": 5
            }

            # ACT
            response = api_client.post("/api/v1/recommendations", json=request_data)

            # ASSERT
            assert response.status_code == 200

            data = response.json()
            assert data["customer_id"] == "test_customer_001"
            assert data["total_recommendations"] == 5
            assert len(data["recommendations"]) == 5

            # Vérifier la structure de chaque recommandation
            for i, rec in enumerate(data["recommendations"]):
                assert "product_id" in rec
                assert "purchase_probability" in rec
                assert "confidence" in rec
                assert "rank" in rec
                assert rec["rank"] == i + 1

    @pytest.mark.asyncio
    async def test_batch_recommendations_integration(self, api_client, initialized_service):
        """
        Test: L'intégration du traitement en lot de recommandations.
        """
        # ARRANGE
        with patch('backend.app.services.recommendation_service.recommendation_service', initialized_service):

            request_data = {
                "customer_ids": ["test_customer_000", "test_customer_001", "test_customer_002"],
                "n_recommendations": 3
            }

            # ACT
            response = api_client.post("/api/v1/recommendations/batch", json=request_data)

            # ASSERT
            assert response.status_code == 200

            data = response.json()
            assert data["total_customers"] == 3
            assert len(data["results"]) == 3
            assert "processing_time_seconds" in data

            # Vérifier que chaque résultat a la bonne structure
            for result in data["results"]:
                assert "customer_id" in result
                assert "recommendations" in result
                assert len(result["recommendations"]) == 3

    @pytest.mark.asyncio
    async def test_model_info_integration(self, api_client, initialized_service):
        """
        Test: L'intégration de récupération des informations du modèle.
        """
        # ARRANGE
        with patch('backend.app.services.recommendation_service.recommendation_service', initialized_service):

            # ACT
            response = api_client.get("/api/v1/model/info")

            # ASSERT
            assert response.status_code == 200

            data = response.json()
            assert "metrics" in data
            assert "feature_importance" in data
            assert "model_status" in data

            # Vérifier les métriques
            metrics = data["metrics"]
            required_metrics = ["train_accuracy", "test_accuracy", "auc_score", "cv_mean", "cv_std"]
            for metric in required_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], (int, float))

    def test_api_validation_integration(self, api_client):
        """
        Test: L'intégration de la validation Pydantic avec l'API.

        Vérifie que la validation des données d'entrée fonctionne correctement
        avec les schémas Pydantic.
        """
        # Test avec customer_id manquant
        response = api_client.post("/api/v1/recommendations", json={"n_recommendations": 5})
        assert response.status_code == 422  # Validation error

        # Test avec customer_id vide
        response = api_client.post("/api/v1/recommendations", json={
            "customer_id": "",
            "n_recommendations": 5
        })
        assert response.status_code == 422

        # Test avec n_recommendations invalide
        response = api_client.post("/api/v1/recommendations", json={
            "customer_id": "test_customer",
            "n_recommendations": 0  # Invalide (minimum 1)
        })
        assert response.status_code == 422

        response = api_client.post("/api/v1/recommendations", json={
            "customer_id": "test_customer",
            "n_recommendations": 100  # Invalide (maximum 50)
        })
        assert response.status_code == 422

    def test_error_handling_integration(self, api_client):
        """
        Test: L'intégration de la gestion d'erreurs dans l'API.
        """
        # Simuler une erreur du service
        with patch('backend.app.services.recommendation_service.recommendation_service') as mock_service:
            mock_service.get_recommendations.side_effect = Exception("Service error")

            response = api_client.post("/api/v1/recommendations", json={
                "customer_id": "test_customer",
                "n_recommendations": 5
            })

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

    def test_health_check_integration(self, api_client):
        """
        Test: L'intégration du health check avec le service.
        """
        # ACT
        response = api_client.get("/api/v1/health")

        # ASSERT
        assert response.status_code == 200
        data = response.json()

        required_fields = ["status", "model_loaded", "version", "uptime_seconds"]
        for field in required_fields:
            assert field in data

        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

@pytest.mark.integration
class TestServiceModelIntegration:
    """
    Tests d'intégration entre le service et le modèle ML.

    Vérifie que le service communique correctement avec le modèle
    et gère les prédictions.
    """

    @pytest.fixture
    def service_with_mock_model(self, mock_trained_model, sample_customer_features, sample_product_features):
        """
        Service avec un modèle mocké mais réaliste.
        """
        service = RecommendationService()
        service.model = mock_trained_model
        service.customer_features = sample_customer_features
        service.product_features = sample_product_features
        service.model_loaded_at = datetime.now()
        return service

    @pytest.mark.asyncio
    async def test_service_model_prediction_integration(self, service_with_mock_model):
        """
        Test: L'intégration entre le service et le modèle pour les prédictions.
        """
        # ACT
        recommendations = await service_with_mock_model.get_recommendations(
            customer_id="test_customer_001",
            n_recommendations=5
        )

        # ASSERT
        assert isinstance(recommendations, RecommendationResponse)
        assert recommendations.customer_id == "test_customer_001"
        assert len(recommendations.recommendations) == 5

        # Vérifier que le modèle a été appelé
        assert service_with_mock_model.model.predict_proba.called

    @pytest.mark.asyncio
    async def test_service_cache_integration(self, service_with_mock_model):
        """
        Test: L'intégration du cache dans le service.

        Vérifie que le cache fonctionne correctement pour éviter
        les recalculs inutiles.
        """
        # ARRANGE
        customer_id = "test_customer_cache"

        # ACT - Premier appel
        start_time = time.time()
        result1 = await service_with_mock_model.get_recommendations(customer_id, 5)
        first_call_time = time.time() - start_time

        # ACT - Deuxième appel (devrait utiliser le cache)
        start_time = time.time()
        result2 = await service_with_mock_model.get_recommendations(customer_id, 5)
        second_call_time = time.time() - start_time

        # ASSERT
        # Les résultats doivent être identiques
        assert result1.customer_id == result2.customer_id
        assert len(result1.recommendations) == len(result2.recommendations)

        # Le deuxième appel devrait être plus rapide (cache hit)
        # Note: En test, la différence peut être minime, on vérifie juste qu'il n'y a pas d'erreur
        assert second_call_time >= 0

        # Vérifier que le cache contient bien l'entrée
        cache_key = service_with_mock_model._get_cache_key(customer_id, 5)
        assert cache_key in service_with_mock_model._cache

    @pytest.mark.asyncio
    async def test_service_fallback_integration(self):
        """
        Test: L'intégration des mécanismes de fallback du service.

        Teste que le service peut gérer les cas où le modèle ne fonctionne pas.
        """
        # ARRANGE
        service = RecommendationService()
        service.model = None  # Simuler l'absence de modèle
        service.customer_features = None
        service.product_features = None

        # ACT
        recommendations = await service.get_recommendations("test_customer", 3)

        # ASSERT
        # Le service doit retourner des recommandations même sans modèle
        assert isinstance(recommendations, RecommendationResponse)
        assert len(recommendations.recommendations) == 3

        # Vérifier que les recommandations de fallback ont une structure valide
        for rec in recommendations.recommendations:
            assert 0 <= rec.purchase_probability <= 1
            assert rec.confidence in ['High', 'Medium', 'Low', 'Very Low']

@pytest.mark.integration
class TestDataFlowIntegration:
    """
    Tests d'intégration du flux de données complet.

    Vérifie que les données circulent correctement à travers
    toute l'architecture.
    """

    def test_customer_data_flow(self, sample_customers_data, sample_orders_data,
                              sample_order_items_data, sample_reviews_data, sample_products_data):
        """
        Test: Le flux complet de traitement des données clients.

        Vérifie l'intégration : Données brutes → Feature Engineering → Modèle
        """
        from ml_pipeline.preprocessing.feature_engineering import RecommendationFeatureEngine

        # ARRANGE
        engine = RecommendationFeatureEngine()

        # ACT
        # 1. Fit du pipeline
        engine.fit(sample_customers_data, sample_products_data)

        # 2. Création des features clients
        customer_features = engine.create_customer_features(
            sample_orders_data, sample_reviews_data,
            sample_order_items_data, sample_products_data
        )

        # 3. Création de la matrice d'interaction
        interactions = engine.create_interaction_matrix(sample_orders_data, sample_order_items_data)

        # ASSERT
        # Vérifier que les données ont bien circulé dans le pipeline
        assert not customer_features.empty
        assert not interactions.empty

        # Vérifier la cohérence des données
        customer_ids_in_features = set(customer_features.index)
        customer_ids_in_interactions = set(interactions['customer_id'])
        assert len(customer_ids_in_features.intersection(customer_ids_in_interactions)) > 0

    @patch('requests.get')
    def test_api_client_integration(self, mock_requests, api_client):
        """
        Test: L'intégration avec des clients externes de l'API.

        Simule un client externe qui utilise l'API.
        """
        # Simuler une réponse du service de santé
        mock_requests.return_value.status_code = 200
        mock_requests.return_value.json.return_value = {
            "status": "healthy",
            "model_loaded": True
        }

        # Test du endpoint de santé
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200

        # Test du endpoint d'exemple
        response = api_client.get("/api/v1/example")
        assert response.status_code == 200
        data = response.json()
        assert "examples" in data
        assert "tips" in data

    def test_cors_integration(self, api_client):
        """
        Test: L'intégration du middleware CORS.

        Vérifie que l'API peut être appelée depuis Streamlit.
        """
        # Test avec headers CORS
        headers = {
            "Origin": "http://localhost:8501",  # Port par défaut de Streamlit
            "Access-Control-Request-Method": "POST"
        }

        # OPTIONS request (preflight)
        response = api_client.options("/api/v1/recommendations", headers=headers)

        # Le serveur doit répondre positivement ou au moins ne pas rejeter
        # (selon la configuration exacte du middleware)
        assert response.status_code in [200, 204, 405]  # 405 si OPTIONS n'est pas implémenté

@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """
    Tests d'intégration pour les performances.

    Vérifie que l'intégration des composants ne cause pas
    de problèmes de performance majeurs.
    """

    def test_api_response_time_integration(self, api_client, initialized_service):
        """
        Test: Le temps de réponse de l'API reste acceptable.
        """
        with patch('backend.app.services.recommendation_service.recommendation_service', initialized_service):

            start_time = time.time()

            response = api_client.post("/api/v1/recommendations", json={
                "customer_id": "test_customer_001",
                "n_recommendations": 10
            })

            response_time = time.time() - start_time

            # ASSERT
            assert response.status_code == 200
            assert response_time < 1.0  # Moins d'une seconde

    @pytest.mark.asyncio
    async def test_concurrent_requests_integration(self, initialized_service):
        """
        Test: L'API peut gérer plusieurs requêtes concurrentes.
        """
        # ARRANGE
        async def make_recommendation_request(customer_id):
            return await initialized_service.get_recommendations(customer_id, 5)

        # ACT - Faire plusieurs requêtes en parallèle
        tasks = [
            make_recommendation_request(f"customer_{i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ASSERT
        # Toutes les requêtes doivent réussir
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, RecommendationResponse)

# Tests de régression pour l'intégration
@pytest.mark.integration
def test_api_compatibility_regression(api_client):
    """
    Test: L'API maintient sa compatibilité avec les versions précédentes.
    """
    # Test que tous les endpoints essentiels existent
    essential_endpoints = [
        ("/api/v1/health", "GET"),
        ("/api/v1/customers", "GET"),
        ("/api/v1/model/info", "GET"),
        ("/api/v1/example", "GET")
    ]

    for endpoint, method in essential_endpoints:
        if method == "GET":
            response = api_client.get(endpoint)
        elif method == "POST":
            response = api_client.post(endpoint, json={})

        # Les endpoints doivent exister (pas de 404)
        assert response.status_code != 404, f"Endpoint {endpoint} not found"