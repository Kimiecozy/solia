# ===============================================
# 🧪 TESTS D'INTÉGRATION - PIPELINE ML
# Master 2 - Data Science Industrielle
# ===============================================

"""
Tests d'intégration pour le pipeline de Machine Learning complet.

Ces tests vérifient que tous les composants ML fonctionnent
correctement ensemble :
- Preprocessing → Feature Engineering → Model Training → Prediction
- Flux de données de bout en bout
- Cohérence des formats entre composants
- Performance du pipeline complet

Usage:
    pytest tests/integration/test_ml_pipeline_integration.py -v
    pytest tests/integration/test_ml_pipeline_integration.py::test_full_ml_pipeline -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
from unittest.mock import patch, MagicMock
import joblib
from datetime import datetime, timedelta

# Import des modules à tester
sys.path.append(str(Path(__file__).parent.parent.parent))
from ml_pipeline.preprocessing.feature_engineering import (
    RecommendationFeatureEngine,
    CustomerFeatureEngineer,
    ProductFeatureEngineer,
    load_and_prepare_data
)
from ml_pipeline.models.recommendation_model import (
    OlistRecommendationModel,
    RecommendationPipeline
)
from config import MLConfig

@pytest.mark.integration
class TestFullMLPipelineIntegration:
    """
    Tests d'intégration pour le pipeline ML complet.

    Vérifie que toutes les étapes du pipeline fonctionnent ensemble :
    Données → Preprocessing → Features → Training → Prediction
    """

    @pytest.fixture
    def complete_test_dataset(self, test_data_dir):
        """
        Fixture qui crée un dataset complet de test dans des fichiers CSV.
        """
        # Créer des données plus complètes pour le pipeline
        np.random.seed(42)

        # Clients
        n_customers = 20
        customers = pd.DataFrame({
            'customer_id': [f'customer_{i:03d}' for i in range(n_customers)],
            'customer_unique_id': [f'unique_{i:03d}' for i in range(n_customers)],
            'customer_zip_code_prefix': np.random.randint(10000, 99999, n_customers),
            'customer_city': np.random.choice(['São Paulo', 'Rio de Janeiro', 'Belo Horizonte'], n_customers),
            'customer_state': np.random.choice(['SP', 'RJ', 'MG'], n_customers)
        })

        # Produits
        n_products = 15
        categories = ['cama_mesa_banho', 'beleza_saude', 'esporte_lazer', 'informatica_acessorios', 'moveis_decoracao']
        products = pd.DataFrame({
            'product_id': [f'product_{i:03d}' for i in range(n_products)],
            'product_category_name': np.random.choice(categories, n_products),
            'product_weight_g': np.random.randint(100, 2000, n_products),
            'product_length_cm': np.random.randint(10, 50, n_products),
            'product_height_cm': np.random.randint(5, 30, n_products),
            'product_width_cm': np.random.randint(10, 40, n_products)
        })

        # Commandes
        n_orders = 50
        base_date = datetime(2023, 1, 1)
        orders = pd.DataFrame({
            'order_id': [f'order_{i:03d}' for i in range(n_orders)],
            'customer_id': np.random.choice(customers['customer_id'], n_orders),
            'order_status': np.random.choice(['delivered', 'shipped'], n_orders, p=[0.8, 0.2]),
            'order_purchase_timestamp': [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_orders)],
            'order_approved_at': [base_date + timedelta(days=np.random.randint(0, 365), hours=1) for _ in range(n_orders)],
            'order_delivered_customer_date': [base_date + timedelta(days=np.random.randint(7, 30)) for _ in range(n_orders)]
        })

        # Items de commande
        n_items = 80
        order_items = pd.DataFrame({
            'order_id': np.random.choice(orders['order_id'], n_items),
            'order_item_id': list(range(1, n_items + 1)),
            'product_id': np.random.choice(products['product_id'], n_items),
            'seller_id': [f'seller_{i%5:03d}' for i in range(n_items)],
            'price': np.random.uniform(20, 500, n_items),
            'freight_value': np.random.uniform(5, 50, n_items)
        })

        # Avis
        delivered_orders = orders[orders['order_status'] == 'delivered']['order_id']
        n_reviews = min(len(delivered_orders), 30)
        reviews = pd.DataFrame({
            'review_id': [f'review_{i:03d}' for i in range(n_reviews)],
            'order_id': np.random.choice(delivered_orders, n_reviews, replace=False),
            'review_score': np.random.choice([1, 2, 3, 4, 5], n_reviews, p=[0.05, 0.05, 0.1, 0.3, 0.5]),
            'review_creation_date': [base_date + timedelta(days=np.random.randint(1, 365)) for _ in range(n_reviews)]
        })

        # Sauvegarder dans le répertoire de test
        raw_dir = test_data_dir / "raw"
        customers.to_csv(raw_dir / 'olist_customers_dataset.csv', index=False)
        products.to_csv(raw_dir / 'olist_products_dataset.csv', index=False)
        orders.to_csv(raw_dir / 'olist_orders_dataset.csv', index=False)
        order_items.to_csv(raw_dir / 'olist_order_items_dataset.csv', index=False)
        reviews.to_csv(raw_dir / 'olist_order_reviews_dataset.csv', index=False)

        return {
            'customers': customers,
            'products': products,
            'orders': orders,
            'order_items': order_items,
            'reviews': reviews,
            'data_dir': raw_dir
        }

    def test_full_ml_pipeline(self, complete_test_dataset):
        """
        Test: Le pipeline ML complet fonctionne de bout en bout.

        Flux testé : Données CSV → Features → Training → Prédiction
        """
        # ARRANGE
        data_dir = complete_test_dataset['data_dir']

        # ACT
        # 1. Chargement des données
        customers, orders, order_items, products, reviews = load_and_prepare_data(data_dir)

        # 2. Feature engineering
        feature_engine = RecommendationFeatureEngine()
        feature_engine.fit(customers, products)

        customer_features = feature_engine.create_customer_features(
            orders, reviews, order_items, products
        )

        # Créer des features produits simples
        product_features = pd.DataFrame({
            'category_encoded': pd.Categorical(products['product_category_name']).codes,
            'weight': products.get('product_weight_g', 100)
        }, index=products['product_id'])

        interactions = feature_engine.create_interaction_matrix(orders, order_items)

        # 3. Entraînement du modèle
        model = OlistRecommendationModel()
        X, y = model.prepare_training_data(customer_features, product_features, interactions)
        model.fit(X, y)

        # 4. Prédictions
        test_customer_id = customer_features.index[0]
        test_customer_features = customer_features.loc[test_customer_id].to_dict()

        recommendations = model.get_recommendations(
            test_customer_id, test_customer_features, product_features, n_recommendations=5
        )

        # ASSERT
        # Vérifier que chaque étape a fonctionné
        assert not customer_features.empty
        assert not product_features.empty
        assert not interactions.empty
        assert len(X) > 0
        assert len(y) > 0
        assert model.is_trained
        assert len(recommendations) == 5

        # Vérifier la cohérence des données
        assert all(0 <= rec['purchase_probability'] <= 1 for rec in recommendations)
        assert all(rec['customer_id'] == test_customer_id for rec in recommendations)

        # Vérifier que les métriques sont raisonnables
        performance = model.get_model_performance()
        assert 0 <= performance['train_accuracy'] <= 1
        assert 0 <= performance['test_accuracy'] <= 1
        assert 0 <= performance['auc_score'] <= 1

    def test_pipeline_data_consistency(self, complete_test_dataset):
        """
        Test: La cohérence des données à travers tout le pipeline.

        Vérifie que les IDs et relations sont préservés correctement.
        """
        # ARRANGE
        data_dir = complete_test_dataset['data_dir']

        # ACT
        customers, orders, order_items, products, reviews = load_and_prepare_data(data_dir)

        feature_engine = RecommendationFeatureEngine()
        feature_engine.fit(customers, products)

        customer_features = feature_engine.create_customer_features(
            orders, reviews, order_items, products
        )
        interactions = feature_engine.create_interaction_matrix(orders, order_items)

        # ASSERT
        # Vérifier la cohérence des customer_ids
        original_customer_ids = set(customers['customer_id'])
        orders_customer_ids = set(orders['customer_id'])
        features_customer_ids = set(customer_features.index)
        interactions_customer_ids = set(interactions['customer_id'])

        # Les customer_ids doivent être cohérents (avec intersections non vides)
        assert len(original_customer_ids.intersection(orders_customer_ids)) > 0
        assert len(features_customer_ids.intersection(interactions_customer_ids)) > 0

        # Vérifier la cohérence des product_ids
        original_product_ids = set(products['product_id'])
        items_product_ids = set(order_items['product_id'])
        interactions_product_ids = set(interactions['product_id'])

        assert len(original_product_ids.intersection(items_product_ids)) > 0
        assert len(items_product_ids.intersection(interactions_product_ids)) > 0

        # Vérifier la cohérence des order_ids
        original_order_ids = set(orders['order_id'])
        items_order_ids = set(order_items['order_id'])
        reviews_order_ids = set(reviews['order_id']) if not reviews.empty else set()

        assert len(original_order_ids.intersection(items_order_ids)) > 0
        if reviews_order_ids:
            assert len(original_order_ids.intersection(reviews_order_ids)) > 0

    def test_pipeline_error_resilience(self, complete_test_dataset):
        """
        Test: La résilience du pipeline aux erreurs et données manquantes.
        """
        data_dir = complete_test_dataset['data_dir']

        # ARRANGE - Modifier les données pour introduire des problèmes
        customers, orders, order_items, products, reviews = load_and_prepare_data(data_dir)

        # Introduire des customer_ids manquants dans orders
        orders.loc[0, 'customer_id'] = 'nonexistent_customer'

        # Introduire des product_ids manquants dans order_items
        order_items.loc[0, 'product_id'] = 'nonexistent_product'

        # ACT & ASSERT
        feature_engine = RecommendationFeatureEngine()
        feature_engine.fit(customers, products)

        # Le pipeline doit gérer gracieusement les IDs manquants
        customer_features = feature_engine.create_customer_features(
            orders, reviews, order_items, products
        )

        # Les features doivent être créées même avec des données imparfaites
        assert not customer_features.empty

        # Tester avec des reviews complètement vides
        empty_reviews = pd.DataFrame(columns=['review_id', 'order_id', 'review_score'])
        customer_features_no_reviews = feature_engine.create_customer_features(
            orders, empty_reviews, order_items, products
        )
        assert not customer_features_no_reviews.empty

@pytest.mark.integration
class TestFeatureEngineeringIntegration:
    """
    Tests d'intégration spécifiques au feature engineering.
    """

    def test_customer_product_feature_integration(self, sample_customers_data, sample_products_data):
        """
        Test: L'intégration entre les features clients et produits.
        """
        # ARRANGE
        customer_engineer = CustomerFeatureEngineer()
        product_engineer = ProductFeatureEngineer()

        # Données factices pour tester l'intégration
        customer_data = pd.DataFrame({
            'customer_id': ['c1', 'c2'],
            'total_orders': [3, 5],
            'total_spent': [150, 300],
            'favorite_category': ['cat1', 'cat2']
        }).set_index('customer_id')

        # ACT
        customer_engineer.fit(customer_data)
        product_engineer.fit(sample_products_data)

        customer_features = customer_engineer.transform(customer_data)
        product_features = product_engineer.transform(sample_products_data)

        # ASSERT
        # Les deux transformateurs doivent pouvoir être utilisés ensemble
        assert not customer_features.empty
        assert not product_features.empty

        # Les formats doivent être compatibles pour le modèle
        assert isinstance(customer_features, pd.DataFrame)
        assert isinstance(product_features, pd.DataFrame)

        # Les types de données doivent être numériques (pour le ML)
        numeric_customer_cols = customer_features.select_dtypes(include=[np.number]).columns
        numeric_product_cols = product_features.select_dtypes(include=[np.number]).columns

        assert len(numeric_customer_cols) > 0
        assert len(numeric_product_cols) > 0

    def test_feature_scaling_integration(self, sample_products_data):
        """
        Test: L'intégration du scaling dans le feature engineering.

        Vérifie que le scaling est appliqué de manière cohérente.
        """
        # ARRANGE
        engineer = ProductFeatureEngineer()

        # ACT
        engineer.fit(sample_products_data)
        features1 = engineer.transform(sample_products_data)

        # Transformer d'autres données avec le même scaler
        new_product_data = sample_products_data.iloc[:3].copy()
        features2 = engineer.transform(new_product_data)

        # ASSERT
        # Le scaling doit être cohérent
        numeric_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
        available_cols = [col for col in numeric_cols if col in features1.columns]

        for col in available_cols:
            # Les features transformées ne doivent pas être identiques aux originales
            assert not np.array_equal(
                sample_products_data[col].values,
                features1[col].values
            )

            # Le scaling doit être reproductible
            original_subset = sample_products_data.iloc[:3][col].values
            transformed_subset = features2[col].values
            expected_subset = features1.iloc[:3][col].values

            np.testing.assert_array_almost_equal(transformed_subset, expected_subset)

@pytest.mark.integration
class TestModelTrainingIntegration:
    """
    Tests d'intégration pour l'entraînement de modèle.
    """

    def test_training_with_real_features(self, complete_test_dataset):
        """
        Test: L'entraînement avec des features réelles du pipeline.
        """
        # ARRANGE
        data_dir = complete_test_dataset['data_dir']
        customers, orders, order_items, products, reviews = load_and_prepare_data(data_dir)

        feature_engine = RecommendationFeatureEngine()
        feature_engine.fit(customers, products)

        customer_features = feature_engine.create_customer_features(
            orders, reviews, order_items, products
        )

        product_features = pd.DataFrame({
            'category_encoded': pd.Categorical(products['product_category_name']).codes,
            'weight_normalized': (products['product_weight_g'] - products['product_weight_g'].mean()) / products['product_weight_g'].std()
        }, index=products['product_id'])

        interactions = feature_engine.create_interaction_matrix(orders, order_items)

        # ACT
        model = OlistRecommendationModel()
        X, y = model.prepare_training_data(customer_features, product_features, interactions)

        # Vérifier les dimensions avant training
        assert len(X.columns) > 0
        assert len(X) == len(y)

        # Entraînement
        model.fit(X, y)

        # ASSERT
        assert model.is_trained

        # Le modèle doit avoir des métriques raisonnables
        performance = model.get_model_performance()
        assert performance['auc_score'] > 0.4  # Score minimal acceptable

        # Le modèle doit pouvoir faire des prédictions
        test_customer = customer_features.iloc[0].to_dict()
        predictions = model.predict_proba(test_customer, product_features)

        assert len(predictions) == len(product_features)
        assert all(0 <= prob <= 1 for prob in predictions['purchase_probability'])

    def test_model_persistence_integration(self, mock_trained_model, test_data_dir):
        """
        Test: L'intégration de la persistence du modèle avec le pipeline.
        """
        # ARRANGE
        model_path = test_data_dir / "models" / "test_model.joblib"
        model_path.parent.mkdir(exist_ok=True)

        # ACT
        # Sauvegarder
        mock_trained_model.save_model(model_path)
        assert model_path.exists()

        # Charger
        loaded_model = OlistRecommendationModel.load_model(model_path)

        # ASSERT
        # Le modèle chargé doit avoir les mêmes propriétés
        assert loaded_model.is_trained == mock_trained_model.is_trained
        assert loaded_model.feature_columns == mock_trained_model.feature_columns

        # Les métriques doivent être identiques
        original_metrics = mock_trained_model.get_model_performance()
        loaded_metrics = loaded_model.get_model_performance()
        assert original_metrics == loaded_metrics

@pytest.mark.integration
class TestPipelineScalabilityIntegration:
    """
    Tests d'intégration pour la scalabilité du pipeline.
    """

    @pytest.mark.slow
    def test_pipeline_with_larger_dataset(self):
        """
        Test: Le pipeline fonctionne avec des datasets plus grands.
        """
        # ARRANGE - Créer un dataset plus grand
        np.random.seed(42)

        n_customers = 100
        n_products = 50
        n_orders = 300
        n_items = 500

        # Créer des données factices plus volumineuses
        customers = pd.DataFrame({
            'customer_id': [f'customer_{i:04d}' for i in range(n_customers)]
        })

        products = pd.DataFrame({
            'product_id': [f'product_{i:04d}' for i in range(n_products)],
            'product_category_name': np.random.choice(['cat1', 'cat2', 'cat3', 'cat4', 'cat5'], n_products)
        })

        orders = pd.DataFrame({
            'order_id': [f'order_{i:04d}' for i in range(n_orders)],
            'customer_id': np.random.choice(customers['customer_id'], n_orders),
            'order_status': np.random.choice(['delivered', 'shipped'], n_orders, p=[0.8, 0.2]),
            'order_purchase_timestamp': [datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(n_orders)]
        })

        order_items = pd.DataFrame({
            'order_id': np.random.choice(orders['order_id'], n_items),
            'order_item_id': list(range(n_items)),
            'product_id': np.random.choice(products['product_id'], n_items),
            'price': np.random.uniform(10, 1000, n_items)
        })

        reviews = pd.DataFrame({
            'review_id': [f'review_{i:04d}' for i in range(min(200, n_orders))],
            'order_id': np.random.choice(orders['order_id'], min(200, n_orders), replace=False),
            'review_score': np.random.choice([1, 2, 3, 4, 5], min(200, n_orders))
        })

        # ACT
        feature_engine = RecommendationFeatureEngine()
        feature_engine.fit(customers, products)

        customer_features = feature_engine.create_customer_features(
            orders, reviews, order_items, products
        )

        # ASSERT
        # Le pipeline doit gérer des datasets plus grands sans erreur
        assert len(customer_features) > 0

        # Les performances doivent rester raisonnables
        import time
        start_time = time.time()

        interactions = feature_engine.create_interaction_matrix(orders, order_items)
        processing_time = time.time() - start_time

        assert len(interactions) > 0
        assert processing_time < 10.0  # Moins de 10 secondes pour ce dataset

@pytest.mark.integration
def test_pipeline_configuration_integration():
    """
    Test: L'intégration avec le système de configuration.
    """
    # ARRANGE
    original_params = MLConfig.RANDOM_FOREST_PARAMS.copy()

    # ACT
    model = OlistRecommendationModel()

    # ASSERT
    # Le modèle doit utiliser la configuration globale
    assert model.rf_params == original_params

    # Tester la modification de configuration
    new_params = original_params.copy()
    new_params['n_estimators'] = 150

    model_with_custom = OlistRecommendationModel(new_params)
    assert model_with_custom.rf_params['n_estimators'] == 150