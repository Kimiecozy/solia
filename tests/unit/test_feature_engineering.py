# ===============================================
# 🧪 TESTS UNITAIRES - FEATURE ENGINEERING
# Master 2 - Data Science Industrielle
# ===============================================

"""
Tests unitaires pour le module de feature engineering.

Ces tests vérifient que chaque fonction et classe de feature engineering
fonctionne correctement de manière isolée.

Concepts testés:
- Transformation des données clients
- Calcul des métriques RFM
- Segmentation des clients
- Validation des outputs

Usage:
    pytest tests/unit/test_feature_engineering.py -v
    pytest tests/unit/test_feature_engineering.py::test_customer_feature_engineer_basic -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Import du module à tester
sys.path.append(str(Path(__file__).parent.parent.parent))
from ml_pipeline.preprocessing.feature_engineering import (
    CustomerFeatureEngineer,
    ProductFeatureEngineer,
    RecommendationFeatureEngine
)

@pytest.mark.unit
class TestCustomerFeatureEngineer:
    """
    Tests unitaires pour CustomerFeatureEngineer.

    Teste la transformation des données clients en features ML.
    """

    def test_initialization(self):
        """
        Test: L'initialisation du CustomerFeatureEngineer fonctionne correctement.
        """
        # ARRANGE & ACT
        engineer = CustomerFeatureEngineer()

        # ASSERT
        assert engineer.reference_date is None
        assert engineer.category_encoder is not None
        assert hasattr(engineer, 'fit')
        assert hasattr(engineer, 'transform')

    def test_fit_basic(self, sample_customer_features):
        """
        Test: La méthode fit apprend correctement les paramètres.
        """
        # ARRANGE
        engineer = CustomerFeatureEngineer()

        # ACT
        result = engineer.fit(sample_customer_features)

        # ASSERT
        assert result == engineer  # fit doit retourner self
        assert engineer.reference_date is not None
        assert hasattr(engineer.category_encoder, 'classes_')

    def test_transform_single_customer(self):
        """
        Test: La transformation d'un client unique fonctionne correctement.
        """
        # ARRANGE
        engineer = CustomerFeatureEngineer()
        engineer.reference_date = datetime(2023, 12, 31)

        customer_data = {
            'total_orders': 5,
            'total_spent': 250.0,
            'last_order_date': datetime(2023, 11, 15),
            'favorite_category': 'cama_mesa_banho',
            'avg_review_score': 4.2,
            'unique_products_bought': 3
        }

        # Fit avec des données factices pour l'encodeur
        dummy_df = pd.DataFrame({
            'favorite_category': ['cama_mesa_banho', 'beleza_saude']
        })
        engineer.fit(dummy_df)

        # ACT
        features = engineer.transform(customer_data)

        # ASSERT
        assert isinstance(features, dict)
        assert 'total_orders' in features
        assert 'total_spent' in features
        assert 'avg_order_value' in features
        assert 'days_since_last_order' in features

        # Vérifications des calculs
        expected_avg_order = 250.0 / 5
        assert features['avg_order_value'] == expected_avg_order

        expected_days_since = (engineer.reference_date - customer_data['last_order_date']).days
        assert features['days_since_last_order'] == expected_days_since

    def test_rfm_segments_calculation(self, sample_customer_features):
        """
        Test: Le calcul des segments RFM fonctionne correctement.
        """
        # ARRANGE
        engineer = CustomerFeatureEngineer()
        df = sample_customer_features.copy()
        df['last_order_date'] = [datetime(2023, 11, 1) + timedelta(days=i*10) for i in range(len(df))]

        # ACT
        engineer.fit(df)
        result = engineer.transform(df)

        # ASSERT
        assert 'rfm_score' in result.columns
        assert 'customer_segment' in result.columns
        assert 'recency_quartile' in result.columns
        assert 'frequency_quartile' in result.columns
        assert 'monetary_quartile' in result.columns

        # Vérifier que les quartiles sont bien entre 1 et 4
        for col in ['recency_quartile', 'frequency_quartile', 'monetary_quartile']:
            assert result[col].min() >= 1
            assert result[col].max() <= 4

        # Vérifier que les segments sont valides
        valid_segments = [
            'Champions', 'Loyal Customers', 'Potential Loyalists', 'New Customers',
            'Promising', 'Need Attention', 'About to Sleep', 'At Risk',
            'Cannot Lose', 'Lost'
        ]
        assert all(segment in valid_segments for segment in result['customer_segment'].unique())

    def test_get_customer_segment_logic(self):
        """
        Test: La logique de segmentation RFM est correcte.
        """
        # ARRANGE
        engineer = CustomerFeatureEngineer()

        # ACT & ASSERT - Tester quelques cas spécifiques
        assert engineer._get_customer_segment('111') == 'Champions'
        assert engineer._get_customer_segment('444') == 'Cannot Lose'
        assert engineer._get_customer_segment('123') == 'Potential Loyalists'
        assert engineer._get_customer_segment('333') == 'At Risk'

    def test_transform_handles_missing_data(self):
        """
        Test: La transformation gère correctement les données manquantes.
        """
        # ARRANGE
        engineer = CustomerFeatureEngineer()
        engineer.reference_date = datetime(2023, 12, 31)

        # Données avec valeurs manquantes
        customer_data = {
            'total_orders': 2,
            'total_spent': 100.0,
            # last_order_date manquant intentionnellement
            'favorite_category': None,
            'avg_review_score': None,
            'unique_products_bought': 1
        }

        # Fit avec données factices
        dummy_df = pd.DataFrame({'favorite_category': ['unknown']})
        engineer.fit(dummy_df)

        # ACT
        features = engineer.transform(customer_data)

        # ASSERT
        assert 'days_since_last_order' in features
        assert features['days_since_last_order'] == 365  # Valeur par défaut

@pytest.mark.unit
class TestProductFeatureEngineer:
    """
    Tests unitaires pour ProductFeatureEngineer.
    """

    def test_initialization(self):
        """
        Test: L'initialisation du ProductFeatureEngineer fonctionne.
        """
        # ARRANGE & ACT
        engineer = ProductFeatureEngineer()

        # ASSERT
        assert engineer.scaler is not None
        assert engineer.category_encoder is not None

    def test_fit_and_transform(self, sample_products_data):
        """
        Test: Fit et transform fonctionnent ensemble correctement.
        """
        # ARRANGE
        engineer = ProductFeatureEngineer()

        # ACT
        engineer.fit(sample_products_data)
        result = engineer.transform(sample_products_data)

        # ASSERT
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_products_data)
        assert 'category_encoded' in result.columns

        # Vérifier que les dimensions ont été normalisées
        numeric_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
        available_cols = [col for col in numeric_cols if col in result.columns]

        if available_cols:
            # Les colonnes normalisées ne doivent pas avoir exactement les mêmes valeurs qu'avant
            original_col = available_cols[0]
            original_values = sample_products_data[original_col].values
            normalized_values = result[original_col].values
            assert not np.array_equal(original_values, normalized_values)

    def test_volume_calculation(self):
        """
        Test: Le calcul du volume produit fonctionne correctement.
        """
        # ARRANGE
        engineer = ProductFeatureEngineer()

        data = pd.DataFrame({
            'product_id': ['prod_1', 'prod_2'],
            'product_length_cm': [10.0, 20.0],
            'product_width_cm': [5.0, 10.0],
            'product_height_cm': [2.0, 3.0]
        })

        # ACT
        engineer.fit(data)
        result = engineer.transform(data)

        # ASSERT
        assert 'product_volume' in result.columns

        # Vérifier les calculs de volume (après normalisation, on vérifie le ratio)
        volumes = result['product_volume'].values
        assert len(volumes) == 2
        assert all(vol >= 0 for vol in volumes)  # Volumes doivent être positifs

@pytest.mark.unit
class TestRecommendationFeatureEngine:
    """
    Tests unitaires pour RecommendationFeatureEngine.
    """

    def test_initialization(self):
        """
        Test: L'initialisation du pipeline complet fonctionne.
        """
        # ARRANGE & ACT
        engine = RecommendationFeatureEngine()

        # ASSERT
        assert engine.customer_engineer is not None
        assert engine.product_engineer is not None
        assert not engine.is_fitted

    def test_fit_process(self, sample_customer_features, sample_products_data):
        """
        Test: Le processus de fit du pipeline complet.
        """
        # ARRANGE
        engine = RecommendationFeatureEngine()

        # ACT
        result = engine.fit(sample_customer_features, sample_products_data)

        # ASSERT
        assert result == engine  # fit doit retourner self
        assert engine.is_fitted

    def test_create_interaction_matrix(self, sample_orders_data, sample_order_items_data):
        """
        Test: La création de la matrice d'interaction fonctionne.
        """
        # ARRANGE
        engine = RecommendationFeatureEngine()

        # ACT
        interactions = engine.create_interaction_matrix(sample_orders_data, sample_order_items_data)

        # ASSERT
        assert isinstance(interactions, pd.DataFrame)
        assert 'customer_id' in interactions.columns
        assert 'product_id' in interactions.columns
        assert 'purchased' in interactions.columns
        assert 'rating' in interactions.columns

        # Vérifier que seules les commandes livrées sont incluses
        assert all(interactions['order_status'] == 'delivered')

        # Vérifier que purchased est binaire
        assert set(interactions['purchased'].unique()) == {1}

        # Vérifier que rating est normalisé
        assert all(0 <= rating <= 1 for rating in interactions['rating'])

    def test_create_customer_features_integration(self, sample_orders_data, sample_reviews_data,
                                                sample_order_items_data, sample_products_data):
        """
        Test: La création des features clients intègre correctement toutes les données.
        """
        # ARRANGE
        engine = RecommendationFeatureEngine()
        engine.fit(pd.DataFrame(), sample_products_data)  # Fit minimal pour initialiser

        # ACT
        customer_features = engine.create_customer_features(
            sample_orders_data, sample_reviews_data, sample_order_items_data, sample_products_data
        )

        # ASSERT
        assert isinstance(customer_features, pd.DataFrame)
        assert not customer_features.empty

        # Vérifier que les colonnes essentielles sont présentes
        expected_columns = ['total_orders', 'total_spent', 'favorite_category', 'unique_products_bought']
        for col in expected_columns:
            assert col in customer_features.columns

        # Vérifier les types de données
        assert customer_features['total_orders'].dtype in [np.int64, int]
        assert customer_features['total_spent'].dtype in [np.float64, float]

        # Vérifier la cohérence des données
        assert all(customer_features['total_orders'] >= 0)
        assert all(customer_features['total_spent'] >= 0)

# Tests d'aide et utilitaires
@pytest.mark.unit
def test_feature_validation_helpers():
    """
    Test: Les fonctions helper de validation des features.
    """
    from tests.conftest import assert_dataframe_equals

    # ARRANGE
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df3 = pd.DataFrame({'a': [1, 2, 4], 'b': [4, 5, 6]})

    # ACT & ASSERT
    assert assert_dataframe_equals(df1, df2) is True
    assert assert_dataframe_equals(df1, df3) is False

# Tests de régression
@pytest.mark.unit
def test_feature_engineering_reproducibility(sample_customer_features):
    """
    Test: Les transformations sont reproductibles avec les mêmes données.
    """
    # ARRANGE
    engineer1 = CustomerFeatureEngineer()
    engineer2 = CustomerFeatureEngineer()

    # ACT
    result1 = engineer1.fit_transform(sample_customer_features)
    result2 = engineer2.fit_transform(sample_customer_features)

    # ASSERT
    pd.testing.assert_frame_equal(result1, result2)

# Tests de performance et edge cases
@pytest.mark.unit
def test_feature_engineering_with_empty_data():
    """
    Test: Le feature engineering gère correctement les datasets vides.
    """
    # ARRANGE
    engineer = CustomerFeatureEngineer()
    empty_df = pd.DataFrame()

    # ACT & ASSERT
    # Devrait lever une exception ou gérer gracieusement
    try:
        engineer.fit(empty_df)
        # Si pas d'exception, vérifier que l'état est cohérent
        assert engineer.reference_date is None
    except (ValueError, KeyError):
        # Exception attendue pour données vides
        pass

@pytest.mark.unit
def test_feature_engineering_with_single_row(sample_customer_features):
    """
    Test: Le feature engineering fonctionne avec une seule ligne de données.
    """
    # ARRANGE
    engineer = CustomerFeatureEngineer()
    single_row = sample_customer_features.iloc[:1].copy()

    # ACT
    result = engineer.fit_transform(single_row)

    # ASSERT
    assert len(result) == 1
    assert isinstance(result, pd.DataFrame)
    # Avec une seule ligne, les quartiles peuvent ne pas fonctionner normalement
    # mais le processus ne doit pas crash