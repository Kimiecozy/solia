# ===============================================
# 🧪 TEST DÉMO - OLIST RECOMMENDATION SYSTEM
# Master 2 - Data Science Industrielle
# ===============================================

"""
Tests de démonstration pour vérifier que le système de tests fonctionne.

Ces tests sont simples et rapides, parfaits pour :
- Vérifier que pytest est correctement installé
- Comprendre la structure des tests
- Tester les fixtures de base

Usage:
    pytest tests/test_demo.py -v
"""

import pytest
import pandas as pd
import numpy as np

@pytest.mark.unit
def test_demo_basic():
    """
    Test démo basique - vérifie que pytest fonctionne.
    """
    # ARRANGE
    a = 2
    b = 3

    # ACT
    result = a + b

    # ASSERT
    assert result == 5
    assert isinstance(result, int)

@pytest.mark.unit
def test_demo_with_pandas():
    """
    Test démo avec Pandas - vérifie que les dépendances fonctionnent.
    """
    # ARRANGE
    data = {
        'customer_id': ['A', 'B', 'C'],
        'orders': [1, 2, 3],
        'total_spent': [100, 200, 300]
    }

    # ACT
    df = pd.DataFrame(data)
    avg_spent = df['total_spent'].mean()

    # ASSERT
    assert len(df) == 3
    assert avg_spent == 200.0
    assert 'customer_id' in df.columns

@pytest.mark.unit
def test_demo_with_fixture(sample_customer_features):
    """
    Test démo utilisant une fixture - vérifie que les fixtures fonctionnent.
    """
    # ACT
    # La fixture sample_customer_features est automatiquement injectée
    total_customers = len(sample_customer_features)
    avg_orders = sample_customer_features['total_orders'].mean()

    # ASSERT
    assert total_customers > 0
    assert avg_orders > 0
    assert isinstance(sample_customer_features, pd.DataFrame)

@pytest.mark.unit
def test_demo_error_handling():
    """
    Test démo pour la gestion d'erreurs.
    """
    # Test qu'une division par zéro lève une exception
    with pytest.raises(ZeroDivisionError):
        result = 10 / 0

    # Test avec un message d'erreur spécifique
    with pytest.raises(ValueError, match="invalid literal"):
        int("not_a_number")

@pytest.mark.unit
@pytest.mark.parametrize("input_value,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
    (4, 16),
    (0, 0)
])
def test_demo_parametrized(input_value, expected):
    """
    Test démo paramétrisé - teste plusieurs cas avec une seule fonction.
    """
    # ACT
    result = input_value ** 2

    # ASSERT
    assert result == expected

@pytest.mark.integration
def test_demo_integration():
    """
    Test démo d'intégration - combine plusieurs composants.
    """
    # ARRANGE - Simuler des données de différentes sources
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'city': ['São Paulo', 'Rio', 'Belo Horizonte']
    })

    orders = pd.DataFrame({
        'customer_id': [1, 1, 2, 3],
        'order_value': [100, 150, 200, 75]
    })

    # ACT - Intégrer les données
    customer_spending = orders.groupby('customer_id')['order_value'].sum()
    result = customers.merge(
        customer_spending.to_frame('total_spent'),
        left_on='customer_id',
        right_index=True,
        how='left'
    )
    result['total_spent'] = result['total_spent'].fillna(0)

    # ASSERT
    assert len(result) == 3
    assert result.loc[0, 'total_spent'] == 250  # Customer 1: 100 + 150
    assert result.loc[1, 'total_spent'] == 200  # Customer 2: 200
    assert result.loc[2, 'total_spent'] == 75   # Customer 3: 75

@pytest.mark.slow
def test_demo_slow():
    """
    Test démo lent - marqué comme 'slow' pour être exclu des tests rapides.
    """
    import time

    # Simuler une opération lente
    time.sleep(0.1)  # 100ms

    # Test simple
    assert True

# Test qui échoue intentionnellement (commenté par défaut)
# @pytest.mark.unit
# def test_demo_failure():
#     """
#     Test démo qui échoue - décommentez pour voir un échec de test.
#     """
#     assert 1 == 2, "Ce test échoue intentionnellement pour la démo"

def test_demo_skip():
    """
    Test démo qui est ignoré conditionnellement.
    """
    pytest.skip("Test ignoré pour la démonstration")

@pytest.mark.skipif(not hasattr(pd, 'DataFrame'), reason="Pandas non disponible")
def test_demo_skip_conditional():
    """
    Test démo ignoré conditionnellement si Pandas n'est pas disponible.
    """
    df = pd.DataFrame({'a': [1, 2, 3]})
    assert len(df) == 3

def test_demo_approximate():
    """
    Test démo pour les comparaisons approximatives (nombres flottants).
    """
    # ACT
    result = 0.1 + 0.2

    # ASSERT - Comparaison exacte échouerait à cause de la précision flottante
    # assert result == 0.3  # Ceci pourrait échouer !

    # Comparaison approximative
    assert result == pytest.approx(0.3)
    assert result == pytest.approx(0.3, rel=1e-9)  # Précision relative

def test_demo_data_validation():
    """
    Test démo pour valider des données comme en data science.
    """
    # ARRANGE - Données simulées
    data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003'],
        'age': [25, 35, 45],
        'income': [50000, 75000, 100000],
        'orders': [1, 3, 5]
    })

    # ACT & ASSERT - Validations
    # 1. Pas de valeurs manquantes
    assert not data.isnull().any().any()

    # 2. Types de données corrects
    assert data['age'].dtype in ['int64', 'int32']
    assert data['income'].dtype in ['int64', 'int32', 'float64']

    # 3. Valeurs dans les plages attendues
    assert (data['age'] >= 18).all()
    assert (data['age'] <= 100).all()
    assert (data['income'] > 0).all()
    assert (data['orders'] >= 0).all()

    # 4. Unicité des IDs
    assert data['customer_id'].nunique() == len(data)

    # 5. Cohérence des données
    assert len(data) > 0
    assert all(col in data.columns for col in ['customer_id', 'age', 'income', 'orders'])

# Helper function pour la démo
def calculate_clv(orders, avg_order_value, retention_rate=0.8):
    """
    Calcule la Customer Lifetime Value simplifiée.

    Args:
        orders: Nombre de commandes
        avg_order_value: Valeur moyenne des commandes
        retention_rate: Taux de rétention

    Returns:
        CLV estimée
    """
    if retention_rate <= 0 or retention_rate >= 1:
        raise ValueError("Le taux de rétention doit être entre 0 et 1")

    return (orders * avg_order_value) / (1 - retention_rate)

def test_demo_business_logic():
    """
    Test démo pour la logique métier - calcul de CLV.
    """
    # ARRANGE
    orders = 5
    avg_order_value = 100
    retention_rate = 0.8

    # ACT
    clv = calculate_clv(orders, avg_order_value, retention_rate)

    # ASSERT
    expected_clv = (5 * 100) / (1 - 0.8)  # 500 / 0.2 = 2500
    assert clv == expected_clv
    assert clv > 0

    # Test des cas limites
    with pytest.raises(ValueError):
        calculate_clv(5, 100, 0)  # Retention rate = 0

    with pytest.raises(ValueError):
        calculate_clv(5, 100, 1)  # Retention rate = 1