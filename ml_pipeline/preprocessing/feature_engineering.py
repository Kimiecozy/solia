# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - FEATURE ENGINEERING
# Master 2 - Data Science Industrielle
# ===============================================

"""
Module de feature engineering pour le système de recommandation Olist.

Ce module transforme les données brutes en features utilisables pour
le modèle de machine learning.

Classes:
    - CustomerFeatureEngineer: Calcul des features clients
    - ProductFeatureEngineer: Calcul des features produits
    - RecommendationFeatureEngine: Pipeline complet
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DataConfig

class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformeur pour créer des features clients.

    Features créées:
    - Métriques RFM (Récence, Fréquence, Montant)
    - Comportement d'achat
    - Préférences par catégorie
    """

    def __init__(self):
        self.reference_date = None
        self.category_encoder = LabelEncoder()

    def fit(self, X, y=None):
        """Apprend les paramètres nécessaires."""
        # Date de référence pour calculer la récence
        if 'order_purchase_timestamp' in X.columns:
            self.reference_date = X['order_purchase_timestamp'].max()
        else:
            self.reference_date = datetime.now()

        # Encoder les catégories préférées
        if 'favorite_category' in X.columns:
            self.category_encoder.fit(X['favorite_category'].fillna('unknown'))

        return self

    def transform(self, X):
        """Transforme les données en features."""
        if isinstance(X, dict):
            # Si X est un dictionnaire (données client unique)
            return self._transform_single_customer(X)
        else:
            # Si X est un DataFrame (batch)
            return self._transform_dataframe(X)

    def _transform_single_customer(self, customer_data: Dict) -> Dict:
        """Transforme les données d'un client unique."""
        features = {}

        # Features RFM basiques
        features['total_orders'] = customer_data.get('total_orders', 0)
        features['total_spent'] = customer_data.get('total_spent', 0.0)
        features['avg_order_value'] = (
            features['total_spent'] / max(features['total_orders'], 1)
        )

        # Récence (jours depuis la dernière commande)
        last_order = customer_data.get('last_order_date')
        if last_order:
            if isinstance(last_order, str):
                last_order = pd.to_datetime(last_order)
            features['days_since_last_order'] = (self.reference_date - last_order).days
        else:
            features['days_since_last_order'] = 365  # Client inactif

        # Catégorie préférée
        favorite_cat = customer_data.get('favorite_category', 'unknown')
        features['favorite_category_encoded'] = self.category_encoder.transform([favorite_cat])[0]

        # Score client
        features['avg_review_score'] = customer_data.get('avg_review_score', 3.0)
        features['unique_products_bought'] = customer_data.get('unique_products_bought', 0)

        return features

    def _transform_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforme un DataFrame complet."""
        X_transformed = X.copy()

        # Features RFM
        X_transformed['avg_order_value'] = (
            X_transformed['total_spent'] / X_transformed['total_orders'].clip(lower=1)
        )

        # Récence
        if 'last_order_date' in X_transformed.columns:
            X_transformed['last_order_date'] = pd.to_datetime(X_transformed['last_order_date'])
            X_transformed['days_since_last_order'] = (
                self.reference_date - X_transformed['last_order_date']
            ).dt.days
        else:
            X_transformed['days_since_last_order'] = 365

        # Encoder la catégorie préférée
        if 'favorite_category' in X_transformed.columns:
            X_transformed['favorite_category_encoded'] = self.category_encoder.transform(
                X_transformed['favorite_category'].fillna('unknown')
            )

        # Segmentation RFM
        X_transformed = self._add_rfm_segments(X_transformed)

        return X_transformed

    def _add_rfm_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les segments RFM."""
        # Créer des quartiles pour R, F, M
        df['recency_quartile'] = pd.qcut(
            df['days_since_last_order'], q=4, labels=['1', '2', '3', '4']
        ).astype(int)

        df['frequency_quartile'] = pd.qcut(
            df['total_orders'], q=4, labels=['4', '3', '2', '1'], duplicates='drop'
        ).astype(int)

        df['monetary_quartile'] = pd.qcut(
            df['total_spent'], q=4, labels=['4', '3', '2', '1'], duplicates='drop'
        ).astype(int)

        # Score RFM combiné
        df['rfm_score'] = (
            df['recency_quartile'].astype(str) +
            df['frequency_quartile'].astype(str) +
            df['monetary_quartile'].astype(str)
        )

        # Segments business
        df['customer_segment'] = df['rfm_score'].apply(self._get_customer_segment)

        return df

    @staticmethod
    def _get_customer_segment(rfm_score: str) -> str:
        """Détermine le segment client selon le score RFM."""
        if rfm_score in ['111', '112', '121', '211']:
            return 'Champions'
        elif rfm_score in ['113', '114', '122', '131', '141', '212', '221']:
            return 'Loyal Customers'
        elif rfm_score in ['123', '124', '132', '142', '213', '222', '231']:
            return 'Potential Loyalists'
        elif rfm_score in ['133', '134', '143', '144', '223', '232', '241', '242']:
            return 'New Customers'
        elif rfm_score in ['311', '312', '321', '411']:
            return 'Promising'
        elif rfm_score in ['313', '314', '322', '331', '341', '412', '421']:
            return 'Need Attention'
        elif rfm_score in ['323', '324', '332', '342', '413', '422', '431']:
            return 'About to Sleep'
        elif rfm_score in ['333', '334', '343', '344', '423', '432', '441', '442']:
            return 'At Risk'
        elif rfm_score in ['444', '443', '434']:
            return 'Cannot Lose'
        else:
            return 'Lost'

class ProductFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformeur pour créer des features produits.

    Features créées:
    - Popularité du produit
    - Score moyen des avis
    - Caractéristiques physiques normalisées
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.category_encoder = LabelEncoder()

    def fit(self, X, y=None):
        """Apprend les paramètres de normalisation."""
        numeric_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
        available_cols = [col for col in numeric_cols if col in X.columns]

        if available_cols:
            self.scaler.fit(X[available_cols].fillna(0))

        if 'product_category_name' in X.columns:
            self.category_encoder.fit(X['product_category_name'].fillna('unknown'))

        return self

    def transform(self, X):
        """Transforme les données produits."""
        X_transformed = X.copy()

        # Normaliser les dimensions
        numeric_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
        available_cols = [col for col in numeric_cols if col in X_transformed.columns]

        if available_cols:
            X_transformed[available_cols] = self.scaler.transform(
                X_transformed[available_cols].fillna(0)
            )

        # Encoder la catégorie
        if 'product_category_name' in X_transformed.columns:
            X_transformed['category_encoded'] = self.category_encoder.transform(
                X_transformed['product_category_name'].fillna('unknown')
            )

        # Features calculées
        if all(col in X_transformed.columns for col in ['product_length_cm', 'product_width_cm', 'product_height_cm']):
            X_transformed['product_volume'] = (
                X_transformed['product_length_cm'] *
                X_transformed['product_width_cm'] *
                X_transformed['product_height_cm']
            )

        return X_transformed

class RecommendationFeatureEngine:
    """
    Pipeline complet de feature engineering pour les recommandations.
    """

    def __init__(self):
        self.customer_engineer = CustomerFeatureEngineer()
        self.product_engineer = ProductFeatureEngineer()
        self.is_fitted = False

    def fit(self, customer_data: pd.DataFrame, product_data: pd.DataFrame):
        """Entraîne les transformeurs sur les données."""
        self.customer_engineer.fit(customer_data)
        self.product_engineer.fit(product_data)
        self.is_fitted = True
        return self

    def create_customer_features(self, orders_df: pd.DataFrame, reviews_df: pd.DataFrame,
                                order_items_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les features clients à partir des données brutes.
        """
        print("🔨 Création des features clients...")

        # Agrégations par client
        customer_orders = orders_df.groupby('customer_id').agg({
            'order_id': 'count',
            'order_purchase_timestamp': 'max'
        }).rename(columns={
            'order_id': 'total_orders',
            'order_purchase_timestamp': 'last_order_date'
        })

        # Montant total dépensé
        customer_spending = order_items_df.merge(
            orders_df[['order_id', 'customer_id']], on='order_id'
        ).groupby('customer_id')['price'].sum().to_frame('total_spent')

        # Catégorie préférée
        customer_categories = order_items_df.merge(
            orders_df[['order_id', 'customer_id']], on='order_id'
        ).merge(
            products_df[['product_id', 'product_category_name']], on='product_id'
        ).groupby('customer_id')['product_category_name'].agg(
            lambda x: x.mode().iloc[0] if not x.empty else 'unknown'
        ).to_frame('favorite_category')

        # Scores des avis
        if not reviews_df.empty:
            customer_reviews = reviews_df.merge(
                orders_df[['order_id', 'customer_id']], on='order_id'
            ).groupby('customer_id')['review_score'].mean().to_frame('avg_review_score')
        else:
            customer_reviews = pd.DataFrame(columns=['avg_review_score'])

        # Nombre de produits uniques
        customer_products = order_items_df.merge(
            orders_df[['order_id', 'customer_id']], on='order_id'
        ).groupby('customer_id')['product_id'].nunique().to_frame('unique_products_bought')

        # Combiner toutes les features
        customer_features = customer_orders.join([
            customer_spending, customer_categories, customer_reviews, customer_products
        ], how='left')

        # Remplir les valeurs manquantes
        customer_features = customer_features.fillna({
            'total_spent': 0,
            'favorite_category': 'unknown',
            'avg_review_score': 3.0,
            'unique_products_bought': 0
        })

        # Appliquer les transformations
        if self.is_fitted:
            customer_features = self.customer_engineer.transform(customer_features)

        print(f"   ✅ Features créées pour {len(customer_features)} clients")
        return customer_features

    def create_interaction_matrix(self, orders_df: pd.DataFrame, order_items_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée une matrice d'interaction client-produit pour l'entraînement.
        """
        print("🔗 Création de la matrice d'interaction...")

        # Joindre les commandes et items
        interactions = order_items_df.merge(
            orders_df[['order_id', 'customer_id', 'order_status']], on='order_id'
        )

        # Ne garder que les commandes livrées pour l'entraînement
        interactions = interactions[interactions['order_status'] == 'delivered']

        # Créer des features d'interaction
        interactions['purchased'] = 1  # Variable cible binaire
        interactions['rating'] = interactions['price'] / interactions['price'].max()  # Rating normalisé

        # Ajouter des features temporelles si disponible
        if 'order_purchase_timestamp' in orders_df.columns:
            interactions = interactions.merge(
                orders_df[['order_id', 'order_purchase_timestamp']], on='order_id'
            )
            interactions['order_purchase_timestamp'] = pd.to_datetime(interactions['order_purchase_timestamp'])
            interactions['days_ago'] = (
                interactions['order_purchase_timestamp'].max() - interactions['order_purchase_timestamp']
            ).dt.days

        print(f"   ✅ Matrice créée avec {len(interactions)} interactions")
        return interactions

def load_and_prepare_data(raw_data_dir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge et prépare toutes les données nécessaires.
    """
    print("📁 Chargement des données...")

    # Charger tous les datasets
    customers = pd.read_csv(raw_data_dir / 'olist_customers_dataset.csv')
    orders = pd.read_csv(raw_data_dir / 'olist_orders_dataset.csv')
    order_items = pd.read_csv(raw_data_dir / 'olist_order_items_dataset.csv')
    products = pd.read_csv(raw_data_dir / 'olist_products_dataset.csv')

    # Reviews optionnelles
    try:
        reviews = pd.read_csv(raw_data_dir / 'olist_order_reviews_dataset.csv')
    except FileNotFoundError:
        print("   ⚠️ Fichier reviews non trouvé, création d'un DataFrame vide")
        reviews = pd.DataFrame(columns=['review_id', 'order_id', 'review_score'])

    # Conversion des dates
    date_columns = ['order_purchase_timestamp', 'order_approved_at',
                   'order_delivered_carrier_date', 'order_delivered_customer_date']

    for col in date_columns:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col])

    print(f"   ✅ Chargé: {len(customers)} clients, {len(orders)} commandes, {len(order_items)} items, {len(products)} produits, {len(reviews)} avis")

    return customers, orders, order_items, products, reviews