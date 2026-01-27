# ===============================================
# OLIST RECOMMENDATION SYSTEM - FEATURE ENGINEERING
# Master 2 - SEP
# Version Simplifiée pour Apprentissage
# ===============================================


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MLConfig


class RecommendationFeatureEngine:
    """
    Pipeline complet de feature engineering.

    Ce pipeline orchestre la création de toutes les features:
    1. Features clients (RFM, comportement)
    2. Features produits (catégorie, dimensions)
    3. Matrice d'interaction (qui a acheté quoi?)

    ARCHITECTURE:
    Données brutes -> Feature Engineering -> Modèle ML -> Recommandations
    """

    def __init__(self):

        self.is_fitted = False

    def create_seller_features(self, sellers, orders, items, payments, reviews, products):
        print("🏗️ Fusion et calcul de la solvabilité par vendeur...")

        # Ta logique de merge
        df = items.merge(orders, on='order_id2')
        df = df.merge(payments, on='order_id2', how='left')
        df = df.merge(reviews, on='order_id2', how='left')
        df = df.merge(products, on='product_id2', how='left')

        # Calcul du taux de retard
        df['is_late'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)

        # Agrégation par SELLER_ID2
        seller_stats = df.groupby('seller_id2').agg({
            'price': 'sum',
            'review_score': 'mean',
            'is_late': 'mean',
            'payment_installments': 'mean',
            'order_purchase_timestamp': ['min', 'max']
        }).reset_index()

        seller_stats.columns = ['seller_id2','total_revenue', 'avg_review_score', 'late_rate', 'avg_installments', 'first_order', 'last_order']

        # Calcul de l'ancienneté
        seller_stats['active_months'] = ((seller_stats['last_order'] - seller_stats['first_order']).dt.days / 30).clip(lower=1)
        
        # Formule de solvabilité
        norm_revenue = (seller_stats['total_revenue'] - seller_stats['total_revenue'].min()) / (seller_stats['total_revenue'].max() - seller_stats['total_revenue'].min())
        norm_reviews = seller_stats['avg_review_score'] / 5
        norm_punctuality = 1 - seller_stats['late_rate']

        seller_stats['solvability_score'] = (
            (norm_revenue * 40) + 
            (norm_reviews * 30) + 
            (norm_punctuality * 30)
        ).round(2)

        # Ajout du state (car ta collègue a mis seller_id en index)
        #seller_stats = seller_stats.join(sellers['seller_state'], how='left')
        seller_stats = seller_stats.merge(sellers[['seller_id2', 'seller_state','seller_name']],on='seller_id2',how='left')
        
        #on retourne le truc 
        seller_stats.to_csv(MLConfig.SELLER_FEATURES_FILE, index=False)

        return seller_stats

def load_and_prepare_data(raw_data_dir) -> Tuple:
    """
    Charge les 6 bases nécessaires au Scoring Vendeur.
    Version optimisée avec les colonnes spécifiques.
    """
    print("📂 Chargement des données pour le Scoring Vendeur...")


    # 1. Sellers (pour le state)
    sellers = pd.read_csv(raw_data_dir / 'olist_sellers_dataset.csv', 
                         usecols=['seller_id', 'seller_state','seller_name'])
    
    # 2. Orders (pour les dates et le statut)
    orders = pd.read_csv(raw_data_dir / 'olist_orders_dataset.csv',
                        usecols=['order_id', 'order_status', 'order_purchase_timestamp', 
                                'order_delivered_customer_date', 'order_estimated_delivery_date'])
    
    # 3. Items (pour le prix et le lien vendeur/produit)
    items = pd.read_csv(raw_data_dir / 'olist_order_items_dataset.csv',
                       usecols=['seller_id', 'order_id', 'product_id', 'price', 'freight_value'])
    
    # 4. Payments (pour les mensualités / installments)
    payments = pd.read_csv(raw_data_dir / 'olist_order_payments_dataset.csv',
                          usecols=['order_id', 'payment_installments'])
    
    # 5. Reviews (pour la satisfaction client)
    reviews = pd.read_csv(raw_data_dir / 'olist_order_reviews_dataset.csv',
                         usecols=['order_id', 'review_score'])
    
    # 6. Products (pour la catégorie)
    products = pd.read_csv(raw_data_dir / 'olist_products_dataset.csv',
                          usecols=['product_id', 'product_category_name'])

    # Indexation des id pour plus de lisibilité
    sellers['seller_id2'] = pd.factorize(sellers['seller_id'])[0] + 1
    items = items.merge( sellers[['seller_id', 'seller_id2']],on='seller_id',how='left')

    orders['order_id2'] = pd.factorize(orders['order_id'])[0] + 1
    items = items.merge( orders[['order_id', 'order_id2']],on='order_id',how='left')
    reviews = reviews.merge( orders[['order_id', 'order_id2']],on='order_id',how='left')
    payments = payments.merge( orders[['order_id', 'order_id2']],on='order_id',how='left')

    products['product_id2'] = pd.factorize(products['product_id'])[0] + 1
    items = items.merge( products[['product_id', 'product_id2']],on='product_id',how='left')

    # Suppression des ancien id
    sellers.drop(columns=['seller_id'], inplace=True)
    items.drop(columns=['seller_id', 'order_id', 'product_id'], inplace=True)
    orders.drop(columns=['order_id'], inplace=True)
    products.drop(columns=['product_id'], inplace=True)

    # ajout des noms de vendeurs
    #sellers['seller_name'] = [f"vendeur{i+1}" for i in range(len(sellers))]

    # Conversion des dates (important pour calculer la récence!)
    date_columns = ['order_purchase_timestamp',
                    'order_delivered_customer_date','order_estimated_delivery_date']

    for col in date_columns:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col])

    print(f" {len(orders)} commandes, "
          f"{len(items)} items, {len(products)} produits, {len(reviews)} avis")

    # L'ORDRE : create_seller_features(self, sellers, orders, items, payments, reviews, products)
    return sellers, orders, items, payments, reviews, products
