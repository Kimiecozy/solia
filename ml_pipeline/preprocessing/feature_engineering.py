# ===============================================
# OLIST RECOMMENDATION SYSTEM - FEATURE ENGINEERING
# Master 2 - SEP
# Version Simplifiée pour Apprentissage
# ===============================================

"""
Module de feature engineering pour le système de recommandation Olist.

Ce module transforme les données brutes en features utilisables pour
le modèle de machine learning.

OBJECTIFS PÉDAGOGIQUES:
    - Comprendre les features RFM (Récence, Fréquence, Montant)
    - Apprendre la normalisation des données
    - Découvrir l'encodage des variables catégorielles

FEATURES IMPLÉMENTÉES (SIMPLES):
    - Total de commandes par client
    - Montant total dépensé
    - Valeur moyenne des commandes
    - Jours depuis la dernière commande (Récence)
    - Catégorie préférée (encodée)

FEATURES AVANCÉES (COMMENTÉES - À EXPLORER):
    - Segmentation RFM complète
    - Score de fidélité client
    - Tendances d'achat saisonnières
    - Diversité des produits achetés
"""

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


class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformeur pour créer des features clients.

    CONCEPTS CLÉS:
    - RFM: Récence (quand?), Fréquence (combien?), Montant (€?)
    - Ces 3 métriques permettent de segmenter les clients

    POUR ALLER PLUS LOIN:
    - Ajouter des features temporelles (saisonnalité)
    - Calculer des ratios (ex: panier moyen vs médiane)
    - Créer des scores de fidélité personnalisés
    """

    def __init__(self):
        self.reference_date = None
        self.category_encoder = LabelEncoder()
        self._is_fitted = False

    def fit(self, X, y=None):
        """
        Apprend les paramètres nécessaires sur les données d'entraînement.

        NOTE: Cette étape est cruciale pour éviter le "data leakage"
        On apprend les paramètres UNIQUEMENT sur le train set.
        """
        if isinstance(X, pd.DataFrame):
            # Déterminer la date de référence pour calculer la récence
            if 'order_purchase_timestamp' in X.columns:
                self.reference_date = X['order_purchase_timestamp'].max()
            elif 'last_order_date' in X.columns:
                self.reference_date = pd.to_datetime(X['last_order_date']).max()
            else:
                self.reference_date = datetime.now()

            # Encoder les catégories (transformation texte -> nombre)
            if 'favorite_category' in X.columns:
                categories = X['favorite_category'].fillna('unknown').astype(str)
                print(f"      Apprentissage de {categories.nunique()} catégories")
                self.category_encoder.fit(categories)
            else:
                self.category_encoder.fit(['unknown'])
        else:
            self.reference_date = datetime.now()
            self.category_encoder.fit(['unknown'])

        self._is_fitted = True
        return self

    def transform(self, X):
        """Transforme les données en features."""
        if isinstance(X, dict):
            return self._transform_single_customer(X)
        else:
            return self._transform_dataframe(X)

    def _transform_single_customer(self, customer_data: Dict) -> Dict:
        """
        Transforme les données d'un client unique.
        Utilisé pour faire des prédictions en temps réel.
        """
        features = {}

        # ============================================
        # FEATURES DE BASE (IMPLÉMENTÉES)
        # ============================================

        # 1. Fréquence: Nombre total de commandes
        features['total_orders'] = customer_data.get('total_orders', 0)

        # 2. Montant: Total dépensé
        features['total_spent'] = customer_data.get('total_spent', 0.0)

        # 3. Panier moyen
        features['avg_order_value'] = (
                features['total_spent'] / max(features['total_orders'], 1)
        )

        # 4. Récence: Jours depuis la dernière commande
        last_order = customer_data.get('last_order_date')
        if last_order:
            if isinstance(last_order, str):
                last_order = pd.to_datetime(last_order)
            features['days_since_last_order'] = (self.reference_date - last_order).days
        else:
            features['days_since_last_order'] = 365

        # 5. Catégorie préférée (encodée en nombre)
        favorite_cat = customer_data.get('favorite_category', 'unknown')
        if self._is_fitted:
            features['favorite_category_encoded'] = self.category_encoder.transform([favorite_cat])[0]
        else:
            features['favorite_category_encoded'] = 0

        # ============================================
        # FEATURES AVANCÉES (À IMPLÉMENTER)
        # ============================================

        # IDÉE 1: Score de satisfaction client
        # features['avg_review_score'] = customer_data.get('avg_review_score', 3.0)
        # Pourquoi utile? Les clients satisfaits achètent plus

        # IDÉE 2: Diversité des achats
        # features['unique_products_bought'] = customer_data.get('unique_products_bought', 0)
        # Pourquoi utile? Mesure l'exploration vs fidélité à certains produits

        # IDÉE 3: Tendance d'achat
        # features['purchase_trend'] = self._calculate_purchase_trend(customer_data)
        # Pourquoi utile? Détecte si le client achète de + en + ou de - en -

        return features

    def _transform_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme un DataFrame complet (batch de clients).
        Utilisé pendant l'entraînement du modèle.
        """
        X_transformed = X.copy()

        # ============================================
        # FEATURES DE BASE (IMPLÉMENTÉES)
        # ============================================

        # Panier moyen = Montant total / Nombre de commandes
        X_transformed['avg_order_value'] = (
                X_transformed['total_spent'] / X_transformed['total_orders'].clip(lower=1)
        )

        # Récence: Nombre de jours depuis la dernière commande
        if 'last_order_date' in X_transformed.columns:
            X_transformed['last_order_date'] = pd.to_datetime(X_transformed['last_order_date'])
            X_transformed['days_since_last_order'] = (
                    self.reference_date - X_transformed['last_order_date']
            ).dt.days
        else:
            X_transformed['days_since_last_order'] = 365

        # Encoder la catégorie préférée
        if 'favorite_category' in X_transformed.columns and self._is_fitted:
            try:
                categories = X_transformed['favorite_category'].fillna('unknown').astype(str)
                X_transformed['favorite_category_encoded'] = self.category_encoder.transform(categories)
            except ValueError as e:
                print(f"      Catégories inconnues détectées, utilisation de 0 par défaut")
                known_categories = set(self.category_encoder.classes_)
                X_transformed['favorite_category_encoded'] = categories.apply(
                    lambda x: self.category_encoder.transform([x])[0] if x in known_categories else 0
                )
        else:
            X_transformed['favorite_category_encoded'] = 0

        # ============================================
        # SEGMENTATION RFM (COMMENTÉE)
        # ============================================
        # À DÉCOMMENTER pour aller plus loin
        # X_transformed = self._add_rfm_segments(X_transformed)

        # EXERCICE POUR L'ÉTUDIANT:
        # 1. Décommenter la ligne ci-dessus
        # 2. Observer les nouveaux segments créés
        # 3. Analyser quelle est la distribution des clients par segment
        # 4. Question: Quels segments sont les plus rentables?

        return X_transformed

    def _add_rfm_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FONCTION AVANCÉE: Segmentation RFM

        Cette fonction crée des segments de clients basés sur RFM:
        - R (Recency): À quel point le client est récent?
        - F (Frequency): À quelle fréquence achète-t-il?
        - M (Monetary): Combien dépense-t-il?

        RÉSULTAT: Chaque client reçoit un score RFM (ex: "111" = Champion)

        POUR L'ÉTUDIANT:
        - Essayez de modifier le nombre de quartiles (q=4 -> q=3 ou q=5)
        - Observez comment cela change la distribution des segments
        - Créez vos propres règles de segmentation personnalisées
        """
        try:
            df['recency_quartile'] = pd.qcut(
                df['days_since_last_order'], q=4, labels=False, duplicates='drop'
            ) + 1
        except ValueError:
            df['recency_quartile'] = pd.cut(
                df['days_since_last_order'], bins=4, labels=False, duplicates='drop'
            ) + 1

        try:
            freq_quartiles = pd.qcut(
                df['total_orders'], q=4, labels=False, duplicates='drop'
            )
            df['frequency_quartile'] = 4 - freq_quartiles
        except ValueError:
            freq_bins = pd.cut(
                df['total_orders'], bins=4, labels=False, duplicates='drop'
            )
            df['frequency_quartile'] = 4 - freq_bins

        try:
            mon_quartiles = pd.qcut(
                df['total_spent'], q=4, labels=False, duplicates='drop'
            )
            df['monetary_quartile'] = 4 - mon_quartiles
        except ValueError:
            mon_bins = pd.cut(
                df['total_spent'], bins=4, labels=False, duplicates='drop'
            )
            df['monetary_quartile'] = 4 - mon_bins

        df['recency_quartile'] = df['recency_quartile'].fillna(1).astype(int)
        df['frequency_quartile'] = df['frequency_quartile'].fillna(1).astype(int)
        df['monetary_quartile'] = df['monetary_quartile'].fillna(1).astype(int)

        # Score RFM combiné (ex: "111", "444", etc.)
        df['rfm_score'] = (
                df['recency_quartile'].astype(str) +
                df['frequency_quartile'].astype(str) +
                df['monetary_quartile'].astype(str)
        )

        # Segments business lisibles
        df['customer_segment'] = df['rfm_score'].apply(self._get_customer_segment)

        return df

    @staticmethod
    def _get_customer_segment(rfm_score: str) -> str:
        """
        Mapping des scores RFM vers des segments business.

        LÉGENDE:
        - Champions: Achètent souvent, récemment, et beaucoup
        - At Risk: Bons clients qui n'ont pas acheté récemment
        - Lost: Clients inactifs depuis longtemps

        EXERCICE: Créez vos propres règles de segmentation!
        """
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
    Transformeur pour créer des features produits (VERSION SIMPLIFIÉE).

    FEATURES IMPLÉMENTÉES:
    - Catégorie du produit (encodée)
    - Dimensions physiques (normalisées)

    FEATURES À EXPLORER:
    - Popularité du produit (nombre de ventes)
    - Score moyen des avis
    - Prix relatif par catégorie
    - Taux de retour
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.category_encoder = LabelEncoder()
        self._is_fitted = False

    def fit(self, X, y=None):
        """Apprend les paramètres de normalisation."""
        numeric_cols = ['product_weight_g', 'product_length_cm',
                        'product_height_cm', 'product_width_cm']
        available_cols = [col for col in numeric_cols if col in X.columns]

        if available_cols:
            print(f"      Normalisation de {len(available_cols)} dimensions produit")
            self.scaler.fit(X[available_cols].fillna(0))

        if 'product_category_name' in X.columns:
            self.category_encoder.fit(X['product_category_name'].fillna('unknown'))

        self._is_fitted = True
        return self

    def transform(self, X):
        """Transforme les données produits."""
        X_transformed = X.copy()

        # Normaliser les dimensions (StandardScaler: moyenne=0, écart-type=1)
        numeric_cols = ['product_weight_g', 'product_length_cm',
                        'product_height_cm', 'product_width_cm']
        available_cols = [col for col in numeric_cols if col in X_transformed.columns]

        if available_cols and self._is_fitted:
            X_transformed[available_cols] = self.scaler.transform(
                X_transformed[available_cols].fillna(0)
            )

        # Encoder la catégorie
        if 'product_category_name' in X_transformed.columns and self._is_fitted:
            X_transformed['category_encoded'] = self.category_encoder.transform(
                X_transformed['product_category_name'].fillna('unknown')
            )

        # ============================================
        # FEATURE CALCULÉE: Volume du produit
        # ============================================
        # EXERCICE: Quelles autres features géométriques pourrait-on créer?
        # Idées: surface, ratio longueur/largeur, densité (poids/volume), etc.

        # if all(col in X_transformed.columns for col in ['product_length_cm', 'product_width_cm', 'product_height_cm']):
        #     X_transformed['product_volume'] = (
        #         X_transformed['product_length_cm'] *
        #         X_transformed['product_width_cm'] *
        #         X_transformed['product_height_cm']
        #     )

        return X_transformed


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
        self.customer_engineer = CustomerFeatureEngineer()
        self.product_engineer = ProductFeatureEngineer()
        self.is_fitted = False

    def create_seller_features(self, sellers, orders, items, payments, reviews, products):
        print("🏗️ Fusion et calcul de la solvabilité par vendeur...")

        # Ta logique de merge
        df = items.merge(orders, on='order_id')
        df = df.merge(payments, on='order_id', how='left')
        df = df.merge(reviews, on='order_id', how='left')
        df = df.merge(products, on='product_id', how='left')

        # Calcul du taux de retard
        df['is_late'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)

        # Agrégation par SELLER_ID2
        seller_stats = df.groupby('seller_id2').agg({
            'price': 'sum',
            'review_score': 'mean',
            'is_late': 'mean',
            'payment_installments': 'mean',
            'order_purchase_timestamp': ['min', 'max']
        })

        seller_stats.columns = ['total_revenue', 'avg_review_score', 'late_rate', 'avg_installments', 'first_order', 'last_order']

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
        seller_stats = seller_stats.join(sellers['seller_state'], how='left')

        return seller_stats

    def fit(self, customer_data: pd.DataFrame, product_data: pd.DataFrame):
        """Entraîne les transformeurs sur les données."""
        print("   Entraînement des feature engineers...")
        self.customer_engineer.fit(customer_data)
        self.product_engineer.fit(product_data)
        self.is_fitted = True
        return self

    def create_customer_features(self, orders_df: pd.DataFrame, reviews_df: pd.DataFrame,
                                 order_items_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les features clients à partir des données brutes.

        AGRÉGATIONS RÉALISÉES:
        1. Nombre de commandes par client
        2. Montant total dépensé
        3. Date de la dernière commande
        4. Catégorie préférée (mode statistique)
        5. Score moyen des avis (si disponible)

        CONCEPTS CLÉS:
        - groupby: Agrégation de données par client
        - merge/join: Combinaison de plusieurs tables
        - mode: Valeur la plus fréquente (catégorie préférée)
        """
        print("Création des features clients...")

        # ============================================
        # AGRÉGATION 1: Statistiques de commandes
        # ============================================
        customer_orders = orders_df.groupby('customer_id').agg({
            'order_id': 'count',  # Nombre de commandes
            'order_purchase_timestamp': 'max'  # Dernière commande
        }).rename(columns={
            'order_id': 'total_orders',
            'order_purchase_timestamp': 'last_order_date'
        })

        # ============================================
        # AGRÉGATION 2: Montant total dépensé
        # ============================================
        customer_spending = order_items_df.merge(
            orders_df[['order_id', 'customer_id']], on='order_id'
        ).groupby('customer_id')['price'].sum().to_frame('total_spent')

        # ============================================
        # AGRÉGATION 3: Catégorie préférée
        # ============================================
        # On utilise mode() = valeur la plus fréquente
        customer_categories = order_items_df.merge(
            orders_df[['order_id', 'customer_id']], on='order_id'
        ).merge(
            products_df[['product_id', 'product_category_name']], on='product_id'
        ).groupby('customer_id')['product_category_name'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        ).to_frame('favorite_category')

        # ============================================
        # AGRÉGATION 4: Score moyen des avis
        # ============================================
        if not reviews_df.empty:
            customer_reviews = reviews_df.merge(
                orders_df[['order_id', 'customer_id']], on='order_id'
            ).groupby('customer_id')['review_score'].mean().to_frame('avg_review_score')
        else:
            customer_reviews = pd.DataFrame(columns=['avg_review_score'])

        # ============================================
        # AGRÉGATION 5: Diversité des produits
        # ============================================
        # FEATURE AVANCÉE (commentée)
        # customer_products = order_items_df.merge(
        #     orders_df[['order_id', 'customer_id']], on='order_id'
        # ).groupby('customer_id')['product_id'].nunique().to_frame('unique_products_bought')

        # ============================================
        # COMBINAISON de toutes les features
        # ============================================
        customer_features = customer_orders.join([
            customer_spending,
            customer_categories,
            customer_reviews,
            # customer_products  # Décommenter si activé
        ], how='left')

        # Remplir les valeurs manquantes avec des valeurs par défaut
        customer_features['total_spent'] = customer_features['total_spent'].fillna(0)
        customer_features['favorite_category'] = customer_features['favorite_category'].fillna('unknown')
        customer_features['avg_review_score'] = customer_features['avg_review_score'].fillna(3.0)
        # customer_features['unique_products_bought'] = customer_features['unique_products_bought'].fillna(0)

        # ============================================
        # TRANSFORMATION FINALE
        # ============================================
        print("   Re-fit avec les vraies données...")
        self.customer_engineer.fit(customer_features)
        self.is_fitted = True

        print("   Transformation des features...")
        customer_features = self.customer_engineer.transform(customer_features)

        print(f"   {len(customer_features)} clients traités")
        print(f"   {len(customer_features.columns)} features créées")
        return customer_features
    

    def create_interaction_matrix(self, orders_df: pd.DataFrame,
                                  order_items_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée la matrice d'interaction client-produit.

        CONCEPT: Collaborative Filtering
        Cette matrice enregistre "qui a acheté quoi" et sert de base
        pour trouver des patterns d'achat similaires.

        STRUCTURE:
        - Chaque ligne = une interaction (achat)
        - Colonnes: customer_id, product_id, purchased (1), rating, etc.

        AMÉLIORATION POSSIBLE:
        - Ajouter un poids basé sur le nombre d'achats du même produit
        - Intégrer les avis clients comme score de qualité
        - Tenir compte du temps (achats récents = plus pertinents)
        """
        print("Création de la matrice d'interaction...")

        # Joindre commandes et items
        interactions = items_df.merge(
            orders_df[['order_id', 'customer_id', 'order_status']], on='order_id'
        )

        # IMPORTANT: Ne garder que les commandes livrées
        # Pourquoi? Commandes annulées/retournées != vraies préférences
        interactions = interactions[interactions['order_status'] == 'delivered']

        # Features d'interaction de base
        interactions['purchased'] = 1  # Variable cible binaire

        # FEATURE AVANCÉE: Rating normalisé basé sur le prix
        # Hypothèse: Prix élevé = plus d'engagement/intérêt
        # interactions['rating'] = interactions['price'] / interactions['price'].max()

        # FEATURE TEMPORELLE (commentée)
        # if 'order_purchase_timestamp' in orders_df.columns:
        #     interactions = interactions.merge(
        #         orders_df[['order_id', 'order_purchase_timestamp']], on='order_id'
        #     )
        #     interactions['order_purchase_timestamp'] = pd.to_datetime(interactions['order_purchase_timestamp'])
        #     interactions['days_ago'] = (
        #         interactions['order_purchase_timestamp'].max() - interactions['order_purchase_timestamp']
        #     ).dt.days

        print(f"   {len(interactions)} interactions créées")
        print(f"   {interactions['customer_id'].nunique()} clients uniques")
        print(f"   {interactions['product_id'].nunique()} produits uniques")

        return interactions


def load_and_prepare_data(raw_data_dir) -> Tuple:
    """
    Charge les 6 bases nécessaires au Scoring Vendeur.
    Version optimisée avec les colonnes spécifiques.
    """
    print("📂 Chargement des données pour le Scoring Vendeur...")


    # 1. Sellers (pour le state)
    sellers = pd.read_csv(raw_data_dir / 'olist_sellers_dataset.csv', 
                         usecols=['seller_id', 'seller_state'])
    
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

    # Conversion des dates (important pour calculer la récence!)
    date_columns = ['order_purchase_timestamp',
                    'order_delivered_customer_date','order_estimated_delivery_date']

    for col in date_columns:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col])

    print(f" {len(orders)} commandes, "
          f"{len(items)} items, {len(products)} produits, {len(reviews)} avis")

    # L'ORDRE : create_seller_features(self, sellers, orders, items, payments, reviews, products)
    return sellers, orders, items, products, reviews, payments

