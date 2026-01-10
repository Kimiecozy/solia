# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - SERVICE
# Master 2 - Data Science Industrielle
# ===============================================

"""
Service de recommandation pour l'API FastAPI.

Ce module fait l'interface entre l'API REST et le modèle de machine learning,
gérant le chargement du modèle, la cache, et les prédictions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import sys
import logging

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from config import MLConfig, PROCESSED_DATA_DIR
from ml_pipeline.models import OlistRecommendationModel
from ml_pipeline.preprocessing import CustomerFeatureEngineer
from backend.app.schemas.recommendation import (
    Recommendation, RecommendationResponse, ModelMetrics,
    FeatureImportance, ModelInfoResponse
)

logger = logging.getLogger(__name__)

class RecommendationService:
    """
    Service de recommandation centralisé.

    Gère:
    - Le chargement et la cache du modèle ML
    - La génération de recommandations
    - La gestion des features clients
    - Les métriques de performance
    """

    def __init__(self):
        self.model: Optional[OlistRecommendationModel] = None
        self.customer_features: Optional[pd.DataFrame] = None
        self.product_features: Optional[pd.DataFrame] = None
        self.feature_engineer: Optional[CustomerFeatureEngineer] = None
        self.model_loaded_at: Optional[datetime] = None
        self._cache: Dict = {}
        self._cache_ttl = timedelta(hours=1)

    async def initialize(self) -> bool:
        """
        Initialise le service en chargeant le modèle et les données.

        Returns:
            True si l'initialisation réussit, False sinon
        """
        try:
            logger.info("🔄 Initialisation du service de recommandation...")

            # Charger le modèle pré-entraîné
            await self._load_model()

            # Charger les features clients
            await self._load_customer_features()

            # Créer des features produits simulées
            await self._load_product_features()

            self.model_loaded_at = datetime.now()
            logger.info("✅ Service de recommandation initialisé avec succès")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {e}")
            return False

    async def _load_model(self):
        """Charge le modèle pré-entraîné."""
        try:
            self.model = OlistRecommendationModel.load_model()
            self.feature_engineer = CustomerFeatureEngineer()
            logger.info("✅ Modèle chargé avec succès")
        except FileNotFoundError:
            logger.warning("⚠️ Modèle non trouvé, création d'un modèle fictif pour la démo")
            await self._create_dummy_model()

    async def _create_dummy_model(self):
        """Crée un modèle fictif pour la démonstration."""
        self.model = OlistRecommendationModel()
        self.model.is_trained = True
        self.model.feature_columns = [
            'total_orders', 'total_spent', 'avg_order_value',
            'days_since_last_order', 'avg_review_score'
        ]
        self.model.training_score_ = {
            'train_accuracy': 0.85,
            'test_accuracy': 0.78,
            'auc_score': 0.82,
            'cv_mean': 0.80,
            'cv_std': 0.03
        }
        self.feature_engineer = CustomerFeatureEngineer()
        logger.info("✅ Modèle fictif créé pour la démo")

    async def _load_customer_features(self):
        """Charge les features clients pré-calculées."""
        try:
            customer_features_path = MLConfig.CUSTOMER_FEATURES_FILE
            if customer_features_path.exists():
                self.customer_features = pd.read_csv(customer_features_path, index_col=0)
                logger.info(f"✅ Features clients chargées: {len(self.customer_features)} clients")
            else:
                logger.warning("⚠️ Features clients non trouvées, création d'un dataset fictif")
                await self._create_dummy_customer_features()
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des features clients: {e}")
            await self._create_dummy_customer_features()

    async def _create_dummy_customer_features(self):
        """Crée des features clients fictives pour la démo."""
        np.random.seed(42)
        n_customers = 50

        # Créer des IDs clients factices
        customer_ids = [f"customer_{i:03d}" for i in range(n_customers)]

        # Générer des features aléatoires mais réalistes
        self.customer_features = pd.DataFrame({
            'total_orders': np.random.poisson(3, n_customers) + 1,
            'total_spent': np.random.exponential(200, n_customers) + 50,
            'days_since_last_order': np.random.exponential(30, n_customers) + 1,
            'avg_review_score': np.random.normal(4.0, 0.8, n_customers).clip(1, 5),
            'unique_products_bought': np.random.poisson(2, n_customers) + 1,
        }, index=customer_ids)

        # Calculer avg_order_value
        self.customer_features['avg_order_value'] = (
            self.customer_features['total_spent'] /
            self.customer_features['total_orders']
        )

        logger.info("✅ Features clients fictives créées")

    async def _load_product_features(self):
        """Charge ou crée les features produits."""
        np.random.seed(42)
        n_products = 30

        categories = [
            'cama_mesa_banho', 'beleza_saude', 'esporte_lazer',
            'informatica_acessorios', 'moveis_decoracao', 'utilidades_domesticas'
        ]

        product_ids = [f"product_{i:03d}" for i in range(n_products)]

        self.product_features = pd.DataFrame({
            'category_encoded': np.random.randint(0, len(categories), n_products),
            'weight': np.random.exponential(500, n_products) + 100,
            'popularity_score': np.random.beta(2, 5, n_products),
        }, index=product_ids)

        logger.info("✅ Features produits créées")

    def _get_cache_key(self, customer_id: str, n_recommendations: int) -> str:
        """Génère une clé de cache."""
        return f"{customer_id}_{n_recommendations}"

    def _is_cache_valid(self, cached_at: datetime) -> bool:
        """Vérifie si le cache est encore valide."""
        return datetime.now() - cached_at < self._cache_ttl

    async def get_recommendations(self, customer_id: str, n_recommendations: int = 10) -> RecommendationResponse:
        """
        Génère des recommandations personnalisées pour un client.

        Args:
            customer_id: ID du client
            n_recommendations: Nombre de recommandations

        Returns:
            Réponse avec les recommandations
        """
        # Vérifier le cache
        cache_key = self._get_cache_key(customer_id, n_recommendations)
        if cache_key in self._cache:
            cached_result, cached_at = self._cache[cache_key]
            if self._is_cache_valid(cached_at):
                logger.info(f"📋 Recommandations servies depuis le cache pour {customer_id}")
                return cached_result

        try:
            # Obtenir les features du client
            customer_features = await self._get_customer_features(customer_id)

            # Générer les recommandations
            if self.model and self.model.is_trained:
                recommendations = await self._generate_ml_recommendations(
                    customer_id, customer_features, n_recommendations
                )
            else:
                recommendations = await self._generate_dummy_recommendations(
                    customer_id, n_recommendations
                )

            # Créer la réponse
            response = RecommendationResponse(
                customer_id=customer_id,
                recommendations=recommendations,
                total_recommendations=len(recommendations)
            )

            # Mettre en cache
            self._cache[cache_key] = (response, datetime.now())

            logger.info(f"✅ {len(recommendations)} recommandations générées pour {customer_id}")
            return response

        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de recommandations: {e}")
            # Retourner des recommandations par défaut
            return await self._generate_fallback_recommendations(customer_id, n_recommendations)

    async def _get_customer_features(self, customer_id: str) -> Dict:
        """Récupère les features d'un client."""
        if customer_id in self.customer_features.index:
            return self.customer_features.loc[customer_id].to_dict()
        else:
            logger.warning(f"⚠️ Client {customer_id} non trouvé, utilisation de features par défaut")
            return {
                'total_orders': 1,
                'total_spent': 100.0,
                'avg_order_value': 100.0,
                'days_since_last_order': 30,
                'avg_review_score': 3.5,
                'unique_products_bought': 1
            }

    async def _generate_ml_recommendations(self, customer_id: str, customer_features: Dict, n_recommendations: int) -> List[Recommendation]:
        """Génère des recommandations avec le modèle ML."""
        try:
            # Utiliser le modèle pour obtenir les prédictions
            predictions = self.model.predict_proba(customer_features, self.product_features)

            # Limiter au nombre demandé
            top_predictions = predictions.head(n_recommendations)

            # Convertir en objets Recommendation
            recommendations = []
            for rank, (_, row) in enumerate(top_predictions.iterrows(), 1):
                product_id = row['product_id']
                probability = row['purchase_probability']

                recommendation = Recommendation(
                    customer_id=customer_id,
                    product_id=product_id,
                    purchase_probability=float(probability),
                    confidence=self._calculate_confidence(probability),
                    rank=rank
                )
                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"❌ Erreur dans les recommandations ML: {e}")
            return await self._generate_dummy_recommendations(customer_id, n_recommendations)

    async def _generate_dummy_recommendations(self, customer_id: str, n_recommendations: int) -> List[Recommendation]:
        """Génère des recommandations factices pour la démo."""
        np.random.seed(hash(customer_id) % 2**32)  # Seed basé sur customer_id pour la reproductibilité

        recommendations = []
        available_products = list(self.product_features.index) if self.product_features is not None else [f"product_{i:03d}" for i in range(30)]

        # Sélectionner des produits aléatoirement
        selected_products = np.random.choice(
            available_products,
            size=min(n_recommendations, len(available_products)),
            replace=False
        )

        for rank, product_id in enumerate(selected_products, 1):
            # Probabilité décroissante selon le rang
            probability = max(0.1, 0.9 - (rank - 1) * 0.1 + np.random.normal(0, 0.05))
            probability = min(0.95, max(0.05, probability))

            recommendation = Recommendation(
                customer_id=customer_id,
                product_id=product_id,
                purchase_probability=float(probability),
                confidence=self._calculate_confidence(probability),
                rank=rank
            )
            recommendations.append(recommendation)

        return recommendations

    async def _generate_fallback_recommendations(self, customer_id: str, n_recommendations: int) -> RecommendationResponse:
        """Génère des recommandations de secours en cas d'erreur."""
        recommendations = [
            Recommendation(
                customer_id=customer_id,
                product_id=f"product_{i:03d}",
                purchase_probability=0.5,
                confidence="Medium",
                rank=i+1
            )
            for i in range(n_recommendations)
        ]

        return RecommendationResponse(
            customer_id=customer_id,
            recommendations=recommendations,
            total_recommendations=len(recommendations)
        )

    def _calculate_confidence(self, probability: float) -> str:
        """Calcule le niveau de confiance basé sur la probabilité."""
        if probability >= 0.8:
            return 'High'
        elif probability >= 0.6:
            return 'Medium'
        elif probability >= 0.4:
            return 'Low'
        else:
            return 'Very Low'

    async def get_model_info(self) -> ModelInfoResponse:
        """
        Retourne les informations sur le modèle.

        Returns:
            Informations détaillées sur le modèle
        """
        if not self.model or not self.model.is_trained:
            # Modèle non chargé, retourner des infos par défaut
            metrics = ModelMetrics(
                train_accuracy=0.0,
                test_accuracy=0.0,
                auc_score=0.0,
                cv_mean=0.0,
                cv_std=0.0
            )
            return ModelInfoResponse(
                metrics=metrics,
                feature_importance=[],
                model_status="not_loaded"
            )

        # Métriques du modèle
        perf = self.model.get_model_performance()
        metrics = ModelMetrics(
            train_accuracy=perf['train_accuracy'],
            test_accuracy=perf['test_accuracy'],
            auc_score=perf['auc_score'],
            cv_mean=perf['cv_mean'],
            cv_std=perf['cv_std'],
            last_trained=self.model_loaded_at
        )

        # Importance des features
        feature_importance = []
        if hasattr(self.model, 'get_feature_importance'):
            importance_df = self.model.get_feature_importance()
            for _, row in importance_df.iterrows():
                feature_importance.append(FeatureImportance(
                    feature=row['feature'],
                    importance=row['importance']
                ))

        return ModelInfoResponse(
            metrics=metrics,
            feature_importance=feature_importance,
            model_status="ready"
        )

    async def get_customer_list(self) -> List[str]:
        """Retourne la liste des clients disponibles."""
        if self.customer_features is not None:
            return self.customer_features.index.tolist()
        else:
            return [f"customer_{i:03d}" for i in range(10)]

    def is_healthy(self) -> bool:
        """Vérifie si le service est en bonne santé."""
        return (
            self.model is not None and
            self.customer_features is not None and
            self.product_features is not None
        )

# Instance singleton du service
recommendation_service = RecommendationService()