# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - STREAMLIT APP
# Master 2 - SEP
# ===============================================

"""
Interface utilisateur Streamlit pour le système de recommandation Olist.

Cette application web permet de:
- Tester les recommandations personnalisées
- Visualiser les performances du modèle
- Explorer les données et résultats
- Démontrer le système complet aux étudiants

Architecture:
- Streamlit pour l'interface utilisateur
- Appels API REST vers le backend FastAPI
- Visualisations interactives avec Plotly
- Cache pour optimiser les performances
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import sys
from pathlib import Path
import os

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

# 1. On importe la classe MLConfig au lieu des variables directes
from config import MLConfig

# Configuration de l'API
API_BASE_URL = "http://localhost:8000/api/v1"

# 2. On utilise le préfixe MLConfig. pour accéder aux chemins
@st.cache_data
def load_seller_data():
    df = pd.read_csv(MLConfig.SELLER_FEATURES_FILE, sep=";", index_col="seller_id")
    # Ajouter le mapping name → id
    df['seller_name'] = [f"vendeur{i+1}" for i in range(len(df))]
    return df



@st.cache_resource
def load_credit_model():
    """Charge le modèle de prédiction de revenus."""
    return joblib.load(MLConfig.REVENUE_MODEL_FILE)

# Initialisation des données
df_sellers = load_seller_data()
model_revenue = load_credit_model()

def get_recommendations(customer_id, n_recommendations=10):
    """Obtient les recommandations pour un client."""
    try:
        payload = {
            "customer_id": customer_id,
            "n_recommendations": n_recommendations
        }
        response = requests.post(f"{API_BASE_URL}/recommendations", json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération de recommandations: {e}")
        return None

def check_api_health():
    """Vérifie la santé de l'API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except:
        return False

def main():
    st.title("🏦 SolIA - Scoring de Crédit Vendeur")
    st.markdown("### Analyse de solvabilité pour les vendeurs Olist")

    with st.sidebar:
        st.markdown("## 🔍 Sélection")

        df_sellers = load_seller_data()
        name_to_id = {df_sellers.loc[idx, 'seller_name']: idx for idx in df_sellers.index}
    
        seller_name = st.selectbox("Choisir un Vendeur", options=list(name_to_id.keys()))
        seller_id = name_to_id[seller_name]
    
        
        st.markdown("---")
        page = st.radio("Navigation", ["🎯 Verdict Crédit", "📊 Analyse du Modèle"])

    vendeur = df_sellers.loc[seller_id]

    if page == "🎯 Verdict Crédit":
        show_credit_scoring_page(vendeur)
    else:
        show_model_analysis_page()


def show_model_analysis_page():
    """Affiche les performances techniques du modèle de régression."""
    st.header("📊 Analyse technique du modèle")

    # 1. Explication des métriques
    st.markdown("""
    Le modèle utilise un algorithme **Random Forest Regressor** pour prédire le chiffre d'affaires 
    futur d'un vendeur en se basant sur son historique de performance.
    """)

    # 2. Importance des Features (Le pourquoi du score)
    st.subheader("🔍 Importance des critères")
    
    # On récupère l'importance des variables directement depuis le modèle chargé
    importances = model_revenue.feature_importances_
    features = ['Note Moyenne', 'Taux de Retard', 'Mensualités Moy.', 'Ancienneté', 'Score de Solvabilité']
    
    df_importance = pd.DataFrame({
        'Critère': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    fig = px.bar(
        df_importance, 
        x='Importance', 
        y='Critère', 
        orientation='h',
        title="Qu'est-ce qui influence le plus la prédiction ?",
        color_discrete_sequence=['#2ecc71']
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Rappel des performances globales (Métriques de l'entraînement)
    st.subheader("🎯 Précision du système")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("R² Score (Fiabilité)", "0.95", help="Plus le score est proche de 1, plus le modèle est précis.")
        st.write("Le modèle explique 95% des variations de revenus.")
        
    with col2:
        st.metric("Erreur Moyenne (MAE)", "1779 R$", help="Écart moyen entre la prédiction et la réalité.")
        st.write("L'incertitude moyenne sur le CA prédit.")



def show_credit_scoring_page(vendeur):
    st.header(f"Analyse du Vendeur : **{vendeur['seller_name']}** (ID: {vendeur.name[:8]}...)")


    
    # 1. Le Score de Solvabilité
    score = vendeur['solvability_score']
    
    if score >= 75:
        st.success(f"### ✅ ÉLIGIBLE AU PRÊT (Score : {score}/100)")
        st.balloons()
    elif score >= 50:
        st.warning(f"### ⚠️ DOSSIER À ÉTUDIER (Score : {score}/100)")
    else:
        st.error(f"### ❌ PRÊT REFUSÉ (Score : {score}/100)")

    # 2. Indicateurs Clés
    col1, col2, col3 = st.columns(3)
    col1.metric("CA Total", f"{vendeur['total_revenue']:.2f} R$")
    col2.metric("Note Moyenne", f"{vendeur['avg_review_score']:.1f} / 5")
    col3.metric("Taux de Retard", f"{vendeur['late_rate']*100:.1f} %")

    st.markdown("---")
    
    # 3. Prédiction du CA Futur
    st.subheader("🔮 Capacité de Remboursement")
    # Préparer les données pour le modèle (doit être le même ordre que dans train_model.py)
    input_data = pd.DataFrame([[
        vendeur['avg_review_score'], 
        vendeur['late_rate'], 
        vendeur['avg_installments'], 
        vendeur['active_months'], 
        vendeur['solvability_score']
    ]])
    prediction = model_revenue.predict(input_data)[0]
    
    st.write(f"Notre IA estime que ce vendeur peut générer **{prediction:.2f} R$** de revenus futurs.")
    st.info(f"Mensualité maximale conseillée : **{(prediction * 0.3):.2f} R$** (30% du CA prédit)")

def show_recommendations_page():
    """Page principale de génération de recommandations."""

    st.markdown("## 🎯 Recommandations Personnalisées")

    # Configuration des recommandations
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Sélection du client")

        # Charger la liste des clients
        customers = get_customers()
        if not customers:
            st.warning("Aucun client disponible")
            return

        customer_id = st.selectbox(
            "Client à analyser",
            customers,
            help="Sélectionnez un client pour générer ses recommandations personnalisées"
        )

    with col2:
        st.markdown("### Paramètres")
        n_recommendations = st.slider(
            "Nombre de recommandations",
            min_value=1,
            max_value=20,
            value=10,
            help="Nombre de produits à recommander"
        )

    # Bouton de génération
    if st.button("🚀 Générer les recommandations", type="primary"):
        with st.spinner("Génération des recommandations..."):
            recommendations_data = get_recommendations(customer_id, n_recommendations)

        if recommendations_data:
            display_recommendations(recommendations_data)
        else:
            st.error("Impossible de générer les recommandations")

def display_recommendations(data):
    """Affiche les recommandations de manière interactive."""

    st.markdown("---")
    st.markdown("## 🎁 Recommandations Générées")

    # Informations générales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Client ID", data["customer_id"])
    with col2:
        st.metric("Recommandations", data["total_recommendations"])
    with col3:
        st.metric("Généré le", data["generated_at"][:10])

    # Graphique des probabilités
    recommendations = data["recommendations"]
    df_recs = pd.DataFrame(recommendations)

    fig = px.bar(
        df_recs,
        x="rank",
        y="purchase_probability",
        color="confidence",
        title="📈 Probabilités d'achat par produit",
        labels={
            "rank": "Rang de la recommandation",
            "purchase_probability": "Probabilité d'achat",
            "confidence": "Niveau de confiance"
        }
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Tableau détaillé
    st.markdown("### 📋 Détail des recommandations")

    for i, rec in enumerate(recommendations):
        with st.expander(f"#{rec['rank']} - {rec['product_id']} (Probabilité: {rec['purchase_probability']:.3f})"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Product ID:** {rec['product_id']}")
                st.write(f"**Probabilité:** {rec['purchase_probability']:.3f}")
                st.write(f"**Confiance:** {rec['confidence']}")



def show_model_performance_page():
    """Page d'analyse des performances du modèle."""

    st.markdown("## 📊 Performance du Modèle ML")

    model_info = get_model_info()
    if not model_info:
        st.error("Impossible de récupérer les informations du modèle")
        return

    metrics = model_info.get("metrics", {})
    feature_importance = model_info.get("feature_importance", [])

    # Métriques principales
    st.markdown("### 🎯 Métriques de Performance")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Précision Train",
            f"{metrics.get('train_accuracy', 0):.3f}",
            help="Précision sur les données d'entraînement"
        )
    with col2:
        st.metric(
            "Précision Test",
            f"{metrics.get('test_accuracy', 0):.3f}",
            help="Précision sur les données de test"
        )
    with col3:
        st.metric(
            "Score AUC",
            f"{metrics.get('auc_score', 0):.3f}",
            help="Area Under the ROC Curve"
        )
    with col4:
        st.metric(
            "CV Score",
            f"{metrics.get('cv_mean', 0):.3f}",
            help="Score de validation croisée"
        )

    # Interprétation des résultats
    auc_score = metrics.get('auc_score', 0)
    if auc_score >= 0.9:
        st.success("🏆 Performance excellente! Le modèle distingue très bien les clients qui vont acheter.")
    elif auc_score >= 0.8:
        st.success("👍 Très bonne performance! Le modèle est fiable pour les recommandations.")
    elif auc_score >= 0.7:
        st.info("✅ Performance correcte. Le modèle peut être amélioré.")
    else:
        st.warning("⚠️ Performance faible. Considérez l'amélioration du modèle.")

    # Importance des features
    if feature_importance:
        st.markdown("### 🔍 Importance des Features")

        df_importance = pd.DataFrame(feature_importance)
        fig_importance = px.bar(
            df_importance.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 des features les plus importantes",
            labels={"importance": "Importance", "feature": "Feature"}
        )
        fig_importance.update_layout(height=500)
        st.plotly_chart(fig_importance, use_container_width=True)

        # Explication pédagogique
        st.markdown("#### 💡 Interprétation")
        st.write("""
        **L'importance des features nous indique:**
        - Quelles informations client sont les plus prédictives
        - Comment améliorer le modèle en collectant de meilleures données
        - Quels aspects du comportement client privilégier

        **Features typiques importantes:**
        - `total_spent`: Montant total dépensé par le client
        - `total_orders`: Nombre de commandes passées
        - `avg_review_score`: Satisfaction moyenne du client
        - `days_since_last_order`: Récence de la dernière commande
        """)

def show_data_analysis_page():
    """Page d'analyse exploratoire des données."""

    st.markdown("## 🔍 Analyse des Données")

    # Simuler quelques analyses avec des données factices
    st.markdown("### 👥 Distribution des Clients")

    # Génération de données simulées pour la démo
    import numpy as np
    np.random.seed(42)

    n_customers = 50
    customer_data = {
        'Total Orders': np.random.poisson(3, n_customers) + 1,
        'Total Spent': np.random.exponential(200, n_customers) + 50,
        'Avg Review Score': np.random.normal(4.0, 0.8, n_customers).clip(1, 5),
        'Days Since Last Order': np.random.exponential(30, n_customers) + 1
    }

    df_customers = pd.DataFrame(customer_data)

    col1, col2 = st.columns(2)

    with col1:
        # Distribution du nombre de commandes
        fig_orders = px.histogram(
            df_customers,
            x='Total Orders',
            title="Distribution du nombre de commandes",
            labels={'Total Orders': 'Nombre de commandes', 'count': 'Nombre de clients'}
        )
        st.plotly_chart(fig_orders, use_container_width=True)

    with col2:
        # Distribution des montants dépensés
        fig_spent = px.histogram(
            df_customers,
            x='Total Spent',
            title="Distribution des montants dépensés",
            labels={'Total Spent': 'Montant dépensé (€)', 'count': 'Nombre de clients'}
        )
        st.plotly_chart(fig_spent, use_container_width=True)

    # Corrélations
    st.markdown("### 🔗 Analyse des Corrélations")
    correlation_matrix = df_customers.corr()

    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Matrice de corrélation des features clients"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Segmentation RFM simplifiée
    st.markdown("### 📊 Segmentation RFM")

    # Calculer des quartiles
    df_customers['Recency_Score'] = pd.qcut(df_customers['Days Since Last Order'], 4, labels=['4', '3', '2', '1'])
    df_customers['Frequency_Score'] = pd.qcut(df_customers['Total Orders'], 4, labels=['1', '2', '3', '4'], duplicates='drop')
    df_customers['Monetary_Score'] = pd.qcut(df_customers['Total Spent'], 4, labels=['1', '2', '3', '4'], duplicates='drop')

    # Distribution des segments
    segment_counts = df_customers.groupby(['Frequency_Score', 'Monetary_Score']).size().reset_index(name='Count')

    fig_segments = px.scatter(
        segment_counts,
        x='Frequency_Score',
        y='Monetary_Score',
        size='Count',
        title="Segmentation Fréquence vs Montant",
        labels={
            'Frequency_Score': 'Score de Fréquence',
            'Monetary_Score': 'Score Monétaire'
        }
    )
    st.plotly_chart(fig_segments, use_container_width=True)

if __name__ == "__main__":
    main()
    os.system('streamlit run app.py')