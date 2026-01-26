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


# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))


# 1. On importe la classe MLConfig au lieu des variables directes
from config import MLConfig, RAW_DATA_DIR
from ml_pipeline.preprocessing.feature_engineering import load_and_prepare_data

# Configuration de l'API
API_BASE_URL = "http://localhost:8000/api/v1"

# 2. On utilise le préfixe MLConfig. pour accéder aux chemins
@st.cache_data
def load_seller_data():
    """Charge la base des vendeurs générée par le train_model.py."""
    return pd.read_csv(MLConfig.SELLER_FEATURES_FILE)

@st.cache_resource
def load_credit_model():
    """Charge le modèle de prédiction de revenus."""
    return joblib.load(MLConfig.REVENUE_MODEL_FILE)

# Initialisation des données
#sellers = load_seller_data()
sellers = load_and_prepare_data(RAW_DATA_DIR)
model_revenue = load_credit_model()

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
        # Sélection du vendeur via l'index (ID)
        seller_id2 = st.selectbox("Choisir un Vendeur", options=sellers['seller_id2'])
        
        st.markdown("---")
        page = st.radio("Navigation", ["🎯 Verdict Crédit", "📊 Analyse du Modèle"])

    vendeur = sellers.loc[seller_id2]

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
    st.header(f"Analyse du Vendeur : {vendeur.name}")
    
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

if __name__ == "__main__":
    main()