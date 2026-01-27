# ===============================================
# SolIA
# Master 2 - SEP
# ===============================================

"""
Interface utilisateur Streamlit pour le système de recommandation Olist.

Cette application web permet de:
- Tester les recommandations personnalisées
- Visualiser les performances du modèle
- Explorer les données et résultats
- Démontrer le système complet aux étudiants
- CHATBOT pour requêtes naturelles sur solvabilité/revenue

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
import re

# Configuration de la page (doit être la 1ère commande Streamlit)
st.set_page_config(page_title="SolIA | Decision Support", layout="wide")

# Injection du CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    local_css(css_path)

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))
from config import MLConfig

# Configuration de l'API
API_BASE_URL = "http://localhost:8000/api/v1"

# ========================================
# CHATBOT FONCTIONS
# ========================================
def chatbot_parse_query(query: str):
    """Parseur robuste gérant les commandes des boutons et le texte libre."""
    q = query.strip()
    q_low = q.lower()
    res = {'type': 'unknown', 'target_seller': None}
    
    # 1. COMMANDES DES BOUTONS (Priorité #1)
    if q == "ACTION_AUDIT_COMPLET": return {'type': 'full_audit'}
    if q == "ACTION_RISQUE_PAIEMENT": return {'type': 'payment_risk'}
    if q == "ACTION_TOP_5": return {'type': 'top_performers'}
    if q == "ACTION_BENCHMARK_GLOBAL": return {'type': 'global_stats'}

    # 2. DÉTECTION DE VENDEUR SPÉCIFIQUE (ex: "vendeur 15")
    seller_match = re.search(r'vendeur\s*(\d+)', q_low)
    if seller_match:
        res['target_seller'] = f"vendeur{seller_match.group(1)}"
        res['type'] = 'full_audit'
        return res

    # 3. ANALYSE DU LANGAGE NATUREL
    if any(w in q_low for w in ['audit', 'analyse', 'pourquoi', 'score']):
        res['type'] = 'full_audit'
    elif any(w in q_low for w in ['paiement', 'mensualit', 'liquidit', 'argent', 'fraction']):
        res['type'] = 'payment_risk'
    elif any(w in q_low for w in ['top', 'meilleur']):
        res['type'] = 'top_performers'
    elif any(w in q_low for w in ['stats', 'global', 'plateforme', 'benchmark']):
        res['type'] = 'global_stats'
        
    return res

def chatbot_execute_query(df, query_dict, sidebar_vendeur):
    """Moteur d'exécution unique : gère l'analyse contextuelle et globale."""
    q_type = query_dict['type']
    
    # --- LOGIQUE DE CIBLE (QUI ANALYSE-T-ON ?) ---
    v = None
    if query_dict.get('target_seller'):
        target = query_dict['target_seller']
        match = df[df['seller_name'].str.lower() == target.lower()]
        if not match.empty: v = match.iloc[0]
    else:
        v = sidebar_vendeur

    # --- RÉPONSES SELON L'INTENTION ---
    
    # 1. Audit Complet (360°)
    if q_type == 'full_audit' and v is not None:
        score = v['solvability_score']
        barre = "🟦" * int(score/10) + "⬜" * (10 - int(score/10))
        reponse = f"### AUDIT COMPLET : {v['seller_name'].upper()}\n"
        reponse += f"**Verdict SolIA :** `{score:.1f}/100` | {barre}\n\n"
        reponse += f"- **Logistique :** {v['late_rate']*100:.1f}% de retards opérationnels.\n"
        reponse += f"- **Réputation :** Note moyenne de {v['avg_review_score']:.1f}/5.\n"
        reponse += f"- **Revenu :** {v['total_revenue']:,.0f} R$ générés au total."
        return pd.DataFrame([v[['solvability_score', 'late_rate', 'avg_review_score']]]), reponse

    # 2. Risque de Paiement (Trésorerie)
    elif q_type == 'payment_risk' and v is not None:
        inst = v['avg_installments']
        reponse = f"### ANALYSE FINANCIÈRE : {v['seller_name'].upper()}\n"
        reponse += f"Structure des paiements : **{inst:.1f} mensualités** en moyenne.\n\n"
        if inst > 6:
            reponse += "> **ALERTE LIQUIDITÉ :** Très dépendant des paiements longs. Risque d'impayés élevé."
        else:
            reponse += "> **TRÉSORERIE SAINE :** Cycle d'encaissement court et sécurisé."
        return pd.DataFrame([v[['avg_installments', 'total_revenue']]]), reponse

    # 3. Top Performers
    elif q_type == 'top_performers':
        res = df.nlargest(5, 'solvability_score')[['seller_name', 'solvability_score', 'total_revenue']].reset_index()
        return res, "### TOP 5 DES PROFILS LES PLUS SOLVABLES"

    # 4. Benchmark Marché (Global)
    elif q_type == 'global_stats':
        avg_s = df['solvability_score'].mean()
        stats = pd.DataFrame([{
            'Score Moyen': f"{avg_s:.1f}/100",
            'CA Total': f"{df['total_revenue'].sum():,.0f} R$",
            'Vendeurs': len(df)
        }])
        return stats, "**BENCHMARK GLOBAL : État de la plateforme Olist**"

    return None, "Je n'ai pas bien compris. Essayez 'Audit', 'Risque' ou 'Stats'."

# ========================================
# INTERFACE : PAGE CHATBOT
# ========================================

def show_chatbot_page(df, current_vendeur):
    st.markdown("<h1>Assistant SolIA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Expert en Risque Crédit & Analyse de Données</p>", unsafe_allow_html=True)

    # Bandeau de contexte
    st.info(f"Analyse ciblée sur : **{current_vendeur['seller_name']}** (Score: {current_vendeur['solvability_score']:.1f})")

    # --- BOUTONS ---
    st.write("---")
    c1, c2, c3, c4 = st.columns(4)
    btn_diag = c1.button("Audit Complet", use_container_width=True)
    btn_risk = c2.button("Risque Paiement", use_container_width=True)
    btn_top = c3.button("Top 5", use_container_width=True)
    btn_market = c4.button("Benchmark", use_container_width=True)

    # Historique
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message.get("dataframe") is not None:
                st.dataframe(message["dataframe"], use_container_width=True)

    # Logique Input
    prompt = st.chat_input("Posez votre question...")
    
    # Override du prompt si bouton cliqué
    if btn_diag: prompt = "ACTION_AUDIT_COMPLET"
    if btn_risk: prompt = "ACTION_RISQUE_PAIEMENT"
    if btn_top: prompt = "ACTION_TOP_5"
    if btn_market: prompt = "ACTION_BENCHMARK_GLOBAL"

    if prompt:
        # User message
        st.session_state.chat_messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant message
        with st.chat_message("assistant"):
            with st.spinner("Analyse des algorithmes..."):
                query_dict = chatbot_parse_query(prompt)
                result_df, response_text = chatbot_execute_query(df, query_dict, current_vendeur)
            
            st.markdown(response_text)
            if result_df is not None:
                st.dataframe(result_df, use_container_width=True)
            
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "text": response_text, 
                "dataframe": result_df
            })
        st.rerun()

# ========================================
# FIN CHATBOT
# ========================================

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
sellers = load_seller_data()
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
        st.error(f" Erreur lors de la génération de recommandations: {e}")
        return None

def check_api_health():
    """Vérifie la santé de l'API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except:
        return False

def main():
    # --- 1. TITRE PRINCIPAL ---
    st.title("SolIA - Scoring de Crédit Vendeur")

    # --- 2. BARRE LATÉRALE ---
    with st.sidebar:
        st.markdown('<h2 style="font-size: 1.5rem;">Navigation</h2>', unsafe_allow_html=True)
        page = st.radio("Aller vers :", ["Verdict Crédit", "Fiabilité du Modèle", "Assistant Chatbot"], label_visibility="collapsed")
        
        st.divider()
        st.markdown('<h3>Filtres de recherche</h3>', unsafe_allow_html=True)
        search_query = st.text_input("ID Vendeur (UUID)", "").strip()
        min_score = st.slider("Solvabilité minimum", 0, 100, 0)
        min_revenue = st.slider("CA minimum (R$)", 0, int(sellers['total_revenue'].max()), 0)
        
        st.divider()

        filtered_df = sellers.copy()
        filtered_df = filtered_df[(filtered_df['solvability_score'] >= min_score) & (filtered_df['total_revenue'] >= min_revenue)]
        if search_query:
            #filtered_df = filtered_df[filtered_df.index.str.contains(search_query, case=False)]
            filtered_df = filtered_df[filtered_df.index.astype(str).str.contains(search_query, case=False)]


        st.markdown(f"<p style='font-family: monospace; font-size: 0.8rem; color: #64748b; margin-top: 10px;'>UNITÉS FILTRÉES : {len(filtered_df)}</p>", unsafe_allow_html=True)

        if not filtered_df.empty:
            name_to_id = {filtered_df.loc[idx, 'seller_name']: idx for idx in filtered_df.index}
            selected_name = st.selectbox("Choisir le profil :", options=list(name_to_id.keys()))
            seller_id2 = name_to_id[selected_name]
            vendeur_data = filtered_df.loc[seller_id2]
        else:
            st.error("Aucun résultat.")
            st.stop()

    # --- 3. AFFICHAGE DES PAGES ---
    if page == "Verdict Crédit":
        show_credit_scoring_page(vendeur_data, selected_name)
    elif page == "Fiabilité du Modèle":
        show_model_analysis_page()
    elif page == "Assistant Chatbot":
        show_chatbot_page(sellers, vendeur_data)

    # --- 3. AFFICHAGE DES PAGES ---
    if page == "Verdict Crédit":
        show_credit_scoring_page(vendeur_data, selected_name)
    elif page == "Fiabilité du Modèle":
        show_model_analysis_page()
    elif page == "Assistant Chatbot":
        show_chatbot_page(df_sellers, vendeur_data)

def show_model_analysis_page():
    """Affiche les performances techniques du modèle de régression."""
    st.header(" Analyse technique du modèle")

    # 1. Explication des métriques
    st.markdown("""
    Le modèle utilise un algorithme **Random Forest Regressor** pour prédire le chiffre d'affaires 
    futur d'un vendeur en se basant sur son historique de performance.
    """)

    # 2. Importance des Features (Le pourquoi du score)
    st.subheader(" Importance des critères")
    
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
    st.subheader(" Précision du système")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("R² Score (Fiabilité)", "0.95", help="Plus le score est proche de 1, plus le modèle est précis.")
        st.write("Le modèle explique 95% des variations de revenus.")
        
    with col2:
        st.metric("Erreur Moyenne (MAE)", "1779 R$", help="Écart moyen entre la prédiction et la réalité.")
        st.write("L'incertitude moyenne sur le CA prédit.")


def show_credit_scoring_page(vendeur, seller_name):
    st.header(f"Analyse du Vendeur : **{seller_name}**")
    st.caption(f"ID: {vendeur.name[:8]}...")

def show_credit_scoring_page(vendeur, seller_name):
    st.header(f"Analyse du Vendeur : **{seller_name}**")
    st.caption("Nom du vendeur")
    
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
    st.subheader(" Capacité de Remboursement")
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

def show_chatbot_page(df, current_vendeur):
    # --- TITRE AVEC EFFET WAVE (Injecté via CSS) ---
    st.markdown("<h1>Assistant SolIA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6b7280;'>Intelligence Décisionnelle & Analyse de Risque</p>", unsafe_allow_html=True)

    # --- BANDEAU DE CONTEXTE ---
    with st.container():
        col_info, col_reset = st.columns([4, 1])
        with col_info:
            st.info(f"**Focus :** {current_vendeur['seller_name']} | **Score :** {current_vendeur['solvability_score']:.1f}/100")
        with col_reset:
            if st.button("Effacer", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()

    # --- 1. BARRE D'ACTIONS RAPIDES (QUICK ACTIONS) ---
    st.write("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        btn_diag = st.button("Audit Complet", use_container_width=True)
    with c2:
        btn_risk = st.button("Risque Paiement", use_container_width=True)
    with c3:
        btn_top = st.button("Top Performers", use_container_width=True)
    with c4:
        btn_market = st.button("Benchmark", use_container_width=True)

    # --- 2. INITIALISATION ET AFFICHAGE DE L'HISTORIQUE ---
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Zone de défilement du chat
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["text"])
                if message.get("dataframe") is not None:
                    st.dataframe(message["dataframe"], use_container_width=True)
                if message.get("metrics"):
                    # Affichage de mini-métriques dans le chat pour le look pro
                    m = message["metrics"]
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Score", f"{m['score']:.1f}")
                    mc2.metric("Impact CA", f"{m['rev']:.0f} R$")

    # --- 3. GESTION DES ENTRÉES (INPUTS) ---
    prompt = st.chat_input("Une question sur ce dossier ? (ex: Pourquoi ce score ?)")

    # Mapping des boutons vers des prompts textuels

    if btn_diag: prompt = "ACTION_AUDIT_COMPLET"
    if btn_risk: prompt = "ACTION_RISQUE_PAIEMENT"
    if btn_top: prompt = "ACTION_TOP_5"
    if btn_market: prompt = "ACTION_BENCHMARK_GLOBAL"

    if prompt:
        # Affichage immédiat du message utilisateur
        st.session_state.chat_messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Logique de réponse de l'assistant
        with st.chat_message("assistant"):
            with st.spinner("Consultation des algorithmes de risque..."):
                # On utilise ton parser boosté
                query_dict = chatbot_parse_query(prompt)
                
                # Exécution de la requête avec contexte
                result_df, response_text = chatbot_execute_query(df, query_dict, current_vendeur)
                
                # Optionnel : Extraction de métriques pour affichage visuel
                metrics_data = None
                if query_dict['type'] == 'contextual_analysis':
                    metrics_data = {
                        "score": current_vendeur['solvability_score'],
                        "rev": current_vendeur['total_revenue']
                    }

            st.markdown(response_text)
            if result_df is not None:
                st.dataframe(result_df, use_container_width=True)
            
            # Sauvegarde dans l'historique
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "text": response_text, 
                "dataframe": result_df,
                "metrics": metrics_data
            })
        
        # Rafraîchissement pour fluidité
        st.rerun()

def show_model_performance_page():
    """Page d'analyse des performances du modèle."""

    st.markdown("##  Performance du Modèle ML")

    model_info = get_model_info()
    if not model_info:
        st.error("Impossible de récupérer les informations du modèle")
        return

    metrics = model_info.get("metrics", {})
    feature_importance = model_info.get("feature_importance", [])

    # Métriques principales
    st.markdown("###  Métriques de Performance")

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
        st.markdown("###  Importance des Features")

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

if __name__ == "__main__":
    main()
