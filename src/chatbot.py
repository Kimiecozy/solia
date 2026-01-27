import streamlit as st
import pandas as pd
import re
from typing import Dict, List, Tuple

@st.cache_data
def load_data():
    df = pd.read_csv("seller_features.csv", sep=";")
    df['seller_name'] = [f"seller{i+1}" for i in range(len(df))]
    return df

def parse_query(query: str) -> Dict:
    """Parse les requêtes naturelles"""
    query_lower = query.lower()
    
    # Solvabilité
    solv_match = re.search(r'solvabilit[éey]?\s*(>|<|=|plus|moins)\s*(\d+)', query_lower)
    if solv_match:
        op, value = solv_match.groups()
        value = float(value)
        if op in ['plus', '>']: op = '>'
        elif op in ['moins', '<']: op = '<'
        else: op = '='
        return {'type': 'solvability', 'op': op, 'value': value}
    
    # Top/Bottom
    if 'top' in query_lower or 'meilleur' in query_lower or 'plus haut' in query_lower:
        n = re.search(r'top\s*(\d+)', query_lower)
        return {'type': 'top_solv', 'n': int(n.group(1)) if n else 5}
    
    if 'moins bon' in query_lower or 'plus bas' in query_lower:
        n = re.search(r'(\d+)', query_lower)
        return {'type': 'bottom_solv', 'n': int(n.group(1)) if n else 5}
    
    # Revenue
    rev_match = re.search(r'revenu[sse]?\s*(>|<|=|plus|moins)\s*(\d+(?:[kKmM]?)?)', query_lower)
    if rev_match:
        op, value = rev_match.groups()
        value = float(value.replace('k', '000').replace('m', '000000'))
        if op in ['plus', '>']: op = '>'
        elif op in ['moins', '<']: op = '<'
        else: op = '='
        return {'type': 'revenue', 'op': op, 'value': value}
    
    return {'type': 'stats'}

def execute_query(df: pd.DataFrame, query_dict: Dict) -> Tuple[pd.DataFrame, str]:
    """Exécute la requête et retourne les résultats"""
    if query_dict['type'] == 'solvability':
        mask = df['solvability_score'].__getattribute__(query_dict['op'])(query_dict['value'])
        result = df[mask][['seller_name', 'seller_id', 'solvability_score', 'total_revenue']].round(2)
        return result, f"**{len(result)} vendeurs** {query_dict['op']} {query_dict['value']}"
    
    elif query_dict['type'] == 'top_solv':
        result = df.nlargest(query_dict['n'], 'solvability_score')[['seller_name', 'seller_id', 'solvability_score', 'total_revenue']].round(2)
        return result, f"**Top {query_dict['n']} solvabilité**"
    
    elif query_dict['type'] == 'bottom_solv':
        result = df.nsmallest(query_dict['n'], 'solvability_score')[['seller_name', 'seller_id', 'solvability_score', 'total_revenue']].round(2)
        return result, f"**Bottom {query_dict['n']} solvabilité**"
    
    elif query_dict['type'] == 'revenue':
        mask = df['total_revenue'].__getattribute__(query_dict['op'])(query_dict['value'])
        result = df[mask][['seller_name', 'seller_id', 'solvability_score', 'total_revenue']].round(2)
        return result, f"**{len(result)} vendeurs** {query_dict['op']} {query_dict['value']}$ revenue"
    
    else:
        stats = {
            'Moyenne solvabilité': df['solvability_score'].mean(),
            'Meilleure solvabilité': df['solvability_score'].max(),
            'Pire solvabilité': df['solvability_score'].min(),
            'Total revenue': df['total_revenue'].sum(),
            'Nb vendeurs': len(df)
        }
        return pd.DataFrame([stats]), "📊 **Statistiques générales**"

# Interface
st.title("🤖 Chatbot Vendeurs")
st.caption("Ex: 'vendeurs solvabilité > 50', 'top 10', 'revenue > 100k'")

df = load_data()
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pose ta question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        query_dict = parse_query(prompt)
        result_df, title = execute_query(df, query_dict)
        
        st.markdown(f"### {title}")
        st.dataframe(result_df, use_container_width=True)
        
        st.session_state.messages.append({"role": "assistant", "content": st.container()})
