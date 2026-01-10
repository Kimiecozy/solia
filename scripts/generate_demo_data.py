# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - DEMO DATA
# Master 2 - Data Science Industrielle
# ===============================================

"""
Génère des données Olist simplifiées pour la démonstration.

Ce script crée un dataset minimal mais réaliste pour permettre aux
étudiants de tester rapidement le système de recommandation.

Usage:
    python scripts/generate_demo_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import sys

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR

# Configuration des données de démo
np.random.seed(42)
N_CUSTOMERS = 50
N_ORDERS = 150
N_PRODUCTS = 30
N_REVIEWS = 120

def generate_customers():
    """Génère des données clients réalistes."""
    print("👥 Génération des clients...")

    states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'GO', 'PE', 'BA', 'MT']
    cities = {
        'SP': ['São Paulo', 'Campinas', 'Santos', 'Ribeirão Preto'],
        'RJ': ['Rio de Janeiro', 'Niterói', 'Nova Iguaçu'],
        'MG': ['Belo Horizonte', 'Uberlândia', 'Juiz de Fora'],
        'RS': ['Porto Alegre', 'Caxias do Sul', 'Pelotas'],
        'PR': ['Curitiba', 'Londrina', 'Maringá']
    }

    customers = []
    for i in range(N_CUSTOMERS):
        state = np.random.choice(states)
        city_list = cities.get(state, [f"Cidade_{state}"])
        city = np.random.choice(city_list) if city_list else f"Cidade_{state}"

        customer = {
            'customer_id': str(uuid.uuid4()),
            'customer_unique_id': str(uuid.uuid4()),
            'customer_zip_code_prefix': np.random.randint(10000, 99999),
            'customer_city': city,
            'customer_state': state
        }
        customers.append(customer)

    df = pd.DataFrame(customers)
    print(f"   ✅ {len(df)} clients générés")
    return df

def generate_products():
    """Génère des données produits réalistes."""
    print("📦 Génération des produits...")

    categories = [
        'cama_mesa_banho', 'beleza_saude', 'esporte_lazer', 'informatica_acessorios',
        'moveis_decoracao', 'utilidades_domesticas', 'relogios_presentes', 'telefonia',
        'automotivo', 'brinquedos', 'cool_stuff', 'ferramentas_jardim'
    ]

    products = []
    for i in range(N_PRODUCTS):
        category = np.random.choice(categories)
        product = {
            'product_id': str(uuid.uuid4()),
            'product_category_name': category,
            'product_name_lenght': np.random.randint(10, 80),
            'product_description_lenght': np.random.randint(100, 2000),
            'product_photos_qty': np.random.randint(1, 10),
            'product_weight_g': np.random.randint(50, 5000),
            'product_length_cm': np.random.randint(5, 50),
            'product_height_cm': np.random.randint(2, 30),
            'product_width_cm': np.random.randint(5, 40)
        }
        products.append(product)

    df = pd.DataFrame(products)
    print(f"   ✅ {len(df)} produits générés")
    return df

def generate_orders(customers_df):
    """Génère des commandes réalistes."""
    print("🛒 Génération des commandes...")

    order_statuses = ['delivered', 'shipped', 'processing', 'canceled']
    status_weights = [0.7, 0.15, 0.1, 0.05]  # 70% delivered

    orders = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    for i in range(N_ORDERS):
        # Date de commande aléatoire
        order_date = start_date + timedelta(
            days=np.random.randint(0, (end_date - start_date).days)
        )

        status = np.random.choice(order_statuses, p=status_weights)

        # Dates de livraison selon le statut
        if status == 'delivered':
            approved_date = order_date + timedelta(hours=np.random.randint(1, 48))
            carrier_date = approved_date + timedelta(days=np.random.randint(1, 5))
            delivered_date = carrier_date + timedelta(days=np.random.randint(1, 20))
        elif status == 'shipped':
            approved_date = order_date + timedelta(hours=np.random.randint(1, 48))
            carrier_date = approved_date + timedelta(days=np.random.randint(1, 5))
            delivered_date = None
        else:
            approved_date = order_date + timedelta(hours=np.random.randint(1, 72))
            carrier_date = None
            delivered_date = None

        order = {
            'order_id': str(uuid.uuid4()),
            'customer_id': np.random.choice(customers_df['customer_id']),
            'order_status': status,
            'order_purchase_timestamp': order_date,
            'order_approved_at': approved_date,
            'order_delivered_carrier_date': carrier_date,
            'order_delivered_customer_date': delivered_date,
            'order_estimated_delivery_date': order_date + timedelta(days=np.random.randint(7, 30))
        }
        orders.append(order)

    df = pd.DataFrame(orders)
    print(f"   ✅ {len(df)} commandes générées")
    return df

def generate_order_items(orders_df, products_df):
    """Génère les items de commandes."""
    print("📋 Génération des items de commandes...")

    order_items = []
    for _, order in orders_df.iterrows():
        # Nombre d'items par commande (1-5)
        n_items = np.random.randint(1, 6)

        # Sélectionner des produits aléatoirement
        selected_products = np.random.choice(
            products_df['product_id'], size=min(n_items, len(products_df)), replace=False
        )

        for i, product_id in enumerate(selected_products):
            # Prix réaliste selon la catégorie
            product_cat = products_df[products_df['product_id'] == product_id]['product_category_name'].iloc[0]

            if product_cat in ['informatica_acessorios', 'telefonia']:
                base_price = np.random.uniform(200, 2000)
            elif product_cat in ['moveis_decoracao', 'automotivo']:
                base_price = np.random.uniform(100, 1000)
            else:
                base_price = np.random.uniform(20, 300)

            freight_value = base_price * np.random.uniform(0.05, 0.25)  # 5-25% du prix

            item = {
                'order_id': order['order_id'],
                'order_item_id': i + 1,
                'product_id': product_id,
                'seller_id': str(uuid.uuid4()),
                'shipping_limit_date': order['order_purchase_timestamp'] + timedelta(days=np.random.randint(1, 10)),
                'price': round(base_price, 2),
                'freight_value': round(freight_value, 2)
            }
            order_items.append(item)

    df = pd.DataFrame(order_items)
    print(f"   ✅ {len(df)} items générés")
    return df

def generate_reviews(orders_df):
    """Génère des avis clients réalistes."""
    print("⭐ Génération des avis clients...")

    # Sélectionner des commandes livrées pour les avis
    delivered_orders = orders_df[orders_df['order_status'] == 'delivered'].copy()

    if len(delivered_orders) == 0:
        print("   ⚠️ Aucune commande livrée trouvée")
        return pd.DataFrame()

    # Prendre un échantillon pour les avis (pas tous les clients laissent un avis)
    sample_orders = delivered_orders.sample(min(N_REVIEWS, len(delivered_orders)))

    reviews = []
    for _, order in sample_orders.iterrows():
        # Score pondéré (plus d'avis positifs)
        score = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.05, 0.1, 0.3, 0.5])

        # Date d'avis après livraison
        review_date = order['order_delivered_customer_date'] + timedelta(
            days=np.random.randint(1, 30)
        )

        # Titre et commentaire selon le score
        if score >= 4:
            titles = ["Muito bom!", "Excelente produto", "Recomendo", "Ótima compra"]
            comments = [
                "Produto chegou rápido e conforme descrito.",
                "Excelente qualidade, recomendo!",
                "Muito satisfeito com a compra.",
                "Produto de qualidade, entrega rápida."
            ]
        elif score == 3:
            titles = ["Produto OK", "Atendeu expectativas", "Razoável"]
            comments = [
                "Produto está OK, nada demais.",
                "Atendeu as expectativas básicas.",
                "Produto médio, poderia ser melhor."
            ]
        else:
            titles = ["Não recomendo", "Produto ruim", "Decepcionante"]
            comments = [
                "Produto não corresponde à descrição.",
                "Qualidade abaixo do esperado.",
                "Não recomendo, tive problemas."
            ]

        review = {
            'review_id': str(uuid.uuid4()),
            'order_id': order['order_id'],
            'review_score': score,
            'review_comment_title': np.random.choice(titles) if np.random.random() > 0.3 else None,
            'review_comment_message': np.random.choice(comments) if np.random.random() > 0.4 else None,
            'review_creation_date': review_date,
            'review_answer_timestamp': review_date
        }
        reviews.append(review)

    df = pd.DataFrame(reviews)
    print(f"   ✅ {len(df)} avis générés")
    return df

def save_datasets(customers, products, orders, order_items, reviews):
    """Sauvegarde tous les datasets."""
    print("💾 Sauvegarde des datasets...")

    # S'assurer que le répertoire existe
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        'olist_customers_dataset.csv': customers,
        'olist_products_dataset.csv': products,
        'olist_orders_dataset.csv': orders,
        'olist_order_items_dataset.csv': order_items,
        'olist_order_reviews_dataset.csv': reviews
    }

    for filename, df in datasets.items():
        filepath = RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"   ✅ {filename} - {len(df)} lignes")

    print(f"   🎉 Tous les datasets sauvés dans {RAW_DATA_DIR}")

def main():
    """Fonction principale."""
    print("🚀 " + "="*50)
    print("🚀 GÉNÉRATION DONNÉES DEMO OLIST")
    print("🚀 " + "="*50 + "\n")

    # Générer tous les datasets
    customers = generate_customers()
    products = generate_products()
    orders = generate_orders(customers)
    order_items = generate_order_items(orders, products)
    reviews = generate_reviews(orders)

    # Sauvegarder
    save_datasets(customers, products, orders, order_items, reviews)

    # Statistiques finales
    print("\n📊 STATISTIQUES GÉNÉRÉES:")
    print(f"   👥 Clients: {len(customers)}")
    print(f"   📦 Produits: {len(products)}")
    print(f"   🛒 Commandes: {len(orders)}")
    print(f"   📋 Items: {len(order_items)}")
    print(f"   ⭐ Avis: {len(reviews)}")

    print(f"\n💰 Valeur totale des commandes: {order_items['price'].sum():.2f}€")
    print(f"📊 Note moyenne des avis: {reviews['review_score'].mean():.1f}/5")

    print("\n🎉 " + "="*50)
    print("🎉 DONNÉES DEMO GÉNÉRÉES AVEC SUCCÈS!")
    print("🎉 " + "="*50)

if __name__ == "__main__":
    main()