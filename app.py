import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dynamic Pricing AI", layout="wide")
st.title("Dynamic Pricing AI")
st.caption("Demand forecasting + elasticity scenarios + inventory-aware price optimization")
st.markdown("""
This tool recommends an optimal selling price for a product using three stages:
1. **Demand forecasting** — predicts expected weekly demand from historical sales (XGBoost, trained on the M5 Walmart dataset)
2. **Elasticity scenario** — applies a chosen price-sensitivity assumption to estimate how demand shifts with price
3. **Inventory constraint** — caps achievable sales at current stock, so the price reflects real availability
""")

baseline = joblib.load('baseline_model.pkl')
latest = pd.read_csv('latest_products.csv')


SCENARIOS = {'Conservative': -0.5, 'Moderate': -1.0, 'Aggressive': -1.5}

def generate_price_candidates(p_base, q_base, elasticity, discount_range):
    results = []
    for d in discount_range:
        p_new = p_base * (1 - d)
        q_new = q_base * (p_new / p_base) ** elasticity
        revenue = p_new * q_new
        results.append({'discount': d, 'price': p_new, 'predicted_demand': q_new, 'revenue': revenue})
    return pd.DataFrame(results)

def apply_inventory_constraint(candidates, stock):
    candidates = candidates.copy()
    candidates['sellable'] = np.minimum(candidates['predicted_demand'], stock)
    candidates['constrained_revenue'] = candidates['price'] * candidates['sellable']
    return candidates

def recommend_price(item_id, scenario, stock, max_increase=0.2, max_discount=0.3):
    if stock <= 0:
        return {'status': 'out_of_stock', 'item_id': item_id}
    elasticity = SCENARIOS[scenario]
    row = latest[latest['item_id'] == item_id].iloc[0]
    discount_range = np.arange(-max_increase, max_discount + 0.01, 0.05)
    candidates = generate_price_candidates(row['P_base'], row['Q_base'], elasticity, discount_range)
    constrained = apply_inventory_constraint(candidates, stock)
    best = constrained.loc[constrained['constrained_revenue'].idxmax()]
    return {'status': 'ok', **best.to_dict()}


def plot_price_curve(item_id, scenario, stock, max_increase=0.2, max_discount=0.3):
    elasticity = SCENARIOS[scenario]
    row = latest[latest['item_id'] == item_id].iloc[0]
    discount_range = np.arange(-max_increase, max_discount + 0.01, 0.05)
    candidates = generate_price_candidates(row['P_base'], row['Q_base'], elasticity, discount_range)
    constrained = apply_inventory_constraint(candidates, stock)
    
    best_idx = constrained['constrained_revenue'].idxmax()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(constrained['price'], constrained['constrained_revenue'], marker='o', color='#1f77b4')
    ax.scatter(constrained.loc[best_idx, 'price'], constrained.loc[best_idx, 'constrained_revenue'],
               color='#d62728', s=150, zorder=5, label='Recommended price')
    ax.set_xlabel('Price (₹)')
    ax.set_ylabel('Expected Revenue (₹)')
    ax.set_title(f'{item_id} — Revenue vs Price ({scenario}, stock={stock})')
    ax.legend()
    plt.tight_layout()
    return fig



with st.sidebar:
    st.header("Price Recommendation")
    item_id = st.selectbox("Select product", latest['item_id'].unique())
    scenario = st.selectbox("Elasticity scenario", list(SCENARIOS.keys()))
    stock = st.number_input("Current stock", min_value=0, value=10)
    max_increase = st.slider("Maximum price increase", 0.0, 0.5, 0.2)
    max_discount = st.slider("Maximum discount", 0.0, 0.5, 0.3)
    get_rec = st.button("Get Recommendation", use_container_width=True)



if get_rec:
    result = recommend_price(item_id, scenario, stock, max_increase, max_discount)
    
    if result['status'] == 'out_of_stock':
        st.error(f"{item_id} is out of stock — no price recommendation available.")
    else:
        current_price = latest[latest['item_id'] == item_id]['P_base'].values[0]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"₹{current_price:.2f}")
        col2.metric("Recommended Price", f"₹{result['price']:.2f}", f"{-result['discount']*100:.0f}%")
        col3.metric("Expected Revenue", f"₹{result['constrained_revenue']:.2f}")
        
        st.write(f"**Expected demand:** {result['predicted_demand']:.1f} units")
        st.write(f"**Available inventory:** {stock} units")
        st.write(f"**Units expected to sell:** {result['sellable']:.1f}")
        st.write(f"**Scenario used:** {scenario} (E = {SCENARIOS[scenario]})")
        st.subheader("Price vs Revenue Curve")
        fig = plot_price_curve(item_id, scenario, stock, max_increase, max_discount)
        st.pyplot(fig)
        
        