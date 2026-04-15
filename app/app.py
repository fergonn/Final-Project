import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from openai import OpenAI

# ── Setup ──
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = joblib.load('../models/tuned_xgboost.pkl')
label_encoders = joblib.load('../models/label_encoders.pkl')

# ── Page Config ──
st.set_page_config(
    page_title="Supply Chain Risk Predictor",
    page_icon="🚚",
    layout="centered"
)

# ── Title ──
st.title("🚚 Supply Chain Late Delivery Risk Predictor")
st.markdown("Fill in the order details below to predict delivery risk and get AI-powered recovery actions.")
st.divider()

# ── Input Form ──
st.subheader("📦 Order Details")

col1, col2 = st.columns(2)

with col1:
    shipping_mode = st.selectbox("Shipping Mode", 
        ["First Class", "Same Day", "Second Class", "Standard Class"])
    
    scheduled_days = st.selectbox("Scheduled Delivery Days", [0, 1, 2, 4])
    
    market = st.selectbox("Market", 
        ["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"])
    
    order_region = st.selectbox("Order Region",
        ["Canada", "Caribbean", "Central Africa", "Central America",
         "Central Asia", "East Africa", "East of USA", "Eastern Asia",
         "Eastern Europe", "North Africa", "Northern Europe", "Oceania",
         "South America", "South Asia", "South of USA", "Southeast Asia",
         "Southern Africa", "Southern Europe", "US Center", "West Africa",
         "West Asia", "West of USA", "Western Europe"])

with col2:
    customer_segment = st.selectbox("Customer Segment",
        ["Consumer", "Corporate", "Home Office"])
    
    department = st.selectbox("Department",
        ["Apparel", "Fan Shop", "Fitness", "Footwear", "Golf", "Outdoors"])
    
    payment_type = st.selectbox("Payment Type",
        ["CASH", "DEBIT", "PAYMENT", "TRANSFER"])
    
    product_name = st.selectbox("Product Category",
        sorted(label_encoders['Product Name'].classes_.tolist()))

col3, col4 = st.columns(2)

with col3:
    sales = st.number_input("Sales Amount ($)", min_value=0.0, value=150.0, step=10.0)
    quantity = st.number_input("Order Quantity", min_value=1, max_value=5, value=1)

with col4:
    discount_rate = st.slider("Discount Rate", 0.0, 0.25, 0.05)
    product_price = st.number_input("Product Price ($)", min_value=0.0, value=99.99, step=10.0)

st.divider()

# ── Predict Button ──
if st.button("🔍 Predict Delivery Risk", use_container_width=True):

    # Build feature vector
    input_data = {
        'Days for shipment (scheduled)': scheduled_days,
        'Category Name': 0,
        'Latitude': 0.0,
        'Longitude': 0.0,
        'Order Item Discount Rate': discount_rate,
        'Order Item Quantity': quantity,
        'Sales': sales,
        'Order Region': label_encoders['Order Region'].transform([order_region])[0],
        'Product Name': label_encoders['Product Name'].transform([product_name])[0],
        'Product Price': product_price,
        'order_month': pd.Timestamp.now().month,
        'order_dayofweek': pd.Timestamp.now().dayofweek,
        'order_quarter': pd.Timestamp.now().quarter,
        'order_year': pd.Timestamp.now().year,
        'order_hour': pd.Timestamp.now().hour,
        'Type_CASH': 1 if payment_type == 'CASH' else 0,
        'Type_DEBIT': 1 if payment_type == 'DEBIT' else 0,
        'Type_PAYMENT': 1 if payment_type == 'PAYMENT' else 0,
        'Type_TRANSFER': 1 if payment_type == 'TRANSFER' else 0,
        'Customer Segment_Consumer': 1 if customer_segment == 'Consumer' else 0,
        'Customer Segment_Corporate': 1 if customer_segment == 'Corporate' else 0,
        'Customer Segment_Home Office': 1 if customer_segment == 'Home Office' else 0,
        'Department Name_Apparel': 1 if department == 'Apparel' else 0,
        'Department Name_Fan Shop': 1 if department == 'Fan Shop' else 0,
        'Department Name_Fitness': 1 if department == 'Fitness' else 0,
        'Department Name_Footwear': 1 if department == 'Footwear' else 0,
        'Department Name_Golf': 1 if department == 'Golf' else 0,
        'Department Name_Outdoors': 1 if department == 'Outdoors' else 0,
        'Market_Africa': 1 if market == 'Africa' else 0,
        'Market_Europe': 1 if market == 'Europe' else 0,
        'Market_LATAM': 1 if market == 'LATAM' else 0,
        'Market_Pacific Asia': 1 if market == 'Pacific Asia' else 0,
        'Market_USCA': 1 if market == 'USCA' else 0,
        'Shipping Mode_First Class': 1 if shipping_mode == 'First Class' else 0,
        'Shipping Mode_Same Day': 1 if shipping_mode == 'Same Day' else 0,
        'Shipping Mode_Second Class': 1 if shipping_mode == 'Second Class' else 0,
        'Shipping Mode_Standard Class': 1 if shipping_mode == 'Standard Class' else 0,
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model.get_booster().feature_names, fill_value=0)

    # Predict
    risk_prob = model.predict_proba(input_df)[0][1]
    risk_pred = model.predict(input_df)[0]

    # ── Results ──
    st.subheader("📊 Prediction Result")

    if risk_pred == 1:
        st.error(f"⚠️ HIGH RISK — {risk_prob:.1%} probability of late delivery")
    else:
        st.success(f"✅ LOW RISK — {risk_prob:.1%} probability of late delivery")

    st.progress(float(risk_prob))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Risk Probability", f"{risk_prob:.1%}")
    col_b.metric("Prediction", "HIGH RISK" if risk_pred == 1 else "LOW RISK")
    col_c.metric("Scheduled Days", scheduled_days)

    # ── Gen AI Recovery Action ──
    if risk_pred == 1:
        st.divider()
        st.subheader("🤖 AI Recovery Action")

        with st.spinner("Generating recovery plan..."):
            prompt = f"""
You are a supply chain operations assistant. An order has been flagged as HIGH RISK for late delivery.

ORDER DETAILS:
- Product: {product_name}
- Customer Segment: {customer_segment}
- Shipping Mode: {shipping_mode}
- Scheduled Delivery Days: {scheduled_days}
- Order Region: {order_region}
- Market: {market}
- Late Delivery Probability: {risk_prob:.1%}

Generate a concise proactive recovery plan with:
1. A short customer apology message (2-3 sentences)
2. Two specific operational actions for the supply chain team
3. A recommended shipping upgrade if applicable
Keep under 150 words.
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            recovery_action = response.choices[0].message.content

        st.markdown(recovery_action)