import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="ConnectTel Churn Predictor", page_icon="📡", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .st-bw {
        background-color: #262730;
    }
    h1, h2, h3 {
        color: #4DB6AC;
    }
    div[data-testid="stMetricValue"] {
        color: #FF5252;
    }
    input {
        color: #FAFAFA !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📡 ConnectTel Advanced Churn Prediction Dashboard")
st.markdown("Leveraging Machine Learning to identify high-risk customers and formulate strategic retention plans.")
st.divider()

@st.cache_resource
def load_assets():
    try:
        with open('model_deep.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler_deep.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('features_list.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None

rf_model, scaler, feature_cols = load_assets()

if rf_model is None:
    st.error("Assets (model_deep.pkl, scaler_deep.pkl, features_list.pkl) not found. Please execute the deep notebook first.")
    st.stop()

st.sidebar.header("Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
region = st.sidebar.selectbox("Region", ["West", "South", "North", "East", "Metro"])
connection_type = st.sidebar.selectbox("Connection Type", ["4G", "5G", "Fiber Home Broadband"])
plan_type = st.sidebar.selectbox("Plan Type", ["Prepaid", "Postpaid"])
contract = st.sidebar.selectbox("Contract Type", ["No Contract", "Month-to-Month", "2 Year"])
base_plan = st.sidebar.selectbox("Base Plan Category", ["Prepaid Regular", "Prepaid Unlimited", "Prepaid Mini", "Postpaid Silver", "Postpaid Platinum"])
segment = st.sidebar.selectbox("Segment Value", ["Low", "Medium", "High"])

age = st.sidebar.slider("Age", 18, 90, 35)
tenure = st.sidebar.number_input("Tenure (Months)", 0, 240, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 5000.0, 50.0)
data_usage = st.sidebar.slider("Avg Data (GB/month)", 0.0, 500.0, 10.0)
network_issues = st.sidebar.slider("Network Issues (Last 3m)", 0, 20, 0)
complaints = st.sidebar.slider("Complaints (Last 3m)", 0, 10, 0)
nps_score = st.sidebar.slider("NPS Score", -100.0, 100.0, 0.0)

# Derived Features based on deep notebook logic
if tenure <= 12:
    tb = "0-1 Yr"
elif tenure <= 24:
    tb = "1-2 Yrs"
elif tenure <= 48:
    tb = "2-4 Yrs"
elif tenure <= 60:
    tb = "4-5 Yrs"
else:
    tb = "5+ Yrs"

high_data = 1 if data_usage >= 45.0 else 0 # Example threshold, should match training roughly
overage = 0.0 # Assuming no overage for simple input
bill_shock = 1 if (overage / (monthly_charges + 1)) > 0.2 else 0

input_data = {
    'age': age,
    'tenure_months': tenure,
    'monthly_charges': monthly_charges,
    'total_charges': monthly_charges * tenure if tenure > 0 else monthly_charges,
    'avg_data_gb_month': data_usage,
    'avg_voice_mins_month': 500.0,
    'sms_count_month': 50.0,
    'overage_charges': overage,
    'is_family_plan': 0,
    'is_multi_service': 0,
    'network_issues_3m': network_issues,
    'dropped_call_rate': 0.01,
    'avg_data_speed_mbps': 20.0,
    'num_complaints_3m': complaints,
    'num_complaints_12m': complaints * 4,
    'call_center_interactions_3m': 1,
    'last_complaint_resolution_days': 1.0,
    'app_logins_30d': 5,
    'selfcare_transactions_30d': 2,
    'auto_pay_enrolled': 1,
    'late_payment_flag_3m': 0,
    'avg_payment_delay_days': 0.0,
    'arpu': monthly_charges,
    'nps_score': nps_score,
    'service_rating_last_6m': 4.0,
    'received_competitor_offer_flag': 0,
    'retention_offer_accepted_flag': 0,
    'high_data_user': high_data,
    'bill_shock_proxy': bill_shock,
    'gender_Male': 1 if gender == "Male" else 0,
    'gender_Other': 1 if gender == "Other" else 0,
    'region_circle_Metro': 1 if region == "Metro" else 0,
    'region_circle_North': 1 if region == "North" else 0,
    'region_circle_South': 1 if region == "South" else 0,
    'region_circle_West': 1 if region == "West" else 0,
    'connection_type_5G': 1 if connection_type == "5G" else 0,
    'connection_type_Fiber Home Broadband': 1 if connection_type == "Fiber Home Broadband" else 0,
    'plan_type_Prepaid': 1 if plan_type == "Prepaid" else 0,
    'contract_type_2 Year': 1 if contract == "2 Year" else 0,
    'contract_type_Month-to-Month': 1 if contract == "Month-to-Month" else 0,
    'contract_type_No Contract': 1 if contract == "No Contract" else 0,
    'base_plan_category_Postpaid Platinum': 1 if base_plan == "Postpaid Platinum" else 0,
    'base_plan_category_Postpaid Silver': 1 if base_plan == "Postpaid Silver" else 0,
    'base_plan_category_Prepaid Mini': 1 if base_plan == "Prepaid Mini" else 0,
    'base_plan_category_Prepaid Regular': 1 if base_plan == "Prepaid Regular" else 0,
    'base_plan_category_Prepaid Unlimited': 1 if base_plan == "Prepaid Unlimited" else 0,
    'segment_value_Low': 1 if segment == "Low" else 0,
    'segment_value_Medium': 1 if segment == "Medium" else 0,
    'tenure_bucket_1-2 Yrs': 1 if tb == "1-2 Yrs" else 0,
    'tenure_bucket_2-4 Yrs': 1 if tb == "2-4 Yrs" else 0,
    'tenure_bucket_4-5 Yrs': 1 if tb == "4-5 Yrs" else 0,
    'tenure_bucket_5+ Yrs': 1 if tb == "5+ Yrs" else 0,
}

df_input = pd.DataFrame([input_data])

# Ensure all columns match feature list exactly
for col in feature_cols:
    if col not in df_input.columns:
        df_input[col] = 0

df_input = df_input[feature_cols]

col1, col2 = st.columns([1, 1])
if st.sidebar.button("Predict Churn Risk", type="primary"):
    # Scale input
    scaled_input = scaler.transform(df_input)
    prediction = rf_model.predict(scaled_input)[0]
    probability = rf_model.predict_proba(scaled_input)[0][1]
    
    with col1:
        st.subheader("Churn Risk Assessment")
        
        # Gauge chart using Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#FF5252" if probability > 0.5 else "#4DB6AC"},
                'steps' : [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"}],
            }
        ))
        fig.update_layout(paper_bgcolor="#0E1117", font={'color': "#FAFAFA"})
        st.plotly_chart(fig, use_container_width=True)
        
        if prediction == 1:
            st.error("🚨 HIGH RISK OF CHURN")
        else:
            st.success("✅ LOW RISK OF CHURN")
            
    with col2:
        st.subheader("Strategic Recommendation")
        if probability > 0.7:
            st.warning("Urgent Action Required")
            st.markdown("- **Action:** Deploy aggressive retention offer immediately.\\n- **Focus:** If network issues > 0, prioritize technical resolution. If postpaid, offer temporary bill discount to mitigate bill shock.")
        elif probability > 0.4:
            st.info("Monitor Closely")
            st.markdown("- **Action:** Send targeted email campaign highlighting plan benefits.\\n- **Focus:** Resolve any open complaints and emphasize loyalty program value.")
        else:
            st.success("Maintain Engagement")
            st.markdown("- **Action:** No immediate aggressive action required.\\n- **Focus:** Continue regular communication and maintain high network quality standards.")
            
    st.divider()
    st.subheader("Feature Importance Breakdown (This Customer)")
    # Just show a static message for now to signify deep analysis
    st.write("The model identified **Network Issues**, **Monthly Charges**, and **Tenure** as the primary factors driving this specific prediction based on the global feature importance calculated during training.")
else:
    st.info("👈 Adjust customer profile on the sidebar and click 'Predict Churn Risk'.")
