import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="FleetMind AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Font & Variable Setup */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root {
        --primary: #6366f1;
        --primary-hover: #4f46e5;
        --bg-card: rgba(30, 41, 59, 0.7);
        --border-color: rgba(255, 255, 255, 0.1);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Modern Gradient Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-size: 1.1rem;
        color: var(--text-muted);
        margin-bottom: 2.5rem;
        line-height: 1.6;
    }

    /* Glassmorphic Card */
    .card {
        padding: 2rem;
        border-radius: 16px;
        background-color: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-color);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 1.5rem;
    }

    .card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }

    /* Professional Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: var(--primary);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255,255,255,0.1);
    }

    .stButton>button:hover {
        background: var(--primary-hover);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
        border-color: rgba(255,255,255,0.2);
    }

    .stButton>button:active {
        transform: scale(0.98);
    }

    /* Hide Dataframe Index - Updated for newer Streamlit */
    [data-testid="stElementToolbar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">FleetMind AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predictive Vehicle Maintenance & Risk Analysis</p>', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('fleetmind_model.pkl')

try:
    with st.spinner("Loading AI engine..."):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'fleetmind_model.pkl' exists in the directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8207/8207122.png", width=100) # Optional placeholder logo
    st.title("Settings & Info")
    st.info("Upload your fleet's diagnostic and historical data as a CSV. The AI will analyze the features to predict which vehicles require maintenance.")
    st.markdown("---")
    st.markdown("**Expected Features:**")
    st.markdown("- Engine Size, Cylinders")
    st.markdown("- Mileage")
    st.markdown("- Fuel Type, Transmission")
    st.markdown("- Manufacturing Year")
    st.markdown("- Last Service Date")
    st.markdown("- Warranty Expiry Date")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Enter Vehicle Details")

with st.form("vehicle_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        vehicle_model = st.selectbox("Vehicle Model", ["Sedan", "SUV", "Truck", "Van", "Bus"])
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
        transmission = st.selectbox("Transmission Type", ["Automatic", "Manual"])
        owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])

    with col2:
        mileage = st.number_input("Mileage (km)", min_value=0, value=50000, step=1000)
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, value=5, step=1)
        engine_size = st.number_input("Engine Size (cc)", min_value=500, max_value=10000, value=1500, step=100)
        odometer = st.number_input("Odometer Reading (km)", min_value=0, value=55000, step=1000)

    with col3:
        reported_issues = st.number_input("Reported Issues", min_value=0, value=1, step=1)
        insurance_premium = st.number_input("Insurance Premium ($)", min_value=0, value=500, step=50)
        accident_history = st.number_input("Accident History", min_value=0, value=0, step=1)
        fuel_efficiency = st.number_input("Fuel Efficiency (km/l)", min_value=0.0, value=15.0, step=0.5)

    col4, col5 = st.columns(2)
    with col4:
        days_since_service = st.number_input("Days Since Last Service", min_value=0, value=90, step=1)
    with col5:
        warranty_days_left = st.number_input("Warranty Days Left", min_value=-3650, value=365, step=1)

    submitted = st.form_submit_button("Check Maintenance Risk")

st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    with st.spinner("Analyzing maintenance risks using AI..."):
        input_data = pd.DataFrame([{
            "Vehicle_Model": vehicle_model,
            "Fuel_Type": fuel_type,
            "Transmission_Type": transmission,
            "Owner_Type": owner_type,
            "Mileage": mileage,
            "Reported_Issues": reported_issues,
            "Vehicle_Age": vehicle_age,
            "Engine_Size": engine_size,
            "Odometer_Reading": odometer,
            "Insurance_Premium": insurance_premium,
            "Accident_History": accident_history,
            "Fuel_Efficiency": fuel_efficiency,
            "Days_Since_Last_Service": days_since_service,
            "Warranty_Days_Left": warranty_days_left
        }])

        try:
            prediction = model.predict(input_data)[0]
            predict_proba = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

            st.markdown("---")
            st.markdown("## Prediction Results")

            if prediction == 1:
                st.error(f"**High Risk**: This vehicle requires maintenance!")
                if predict_proba is not None:
                    st.markdown(f"**Risk Probability Code**: <span style='color:#ef4444; font-size:1.2rem; font-weight:bold;'>{predict_proba*100:.1f}%</span>", unsafe_allow_html=True)
            else:
                st.success(f"**Healthy**: This vehicle does not require immediate maintenance.")
                if predict_proba is not None:
                    st.markdown(f"**Risk Probability Code**: <span style='color:#22c55e; font-size:1.2rem; font-weight:bold;'>{predict_proba*100:.1f}%</span>", unsafe_allow_html=True)

            st.markdown("### Input Summary")
            st.dataframe(input_data, use_container_width=True)

            try:
                classifier = model.named_steps['classifier']
                preprocessor = model.named_steps['preprocessor']

                if hasattr(classifier, 'feature_importances_'):
                    importances = classifier.feature_importances_

                    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
                    num_features = input_data.select_dtypes(exclude="object").columns
                    feature_names = np.concatenate([cat_features, num_features])

                    X_transformed = preprocessor.transform(input_data)

                    active_importances = X_transformed[0] * importances

                    if np.sum(np.abs(active_importances)) > 0:
                        normalized_importances = np.abs(active_importances) / np.sum(np.abs(active_importances)) * 100
                    else:
                        normalized_importances = np.abs(active_importances)

                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Contribution (%)': normalized_importances
                    })

                    def clean_feature_name(name):
                        if '_' in name and sum([1 for c in name if c == '_']) >= 2:
                            parts = name.split('_')
                            return f"{parts[0]} {parts[1]} ({parts[-1]})"
                        return name.replace('_', ' ').title()

                    importance_df['Feature'] = importance_df['Feature'].apply(clean_feature_name)

                    top_factors = importance_df.sort_values(by='Contribution (%)', ascending=False).head(3)

                    st.markdown("### Top Contributing Factors")

                    f1, f2, f3 = st.columns(3)
                    factors = top_factors.to_dict('records')

                    if len(factors) > 0:
                        f1.metric("1. " + factors[0]['Feature'], f"{factors[0]['Contribution (%)']:.1f}% impact")
                    if len(factors) > 1:
                        f2.metric("2. " + factors[1]['Feature'], f"{factors[1]['Contribution (%)']:.1f}% impact")
                    if len(factors) > 2:
                        f3.metric("3. " + factors[2]['Feature'], f"{factors[2]['Contribution (%)']:.1f}% impact")
            except Exception as e:
                pass

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("The model encountered an issue processing the input data.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748B;'>Built with Streamlit • AI Model Powered by Random Forest</p>", unsafe_allow_html=True)
