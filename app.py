import streamlit as st
import pandas as pd
import numpy as np
import joblib
from main import CarPricePrediction

# Page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        car_predictor = CarPricePrediction()
        car_predictor.load_model('car_price_model.pkl')
        return car_predictor
    except FileNotFoundError:
        st.error("Model file not found! Please run main.py first to train the model.")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Price Predictor</h1>', unsafe_allow_html=True)
    
    # Load model
    car_predictor = load_model()
    
    if car_predictor is None:
        st.stop()
    
    # Sidebar for input parameters
    st.sidebar.header("🔧 Car Specifications")
    st.sidebar.markdown("Please enter the car details below:")
    
    # Get unique values for dropdowns
    makes = ['Audi', 'BMW', 'Ford', 'Honda', 'Toyota']
    models = ['Model A', 'Model B', 'Model C', 'Model D', 'Model E']
    fuel_types = ['Diesel', 'Electric', 'Petrol']
    transmissions = ['Automatic', 'Manual']
    
    # Input widgets
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Basic Information")
        
        # Dropdown menus
        make = st.selectbox(
            "🏭 Make",
            options=makes,
            help="Select the car manufacturer"
        )
        
        model = st.selectbox(
            "🚙 Model",
            options=models,
            help="Select the car model"
        )
        
        fuel_type = st.selectbox(
            "⛽ Fuel Type",
            options=fuel_types,
            help="Select the fuel type"
        )
        
        transmission = st.selectbox(
            "⚙️ Transmission",
            options=transmissions,
            help="Select the transmission type"
        )
    
    with col2:
        st.subheader("📊 Technical Specifications")
        
        # Numerical inputs
        year = st.slider(
            "📅 Year",
            min_value=2000,
            max_value=2025,
            value=2015,
            help="Select the manufacturing year"
        )
        
        engine_size = st.slider(
            "🔧 Engine Size (L)",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.1,
            help="Select the engine size in liters"
        )
        
        mileage = st.number_input(
            "🛣️ Mileage (km)",
            min_value=0,
            max_value=300000,
            value=50000,
            step=1000,
            help="Enter the total mileage in kilometers"
        )
    
    # Prediction section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔮 Predict Car Price", use_container_width=True):
            try:
                # Make prediction
                predicted_price = car_predictor.predict_price(
                    make=make,
                    model=model,
                    year=year,
                    engine_size=engine_size,
                    mileage=mileage,
                    fuel_type=fuel_type,
                    transmission=transmission
                )
                
                # Display prediction
                st.markdown("### 💰 Predicted Price")
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<h2 style="color: #1f77b4; text-align: center;">${predicted_price:,.2f}</h2>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Additional information
                st.markdown("### 📈 Prediction Details")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Base Price Range", "$6,705 - $41,781")
                with col2:
                    st.metric("Model Accuracy (R²)", "79.5%")
                with col3:
                    st.metric("Average Error", "$1,916")
                
                # Display input summary
                st.markdown("### 📋 Input Summary")
                input_data = {
                    "Make": make,
                    "Model": model,
                    "Year": year,
                    "Engine Size": f"{engine_size}L",
                    "Mileage": f"{mileage:,} km",
                    "Fuel Type": fuel_type,
                    "Transmission": transmission
                }
                
                st.json(input_data)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
    
    # Information section
    st.markdown("---")
    
    # Model information
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        ### Model Information
        
        - **Algorithm**: Random Forest Regressor
        - **Training Data**: 1,000 car records
        - **Features**: Make, Model, Year, Engine Size, Mileage, Fuel Type, Transmission
        - **Performance**: 79.5% R² score on test data
        - **Most Important Features**: Year, Mileage, Engine Size
        
        ### How It Works
        
        1. **Data Processing**: The model encodes categorical variables and scales numerical features
        2. **Prediction**: Uses ensemble learning with 100 decision trees
        3. **Output**: Provides price estimate based on similar cars in the training data
        
        ### Limitations
        
        - Predictions are based on the training dataset patterns
        - Actual market prices may vary due to factors not in the model
        - Best used as a reference point for price estimation
        """)
    
    # Feature importance
    with st.expander("📊 Feature Importance"):
        st.markdown("""
        Based on the trained model, here's how much each feature influences the price prediction:
        
        1. **Year (37.9%)** - Newer cars tend to be more expensive
        2. **Mileage (37.7%)** - Lower mileage increases the price
        3. **Engine Size (19.4%)** - Larger engines generally cost more
        4. **Model (1.9%)** - Some models have premium pricing
        5. **Make (1.7%)** - Brand influence on pricing
        6. **Fuel Type (0.9%)** - Electric vs Petrol vs Diesel impact
        7. **Transmission (0.6%)** - Automatic vs Manual difference
        """)

if __name__ == "__main__":
    main()
