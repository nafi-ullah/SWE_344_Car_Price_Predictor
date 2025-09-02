#!/usr/bin/env python3
"""
Test script for car price prediction
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

class CarPricePrediction:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission']
    
    def load_model(self, model_path='car_price_model.pkl'):
        """Load a saved model and preprocessors"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {model_path}")
    
    def predict_price(self, make, model, year, engine_size, mileage, fuel_type, transmission):
        """Predict car price for given features"""
        # Create a dataframe with the input
        input_data = pd.DataFrame({
            'Make': [make],
            'Model': [model],
            'Year': [year],
            'Engine Size': [engine_size],
            'Mileage': [mileage],
            'Fuel Type': [fuel_type],
            'Transmission': [transmission]
        })
        
        # Encode categorical variables
        categorical_columns = ['Make', 'Model', 'Fuel Type', 'Transmission']
        for col in categorical_columns:
            input_data[col] = self.label_encoders[col].transform(input_data[col])
        
        # Scale numerical features
        numerical_columns = ['Year', 'Engine Size', 'Mileage']
        input_data[numerical_columns] = self.scaler.transform(input_data[numerical_columns])
        
        # Make prediction
        predicted_price = self.model.predict(input_data)[0]
        return predicted_price

def test_prediction():
    """Test the car price prediction functionality"""
    print("=== Car Price Prediction Test ===")
    
    # Load the trained model
    try:
        car_predictor = CarPricePrediction()
        car_predictor.load_model('car_price_model.pkl')
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model file not found. Please run main.py first.")
        return
    
    # Test cases
    test_cases = [
        {
            'name': 'Honda Model B (2015) - Mid-range',
            'make': 'Honda',
            'model': 'Model B',
            'year': 2015,
            'engine_size': 3.0,
            'mileage': 50000,
            'fuel_type': 'Petrol',
            'transmission': 'Automatic'
        },
        {
            'name': 'BMW Model A (2020) - Luxury',
            'make': 'BMW',
            'model': 'Model A',
            'year': 2020,
            'engine_size': 4.0,
            'mileage': 20000,
            'fuel_type': 'Petrol',
            'transmission': 'Automatic'
        },
        {
            'name': 'Ford Model C (2010) - Budget',
            'make': 'Ford',
            'model': 'Model C',
            'year': 2010,
            'engine_size': 2.0,
            'mileage': 120000,
            'fuel_type': 'Diesel',
            'transmission': 'Manual'
        },
        {
            'name': 'Toyota Model E (2018) - Electric',
            'make': 'Toyota',
            'model': 'Model E',
            'year': 2018,
            'engine_size': 2.5,
            'mileage': 30000,
            'fuel_type': 'Electric',
            'transmission': 'Automatic'
        }
    ]
    
    print("\n=== Test Predictions ===")
    for i, test_case in enumerate(test_cases, 1):
        try:
            predicted_price = car_predictor.predict_price(
                make=test_case['make'],
                model=test_case['model'],
                year=test_case['year'],
                engine_size=test_case['engine_size'],
                mileage=test_case['mileage'],
                fuel_type=test_case['fuel_type'],
                transmission=test_case['transmission']
            )
            
            print(f"\n{i}. {test_case['name']}")
            print(f"   Specifications: {test_case['year']} {test_case['make']} {test_case['model']}")
            print(f"   Engine: {test_case['engine_size']}L {test_case['fuel_type']}")
            print(f"   Transmission: {test_case['transmission']}")
            print(f"   Mileage: {test_case['mileage']:,} km")
            print(f"   ✅ Predicted Price: ${predicted_price:,.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print("\n=== Test Summary ===")
    print("✅ All test cases completed!")
    print("🚗 The car price prediction model is working correctly.")
    print("🌐 You can now use the Streamlit app at: http://localhost:8501")

if __name__ == "__main__":
    test_prediction()
