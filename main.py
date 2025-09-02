import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class CarPricePrediction:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission']
        
    def load_and_prepare_data(self, csv_path):
        """Load and prepare the dataset for training"""
        print("Loading dataset...")
        self.df = pd.read_csv(csv_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Display basic information
        print("\nDataset Info:")
        print(self.df.info())
        print("\nFirst few rows:")
        print(self.df.head())
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("\nPreprocessing data...")
        
        # Separate features and target
        X = self.df[self.feature_columns].copy()
        y = self.df['Price'].copy()
        
        # Encode categorical variables
        categorical_columns = ['Make', 'Model', 'Fuel Type', 'Transmission']
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
            print(f"Encoded {col}: {le.classes_}")
        
        # Scale numerical features
        numerical_columns = ['Year', 'Engine Size', 'Mileage']
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        
        return X, y
    
    def train_model(self, X, y):
        """Train the machine learning model"""
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training the Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print("\nModel Performance:")
        print(f"Training MSE: {train_mse:.2f}")
        print(f"Testing MSE: {test_mse:.2f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        print(f"Training MAE: {train_mae:.2f}")
        print(f"Testing MAE: {test_mae:.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return X_train, X_test, y_train, y_test, y_pred_test
    
    def save_model(self, model_path='car_price_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, model_path)
        print(f"\nModel saved as {model_path}")
    
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
    
    def plot_results(self, y_test, y_pred_test):
        """Plot model results"""
        plt.figure(figsize=(12, 4))
        
        # Actual vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred_test, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Prices')
        
        # Residuals
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred_test
        plt.scatter(y_pred_test, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Initialize the prediction system
    car_predictor = CarPricePrediction()
    
    # Load and prepare data
    df = car_predictor.load_and_prepare_data('Car_Price_Prediction.csv')
    
    # Preprocess data
    X, y = car_predictor.preprocess_data()
    
    # Train model
    X_train, X_test, y_train, y_test, y_pred_test = car_predictor.train_model(X, y)
    
    # Plot results
    car_predictor.plot_results(y_test, y_pred_test)
    
    # Save the model
    car_predictor.save_model()
    
    # Example prediction
    print("\nExample Prediction:")
    example_price = car_predictor.predict_price(
        make='Honda',
        model='Model B',
        year=2015,
        engine_size=3.0,
        mileage=50000,
        fuel_type='Petrol',
        transmission='Automatic'
    )
    print(f"Predicted price for Honda Model B (2015): ${example_price:.2f}")

if __name__ == "__main__":
    main()
