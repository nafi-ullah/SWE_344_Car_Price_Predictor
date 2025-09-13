

# Car Price Prediction Application 🚗

A machine learning application that predicts car prices based on various features using scikit-learn and provides an interactive web interface using Streamlit.[1][2][3][4]

***

## Features

- **Machine Learning Model**: Random Forest Regressor with 79.5% accuracy  
- **Interactive GUI**: Streamlit web application with dropdowns and sliders  
- **Real-time Predictions**: Instant price predictions based on user input  
- **Comprehensive Analysis**: Feature importance and model performance metrics  

***

## Dataset

The application uses a dataset with 1,000 car records containing the following features:  
- **Make**: Audi, BMW, Ford, Honda, Toyota  
- **Model**: Model A, Model B, Model C, Model D, Model E  
- **Year**: 2000-2021  
- **Engine Size**: 1.0-5.0 liters  
- **Mileage**: 0-300,000 km  
- **Fuel Type**: Diesel, Electric, Petrol  
- **Transmission**: Automatic, Manual  
- **Price**: Target variable (\$6,705 - \$41,781)  

***

## Installation

1. **Clone or download the project**:
   ```bash
   cd /path/to/car_price_prediction
   ```
2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```
3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

***

## Usage

### 1. Train the Model
```bash
python main.py
```
This will:
- Load and preprocess the dataset
- Train a Random Forest model
- Display performance metrics
- Save the trained model as `car_price_model.pkl`
- Generate a sample prediction

### 2. Test the Model
```bash
python test_prediction.py
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
The application will be available at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://your-ip:8501

***

## Using the Web Interface

1. **Input Car Details**:
   - Select **Make** from dropdown (Audi, BMW, Ford, Honda, Toyota)
   - Select **Model** from dropdown (Model A-E)
   - Select **Fuel Type** from dropdown (Diesel, Electric, Petrol)
   - Select **Transmission** from dropdown (Automatic, Manual)
   - Adjust **Year** slider (2000-2025)
   - Adjust **Engine Size** slider (1.0-5.0L)
   - Enter **Mileage** (0-300,000 km)

2. **Get Prediction**:
   - Click "🔮 Predict Car Price" button
   - View the predicted price
   - See model accuracy metrics
   - Review input summary

3. **Additional Information**:
   - Expand "About This Model" for technical details
   - Expand "Feature Importance" to understand which factors affect pricing most

***

## Model Performance

- **Algorithm**: Random Forest Regressor (100 trees)
- **Training Accuracy**: 97.2% R²
- **Testing Accuracy**: 79.5% R²
- **Mean Absolute Error**: \$1,916
- **Most Important Features**: Year (37.9%), Mileage (37.7%), Engine Size (19.4%)

### Model Evaluation Plots
The following plots show the model's performance on the test dataset:

![Model Performance]( **Left Plot - Actual vs Predicted Prices**: Shows how well the model's predictions match the actual car prices. Points closer to the red diagonal line indicate better predictions.
- **Right Plot - Residual Plot**: Shows the difference between actual and predicted prices. Points scattered randomly around the horizontal line (y=0) indicate good model performance with no systematic bias.

***

## File Structure

```
car_price_prediction/
├── Car_Price_Prediction.csv    # Dataset
├── main.py                     # Model training script
├── app.py                      # Streamlit web application
├── test_prediction.py          # Test script
├── requirements.txt            # Package dependencies
├── car_price_model.pkl         # Saved trained model (generated)
├── model_performance.png       # Performance plots (generated)
└── README.md                   # This file
```

***

## Requirements

```
pandas==2.0.3
scikit-learn==1.3.0
streamlit==1.25.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
```

***

## Example Predictions

| Car Details | Predicted Price |
|-------------|----------------|
| 2015 Honda Model B, 3.0L Petrol, Automatic, 50k km | $30,284 |
| 2020 BMW Model A, 4.0L Petrol, Automatic, 20k km | $35,453 |
| 2010 Ford Model C, 2.0L Diesel, Manual, 120k km | $21,746 |
| 2018 Toyota Model E, 2.5L Electric, Automatic, 30k km | $32,523 |

***

## Notes

- Predictions are estimates based on the training dataset  
- Actual market prices may vary due to factors not included in the model  
- The model works best for cars within the training data ranges  
- For production use, consider retraining with more recent and comprehensive data  

***

## Troubleshooting

1. **Model file not found**: Run `python main.py` first to train and save the model  
2. **Import errors**: Ensure all packages are installed with `pip install -r requirements.txt`  
3. **Streamlit issues**: Try updating Streamlit with `pip install --upgrade streamlit`  
4. **Permission errors**: Ensure you have write permissions in the project directory  

***

## Future Improvements

- Add more car features (brand reputation, condition, location)
- Include time series analysis for price trends
- Implement ensemble methods with multiple algorithms
- Add data validation and error handling
- Deploy to cloud platforms (Heroku, AWS, etc.)

***

## Image Demonstration

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ff1f9c2-9fef-4703-9697-717c9aee540d" width="400"/>
  <img src="https://github.com/user-attachments/assets/1b62edf6-fbaf-496e-88b6-91ef8798fa7e" width="400"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/d960ff61-5b3b-4e00-8871-348669d60f3f" width="400"/>
  <img src="https://github.com/user-attachments/assets/792c6117-b602-4162-a316-986c449b094d" width="400"/>
</p>

---
