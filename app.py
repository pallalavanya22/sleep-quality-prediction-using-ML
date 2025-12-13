"""
Flask Web Application for Sleep Quality Prediction
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
MODEL_PATH = 'models/sleep_quality_model.pkl'

def load_model():
    """Load the trained model and scaler"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data['scaler'], model_data['feature_columns']

# Load model on startup
try:
    model, scaler, feature_columns = load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_columns = None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle sleep quality prediction requests"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        # Extract features in correct order
        screen_time = float(data.get('screen_time', 0))
        caffeine_intake = float(data.get('caffeine_intake', 0))
        exercise_duration = float(data.get('exercise_duration', 0))
        stress_level = float(data.get('stress_level', 0))
        sleep_duration = float(data.get('sleep_duration', 0))
        
        # Create feature array
        features = np.array([[
            screen_time,
            caffeine_intake,
            exercise_duration,
            stress_level,
            sleep_duration
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Ensure prediction is within 0-100 range
        prediction = max(0, min(100, prediction))
        
        # Get feature importance for insights
        importances = model.feature_importances_
        feature_names = feature_columns
        
        # Create insights
        insights = []
        if screen_time > 6:
            insights.append("High screen time may be affecting your sleep quality.")
        if caffeine_intake > 200:
            insights.append("Reducing caffeine intake, especially in the afternoon, could improve sleep.")
        if exercise_duration < 30:
            insights.append("Regular exercise can significantly improve sleep quality.")
        if stress_level > 7:
            insights.append("High stress levels are negatively impacting your sleep.")
        if sleep_duration < 7 or sleep_duration > 9:
            insights.append(f"Optimal sleep duration is 7-9 hours. You're getting {sleep_duration:.1f} hours.")
        
        if not insights:
            insights.append("Your lifestyle habits look good! Keep maintaining them.")
        
        return jsonify({
            'prediction': round(prediction, 2),
            'insights': insights,
            'feature_importance': {
                feature_names[i]: float(importances[i])
                for i in range(len(feature_names))
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

