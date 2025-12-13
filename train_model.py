"""
Train Sleep Quality Prediction Model
Uses Random Forest and XGBoost regressors
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def load_data(data_path='data/sleep_quality_dataset.csv'):
    """Load the sleep quality dataset"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run data_generator.py first.")
    
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """Preprocess the data for model training"""
    # Separate features and target
    feature_columns = ['screen_time', 'caffeine_intake', 'exercise_duration', 
                       'stress_level', 'sleep_duration']
    X = df[feature_columns]
    y = df['sleep_quality_score']
    
    return X, y

def train_models(X, y, test_size=0.2, random_state=42):
    """Train Random Forest and XGBoost models"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize scaler (optional for tree-based models, but good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Train XGBoost
    print("Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }
        
        print(f"\n{name} Performance:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")
    
    # Select best model based on R² score
    best_model_name = max(results.keys(), key=lambda k: results[k]['R²'])
    best_model = models[best_model_name]
    
    print(f"\nBest Model: {best_model_name} (R² = {results[best_model_name]['R²']:.4f})")
    
    return best_model, scaler, X_train.columns.tolist(), results

def save_model(model, scaler, feature_columns, model_path='models/sleep_quality_model.pkl'):
    """Save the trained model and scaler"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to {model_path}")

def get_feature_importance(model, feature_columns):
    """Get and display feature importance"""
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))
    
    return feature_importance

if __name__ == "__main__":
    print("=" * 50)
    print("Sleep Quality Prediction Model Training")
    print("=" * 50)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    
    # Preprocess
    print("\nPreprocessing data...")
    X, y = preprocess_data(df)
    
    # Train models
    print("\nTraining models...")
    model, scaler, feature_columns, results = train_models(X, y)
    
    # Feature importance
    feature_importance = get_feature_importance(model, feature_columns)
    
    # Save model
    save_model(model, scaler, feature_columns)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)

