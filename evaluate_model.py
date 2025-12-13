"""
Model Evaluation Script
Evaluates the trained model with detailed metrics and visualizations
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os

def load_model_and_data():
    """Load the trained model and dataset"""
    # Load model
    model_path = 'models/sleep_quality_model.pkl'
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    
    # Load data
    df = pd.read_csv('data/sleep_quality_dataset.csv')
    X = df[feature_columns]
    y = df['sleep_quality_score']
    
    return model, scaler, X, y, feature_columns

def evaluate_model(model, scaler, X, y):
    """Evaluate model performance"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R² Score'],
        'Training': [train_mae, train_rmse, train_r2],
        'Testing': [test_mae, test_rmse, test_r2]
    })
    
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print("\nTraining Set Performance:")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print("\nTest Set Performance:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    print("=" * 60)
    
    return {
        'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2,
                  'y_true': y_train, 'y_pred': y_train_pred},
        'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2,
                 'y_true': y_test, 'y_pred': y_test_pred}
    }

def create_visualizations(model, X, y, feature_columns, results):
    """Create evaluation visualizations"""
    os.makedirs('results', exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Feature Importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance in Sleep Quality Prediction', fontsize=16, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction vs Actual (Test Set)
    plt.figure(figsize=(10, 8))
    plt.scatter(results['test']['y_true'], results['test']['y_pred'], 
                alpha=0.6, s=50, color='#6366f1')
    
    # Perfect prediction line
    min_val = min(min(results['test']['y_true']), min(results['test']['y_pred']))
    max_val = max(max(results['test']['y_true']), max(results['test']['y_pred']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Sleep Quality Score', fontsize=12)
    plt.ylabel('Predicted Sleep Quality Score', fontsize=12)
    plt.title('Predicted vs Actual Sleep Quality Score (Test Set)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Residual Plot
    residuals = results['test']['y_true'] - results['test']['y_pred']
    plt.figure(figsize=(10, 6))
    plt.scatter(results['test']['y_pred'], residuals, alpha=0.6, s=50, color='#ec4899')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Sleep Quality Score', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title('Residual Plot (Test Set)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/residual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color='#8b5cf6', alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=16, fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('results/error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualizations saved to 'results' directory:")
    print("  - feature_importance.png")
    print("  - prediction_vs_actual.png")
    print("  - residual_plot.png")
    print("  - error_distribution.png")

if __name__ == "__main__":
    print("Loading model and data...")
    model, scaler, X, y, feature_columns = load_model_and_data()
    
    print("Evaluating model...")
    results = evaluate_model(model, scaler, X, y)
    
    print("\nCreating visualizations...")
    create_visualizations(model, X, y, feature_columns, results)
    
    print("\nEvaluation complete!")

