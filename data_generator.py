"""
Synthetic Sleep Quality Data Generator
Generates realistic lifestyle data and corresponding sleep quality scores
"""

import pandas as pd
import numpy as np
import random

def generate_synthetic_data(n_samples=1000, random_seed=42):
    """
    Generate synthetic sleep quality dataset based on lifestyle factors
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Dataset with lifestyle features and sleep quality scores
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    data = []
    
    for _ in range(n_samples):
        # Generate realistic lifestyle parameters
        screen_time = np.random.normal(6, 2.5)  # hours, mean 6, std 2.5
        screen_time = max(0, min(16, screen_time))  # Clip between 0-16 hours
        
        caffeine = np.random.exponential(150)  # mg/day
        caffeine = min(600, caffeine)  # Cap at 600mg
        
        exercise = np.random.gamma(2, 15)  # minutes/day
        exercise = min(180, exercise)  # Cap at 3 hours
        
        stress = np.random.beta(2, 3) * 10  # Scale 1-10
        stress = max(1, min(10, stress))
        
        # Sleep duration affected by screen time and stress
        base_sleep = 8.0
        sleep_duration = base_sleep - (screen_time * 0.15) - (stress * 0.1) + np.random.normal(0, 0.5)
        sleep_duration = max(4, min(12, sleep_duration))
        
        # Calculate sleep quality score (0-100 scale)
        # Base score
        sleep_score = 50
        
        # Positive factors
        sleep_score += (exercise / 180) * 25  # More exercise = better sleep
        sleep_score += (8 - abs(8 - sleep_duration)) * 5  # Optimal 8 hours
        
        # Negative factors
        sleep_score -= (screen_time / 16) * 30  # Screen time hurts
        sleep_score -= (caffeine / 600) * 20  # Caffeine hurts
        sleep_score -= (stress / 10) * 15  # Stress hurts
        
        # Add some randomness
        sleep_score += np.random.normal(0, 8)
        
        # Clip between 0-100
        sleep_score = max(0, min(100, sleep_score))
        
        data.append({
            'screen_time': round(screen_time, 2),
            'caffeine_intake': round(caffeine, 2),
            'exercise_duration': round(exercise, 2),
            'stress_level': round(stress, 1),
            'sleep_duration': round(sleep_duration, 2),
            'sleep_quality_score': round(sleep_score, 2)
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate dataset
    print("Generating synthetic sleep quality dataset...")
    df = generate_synthetic_data(n_samples=1000, random_seed=42)
    
    # Save to CSV
    df.to_csv('data/sleep_quality_dataset.csv', index=False)
    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset statistics:")
    print(df.describe())

