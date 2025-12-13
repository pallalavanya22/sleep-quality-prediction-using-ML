"""
Setup script to create necessary directories and check dependencies
"""

import os

def create_directories():
    """Create necessary directories for the project"""
    directories = [
        'data',
        'models',
        'results',
        'static/css',
        'static/js',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is NOT installed")
    
    if missing_packages:
        print(f"\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        print(f"\nOr install all requirements:")
        print(f"pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed!")
        return True

if __name__ == "__main__":
    print("=" * 50)
    print("Sleep Quality Predictor - Setup")
    print("=" * 50)
    
    print("\n1. Creating directories...")
    create_directories()
    
    print("\n2. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 50)
    if deps_ok:
        print("Setup complete! You can now:")
        print("  1. Run 'python data_generator.py' to generate data")
        print("  2. Run 'python train_model.py' to train the model")
        print("  3. Run 'python app.py' to start the web app")
    else:
        print("Setup incomplete. Please install missing dependencies first.")
    print("=" * 50)

