# Sleep Quality Predictor

A machine learning web application that predicts an individual's sleep quality score based on lifestyle and behavioral factors such as screen time, caffeine intake, exercise duration, stress levels, and sleep duration.

## 🌙 Overview

This project aims to help users understand how their lifestyle habits affect their sleep quality. By analyzing various parameters, the system provides personalized predictions and actionable insights to improve sleep health.

## ✨ Features

- **Sleep Quality Prediction**: Predicts sleep quality score (0-100) based on lifestyle factors
- **Personalized Insights**: Provides recommendations based on input data
- **Feature Importance**: Shows which factors most impact sleep quality
- **Beautiful Web Interface**: Modern, responsive UI built with HTML, CSS, and JavaScript
- **Machine Learning Models**: Uses Random Forest and XGBoost regressors

## 🛠️ Technologies Used

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript
- **Development**: Jupyter Notebook compatible

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Installation & Setup

### 1. Clone or Download the Project

```bash
cd "Sleep Quality Prediction"
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Activate the virtual environment:**
- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Dataset

Generate the synthetic sleep quality dataset:

```bash
python data_generator.py
```

This will create a `data` directory with `sleep_quality_dataset.csv`.

### 5. Train the Model

Train the machine learning model:

```bash
python train_model.py
```

This will:
- Train both Random Forest and XGBoost models
- Select the best performing model
- Save the trained model to `models/sleep_quality_model.pkl`
- Display model performance metrics

### 6. (Optional) Evaluate the Model

Generate detailed evaluation metrics and visualizations:

```bash
python evaluate_model.py
```

This creates visualization files in the `results` directory.

## 🎯 Running the Application

Start the Flask web server:

```bash
python app.py
```

The application will be available at: **http://localhost:5000**

Open your web browser and navigate to the URL above to access the Sleep Quality Predictor interface.

## 📊 Dataset Features

The model uses the following input features:

1. **Screen Time** (hours/day): Average daily time spent on screens
2. **Caffeine Intake** (mg/day): Total daily caffeine consumption
3. **Exercise Duration** (minutes/day): Daily physical activity time
4. **Stress Level** (1-10): Perceived stress level
5. **Sleep Duration** (hours): Average hours of sleep per night

**Target Variable:**
- **Sleep Quality Score** (0-100): Predicted sleep quality score

## 🎨 Usage

1. Open the web application in your browser
2. Fill in the form with your lifestyle data:
   - Average daily screen time
   - Daily caffeine intake
   - Daily exercise duration
   - Stress level (use the slider)
   - Average sleep duration
3. Click **"Predict Sleep Quality"**
4. View your predicted sleep quality score and personalized recommendations
5. Review which factors most impact your sleep quality

## 📈 Model Performance

The model is evaluated using:
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors
- **R² Score**: Coefficient of determination (higher is better)

## 📁 Project Structure

```
Sleep Quality Prediction/
│
├── data/
│   └── sleep_quality_dataset.csv      # Generated dataset
│
├── models/
│   └── sleep_quality_model.pkl        # Trained model (generated)
│
├── results/                            # Evaluation visualizations (generated)
│   ├── feature_importance.png
│   ├── prediction_vs_actual.png
│   ├── residual_plot.png
│   └── error_distribution.png
│
├── static/
│   ├── css/
│   │   └── style.css                  # Frontend styles
│   └── js/
│       └── script.js                  # Frontend JavaScript
│
├── templates/
│   └── index.html                     # Main web page
│
├── app.py                             # Flask web application
├── data_generator.py                  # Dataset generation script
├── train_model.py                     # Model training script
├── evaluate_model.py                  # Model evaluation script
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 🔍 Key Insights

The model reveals relationships such as:
- **High screen time** → Lower sleep quality
- **Regular exercise** → Improved sleep quality
- **Excessive caffeine** → Negative impact on sleep
- **High stress levels** → Reduced sleep quality
- **Optimal sleep duration** (7-9 hours) → Better sleep scores

## 🚀 Future Enhancements

- Integration with wearable devices for real-time data
- Additional parameters (diet, heart rate, room temperature)
- Web dashboard with personalized improvement suggestions
- Historical tracking and trend analysis
- User accounts and data persistence
- Mobile app version

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

Developed as part of a machine learning project to predict sleep quality from lifestyle data.

## 🙏 Acknowledgments

- Sleep research community for insights on sleep factors
- Open-source ML libraries (Scikit-learn, XGBoost)
- Flask and web development community

---

**Note**: This application uses synthetic data for demonstration purposes. For real-world applications, consider using validated datasets or collecting data from reliable sources.

For questions or issues, please check the code comments or create an issue in the repository.

