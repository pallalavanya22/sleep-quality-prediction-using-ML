// Sleep Quality Predictor JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const stressSlider = document.getElementById('stress_level_slider');
    const stressInput = document.getElementById('stress_level');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');
    
    // Sync slider with input
    stressSlider.addEventListener('input', function() {
        stressInput.value = this.value;
    });
    
    stressInput.addEventListener('input', function() {
        stressSlider.value = this.value;
    });

    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide previous results
        document.getElementById('resultContainer').classList.add('hidden');
        document.getElementById('errorMessage').classList.add('hidden');
        
        // Show loading spinner
        document.getElementById('loadingSpinner').classList.remove('hidden');
        predictBtn.disabled = true;
        
        // Get form data
        const formData = {
            screen_time: parseFloat(document.getElementById('screen_time').value),
            caffeine_intake: parseFloat(document.getElementById('caffeine_intake').value),
            exercise_duration: parseFloat(document.getElementById('exercise_duration').value),
            stress_level: parseFloat(document.getElementById('stress_level').value),
            sleep_duration: parseFloat(document.getElementById('sleep_duration').value)
        };
        
        try {
            // Make prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }
            
            // Hide loading spinner
            document.getElementById('loadingSpinner').classList.add('hidden');
            
            // Display results
            displayResults(data);
            
            // Scroll to results
            document.getElementById('resultContainer').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
            
        } catch (error) {
            console.error('Error:', error);
            document.getElementById('loadingSpinner').classList.add('hidden');
            showError(error.message);
        } finally {
            predictBtn.disabled = false;
        }
    });
    
    // Reset button
    resetBtn.addEventListener('click', function() {
        document.getElementById('resultContainer').classList.add('hidden');
        form.reset();
        // Reset slider
        stressSlider.value = 5;
        stressInput.value = 5;
        // Scroll to form
        form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    
    function displayResults(data) {
        const { prediction, insights, feature_importance } = data;
        
        // Update score
        const scoreValue = document.getElementById('scoreValue');
        const scoreCircle = document.getElementById('scoreCircle');
        const scoreInterpretation = document.getElementById('scoreInterpretation');
        
        // Animate score
        animateValue(scoreValue, 0, prediction, 1500);
        
        // Update score circle color based on score
        updateScoreCircle(scoreCircle, prediction);
        
        // Update interpretation
        scoreInterpretation.textContent = getScoreInterpretation(prediction);
        scoreInterpretation.className = 'score-interpretation ' + getScoreClass(prediction);
        
        // Update insights
        const insightsList = document.getElementById('insightsList');
        insightsList.innerHTML = '';
        insights.forEach(insight => {
            const li = document.createElement('li');
            li.textContent = insight;
            insightsList.appendChild(li);
        });
        
        // Update feature importance
        displayFeatureImportance(feature_importance);
        
        // Show results
        document.getElementById('resultContainer').classList.remove('hidden');
    }
    
    function animateValue(element, start, end, duration) {
        const range = end - start;
        const increment = range / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                element.textContent = Math.round(end);
                clearInterval(timer);
            } else {
                element.textContent = Math.round(current);
            }
        }, 16);
    }
    
    function updateScoreCircle(circle, score) {
        // Remove existing color classes
        circle.classList.remove('score-excellent', 'score-good', 'score-fair', 'score-poor');
        
        // Add appropriate color based on score
        if (score >= 80) {
            circle.style.background = `conic-gradient(
                #10b981 0% ${score}%,
                #e2e8f0 ${score}% 100%
            )`;
        } else if (score >= 60) {
            circle.style.background = `conic-gradient(
                #f59e0b 0% ${score}%,
                #e2e8f0 ${score}% 100%
            )`;
        } else {
            circle.style.background = `conic-gradient(
                #ef4444 0% ${score}%,
                #e2e8f0 ${score}% 100%
            )`;
        }
    }
    
    function getScoreInterpretation(score) {
        if (score >= 80) {
            return 'Excellent! 🌟 Your sleep quality is outstanding.';
        } else if (score >= 60) {
            return 'Good! 👍 Your sleep quality is decent, but there\'s room for improvement.';
        } else if (score >= 40) {
            return 'Fair. 😴 Your sleep quality could be better. Consider making lifestyle changes.';
        } else {
            return 'Poor. 😔 Your sleep quality needs significant improvement. Focus on healthy habits.';
        }
    }
    
    function getScoreClass(score) {
        if (score >= 80) {
            return 'score-excellent';
        } else if (score >= 60) {
            return 'score-good';
        } else if (score >= 40) {
            return 'score-fair';
        } else {
            return 'score-poor';
        }
    }
    
    function displayFeatureImportance(featureImportance) {
        const container = document.getElementById('featureImportance');
        container.innerHTML = '';
        
        // Sort features by importance
        const sortedFeatures = Object.entries(featureImportance)
            .sort((a, b) => b[1] - a[1]);
        
        // Format feature names
        const featureNameMap = {
            'screen_time': '📱 Screen Time',
            'caffeine_intake': '☕ Caffeine Intake',
            'exercise_duration': '🏃 Exercise Duration',
            'stress_level': '😰 Stress Level',
            'sleep_duration': '😴 Sleep Duration'
        };
        
        sortedFeatures.forEach(([feature, importance]) => {
            const featureBar = document.createElement('div');
            featureBar.className = 'feature-bar';
            
            const featureName = document.createElement('div');
            featureName.className = 'feature-name';
            featureName.textContent = featureNameMap[feature] || feature;
            
            const barContainer = document.createElement('div');
            barContainer.className = 'feature-bar-container';
            
            const barFill = document.createElement('div');
            barFill.className = 'feature-bar-fill';
            barFill.style.width = '0%'; // Start at 0 for animation
            barFill.textContent = (importance * 100).toFixed(1) + '%';
            
            // Animate bar fill
            setTimeout(() => {
                barFill.style.width = (importance * 100) + '%';
            }, 100);
            
            barContainer.appendChild(barFill);
            featureBar.appendChild(featureName);
            featureBar.appendChild(barContainer);
            container.appendChild(featureBar);
        });
    }
    
    function showError(message) {
        const errorDiv = document.getElementById('errorMessage');
        errorDiv.textContent = `Error: ${message}`;
        errorDiv.classList.remove('hidden');
    }
});

