import numpy as np
import pickle
from flask import Flask, request, render_template

# Load Machine Learning Model
model = pickle.load(open('model.pkl', 'rb')) 

# Create Web Application
app = Flask(__name__)

# Bind The Home Function to URL
@app.route('/')
def home():
    return render_template('Heart Disease Prediction.html')

# Bind The Predict Function to URL
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data as a dictionary
    features = request.form.to_dict()
    
    # Convert form values to float and store in a list
    feature_values = [float(value) for value in features.values()]
    
    # Convert features to array
    array_features = np.array([feature_values])
    
    # Predict features
    prediction = model.predict(array_features)
    
    output = prediction[0]
    
    # Check the output values and retrieve the result with an HTML tag based on the value
    if output == 1:
        result = 'It is unlikely that the patient has heart disease.'
    else:
        result = 'It is likely that the patient has heart disease.'
    
    return render_template('Heart Disease Prediction.html', result=result)

if __name__ == '__main__':
    # Run the Application
    app.run()
