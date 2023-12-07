from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load your saved model
model = load_model('./static/CardiacRiskPrediction.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/survey')
def survey():
    return render_template('survey.html')

@app.route('/landing_index')
def landing_index():
    return render_template('landing_index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login logic here
        # For now, let's just redirect to the landing page route
        return redirect(url_for('landing_index'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = []
    input_features = ['Gender', 'Chain_smoker', 'Consumes_other_tobacco_products', 'HighBP', 'Obese', 
                      'Diabetes', 'Metabolic_syndrome', 'Use_of_stimulant_drugs', 'Family_history', 
                      'History_of_preeclampsia', 'CABG_history', 'Respiratory_illness']
    
    for feature in input_features:
        value = request.form[feature]
        
        # Convert 'Yes' to 1 and 'No' to 0
        if value == 'Yes':
            input_data.append(1)
        elif value == 'No':
            input_data.append(0)
        else:
            # Handle other cases or validation as needed
            input_data.append(0)

    # Convert input data to numpy array and reshape
    input_data = np.array(input_data).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(input_data)

    # Assign the prediction result to a variable
    prediction_result = 'At Risk' if prediction[0][0] > 0.5 else 'Not at Risk'

    # Redirect to the report page with the prediction result and input data
    return redirect(url_for('report', prediction_result=prediction_result, **request.form))

@app.route('/report')
def report():
    # Render the report template and pass the prediction result and input data
    return render_template('report.html', prediction=request.args.get('prediction_result'), **request.args)

# Load the trained SVM model (replace 'svm_model.pkl' with your actual model file)
with open('./static/svm_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/numeric')
def numeric():
    return render_template('numeric.html')

@app.route('/numeric_predict', methods=['POST'])
def numeric_predict():
    # Collect input data from the form
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    chest_pain = int(request.form['chest_pain'])
    resting_bp = float(request.form['resting_bp'])
    serum_cholesterol = float(request.form['serum_cholesterol'])
    fasting_blood_sugar = int(request.form['fasting_blood_sugar'])
    resting_ecg = int(request.form['resting_ecg'])
    max_heart_rate = float(request.form['max_heart_rate'])
    exercise_angina = int(request.form['exercise_angina'])
    st_depression = float(request.form['st_depression'])
    slope_st_segment = int(request.form['slope_st_segment'])
    num_vessels = int(request.form['num_vessels'])
    thal = int(request.form['thal'])

    # Preprocess input data
    input_data = np.array([[age, sex, chest_pain, resting_bp, serum_cholesterol, fasting_blood_sugar,
                            resting_ecg, max_heart_rate, exercise_angina, st_depression, slope_st_segment,
                            num_vessels, thal]])

    # Make predictions using the loaded model
    prediction = loaded_model.predict(input_data)

    # Display the prediction result
    numeric_result = "Disease" if prediction[0] == 1 else "No Disease"

    # Render numeric_report.html with entered details and prediction result
    return render_template('numeric_report.html', age=age, sex=sex, chest_pain=chest_pain,
                           resting_bp=resting_bp, serum_cholesterol=serum_cholesterol,
                           fasting_blood_sugar=fasting_blood_sugar, resting_ecg=resting_ecg,
                           max_heart_rate=max_heart_rate, exercise_angina=exercise_angina,
                           st_depression=st_depression, slope_st_segment=slope_st_segment,
                           num_vessels=num_vessels, thal=thal, numeric_result=numeric_result)


if __name__ == '__main__':
    app.run(debug=True)
