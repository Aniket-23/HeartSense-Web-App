from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
from werkzeug.utils import secure_filename
import os
import cv2
from tensorflow.keras.models import load_model
import pickle
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Daemon@2247211'

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["heartsense"]  
users_collection = db["users"]

# Load your saved model
model = load_model('./static/CardiacRiskPrediction.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if the passwords match
        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match')

        # Check if the email is unique
        existing_user = users_collection.find_one({'email': email})
        if existing_user:
            return render_template('signup.html', error='Email already exists')

        # Hash the password before saving to the database
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Save the user to the database
        user_data = {
            'name': name,
            'email': email,
            'password': hashed_password,  # Store hashed password
            # Add other user data fields as needed
        }

        try:
            users_collection.insert_one(user_data)
            print("User successfully registered.")
        except Exception as e:
            print(f"Error inserting user into the database: {e}")

        # Redirect to the login page after successful registration
        return redirect(url_for('landing_index'))

    return render_template('signup.html')

@app.route('/survey')
def survey():
    return render_template('survey.html')

@app.route('/landing_index')
def landing_index():
    # Check if the user is in the session
    if 'user' in session:
        user = session['user']
        # Access user information, e.g., user['name'], user['email']
        return render_template('landing_index.html', user=user)
    else:
        # If the user is not in the session, redirect to the login page
        flash('Please log in first', 'danger')
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if email and password:
            # Check if the user exists in the database
            user = users_collection.find_one({'email': email})

            if user and check_password_hash(user['password'], password):
                # If the email and password match, set the user in the session
                session['user'] = {
                    'email': user['email'],
                    'name': user['name']
                    # Add other user data fields as needed
                }
                flash('Login successful', 'success')
                return redirect(url_for('landing_index'))
            else:
                # If login fails, show an error message
                flash('Invalid email or password', 'danger')
        else:
            # If email or password is not provided, show an error message
            flash('Please provide both email and password', 'danger')

    # If the login fails, redirect back to the login page
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

# Path to the uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the 'uploads' directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your pre-trained Keras model
bestmodel = load_model('./static/best_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to get the class labels
def get_class_labels():
    return ['N', 'S', 'V', 'F', 'Q']

# Route for the main page
@app.route('/ecginput')
def ecginput():
    return render_template('ecginput.html')

# Route for handling the uploaded file and displaying the result
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform classification
        input_image = preprocess_image(file_path)
        prediction = bestmodel.predict(input_image)
        class_labels = get_class_labels()
        predicted_class = class_labels[np.argmax(prediction)]

        # Redirect to the result page
        return redirect(url_for('ecg_report', filename=filename, predicted_class=predicted_class))

# Route for the ECG report page
@app.route('/ecgreport/<filename>/<predicted_class>')
def ecg_report(filename, predicted_class):
    return render_template('ecgreport.html', filename=filename, predicted_class=predicted_class)


if __name__ == '__main__':
    app.run(debug=True)
