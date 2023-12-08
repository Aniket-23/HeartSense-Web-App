from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/survey')
def survey():
    return render_template('survey.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login logic here
        # For now, let's just redirect to the landing page route
        return render_template('landing_index.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
