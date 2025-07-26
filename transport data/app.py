from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('transport_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        price = float(request.form['price'])
        population = float(request.form['population'])
        income = float(request.form['income'])
        parking = float(request.form['parking'])

        features = np.array([[price, population, income, parking]])
        prediction = model.predict(features)[0]

        return render_template('result.html', prediction=int(prediction))
    except:
        return render_template('result.html', prediction="Invalid input. Please enter numeric values.")

if __name__ == '__main__':
    app.run(debug=True)
