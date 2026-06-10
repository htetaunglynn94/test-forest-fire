import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, jsonify, request

application = Flask(__name__)
app = application

# Import Ridge Regressor and Standard Scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        # Transform data
        scaled = scaler_model.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result = ridge_model.predict(scaled)
        return render_template('home.html', results=round(result[0],3))

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")