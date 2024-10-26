import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from flask import Flask , render_template, request, jsonify

application = Flask(__name__)
app= application


## importing the pickled models (ridge regression model and standard scaler)
ridge_reg = pickle.load(open('models/ridgereg.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        new_data_scaled = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_reg.predict(new_data_scaled)
        return render_template('home.html',results=result[0])
    
    else:
        return render_template(('home.html'))



if __name__ =="__main__":
    app.run(host= '0.0.0.0',
        debug=True)