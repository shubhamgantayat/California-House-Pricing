from flask import Flask, request, render_template
import joblib

import pandas as pd
from data.preprocessing import Preprocessing
from prediction.predict_value import Prediction

app = Flask(__name__)
model = joblib.load("california_housing.pkl")
preprocessing = Preprocessing()
prediction = Prediction(model, preprocessing.pipeline)


@app.route('/')
def home_page():
    return render_template("form.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                       'total_bedrooms', 'population', 'households', 'median_income',
                       'ocean_proximity']
            values = dict()
            for col in columns:
                if col == 'ocean_proximity':
                    values[col] = [request.form[col]]
                else:
                    values[col] = [float(request.form[col])]
            df = pd.DataFrame.from_dict(values)
            result = prediction.predict(df)
        except Exception as e:
            result = "Please enter valid inputs"
        return render_template("form.html", results=result)


if __name__ == '__main__':
    app.run(debug=True)
