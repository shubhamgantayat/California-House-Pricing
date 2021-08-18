from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from transformers.combined_attributes_adder import CombinedAttributesAdder
from transformers.top_features_selector import TopFeatureSelector

app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template("form.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        california_housing_model = joblib.load("california_housing.pkl")
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
        result = california_housing_model.predict(df)[0]
        return render_template("form.html", results=result)


if __name__ == '__main__':
    app.run(debug=True)
