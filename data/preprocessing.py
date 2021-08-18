import numpy as np
import pandas as pd
import os
from pipelines.custom_piplelines import CustomPipeline


class Preprocessing:

    def __init__(self):
        housing_path = os.path.join("datasets", "housing")
        csv_path = os.path.join(housing_path, "housing.csv")
        df = pd.read_csv(csv_path)
        housing = df.drop(columns=['median_house_value'])
        housing_labels = df["median_house_value"].copy()
        self.pipeline = CustomPipeline().prepare_predict_and_select_pipeline("mean", 16).fit(housing, housing_labels)
