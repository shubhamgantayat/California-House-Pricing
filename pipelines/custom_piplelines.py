import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from transformers.combined_attributes_adder import CombinedAttributesAdder
from transformers.top_features_selector import TopFeatureSelector


class CustomPipeline:

    @staticmethod
    def num_pipeline(strategy):
        return Pipeline([
            ('imputer', SimpleImputer(strategy=strategy)),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler())
        ])

    @staticmethod
    def full_pipeline(strategy):
        num_attribs = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                       'total_bedrooms', 'population', 'households', 'median_income']
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
            ("num", CustomPipeline().num_pipeline(strategy), num_attribs),
            ("cat", OneHotEncoder(), cat_attribs)
        ])
        return full_pipeline

    @staticmethod
    def prepare_predict_and_select_pipeline(strategy, k):
        feature_importances = np.array([6.85944024e-02, 6.43660684e-02, 4.16819920e-02, 1.63422032e-02,
                                        1.49443509e-02, 1.56476712e-02, 1.41037697e-02, 3.85397399e-01,
                                        5.04758037e-02, 1.11174478e-01, 5.32136556e-02, 4.07202120e-03,
                                        1.55063935e-01, 5.73695231e-05, 1.89097445e-03, 2.97390579e-03])
        prepare_select_and_predict_pipeline = Pipeline([
            ('preparation', CustomPipeline().full_pipeline(strategy)),
            ('feature_selection', TopFeatureSelector(feature_importances, k))
        ])
        return prepare_select_and_predict_pipeline
