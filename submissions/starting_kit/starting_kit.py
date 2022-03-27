from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, \
    OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X, y):

        #numeric transformer
        numeric_transformer=SimpleImputer(strategy='median')

        #date transformer
        def process_date(X):
            date = pd.to_datetime(X['date_reception_dpe'])
            return np.c_[(date-min(date)).dt.total_seconds()]

        date_transformer = FunctionTransformer(process_date)

        #label transformer

        label_transformer = LabelEncoder()

        num_cols_w_nan=['tv026_classe_inertie_id','surface_habitable','nombre_niveau']
        date_col=['date_reception_dpe']

        drop_cols=['consommation_energie', 'deperdition_enveloppe', 'deperdition_renouvellement_air', 'code_insee_commune_corrige', 'commune', 'numero_dpe','shon']

        preprocessor = ColumnTransformer(
            transformers=[('date', make_pipeline(date_transformer,SimpleImputer(strategy='median')), date_col),
                
                ('num',make_pipeline(numeric_transformer,StandardScaler()), num_cols_w_nan),

                ('drop cols', 'drop', drop_cols),
            ],remainder='passthrough')
        self.preprocessor = preprocessor
        self.preprocessor.fit(X)

        self.labelencoder=label_transformer
        self.labelencoder.fit(y)
        return self

    def transform(self, X,y):
        return self.preprocessor.transform(X),self.labelencoder.transform(y)

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([
            ('classifier', RandomForestClassifier())
        ])
        self.fe = FeatureExtractor()

    def fit(self, X, y):
        self.fe.fit(X,y)
        Xf,yf=self.fe.transform(X,y)
        self.clf.fit(Xf, yf)

    def predict_proba(self, X):
        Xf=self.fe.transform(X)
        return self.clf.predict_proba(Xf)