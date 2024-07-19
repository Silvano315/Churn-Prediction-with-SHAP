import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, auc

from collections import defaultdict

class ModelPipeline:

    def __init__(self, df, label, models = None):

        self.df = df
        self.label = label
        self.models = models if models else {
            'RandomForest' : RandomForestClassifier(),
            'XGBoost' : XGBClassifier(),
            'LogisticRegression' : LogisticRegression(max_iter=1000)
        }
        self.results = defaultdict(list)
        

    def imbalance_label(self, method = "SMOTE"):

        X = self.df.drop(columns = self.label)
        y = self.df[self.label]

        if method == 'SMOTE':
            over_sampler = SMOTE()
        elif method == 'ADASYN':
            over_sampler = ADASYN()
        elif method == 'KMeansSMOTE':
            over_sampler = KMeansSMOTE()
        else:
            raise ValueError("Invalid method for handling imbalance. Choose 'SMOTE', 'ADASYN', or 'KMeansSMOTE'.")
        
        X_resampled, y_resampled = over_sampler.fit_resample(X,y)
        self.df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        self.df_resampled[self.label] = y_resampled
        

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        pass

    def stratified_k_cv(self, k = 5, imbalance_method = None):
        pass

    def results_viz(self):
        pass

