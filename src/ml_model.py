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
from sklearn.metrics import roc_auc_score, roc_curve, auc

from collections import defaultdict

class ModelPipeline:

    def __init__(self, df, label, models = None):

        """
        Initialize the ModelPipeline with the dataset, label column, ML models and results dictionaries.

        Parameters:
        df (pd.DataFrame): The input dataset.
        label (str): The label column name for binary classification.
        """

        self.df = df
        self.label = label
        self.models = models if models else {
            'RandomForest' : RandomForestClassifier(random_state=59),
            'XGBoost' : XGBClassifier(),
            'LogisticRegression' : LogisticRegression(max_iter=1000)
        }
        self.results = {'train': defaultdict(list), 'test': defaultdict(list)}
        

    def imbalance_label(self,X,y, method = "SMOTE"):

        """
        Balance the dataset using the specified method.

        Parameters:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        method (str, optional): The balancing method ('SMOTE', 'KMeansSMOTE', 'SMOTETomek').

        Returns:
        pd.DataFrame: The balanced feature matrix.
        pd.Series: The balanced target vector.
        """

        if method == 'SMOTE':
            over_sampler = SMOTE()
        elif method == 'ADASYN':
            over_sampler = ADASYN()
        elif method == 'KMeansSMOTE':
            over_sampler = KMeansSMOTE()
        else:
            raise ValueError("Invalid method for handling imbalance. Choose 'SMOTE', 'ADASYN', or 'KMeansSMOTE'.")
        
        X_resampled, y_resampled = over_sampler.fit_resample(X,y)
        
        return X_resampled, y_resampled


    def evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test):

        """
        Train and evaluate a machine learning model, and store the results.

        Parameters:
        model_name (str): The name of the model.
        model: The machine learning model instance.
        X_train (pd.DataFrame): The training feature matrix.
        y_train (pd.Series): The training target vector.
        X_test (pd.DataFrame): The testing feature matrix.
        y_test (pd.Series): The testing target vector.

        Returns:
        dict: The training metrics (accuracy, recall, precision, f1_score, balanced_accuracy, roc_auc).
        dict: The testing metrics (accuracy, recall, precision, f1_score, balanced_accuracy, roc_auc).
        """

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'f1_score': f1_score(y_train, y_train_pred),
            'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
            'roc_auc' : roc_auc_score(y_train, y_train_pred)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_pred)
        }

        self.results['train'][model_name].append(train_metrics)
        self.results['test'][model_name].append(test_metrics)
        
        return train_metrics, test_metrics
        

    def stratified_k_cv(self, k = 5, imbalance_method = None):

        """
        Perform stratified k-fold cross-validation with the specified imbalance method.

        Parameters:
        k (int): Number of folds for cross-validation.
        imbalance_method (str, optional): The balancing method ('SMOTE', 'KMeansSMOTE', 'SMOTETomek', None).
        """

        skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 59)
        X = self.df.drop(columns = self.label)
        y = self.df[self.label]

        if imbalance_method is not None:
            X_resampled, y_resampled = self.imbalance_label(X, y, method=imbalance_method)
        else:
            X_resampled, y_resampled = X, y

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Fold {i+1}:")
            #print(f"    Train: index={train_index}")
            #print(f"    Test:  index={test_index}")
            X_train, y_train = X_resampled.iloc[train_index], y_resampled.iloc[train_index]
            X_test, y_test = X_resampled.iloc[test_index], y_resampled.iloc[test_index]            

            for i, (model_name, model) in enumerate(self.models.items()):
                print(f"Training Model {i+1} {model_name}...")
                self.evaluate_model(model_name, model, X_train, y_train, X_test, y_test)


    def results_viz(self, metrics):

        """
        Visualize the results of cross-validation using violin plots with mean and std in the legend.

        Parameters:
        metrics (list): List of metrics to visualize (e.g., ['accuracy', 'recall', 'precision']).
        """

        num_metrics = len(metrics)
        num_models = len(self.models)

        fig, axes = plt.subplots(num_metrics, num_models, figsize=(6 * num_models, 5 * num_metrics))

        if num_metrics == 1:
            axes = [axes]
        if num_models == 1:
            axes = [[axis] for axis in axes]

        for j, (model_name, model) in enumerate(self.models.items()):
            train_metrics = pd.DataFrame(self.results['train'][model_name])
            test_metrics = pd.DataFrame(self.results['test'][model_name])

            for i, metric in enumerate(metrics):
                train_metrics['Set'] = 'Train'
                test_metrics['Set'] = 'Test'

                combined_metrics = pd.concat([train_metrics[[metric, 'Set']], test_metrics[[metric, 'Set']]])
                sns.violinplot(x='Set', y=metric, data=combined_metrics, ax=axes[i][j], palette={'Train': 'lightblue', 'Test': 'salmon'}, hue='Set', legend=True, inner=None)

                train_mean = train_metrics[metric].mean()
                train_std = train_metrics[metric].std()
                test_mean = test_metrics[metric].mean()
                test_std = test_metrics[metric].std()

                axes[i][j].set_title(f"{model_name} - {metric}")
                axes[i][j].legend([
                    f'Train Mean: {train_mean:.2f}, Std: {train_std:.2f}', 
                    f'Test Mean: {test_mean:.2f}, Std: {test_std:.2f}'
                ])

        plt.tight_layout()
        plt.show()