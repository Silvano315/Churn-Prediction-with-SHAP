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
import json


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
            'RandomForest' : RandomForestClassifier(max_depth = 3, random_state=59),
            'XGBoost' : XGBClassifier(learning_rate= 0.01, max_depth = 3, n_estimators = 1000),
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
            over_sampler = KMeansSMOTE(k_neighbors=5)
        else:
            raise ValueError("Invalid method for handling imbalance. Choose 'SMOTE', 'ADASYN', or 'KMeansSMOTE'.")
        
        X_resampled, y_resampled = over_sampler.fit_resample(X,y)
        
        return X_resampled, y_resampled


    def evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test, imbalance_method = None):
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
        if model_name == 'CatBoost':
            categorical_features_indices = np.where(X_train.dtypes != np.float64)[0]
            model.fit(X_train, y_train, cat_features =categorical_features_indices, eval_set=(X_test,y_test))
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        pos_label = "Yes" if model_name=='CatBoost' else 1

        if imbalance_method == None:
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'recall': recall_score(y_train, y_train_pred, pos_label = pos_label),
                'precision': precision_score(y_train, y_train_pred, pos_label = pos_label),
                'f1_score': f1_score(y_train, y_train_pred, pos_label = pos_label),
                'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
                'roc_auc' : roc_auc_score(y_train, y_train_proba),
                'y_true': y_train,
                'y_score': y_train_proba
            }
            
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred, pos_label = pos_label),
                'precision': precision_score(y_test, y_test_pred, pos_label = pos_label),
                'f1_score': f1_score(y_test, y_test_pred, pos_label = pos_label),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
                'roc_auc': roc_auc_score(y_test, y_test_proba),
                'y_true': y_test,
                'y_score': y_test_proba
            }
        else:
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'recall': recall_score(y_train, y_train_pred, pos_label = pos_label),
                'precision': precision_score(y_train, y_train_pred, pos_label = pos_label),
                'f1_score': f1_score(y_train, y_train_pred, pos_label = pos_label),
                'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred)
            }
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred, pos_label = pos_label),
                'precision': precision_score(y_test, y_test_pred, pos_label = pos_label),
                'f1_score': f1_score(y_test, y_test_pred, pos_label = pos_label),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred)
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

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Fold {i+1}:")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if imbalance_method is not None:
                X_train, y_train = self.imbalance_label(X_train, y_train, method=imbalance_method)

            for i, (model_name, model) in enumerate(self.models.items()):
                
                if imbalance_method is not None:
                    print(f"Training Model {i+1} {model_name} with {imbalance_method}...")
                    self.evaluate_model(model_name, model, X_train, y_train, X_test, y_test, imbalance_method='SMOTE')
                else:
                    print(f"Training Model {i+1} {model_name}...")
                    self.evaluate_model(model_name, model, X_train, y_train, X_test, y_test)


    def results_viz(self, metrics, imbalance_method = None, path_results = None):

        """
        Visualize the results of cross-validation using violin plots with mean and std in the legend.

        Parameters:
        metrics (list): List of metrics to visualize (e.g., ['accuracy', 'recall', 'precision']).
        """

        if imbalance_method is not None:
            with open(path_results, 'r') as f:
                self.results = json.load(f)

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


    def plot_roc_curve(self, k):

        """
        Plot the ROC curve for each model with train and test data on the same plot.
        
        Parameters:
        k (int): Number of folds for cross-validation (used to average ROC curves across folds).
        """
        
        num_models = len(self.models)
        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))

        if num_models == 1:
            axes = [axes]

        for j, (model_name, model) in enumerate(self.models.items()):
            tprs_train = []
            aucs_train = []
            tprs_test = []
            aucs_test = []
            mean_fpr = np.linspace(0, 1, 100)
            
            for i in range(k):
                train_metrics = self.results['train'][model_name][i]
                fpr_train, tpr_train, _ = roc_curve(train_metrics['y_true'], train_metrics['y_score'])
                tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
                tprs_train[-1][0] = 0.0
                aucs_train.append(roc_auc_score(train_metrics['y_true'], train_metrics['y_score']))
                
                test_metrics = self.results['test'][model_name][i]
                fpr_test, tpr_test, _ = roc_curve(test_metrics['y_true'], test_metrics['y_score'])
                tprs_test.append(np.interp(mean_fpr, fpr_test, tpr_test))
                tprs_test[-1][0] = 0.0
                aucs_test.append(roc_auc_score(test_metrics['y_true'], test_metrics['y_score']))

            mean_tpr_train = np.mean(tprs_train, axis=0)
            mean_tpr_train[-1] = 1.0
            mean_auc_train = auc(mean_fpr, mean_tpr_train)
            std_auc_train = np.std(aucs_train)
            axes[j].plot(mean_fpr, mean_tpr_train, color='lightblue', lw=2, label=f'Train (AUC = {mean_auc_train:.2f} ± {std_auc_train:.2f})')

            mean_tpr_test = np.mean(tprs_test, axis=0)
            mean_tpr_test[-1] = 1.0
            mean_auc_test = auc(mean_fpr, mean_tpr_test)
            std_auc_test = np.std(aucs_test)
            axes[j].plot(mean_fpr, mean_tpr_test, color='salmon', lw=2, label=f'Test (AUC = {mean_auc_test:.2f} ± {std_auc_test:.2f})')

            axes[j].plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8)
            axes[j].set_title(f'{model_name} ROC Curve')
            axes[j].legend(loc='lower right')
            axes[j].set_xlim([0.0, 1.0])
            axes[j].set_ylim([0.0, 1.05])
            axes[j].set_xlabel('False Positive Rate')
            axes[j].set_ylabel('True Positive Rate')

        plt.tight_layout()
        plt.show()