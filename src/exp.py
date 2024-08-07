import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import shap


# Function to compute the Shapley Values in a Stratified Cross-Validation
def compute_shap_values(df, target_column, algorithm='xgboost', smote=True, n_splits=10):

    X = df.drop(columns=target_column)
    y = df[target_column]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=59)
    shap_values_list = []
    expected_values_list = []
    indices_list = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print("-"*20)
        print(f"Fold {i+1}:")

        if smote:
            smote = SMOTE(random_state=59)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        if algorithm.lower() == 'xgboost':
            print("Training XGBoost...")
            model = XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000, random_state=59)
            model.fit(X_train, y_train)
            print("-"*20)
        elif algorithm.lower() == 'catboost':
            print("Training CatBoost...")
            categorical_features_indices = np.where(X_train.dtypes != np.float64)[0]
            model = CatBoostClassifier(random_state=59, verbose=False)
            model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_test, y_test))
            print("-"*20)
        else:
            raise ValueError("Invalid algorithm specified. Use 'xgboost' or 'catboost'.")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        expected_values = explainer.expected_value

        shap_values_list.append(shap_values)
        expected_values_list.append(np.repeat(expected_values, len(test_index)))
        indices_list.append(test_index)

    all_shap_values = np.concatenate(shap_values_list, axis=0)
    all_expected_values = np.concatenate(expected_values_list, axis=0)
    all_indices = np.concatenate(indices_list, axis=0)

    shap_df = pd.DataFrame(all_shap_values, index=all_indices, columns=X.columns)
    expected_values_df = pd.DataFrame(all_expected_values, index=all_indices, columns=['Expected_Value'])

    shap_df_sorted = shap_df.sort_index()
    expected_values_df_sorted = expected_values_df.sort_index()

    return shap_df_sorted, expected_values_df_sorted
