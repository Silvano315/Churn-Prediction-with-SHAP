import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif, SelectKBest, chi2
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from BorutaShap import BorutaShap
from xgboost import XGBClassifier


# Apply OneHotEncoder to categorical features and scaling to numerical features
def feature_engineering(df, scale_type='standard'):
    
    df = df.copy()
    label_encoders = {}

    if scale_type == 'standard':
        scaler = StandardScaler()
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scale_type must be either 'standard' or 'minmax'")
    
    numerical_features = df.select_dtypes(include=['number']).columns
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col == "Churn":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            continue
        ohe = OneHotEncoder(sparse_output=False)
        encoded = ohe.fit_transform(df[[col]])
        encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cat}" for cat in ohe.categories_[0]], index=df.index)
        df = pd.concat([df, encoded_df], axis=1).drop(columns=[col])
        label_encoders[col] = {class_: int(code) for code, class_ in enumerate(ohe.categories_[0])}
    
    return df, label_encoders


# Feature selection with ANOVA test for numerical features and chi2 for categorical features
def combined_feature_selection(df, target_label, k='all'):
    X = df.drop(columns=[target_label])
    y = df[target_label]

    numerical_features = X.loc[:,['Tenure', 'Monthly Charges', 'Total Charges']].columns
    categorical_features = X.drop(columns=['Tenure', 'Monthly Charges', 'Total Charges']).columns

    anova_selector = SelectKBest(score_func=f_classif, k=k if k == 'all' else min(len(numerical_features), k))
    anova_selector.fit(X[numerical_features], y)
    anova_scores = anova_selector.scores_
    anova_features = numerical_features

    anova_results = pd.DataFrame({'Feature': anova_features, 'Score': anova_scores})
    anova_results = anova_results.sort_values(by='Score', ascending=False)
    print("ANOVA Results:")
    print(anova_results)

    chi2_selector = SelectKBest(score_func=chi2, k=k if k == 'all' else min(len(categorical_features), k))
    chi2_selector.fit(X[categorical_features], y)
    chi2_scores = chi2_selector.scores_
    chi2_features = categorical_features

    chi2_results = pd.DataFrame({'Feature': chi2_features, 'Score': chi2_scores})
    chi2_results = chi2_results.sort_values(by='Score', ascending=False)
    print("\nChi2 Results:")
    print(chi2_results)

    # Plot the results
    plt.figure(figsize=(12, 16))
    plt.subplot(2, 1, 1)
    sns.barplot(x='Score', y='Feature', data=anova_results, palette='viridis')
    plt.title('ANOVA Feature Importance')
    plt.xlabel('ANOVA F-Score')
    plt.ylabel('Feature')
    plt.subplot(2, 1, 2)
    sns.barplot(x='Score', y='Feature', data=chi2_results, palette='viridis')
    plt.title('Chi2 Feature Importance')
    plt.xlabel('Chi2 Score')
    plt.ylabel('Feature')

    plt.tight_layout()
    plt.show()
    #return pd.concat([anova_results, chi2_results])


# Feature selection with BorutaShap package
def boruta_shap_feature_selection(df, label_column):

    X = df.drop(columns=[label_column])
    y = df[label_column]

    model = XGBClassifier()
    Feature_Selector = BorutaShap(model=model,
                                  importance_measure='shap',
                                  classification=True)
    
    Feature_Selector.fit(X=X, y=y, n_trials=100, sample=False,
                         train_or_test='test', normalize=True,
                         verbose=True)
    selected_features = Feature_Selector.Subset().columns
    print("Selected Features:", selected_features)

    Feature_Selector.plot(which_features='all')
    plt.show()