import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from BorutaShap import BorutaShap
from xgboost import XGBClassifier


# Apply LabelEncoder to categorical features and scaling to numerical features
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
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = {class_: int(code) for class_, code in zip(le.classes_, le.transform(le.classes_))}
        print(f"Label encoding for {col}: {label_encoders[col]}")
    
    return df, label_encoders


# Feature selection with ANOVA test
def anova_feature_selection(df, target_label, k='all'):

    X = df.drop(columns=[target_label])
    y = df[target_label]

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    scores = selector.scores_
    features = X.columns
    
    anova_results = pd.DataFrame({'Feature': features, 'Score': scores})
    anova_results = anova_results.sort_values(by='Score', ascending=False)
    print(anova_results)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Score', y='Feature', data=anova_results, palette='viridis')
    plt.title('ANOVA Feature Importance')
    plt.xlabel('ANOVA F-Score')
    plt.ylabel('Feature')
    plt.show()
    
    #return anova_results


# ANOVA test in Cross-Validation to find the optimal number of features to use
def anova_cross_validated(df, label_column):

    X = df.drop(columns=[label_column])
    y = df[label_column]
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    feature_counts = range(1, X.shape[1] + 1)
    mean_scores = []

    for k in feature_counts:
        pipeline = Pipeline([
            ('anova', SelectKBest(score_func=f_classif, k=k)),
            ('classifier', classifier)
        ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        mean_scores.append(np.mean(scores))
        print(f'Number of features: {k}, Cross-validated accuracy: {np.mean(scores):.4f}')

    plt.figure(figsize=(12, 6))
    plt.plot(feature_counts, mean_scores, marker='o')
    plt.title('Cross-validated Accuracy vs. Number of Selected Features')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Cross-validated Accuracy')
    plt.grid()
    plt.show()


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