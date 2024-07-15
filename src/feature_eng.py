import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


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
