import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Define class EDA with methods to analyse statistics and visualizations from each dataset
class EDA:
    def __init__(self, filepath):
        self.df = pd.read_excel(filepath)
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = self.df.select_dtypes(include=['number']).columns.tolist()

    def basic_info(self):
        print("Shape of the dataset:", self.df.shape)
        print("\nData types and non-null counts:\n", self.df.info())
        print("\nSummary statistics:\n", self.df.describe())
    
    def check_missing_values(self):
        print("\nMissing values:\n", self.df.isnull().sum())
    
    def check_duplicate_values(self):
        print("\nDuplicate rows:", self.df.duplicated().sum())
        if self.df.duplicated().sum() > 0:
            self.df = self.df.drop_duplicates()
            print(f"\nDuplicates removed! {self.df.duplicated().sum()} duplicate rows were removed.")
        else:
            print("\nNothing to remove :)")
    
    def categorical_analysis(self):
        for col in self.categorical_columns:
            print(f"\nValue counts for {col}:\n", self.df[col].value_counts())
            plt.figure(figsize=(8, 4))
            sns.countplot(data=self.df, x=col)
            plt.title(f"Distribution of {col}")
            plt.show()
    
    def numerical_analysis(self):
        print("\nSummary statistics for numerical columns:\n", self.df[self.numerical_columns].describe())
        
        for col in self.numerical_columns:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(self.df[col], kde=True)
            plt.title(f"Histogram of {col}")

            plt.subplot(1, 2, 2)
            sns.boxplot(data=self.df, x=col)
            plt.title(f"Boxplot of {col}")
            plt.show()
    
    def correlation_matrix(self):
        print("\nCorrelation matrix:\n", self.df[self.numerical_columns].corr())
    
    def crosstab_categorical(self):
        for col1 in self.categorical_columns:
            for col2 in self.categorical_columns:
                if col1 != col2:
                    print(f"\nCrosstabulation between {col1} and {col2}:\n", pd.crosstab(self.df[col1], self.df[col2]))
    
    def pairplot_numerical(self):
        sns.pairplot(self.df[self.numerical_columns])
        plt.show()