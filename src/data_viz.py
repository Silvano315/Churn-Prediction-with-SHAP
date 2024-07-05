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
        self.customer_id = None

    def basic_info(self):
        print("Shape of the dataset:", self.df.shape)
        print("\nData types and non-null counts:\n", self.df.info())
        print("\nSummary statistics:\n", self.df.describe())

    def extract_customer_id(self):
        id_columns = [col for col in self.df.columns if col.lower() == 'id' or 'id' in col.lower()]
        if id_columns:
            self.customer_id = self.df[id_columns[0]].copy()
            self.df.drop(columns=id_columns[0], inplace=True)
            print(f"\nCustomer ID column '{id_columns[0]}' extracted and removed from the DataFrame.")
        else:
            print("\nNo ID column found.")
    
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
        for col in self.df.select_dtypes(include=['object', 'category']).columns.tolist():
            plt.figure(figsize=(10, 6))

            value_counts = self.df[col].value_counts()
            percentages = (value_counts / len(self.df) * 100).round(2)
            
            palette = sns.color_palette("viridis", len(value_counts))
            sns.countplot(data=self.df, x=col, palette=palette, order=value_counts.index, width= 0.3, hue = col, legend = False)
            
            """
            for index, value in enumerate(value_counts):
                plt.text(index, value + 0.01 * len(self.df), f'{percentages.iloc[index]}%', ha='center')
            """

            handles = [plt.Rectangle((0,0),1,1, color=palette[i]) for i in range(len(value_counts))]
            labels = [f'{category} ({percentages.iloc[i]}%)' for i, category in enumerate(value_counts.index)]
            plt.legend(handles, labels, title=col)
            
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.show()
    
    def numerical_analysis(self):
        for col in self.numerical_columns:
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            
            if self.df[col].var() > 0:
                sns.kdeplot(data=self.df, x=col, ax=ax[0], fill=True, color='lightblue')
                ax[0].set_title(f'Distribution of {col}')
            else:
                ax[0].text(0.5, 0.5, 'Zero Variance', ha='center', va='center', fontsize=12, color='red')
                ax[0].set_title(f'Distribution of {col} (Zero Variance)')
                ax[0].set_xticks([])
                ax[0].set_yticks([])
            
            sns.boxplot(data=self.df, x=col, ax=ax[1], color='lightblue')
            ax[1].set_title(f'Boxplot of {col}')
            
            plt.tight_layout()
            plt.show()

    def correlation_matrix(self):
        print("\nCorrelation matrix:\n", self.df[self.numerical_columns].corr())
    
    def crosstab_categorical(self):
        for col1 in self.df.select_dtypes(include=['object', 'category']).columns.tolist():
            for col2 in self.df.select_dtypes(include=['object','category']).columns.tolist():
                if col1 != col2:
                    print(f"\nCrosstabulation between {col1} and {col2}:\n", pd.crosstab(self.df[col1], self.df[col2]))
    
    def pairplot_numerical(self):
        sns.set_theme(style="whitegrid", palette="muted")
        
        pairplot = sns.pairplot(
            self.df[self.numerical_columns],
            kind='scatter',  
            diag_kind='hist',  
            plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k'},
            diag_kws={'color': 'g'},
            height=3,  
            aspect=1
        )

        pairplot.fig.subplots_adjust(hspace=0.5, wspace=0.5)
        
        plt.show()
