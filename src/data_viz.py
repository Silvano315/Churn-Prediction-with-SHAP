import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx


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
            self.customer_id = self.df[id_columns].copy()
            self.df.drop(columns=id_columns, inplace=True)
            print(f"\nCustomer ID column '{id_columns}' extracted and removed from the DataFrame.")
        else:
            print("\nNo ID column found.")
    
    def check_missing_values(self):
        print("\nMissing values:\n", self.df.isnull().sum())

    def plot_missing_comparison(self, missing_col, compare_col):
        missing_mask = self.df[missing_col].isna()
        
        missing_data = self.df[missing_mask]
        
        if self.df[compare_col].dtype == 'object':
            plt.figure(figsize=(6, 4))
            sns.countplot(x=compare_col, data=missing_data, palette='viridis', edgecolor='black', lw=0.5, width=0.2, hue=compare_col, legend=True)
            if self.df[compare_col].nunique() > 5:
                plt.xticks(rotation=45)
            plt.show()
        
        else: 
            plt.figure(figsize=(6, 4))
            sns.histplot(data=missing_data, x=compare_col, bins=50, kde=True, color='skyblue', edgecolor='black', lw=0.5)
            plt.title(f"Distribution of '{compare_col}' where '{missing_col}' is Missing")
            plt.xlabel(compare_col)
            plt.ylabel("Frequency")
            plt.show()
    
    def check_duplicate_values(self):
        print("\nDuplicate rows:", self.df.duplicated().sum())
        if self.df.duplicated().sum() > 0:
            self.df = self.df.drop_duplicates()
            print(f"\nDuplicates removed! {self.df.duplicated().sum()} duplicate rows were removed.")
        else:
            print("\nNothing to remove :)")

    def remove_column(self, column_name):
        if column_name in self.df.columns:
            self.df.drop(columns=column_name, inplace=True)
            print(f"\nColumn '{column_name}' removed from the DataFrame.")
        else:
            print(f"\nColumn '{column_name}' not found in the DataFrame.")
    
    def categorical_analysis(self, max_categories=None):
        for col in self.df.select_dtypes(include=['object', 'category']).columns.tolist():
            plt.figure(figsize=(10, 6))

            value_counts = self.df[col].value_counts()
            if max_categories:
                value_counts = value_counts.head(max_categories)
            
            percentages = (value_counts / len(self.df) * 100).round(2)
            
            palette = sns.color_palette("tab20", n_colors=len(value_counts))
            sns.countplot(data=self.df, x=col, palette=palette, order=value_counts.index, width=0.3)

            handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(len(value_counts))]
            labels = [f'{category} ({percentages.loc[category]}%)' for category in value_counts.index]
            plt.legend(handles, labels, title=col, loc='upper right')

            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel('Count')
            if self.df[col].nunique() > 5:
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    
    def numerical_analysis(self):
        for col in self.df.select_dtypes(include=['number']).columns.tolist():
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

    def correlation_matrix_heatmap(self):
        corr_matrix = self.df[self.df.select_dtypes(include=['number']).columns.tolist()].corr()
                
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        plt.show()
    
    def crosstab_categorical(self):
        for col1 in self.df.select_dtypes(include=['object', 'category']).columns.tolist():
            for col2 in self.df.select_dtypes(include=['object','category']).columns.tolist():
                if col1 != col2:
                    print(f"\nCrosstabulation between {col1} and {col2}:\n", pd.crosstab(self.df[col1], self.df[col2]))

    def crosstab_categorical_plot(self):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = len(categorical_cols)
        
        _, axes = plt.subplots(num_cols, num_cols, figsize=(15, 15))
        
        for i, col1 in enumerate(categorical_cols):
            for j, col2 in enumerate(categorical_cols):
                if i != j:
                    crosstab = pd.crosstab(self.df[col1], self.df[col2])
                    
                    sns.heatmap(crosstab, annot=True, fmt="d", cmap="viridis", cbar=False, ax=axes[i, j])
                    axes[i, j].set_xlabel(col2)
                    axes[i, j].set_ylabel(col1)
        
        plt.tight_layout()
        plt.show()

    
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

    def plot_geopoints(self, shapefile_path):
        gdf = gpd.GeoDataFrame(
            self.df, 
            geometry=gpd.points_from_xy(self.df.Longitude, self.df.Latitude),
            crs="EPSG:4326"
        )
        states_gdf = gpd.read_file(shapefile_path)
        states_to_show = ['California', 'Nevada', 'Oregon']
        usa_nearby = states_gdf[states_gdf['name'].isin(states_to_show)]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-14000000, -12500000)  
        ax.set_ylim(3750000, 5300000)
        gdf.to_crs(epsg=3857).plot(ax=ax, markersize=10, color='red', alpha=0.6)
        usa_nearby.to_crs(epsg=3857).plot(ax=ax, edgecolor='black', facecolor='none')       
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        plt.title('Geographical Distribution of Points in California and Nearby States')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def plot_feature_by_churn(self, feature):
        if feature in self.df.select_dtypes(include=['object','category']).columns.tolist():

            plt.figure(figsize=(10, 6))
            ax = sns.countplot(data=self.df, x=feature, hue='Churn', palette='viridis', width=0.4)

            total_counts = self.df.groupby([feature, 'Churn']).size().reset_index(name='count')
            churn_totals = total_counts.groupby('Churn')['count'].transform('sum')
            total_counts['percent'] = total_counts['count'] / churn_totals * 100

            handles, labels = ax.get_legend_handles_labels()
            churn_categories = labels  

            for i, feature_val in enumerate(self.df[feature].unique()):
                for j, churn_category in enumerate(churn_categories):
                    subset = total_counts[(total_counts[feature] == feature_val) & (total_counts['Churn'] == churn_category)]
                    
                    if not subset.empty:
                        percentage = subset['percent'].values[0]
                        
                        x = i + j * 0.2 -0.1 
                        
                        ax.annotate(f'{percentage:.1f}%', 
                                    (x, subset['count'].values[0]),  
                                    ha='center', 
                                    va='bottom',  
                                    xytext=(0, 2), 
                                    textcoords='offset points')

            plt.title(f'Distribution of {feature} by Churn')
            plt.xlabel(feature)
            plt.ylabel('Count')

            if self.df[feature].nunique() > 5:
                plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()

        elif feature in self.df.select_dtypes(include=['number']).columns.tolist():
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df, x='Churn', y=feature, palette='viridis', width=0.2, hue='Churn',legend=True)
            
            means = self.df.groupby('Churn')[feature].mean()
            stds = self.df.groupby('Churn')[feature].std()
            
            legend_labels = [f'{label} (Mean: {mean:.2f}, Std: {std:.2f})' for label, mean, std in zip(means.index, means, stds)]
            plt.legend(labels=legend_labels)
            
            plt.title(f'Boxplot of {feature} by Churn')
            plt.xlabel('Churn')
            plt.ylabel(feature)
            plt.tight_layout()
            plt.show()
        else:
            print(f"{feature} is not found in the DataFrame.")