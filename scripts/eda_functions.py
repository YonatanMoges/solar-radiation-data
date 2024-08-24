import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


def load_data(file_path):
    return pd.read_csv(file_path)

def summary_statistics(df):
    return df.describe()

def data_quality_check(df, columns):
    missing_values = df.isnull().sum()
    negative_values = (df[columns] < 0).sum()
    return {"missing_values": missing_values, "negative_values": negative_values}

def time_series_analysis(df, date_column, value_columns):
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    df[value_columns].plot(subplots=True, figsize=(12, 8))
    plt.show()


def cleaning_impact(df, cleaning_col, mod_columns):
    df['cleaning'] = df[cleaning_col].apply(lambda x: 1 if x == 'clean' else 0)
    df.groupby('cleaning')[mod_columns].mean().plot(kind='bar')
    plt.show()

""" def correlation_analysis(df, columns):
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show() """


#Correlation analysis

def plot_correlation_heatmap(df, columns):
    """
    Plots a heatmap of the correlation matrix for specified columns.
    
    Parameters:
    - df: DataFrame containing the data.
    - columns: List of column names to include in the correlation matrix.
    """
    correlation_matrix = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

def plot_pairplot(df, columns):
    """
    Plots a pairplot for the specified columns to visualize pairwise relationships.
    
    Parameters:
    - df: DataFrame containing the data.
    - columns: List of column names to include in the pair plot.
    """
    sns.pairplot(df[columns], diag_kind='kde', plot_kws={'alpha': 0.5})
    plt.suptitle('Pair Plot', y=1.02)
    plt.show()



def plot_scatter_matrix(df, columns):
    """
    Plots a scatter matrix for the specified columns.
    
    Parameters:
    - df: DataFrame containing the data.
    - columns: List of column names to include in the scatter matrix.
    """
    pd.plotting.scatter_matrix(df[columns], alpha=0.5, figsize=(12, 12), diagonal='kde')
    plt.suptitle('Scatter Matrix', y=1.02)
    plt.show()



def wind_analysis(df, ws_col, wd_col):
    plt.figure(figsize=(8, 8))
    plt.quiver(df[wd_col], df[ws_col])
    plt.show()

def temperature_analysis(df, temp_col, rh_col):
    sns.scatterplot(x=df[temp_col], y=df[rh_col])
    plt.show()

def histograms(df, columns):
    df[columns].hist(bins=20, figsize=(14, 10))
    plt.show()

def z_score_analysis(df, columns):
    z_scores = np.abs(zscore(df[columns]))
    return np.where(z_scores > 3)

def bubble_chart(df, x_col, y_col, size_col, color_col):
    plt.scatter(df[x_col], df[y_col], s=df[size_col], c=df[color_col], alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def clean_data(df, fill_value=0):
    return df.fillna(fill_value)


def preprocess_data(df):
    """
    Preprocess the specified columns in the DataFrame.
    
    Parameters:
    - df: DataFrame containing the data to be preprocessed.
    
    Returns:
    - df: The preprocessed DataFrame.
    """
    columns = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'Tamb', 'RH', 
               'WS', 'WSgust', 'WSstdev', 'WD', 'WDstdev', 
               'BP', 'Cleaning', 'Precipitation', 'TModA', 'TModB']
    
    # Replace negative values in irradiance and temperature columns with NaN
    irradiance_columns = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB']
    for col in irradiance_columns:
        df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
    
    # Handle missing values (example: forward fill or mean imputation)
    df.fillna(method='ffill', inplace=True)  # Forward fill
    # Alternatively, you could use df.fillna(df.mean(), inplace=True) for mean imputation

    # Handle outliers using Z-scores (optional)
    z_threshold = 3
    for col in columns:
        df[col] = df[col].where(np.abs((df[col] - df[col].mean()) / df[col].std()) <= z_threshold, np.nan)

    # Correct the data types
    df['Cleaning'] = df['Cleaning'].astype(int)
    df['WD'] = df['WD'] % 360  # Ensure WD is within 0-360 degrees

    # Remove rows with too many missing values (optional)
    df.dropna(thresh=len(columns) - 3, inplace=True)  # Example: allowing up to 3 missing values
    
    # Fill remaining NaN values with mean (after outlier removal)
    df.fillna(df.mean(), inplace=True)
    
    return df

# Example usage


