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

def correlation_analysis(df, columns):
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
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
