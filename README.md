Exploratory Data Analysis (EDA) on solar radiation data:
Summary Statistics: mean, median, standard deviation, and other statistical measures for each numeric column to understand data distribution.
Data Quality Check: missing values, outliers, or incorrect entries.
Time Series Analysis: line graphs and area plots of GHI, DNI, DHI, and Tamb over time to observe patterns by month, trends throughout day, or anomalies, such as peaks in solar irradiance or temperature fluctuations. 
Impact of cleaning (using the 'Cleaning' column) on the sensor readings (ModA, ModB) over time.
Correlation Analysis: heatmaps or pair plots to visualize the correlations between solar radiation components (GHI, DNI, DHI) and temperature measures (TModA, TModB) and the relationship between wind conditions (WS, WSgust, WD) and solar irradiance using scatter matrices.
Wind Analysis: Polar plots to identify trends and significant wind events by showing the distribution of wind speed and direction, along with how variable the wind direction tends to be.
Temperature Analysis: how relative humidity (RH) might influence temperature readings and solar radiation.
Histograms:  for variables like GHI, DNI, DHI, WS, and temperatures to visualize the frequency distribution of these variables.
Z-Score Analysis: to flag data points that are significantly different from the mean.
Bubble charts to explore complex relationships between variables, such as GHI vs. Tamb vs. WS, with bubble size representing an additional variable like RH or BP (Barometric Pressure).
Data Cleaning