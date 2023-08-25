
# Rocket Digital Assessment Dashboard

This repository contains a Python script for building an interactive
dashboard using Dash and Plotly libraries to visualize insights from the "Rocket Digital 2022 
Q4 Marketing Dataset". The dashboard provides various visualization options and analytical tools 
to explore relationships and patterns within the dataset.

# Table of Contents
1) Prerequisites
2) Installation
3) Usage
4) Features
5) Screenshots
6) License


# Prerequisites
Before running the code, ensure you have the following prerequisites installed:

1) Python 3.6+
2) Pandas
3) Dash
4) Plotly
5) Scikit-learn

#### You can install these dependencies using the following command:

pip install pandas dash plotly scikit-learn or python requirements.txt


### Run the script using the following command:
python main.py


### The dashboard will be accessible through your web browser at http://127.0.0.1:8050/.

# Features:
The dashboard provides the following features:

1) Selection of columns from the dataset for visualization using a dropdown menu.
2) Line chart to visualize the selected column over time.
3) Scatter plot to visualize relationships between the selected column and revenue.
4) Bar chart showing total revenue by marketer (visible only for revenue column).
5) Cluster analysis using KMeans algorithm to group data points based on the selected column and revenue.
6) Regression analysis to find a linear relationship between the selected column and revenue.






