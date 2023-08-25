import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv(r'C:\Users\abbas\Desktop\Marketing Insights Data visualization\2022q4_marketing_dataset2.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("MARKETING INSIGHT: DATA VISUALIZER AND ANALYZER"),
    
    # Dropdown for selecting columns
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in data.columns],
        value='revenue',  # Default column to show
        multi=False
    ),
    
    # Line chart to visualize the selected column over time
    dcc.Graph(id='line-chart'),
    
    # Scatter plot to visualize relationships between columns
    dcc.Graph(id='scatter-plot'),
    
    # Bar chart to show total revenue by marketer_id (visible only for revenue option)
    dcc.Graph(id='revenue-by-marketer', style={'display': 'none'}),
    
    # Clustering results
    dcc.Graph(id='cluster-plot'),
    
    # Regression results
    dcc.Graph(id='regression-plot'),
])

# Callback for updating visualizations and dropdown options
@app.callback(
    [Output('line-chart', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('revenue-by-marketer', 'figure'),
     Output('revenue-by-marketer', 'style'),
     Output('cluster-plot', 'figure'),
     Output('regression-plot', 'figure')],
    [Input('column-dropdown', 'value')]
)
def update_visualizations(selected_column):
    if selected_column == 'marketer_relationship':
        # Relationship between marketer_id and total revenue
        marketer_revenue = data.groupby('marketer_id')['revenue'].sum().reset_index()
        marketer_relationship = px.scatter(
            marketer_revenue, x='marketer_id', y='revenue',
            title="Marketer Relationship: Total Revenue by Marketer"
        )
        return [None] * 5 + [marketer_relationship]
    
    # Line chart
    line_chart = px.line(data, x='action_date', y=selected_column, title=f"{selected_column} over Time")
    line_chart.update_xaxes(tickvals=data['action_date'][::1000], ticktext=data['action_date'][::1000], tickangle=45)
    
    # Scatter plot
    scatter_plot = px.scatter(data, x=selected_column, y='revenue', title=f"Scatter Plot: {selected_column} vs Revenue")
    
    # Bar chart - Total revenue by marketer_id
    if selected_column == 'revenue':
        marketer_revenue = data.groupby('marketer_id')['revenue'].sum().reset_index()
        revenue_by_marketer = px.bar(marketer_revenue, x='marketer_id', y='revenue', title="Total Revenue by Marketer")
        revenue_by_marketer_style = {'display': 'block'}
    else:
        revenue_by_marketer = None
        revenue_by_marketer_style = {'display': 'none'}
    
    # Clustering
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters)
    data['cluster'] = kmeans.fit_predict(data[[selected_column, 'revenue']])
    cluster_plot = px.scatter(data, x=selected_column, y='revenue', color='cluster', title="Cluster Analysis")
    
    # Regression
    X = data[[selected_column]]
    y = data['revenue']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    regression_plot = px.scatter(x=X[selected_column], y=y, title="Regression Analysis")
    regression_plot.add_trace(px.line(x=X[selected_column], y=y_pred, line_shape='linear').data[0])
    
    return line_chart, scatter_plot, revenue_by_marketer, revenue_by_marketer_style, cluster_plot, regression_plot

if __name__ == '__main__':
    app.run_server(debug=True)
