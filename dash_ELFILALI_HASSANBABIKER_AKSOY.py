# Importation des bibliothèques nécessaires
import os
from threading import Timer
import webbrowser
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from dash import Dash, html, dcc, dependencies
import plotly.graph_objects as go

# Charger les fichiers de données
data_file = 'data_CO2_GAP.csv'
df = pd.read_csv(data_file)

data_origin = 'Carbon_(CO2)_Emissions_by_Country.csv'
df_origin = pd.read_csv(data_origin)

coordinates_file = 'country_coordinates.csv'
df_coords = pd.read_csv(coordinates_file)

# Correction des noms des pays pour correspondre aux coordonnées
country_corrections = {
    'Antigua And Barbuda': 'Antigua and Barbuda',
    'Bosnia': 'Bosnia and Herzegovina',
    'Democratic Republic Of Congo': 'Congo (Kinshasa)',
    # Ajouter les autres corrections ici
}
df_origin['Country'] = df_origin['Country'].replace(country_corrections)

# Préparation des données
df_origin['Date'] = pd.to_datetime(df_origin['Date']).dt.year
df_total_kCo2_Years = df_origin.groupby(['Date', 'Region'])['Kilotons of Co2'].sum().reset_index()
df_total_metric_tons_Years = df_origin.groupby(['Date', 'Region'])['Metric Tons Per Capita'].mean().reset_index()

# Fusionner avec les coordonnées
df_coords_cleaned = df_coords.dropna(subset=['latitude', 'longitude'])
df_map = df_origin.merge(df_coords_cleaned, on='Country', how='left')

# App Dash
app = Dash(__name__)

# App Dash avec design amélioré
app.layout = html.Div([
    # Titre principal
    html.H1("Analyse des Émissions de CO₂", style={'textAlign': 'center', 'padding': '20px', 'color': '#333'}),

    # Section 1 : Cartes
    html.Div([
        html.H2("Cartographie des Émissions de CO₂", style={'textAlign': 'center', 'color': '#555'}),
        
        html.Div([
            dcc.Dropdown(
                id='date-dropdown',
                options=[{'label': str(year), 'value': year} for year in sorted(df_origin['Date'].unique())],
                value=df_origin['Date'].min(),
                placeholder='Sélectionnez une année',
                style={'width': '90%', 'margin': 'auto'}
            ),
        ], style={'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='map-plot', style={'flex': '1', 'margin-right': '10px'}),
            dcc.Graph(id='globe-plot', style={'flex': '1'})
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'padding': '20px'}),
    ], style={'margin': '20px', 'border': '1px solid #ddd', 'border-radius': '10px', 'padding': '10px'}),

    # Section 2 : PCA et Cluster
    html.Div([
        html.H2("Visualisation et Clustering", style={'textAlign': 'center', 'color': '#555'}),
        
        html.Div([
            dcc.Dropdown(
                id='year-selector',
                options=[{'label': year, 'value': year} for year in sorted(df['year'].unique())],
                value=df['year'].min(),
                multi=False,
                placeholder="Sélectionnez une année",
                style={'width': '90%', 'margin': 'auto'}
            ),
        ], style={'padding': '10px'}),

        html.Div([
            dcc.Graph(id='pca-cluster', style={'flex': '1', 'margin-right': '10px'}),
            dcc.Graph(id='co2-bar', style={'flex': '1'})
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'padding': '20px'}),
    ], style={'margin': '20px', 'border': '1px solid #ddd', 'border-radius': '10px', 'padding': '10px'}),

    # Section 3 : Évolution des émissions dans le temps
    html.Div([
        html.H2("Évolution des Émissions de CO₂ dans le Temps", style={'textAlign': 'center', 'color': '#555'}),
        
        html.Div([
            dcc.Graph(
                id='emissions-over-time',
                figure=px.line(
                    df_total_kCo2_Years.sort_values(by='Date'),
                    x='Date',
                    y='Kilotons of Co2',
                    color='Region',
                    title="Émissions de CO₂ par continent au fil du temps"
                ),
                style={'flex': '1', 'margin-right': '10px'}
            ),
            dcc.Graph(
                id='metric-over-time',
                figure=px.line(
                    df_total_metric_tons_Years.sort_values(by='Date'),
                    x='Date',
                    y='Metric Tons Per Capita',
                    color='Region',
                    title="Émissions de CO₂ par habitant au fil du temps"
                ),
                style={'flex': '1'}
            ),
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'padding': '20px'}),
    ], style={'margin': '20px', 'border': '1px solid #ddd', 'border-radius': '10px', 'padding': '10px'}),
])

@app.callback(
    [dependencies.Output('pca-cluster', 'figure'),
     dependencies.Output('co2-bar', 'figure'),
     dependencies.Output('map-plot', 'figure'),
     dependencies.Output('globe-plot', 'figure')],
    [dependencies.Input('year-selector', 'value'),
     dependencies.Input('date-dropdown', 'value')]
)
def update_graphs(selected_year, selected_date):
    # PCA et clustering
    filtered_df = df[df['year'] == selected_year]
    features = ['population', 'life_exp', 'gdp_cap', 'Kilotons of Co2', 'Metric Tons Per Capita']
    X = filtered_df[features].dropna()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    filtered_df['PCA1'] = pca_result[:, 0]
    filtered_df['PCA2'] = pca_result[:, 1]
    kmeans = KMeans(n_clusters=3, random_state=42)
    filtered_df['Cluster'] = kmeans.fit_predict(X)

    # Graphique PCA
    pca_fig = px.scatter(
        filtered_df, x='PCA1', y='PCA2', color='Cluster', hover_name='country',
        title=f"PCA et Clustering pour l'année {selected_year}"
    )

    # Bar Chart
    top_emitters = filtered_df.nlargest(10, 'Kilotons of Co2')
    co2_bar_fig = px.bar(
        top_emitters, x='country', y='Kilotons of Co2', color='country',
        title=f"Top 10 des Émetteurs de CO₂ en {selected_year}"
    )

    # Carte et Globe
    filtered_df_map = df_map[df_map['Date'] == selected_date]
  # Map Plot
    map_fig = go.Figure(go.Scattermapbox(
        lat=filtered_df_map['latitude'], 
        lon=filtered_df_map['longitude'],
        mode='markers',
        marker=dict(size=8, color=filtered_df_map['Kilotons of Co2'], showscale=True),
        text=filtered_df_map.apply(
            lambda row: f"{row['Country']}: {row['Kilotons of Co2']} kt CO₂", axis=1
        ),  # Texte enrichi avec les émissions
        hoverinfo='text'
    ))
    map_fig.update_layout(mapbox=dict(style="carto-positron", zoom=1), showlegend=False)

    # Globe Plot
    globe_fig = go.Figure(go.Scattergeo(
        lat=filtered_df_map['latitude'], 
        lon=filtered_df_map['longitude'],
        mode='markers',
        marker=dict(size=8, color=filtered_df_map['Kilotons of Co2'], showscale=True),
        text=filtered_df_map.apply(
            lambda row: f"{row['Country']}: {row['Kilotons of Co2']} kt CO₂", axis=1
        ),  # Texte enrichi avec les émissions
        hoverinfo='text'
    ))
    globe_fig.update_geos(projection_type="orthographic")

    
    return pca_fig, co2_bar_fig, map_fig, globe_fig

def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:1222/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=1222)
