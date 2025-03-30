from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import os
from flights import Flights
from search import find_shortest_path, create_route_map
import plotly.graph_objects as go

# App setup
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Airline Route & Fare Explorer"
server = app.server

# Load data
flights = Flights("Final.csv")
period_map = {
    "Pre-pandemic (2018–2020)": (2018, 2020),
    "During-pandemic (2021–2022)": (2021, 2022),
    "Post-pandemic (2023–2024)": (2023, 2024)
}

df = pd.read_csv("Final.csv")
df = df.dropna(subset=["city1", "airport_1"])
df = df.drop_duplicates(subset=["airport_1"])

airport_options = [
    {
        "label": f"{row['city1'].strip()} ({row['airport_1'].strip()})",
        "value": row["airport_1"].strip()
    }
    for _, row in df.iterrows()
]

# Layout
app.layout = dbc.Container([
    html.H2("Airline Route & Fare Explorer (2018–2025)", className="text-center my-4"),
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='Route Finder', value='tab1'),
        dcc.Tab(label='Fare Trends', value='tab2'),
        dcc.Tab(label='Future Predictions', value='tab3'),
    ]),
    html.Div(id='tab-content')
], fluid=True)

# Tab switch
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab1':
        return route_finder_layout()
    elif tab == 'tab2':
        return fare_trends_layout()
    elif tab == 'tab3':
        return prediction_layout()

# === Tab 1 ===
def route_finder_layout():
    return dbc.Row([
        dbc.Col([
            html.H5("Select Cities and Period"),
            dcc.Dropdown(airport_options, id='origin-airport', placeholder="Origin Airport", value=airport_options[0]["value"]),
            dcc.Dropdown(airport_options, id='dest-airport', placeholder="Destination Airport", value=airport_options[10]["value"]),
            dcc.Dropdown(list(period_map.keys()), id='period-select', value="Pre-pandemic (2018–2020)"),
            dbc.Input(id='max-budget', type='number', placeholder="Max Budget (USD)", min=0, value=500),
            dbc.Checkbox(id='allow-transfers', value=True, label="Allow Transfers"),
            html.Br(),
            dcc.Dropdown(
                id='priority-select',
                options=[
                    {'label': 'Lowest Fare', 'value': 'fare'},
                    {'label': 'Shortest Distance', 'value': 'dist'},
                    {'label': 'Fewest Transfers', 'value': 'transfers'},
                    {'label': 'Fare then Distance', 'value': 'fare_dist'},
                    {'label': 'Fare then Transfers', 'value': 'fare_transfers'},
                ],
                value='fare',
                placeholder='Optimization Priority'
            ),
            html.Br(),
            dbc.Button("Find Route", id='find-route-btn', color="primary")
        ], width=4),
        dbc.Col([
            dcc.Graph(id='route-map'),
            html.Div(id='route-summary')
        ], width=8)
    ])



@app.callback(
    Output('route-map', 'figure'),
    Output('route-summary', 'children'),
    Input('find-route-btn', 'n_clicks'),
    State('origin-airport', 'value'),
    State('dest-airport', 'value'),
    State('period-select', 'value'),
    State('priority-select', 'value')
)
def find_route(n, origin_code, dest_code, period, priority):
    if not n:
        return go.Figure(), ""

    start_year, end_year = period_map[period]
    filtered = flights.filter_by_period(start_year, end_year)

    origin = next((loc for loc in filtered.cities if loc.airport_code == origin_code), None)
    dest = next((loc for loc in filtered.cities if loc.airport_code == dest_code), None)

    if not origin or not dest:
        return go.Figure(), f"Invalid origin or destination."

    if priority == "fare_dist":
        priorities = ["fare", "dist"]
    elif priority == "fare_transfers":
        priorities = ["fare", "transfers"]
    else:
        priorities = [priority]

    result = find_shortest_path(filtered, origin, dest, priorities)

    if not result:
        return go.Figure(), "No route found."

    route = [origin.airport_code] + [r.arrival_loc.airport_code for r in result["path"]]
    coords = {l.airport_code: l.geo_loc for l in filtered.cities}
    lats = [coords[c][0] for c in route if c in coords]
    lons = [coords[c][1] for c in route if c in coords]

    fig = go.Figure(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='lines+markers',
        line=dict(width=2, color='blue'),
        marker=dict(size=6),
        text=route
    ))
    fig.update_layout(title="Optimal Route", geo=dict(scope='usa', projection_type='albers usa'))

    cost = result["cost"]
    return fig, html.Div([
        html.H6("Route Summary"),
        html.P(" → ".join(route)),
        html.P(f"Total Fare: ${cost[0]:.2f}"),
        html.P(f"Total Distance: {int(cost[1])} miles"),
        html.P(f"Transfers: {int(cost[2])}")
    ])



# === Tab 2 ===
def fare_trends_layout():
    years = list(range(2018, 2025))
    avg_fares = []
    for year in years:
        year_flights = flights.filter_by_period(year, year)
        fares = [r.fare for routes in year_flights.flight_routes.values() for r in routes]
        avg_fare = sum(fares) / len(fares) if fares else 0
        avg_fares.append({"year": year, "fare": avg_fare})
    df = pd.DataFrame(avg_fares)

    return dbc.Container([
        html.H5("Average Fare per Year"),
        dcc.Graph(figure={
            'data': [{'x': df['year'], 'y': df['fare'], 'type': 'line', 'name': 'Fare'}],
            'layout': {'title': 'Fare Trends (2018–2024)'}
        }),
        html.Hr(),
        html.H5("Network Maps by Period"),
        dcc.Graph(figure=create_route_map(flights.filter_by_period(2018, 2020), "Pre-pandemic (2018–2020)")),
        dcc.Graph(figure=create_route_map(flights.filter_by_period(2021, 2022), "During-pandemic (2021–2022)")),
        dcc.Graph(figure=create_route_map(flights.filter_by_period(2023, 2024), "Post-pandemic (2023–2024)")),
    ])


# === Tab 3 ===
def prediction_layout():
    return dbc.Row([
        dbc.Col([
            html.H5("Prediction Inputs"),
            dcc.Dropdown(["Q1", "Q2", "Q3", "Q4"], id='quarter-select', value="Q1"),
            dcc.Dropdown(airport_options, id='pred-origin', placeholder="Origin Airport"),
            dcc.Dropdown(airport_options, id='pred-dest', placeholder="Destination Airport"),
            dcc.Dropdown(
                id='pred-priority',
                options=[
                    {'label': 'Lowest Fare', 'value': 'fare'},
                    {'label': 'Shortest Distance', 'value': 'dist'},
                    {'label': 'Fewest Transfers', 'value': 'transfers'},
                    {'label': 'Fare then Distance', 'value': 'fare_dist'},
                    {'label': 'Fare then Transfers', 'value': 'fare_transfers'},
                ],
                value='fare',
                placeholder='Optimization Priority'
            ),
            dbc.Button("Find Predicted Route", id='predict-btn', color="success")
        ], width=4),
        dbc.Col([
            dcc.Graph(id='pred-map'),
            html.Div(id='pred-summary'),
            html.Hr(),
            html.H5("Full Predicted Flight Network"),
            dcc.Graph(id='full-pred-map')
        ], width=8)
    ])



@app.callback(
    Output('pred-map', 'figure'),
    Output('pred-summary', 'children'),
    Output('full-pred-map', 'figure'),
    Input('predict-btn', 'n_clicks'),
    State('quarter-select', 'value'),
    State('pred-origin', 'value'),
    State('pred-dest', 'value'),
    State('pred-priority', 'value')
)
def predict_route(n, quarter, origin_code, dest_code, priority):
    if not n:
        return go.Figure(), "", go.Figure()

    filename = f"prediction_2025{quarter}.csv"
    if not os.path.exists(filename):
        return go.Figure(), f"Prediction file {filename} not found.", go.Figure()

    pred = Flights(filename)

    origin = next((loc for loc in pred.cities if loc.airport_code == origin_code), None)
    dest = next((loc for loc in pred.cities if loc.airport_code == dest_code), None)

    if not origin or not dest:
        return go.Figure(), "Invalid cities.", create_route_map(pred)

    if priority == "fare_dist":
        priorities = ["fare", "dist"]
    elif priority == "fare_transfers":
        priorities = ["fare", "transfers"]
    else:
        priorities = [priority]

    result = find_shortest_path(pred, origin, dest, priorities)

    if not result:
        return go.Figure(), "No predicted route found.", create_route_map(pred)

    route = [origin.airport_code] + [r.arrival_loc.airport_code for r in result["path"]]
    coords = {l.airport_code: l.geo_loc for l in pred.cities}
    lats = [coords[c][0] for c in route if c in coords]
    lons = [coords[c][1] for c in route if c in coords]

    fig = go.Figure(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='lines+markers',
        line=dict(width=2, color='blue'),
        marker=dict(size=6),
        text=route
    ))
    fig.update_layout(title=f"Predicted Route 2025 {quarter}", geo=dict(scope='usa', projection_type='albers usa'))

    cost = result["cost"]
    summary_div = html.Div([
        html.H6("Predicted Route"),
        html.P(" → ".join(route)),
        html.P(f"Predicted Fare: ${cost[0]:.2f}"),
        html.P(f"Distance: {int(cost[1])} miles"),
        html.P(f"Transfers: {int(cost[2])}")
    ])

    return fig, summary_div, create_route_map(pred, title=f"All Predicted Routes for 2025 {quarter}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
