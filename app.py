from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import os
from flights import Flights, Location, Route
from search import find_shortest_path, create_route_map, plot_route, get_default_map
import plotly.graph_objects as go

# App setup
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Airline Route & Fare Explorer"
server = app.server

# Load data
all_flights = Flights("Final.csv")
period_map = {
    "Pre-pandemic (2018-2020)": (2018, 2020),
    "During-pandemic (2021-2022)": (2021, 2022),
    "Post-pandemic (2023-2024)": (2023, 2024)
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
    html.H2("Airline Route & Fare Explorer (2018-2025)", className="text-center my-4"),
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
            html.H4("Flight Route Planner", className="mb-4"),
            dbc.Card([
                dbc.CardBody([
                    # Origin Section
                    dbc.Form([
                        html.H5("Origin Details", className="mb-3"),
                        dbc.Label("City", className="form-label"),
                        dcc.Dropdown(
                            id='origin-city',
                            options=[{'label': loc.city_name, 'value': loc.city_name} 
                                    for loc in sorted(all_flights.cities, key=lambda x: x.city_name)],
                            value="New York",
                            clearable=False,
                            className="mb-2"
                        ),
                        dbc.Label("Airport", className="form-label mt-2"),
                        dcc.Dropdown(
                            id='origin-airport',
                            clearable=False,
                            className="mb-4"
                        )
                    ]),

                    # Destination Section
                    dbc.Form([
                        html.H5("Destination Details", className="mb-3 mt-4"),
                        dbc.Label("City", className="form-label"),
                        dcc.Dropdown(
                            id='dest-city',
                            options=[{'label': loc.city_name, 'value': loc.city_name} 
                                    for loc in sorted(all_flights.cities, key=lambda x: x.city_name)],
                            value="Los Angeles",
                            clearable=False,
                            className="mb-2"
                        ),
                        dbc.Label("Airport", className="form-label mt-2"),
                        dcc.Dropdown(
                            id='dest-airport',
                            clearable=False,
                            className="mb-4"
                        )
                    ]),

                    # Search Parameters
                    dbc.Form([
                        html.H5("Search Parameters", className="mb-3 mt-4"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Time Period", className="form-label"),
                                dcc.Dropdown(
                                    id='period-select',
                                    options=[{'label': k, 'value': k} for k in period_map.keys()],
                                    value="Pre-pandemic (2018-2020)",
                                    clearable=False,
                                    className="mb-3"
                                )
                            ], width=6),
                            
                            dbc.Col([
                                dbc.Label("Max Budget (USD)", className="form-label"),
                                dbc.Input(
                                    id='max-budget',
                                    type='number',
                                    min=0,
                                    step=50,
                                    value=500,
                                    className="mb-3"
                                )
                            ], width=6)
                        ]),

                        # Corrected Checkbox
                        dbc.Row([
                            dbc.Col([
                                dbc.Checkbox(
                                    id='allow-transfers',
                                    value=True,
                                    className="me-2",
                                    label="Allow Transfers",
                                )
                            ], className="d-flex align-items-center mb-3")
                        ]),

                        dbc.Label("Optimization Priority", className="form-label"),
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
                            clearable=False,
                            className="mb-4"
                        ),

                        dbc.Button("Find Optimal Route", 
                                id='find-route-btn', 
                                color="primary",
                                className="w-100 mt-2",
                                size="lg")
                    ])
                ])
            ])
        ], width=12, lg=4, className="pe-lg-3"),

        # Results Column
        dbc.Col([
            dcc.Loading(
                id="loading-results",
                children=[
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='route-map', 
                                    config={'displayModeBar': False},
                                    style={'height': '600px'})
                        ])
                    ], className="mb-3"),
                    html.Div(id='route-summary', className="mt-3")
                ],
                type="circle"
            )
        ], width=12, lg=8, className="ps-lg-3")
    ], className="g-4")


@app.callback(
    Output('origin-airport', 'options'),
    Output('origin-airport', 'value'),
    Input('origin-city', 'value'),
)
def update_origin_airports(selected_city_value):
    if not selected_city_value:
        return [], None
    
    # Find the matching location object
    selected_location = next(
        (loc for loc in all_flights.cities 
         if loc.city_name == selected_city_value),
        None
    )
    
    if not selected_location or not selected_location.airport_codes:
        return [], None
    
    assert isinstance(selected_location, Location), f"{selected_location = }"
    airports = ["Any"] + sorted(selected_location.airport_codes)
    options = [{'label': code, 'value': code} for code in airports]


    return options, airports[0]

@app.callback(
    Output('dest-airport', 'options'),
    Output('dest-airport', 'value'),
    Input('dest-city', 'value'),
)
def update_destination_airports(selected_city_value):
    if not selected_city_value:
        return [], None
    
    # Find the matching location object
    selected_location = next(
        (loc for loc in all_flights.cities 
         if loc.city_name == selected_city_value),
        None
    )
    
    if not selected_location or not selected_location.airport_codes:
        return [], None
    
    assert isinstance(selected_location, Location), f"{selected_location = }"
    airports = ["Any"] + sorted(selected_location.airport_codes)
    options = [{'label': code, 'value': code} for code in airports]

    return options, airports[0]


@app.callback(
    Output('route-map', 'figure'),
    Output('route-summary', 'children'),
    Input('find-route-btn', 'n_clicks'),
    State('origin-city', 'value'),
    State('origin-airport', 'value'),
    State('dest-city', 'value'),
    State('dest-airport', 'value'),
    State('period-select', 'value'),
    State('priority-select', 'value'),
    State('allow-transfers', 'value')
)
def find_route(n, origin_city, origin_airport, dest_city, dest_airport, period, priority, allow_transfers):
    if not n:
        return get_default_map(all_flights), ""

    start_year, end_year = period_map[period]

    def filter_routes(r: Route) -> bool:
        if not (start_year <= r.year <= end_year):
            return False
        
        # filter start and end locations to match airport codes
        if origin_airport in (r.arrival_loc.airport_codes).union(r.depart_loc.airport_codes):
            return origin_airport in [r.arrival_airport, r.depart_airport]
        
        if dest_airport in (r.arrival_loc.airport_codes).union(r.depart_loc.airport_codes):
            return dest_airport in [r.arrival_airport, r.depart_airport]
        
        return True


    origin = next((loc for loc in all_flights.cities \
                   if ((origin_airport in loc.airport_codes) or 
                       (origin_airport == 'Any' and loc.city_name == origin_city))), None)

    dest = next((loc for loc in all_flights.cities \
                   if ((dest_airport in loc.airport_codes) or 
                       (dest_airport == 'Any' and loc.city_name == dest_city))), None)
    
    if not origin or not dest:
        return get_default_map(all_flights), f"Invalid origin or destination."

    if priority == "fare_dist":
        priorities = ["fare", "dist"]
    elif priority == "fare_transfers":
        priorities = ["fare", "transfers"]
    elif priority == "transfers":
        priorities = ["transfers", "fare"]
    else:
        priorities = [priority]

    if not allow_transfers:
        if "transfers" in priorities:
            priorities.pop(priorities.index("transfers"))
        priorities = ["transfers"] + priorities
    
    print(f"Searching for shortest path with: \n{origin} TO {dest}\n{priorities = }")
    dist, final_route = find_shortest_path(all_flights, origin, dest, priorities, valid=filter_routes)
    print(f"Results: {dist}")

    if not allow_transfers and len(final_route) > 1:
        final_route = []

    if not final_route:
        return get_default_map(all_flights), "No route found."

    fig = plot_route(all_flights, final_route)
    
    output_route = Route.get_route_path_string(final_route)
    return fig, html.Div([
        html.H6("Route Summary"),
        html.P(output_route),
        html.P(f"Total Fare: ${dist["fare"]:.2f}"),
        html.P(f"Total Distance: {int(dist["dist"])} miles"),
        html.P(f"Transfers: {int(dist["transfers"]) - 1}")
    ])



# === Tab 2 ===
def fare_trends_layout():
    years = list(range(2018, 2025))
    avg_fares = []
    for year in years:
        year_all_flights = all_flights.filter_by_period(year, year)
        fares = [r.fare for routes in year_all_flights.flight_routes.values() for r in routes]
        avg_fare = sum(fares) / len(fares) if fares else 0
        avg_fares.append({"year": year, "fare": avg_fare})
    df = pd.DataFrame(avg_fares)

    return dbc.Container([
        html.H5("Average Fare per Year"),
        dcc.Graph(figure={
            'data': [{'x': df['year'], 'y': df['fare'], 'type': 'line', 'name': 'Fare'}],
            'layout': {'title': 'Fare Trends (2018-2024)'}
        }),
        html.Hr(),
        html.H5("Network Maps by Period"),
        dcc.Graph(figure=create_route_map(all_flights.filter_by_period(2018, 2020), "Pre-pandemic (2018-2020)")),
        dcc.Graph(figure=create_route_map(all_flights.filter_by_period(2021, 2022), "During-pandemic (2021-2022)")),
        dcc.Graph(figure=create_route_map(all_flights.filter_by_period(2023, 2024), "Post-pandemic (2023-2024)")),
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
