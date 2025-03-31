"""
Dash application for visualizing and predicting airline routes and fares (2018–2025).
Provides interactive tools to explore historical trends and forecast future airfare behavior.
"""

import os
from typing import Dict, Tuple, List, Optional, Union, Callable

import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State
from dash.development.base_component import Component

from flights import Flights, Location, Route
from search import find_shortest_path, plot_route, get_default_map, plot_all_routes


# App setup
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Airline Route & Fare Explorer"
server = app.server

# Load data
all_flights = Flights("USA_Filtered_Airline_2018-2024_FILLED.csv")
period_map = {
    "Pre-pandemic (2018-2020)": (2018, 2020),
    "During-pandemic (2021-2022)": (2021, 2022),
    "Post-pandemic (2023-2024)": (2023, 2024)
}

df = pd.read_csv("USA_Filtered_Airline_2018-2024_FILLED.csv")
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
def render_tab(tab: str) -> Component:
    """Render the layout for the selected tab."""
    if tab == 'tab1':
        return route_finder_layout()
    elif tab == 'tab2':
        return fare_trends_layout()
    elif tab == 'tab3':
        return prediction_layout()
    return html.Div("Invalid tab selected.")


# === Tab 1 ===
def route_finder_layout() -> Component:
    """Create the layout for the route finder tab, including origin, destination, filters, and result display."""
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
                            options=[
                                {'label': loc.city_name, 'value': loc.city_name}
                                for loc in sorted(all_flights.cities, key=lambda x: x.city_name)
                            ],
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
                            options=[
                                {'label': loc.city_name, 'value': loc.city_name}
                                for loc in sorted(all_flights.cities, key=lambda x: x.city_name)
                            ],
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
                                    options=[
                                        {'label': k, 'value': k}
                                        for k in period_map
                                    ],
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

                        # Checkbox
                        dbc.Row([
                            dbc.Col([
                                dbc.Checkbox(
                                    id='allow-transfers',
                                    value=True,
                                    className="me-2",
                                    label="Allow Transfers"
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
                                {'label': 'Fare then Transfers', 'value': 'fare_transfers'}
                            ],
                            value='fare',
                            clearable=False,
                            className="mb-4"
                        ),

                        dbc.Button(
                            "Find Optimal Route",
                            id='find-route-btn',
                            color="primary",
                            className="w-100 mt-2",
                            size="lg"
                        )
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
                            dcc.Graph(
                                id='route-map',
                                config={'displayModeBar': False},
                                style={'height': '600px'}
                            )
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
def update_origin_airports(selected_city_value: Optional[str]) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Update airport dropdown options based on selected origin city."""
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
    assert isinstance(selected_location, Location), f"{selected_location =}"
    airports = ["Any"] + sorted(selected_location.airport_codes)
    options = [{'label': code, 'value': code} for code in airports]
    return options, airports[0]


@app.callback(
    Output('dest-airport', 'options'),
    Output('dest-airport', 'value'),
    Input('dest-city', 'value'),
)
def update_destination_airports(selected_city_value: Optional[str]) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Update airport dropdown options based on selected destination city."""
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
    assert isinstance(selected_location, Location), f"{selected_location =}"
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
def find_route(n: Optional[str], origin_city: str, origin_airport: str, dest_city: str,
               dest_airport: str,
               period: str,
               priority: str,
               allow_transfers: bool) -> tuple:
    """
    Callback function to compute and return the optimal route map and route summary
    based on user input such as cities, airports, time period, and optimization priority.
    """
    if not n:
        return get_default_map(all_flights), ""

    start_year, end_year = period_map[period]

    def filter_routes(r: Route) -> bool:
        if not start_year <= r.year <= end_year:
            return False
        # filter start and end locations to match airport codes
        if origin_airport in r.arrival_loc.airport_codes.union(r.depart_loc.airport_codes):
            return origin_airport in [r.arrival_airport, r.depart_airport]
        if dest_airport in r.arrival_loc.airport_codes.union(r.depart_loc.airport_codes):
            return dest_airport in [r.arrival_airport, r.depart_airport]
        return True

    origin = next((
        loc for loc in all_flights.cities
        if (origin_airport in loc.airport_codes
            or (origin_airport == 'Any' and loc.city_name == origin_city))
    ), None)

    dest = next((
        loc for loc in all_flights.cities
        if (dest_airport in loc.airport_codes
            or (dest_airport == 'Any' and loc.city_name == dest_city))
    ), None)

    if not origin or not dest:
        return get_default_map(all_flights), "Invalid origin or destination."

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
    print(f"Searching for shortest path with: \n{origin} TO {dest}\n{priorities=}")
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
def fare_trends_layout() -> Component:
    """Generate the layout for visualizing average fare trends over time with historical context."""
    years = list(range(2018, 2025))
    avg_fares = []
    for year in years:
        for quarter in [1, 2, 3, 4]:
            if year == 2024 and quarter > 1:
                continue
            filtered_flights = [
                r for routes in all_flights.flight_routes.values()
                for r in routes
                if r.year == year and r.quarter == quarter
            ]
            fares = [r.fare for r in filtered_flights]
            avg_fare = sum(fares) / len(fares) if fares else 0
            avg_fares.append({"year": year + (quarter / 4), "fare": avg_fare})

    df = pd.DataFrame(avg_fares)

    return dbc.Container([
        html.H5("Average Fare per Year"),
        dcc.Graph(
            figure={
                'data': [{
                    'x': df['year'],
                    'y': df['fare'],
                    'type': 'line',
                    'name': 'Fare'
                }],
                'layout': {
                    'title': 'Fare Trends (2018-2024)',
                    'shapes': [
                        {
                            'type': 'line', 'x0': 2020.25, 'x1': 2020.25,
                            'y0': 0, 'y1': 1, 'yref': 'paper',
                            'line': {'color': 'red', 'width': 2, 'dash': 'dash'}
                        },
                        {
                            'type': 'line', 'x0': 2022.5, 'x1': 2022.5,
                            'y0': 0, 'y1': 1, 'yref': 'paper',
                            'line': {'color': 'red', 'width': 2, 'dash': 'dash'}
                        }
                    ],
                    'annotations': [
                        {
                            'x': 2020.25, 'y': 1,
                            'yanchor': 'bottom',
                            'text': 'COVID Start',
                            'showarrow': False,
                            'font': {'color': 'red'}
                        },
                        {
                            'x': 2022.5, 'y': 1,
                            'yanchor': 'bottom',
                            'text': 'COVID End',
                            'showarrow': False,
                            'font': {'color': 'red'}
                        }
                    ]
                }
            }
        ),
        html.Hr(),
        html.Div([
            html.H5("Network Map Timeline"),
            dcc.Graph(
                id='network-map-timeline',
                figure=plot_all_routes(
                    all_flights,
                    title="Pre-pandemic: 2018 (Q1)",
                    valid=lambda x: (x.year == 2018 and x.quarter == 1)
                ),
                style={
                    'width': '100%',
                    'height': '80vh',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'margin': '0 auto'
                },
                config={
                    'scrollZoom': True,
                    'displayModeBar': False
                }
            ),
            dcc.Slider(
                id='year-slider',
                min=2018.25,
                max=2024.25,
                step=0.25,
                value=2018.25,
                marks={
                    y: {'label': f"{int(y)} Q1", 'style': {'transform': 'rotate(45deg)'}}
                    for y in [2018.25, 2019.25, 2020.25, 2021.25, 2022.25, 2023.25, 2024.25]
                },
                tooltip={
                    "placement": "bottom",
                    "always_visible": False,
                    "template": "{value}"
                },
                included=False,
                className="mt-3 mb-5"
            )
        ], style={
            'maxWidth': '1200px',
            'margin': '0 auto',
            'padding': '20px'
        })
    ])


@app.callback(
    Output('network-map-timeline', 'figure'),
    Input('year-slider', 'value')
)
def update_network_map(selected_year: float) -> go.Figure:
    """Update the interactive route map for the selected year/quarter from the timeline slider."""
    selected_quarter = round((selected_year - int(selected_year - 0.25)) * 4)

    def filter_by_year(year: int, quarter: int) -> Callable[[Route], bool]:
        return lambda x: (x.year == year and x.quarter == quarter)
    if selected_year - 0.25 <= 2020:
        title = "Pre-pandemic"
    elif 2020 <= selected_year - 0.25 <= 2022:
        title = "During-pandemic"
    else:
        title = "Post-pandemic"
    return plot_all_routes(
        all_flights,
        title=f"{title}: {int(selected_year - 0.25)} (Q{selected_quarter})",
        valid=filter_by_year(int(selected_year - 0.25), selected_quarter)
    )


# === Tab 3 ===
def prediction_layout() -> Component:
    """Generate the layout for the fare prediction tab, allowing selection of quarter and endpoints."""
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
def predict_route(
    n: Optional[str],
    quarter: int,
    origin_code: str,
    dest_code: str,
    priority: str
) -> Tuple[go.Figure, Union[str, html.Div], go.Figure]:
    """
    Callback to compute and visualize predicted routes for 2025 based on historical patterns and user priority.
    Returns a predicted route map, summary text or component, and a full network visualization.
    """
    if not n:
        return get_default_map(), "", get_default_map()

    filename = "Predicted_Airlines_2025_FILLED.csv"
    if not os.path.exists(filename):
        return get_default_map(), f"Prediction file {filename} not found.", get_default_map()

    pred = Flights(filename)

    origin = next((loc for loc in pred.cities if origin_code in loc.airport_codes), None)
    dest = next((loc for loc in pred.cities if dest_code in loc.airport_codes), None)

    if not origin or not dest:
        return get_default_map(pred), "Invalid cities.", plot_all_routes(pred, "")

    if priority == "fare_dist":
        priorities = ["fare", "dist"]
    elif priority == "fare_transfers":
        priorities = ["fare", "transfers"]
    elif priority == "transfers":
        priorities = ["transfers", "fare"]
    else:
        priorities = [priority]

    def filter_routes(r: Route) -> bool:
        # filter start and end locations to match airport codes

        if {origin_code, dest_code} <= r.arrival_loc.airport_codes.union(r.depart_loc.airport_codes):
            return {origin_code, dest_code} == {r.arrival_airport, r.depart_airport}

        if origin_code in r.arrival_loc.airport_codes.union(r.depart_loc.airport_codes):
            return origin_code in [r.arrival_airport, r.depart_airport]
        if dest_code in r.arrival_loc.airport_codes.union(r.depart_loc.airport_codes):
            return dest_code in [r.arrival_airport, r.depart_airport]
        return True

    dist, final_route = find_shortest_path(pred, origin, dest, priorities, valid=filter_routes)
    if not final_route:
        return get_default_map(), "No predicted route found.", plot_all_routes(pred, "")

    fig = plot_route(pred, final_route)
    output_route = Route.get_route_path_string(final_route)

    summary_div = html.Div([
        html.H6("Route Summary"),
        html.P(output_route),
        html.P(f"Total Fare: ${dist["fare"]:.2f}"),
        html.P(f"Total Distance: {int(dist["dist"])} miles"),
        html.P(f"Transfers: {int(dist["transfers"]) - 1}")
    ])

    return fig, summary_div, plot_all_routes(pred, title=f"All Predicted Routes for 2025 {quarter}")


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

    # When you are ready to check your work with python_ta, uncomment the following lines.
    # (In PyCharm, select the lines below and press Ctrl/Cmd + / to toggle comments.)
    # You can use "Run file in Python Console" to run both pytest and PythonTA,
    # and then also test your methods manually in the console.
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ["os", "plotly.graph_objects", "dash_bootstrap_components", "dash",
                          "dash.development.base_component", "flights", "search", "pandas"],
        'allowed-io': ["filter_route", "find_route"],  # the names (strs) of functions that call print/open/input
        'max-line-length': 120,
        'disable': ["E9997", "R0913", "R0914", "C9103", "W0621", "E9992", "E1120"]
    })
