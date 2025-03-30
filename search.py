from flights import Flights, Location, Route
from queue import PriorityQueue
from collections import defaultdict
from typing import Optional, Dict, List
import plotly.graph_objects as go


def find_shortest_path(flights: Flights, start: Location, end: Location, priorities: List[str], valid=lambda *args: True) -> tuple[dict, list[Route]]:
    """
    Finds the shortest path between start and end locations using Dijkstra's algorithm,
    prioritizing the given attributes in order.

    Args:
        flights: The Flights object containing all locations and routes.
        start: The starting Location.
        end: The target Location.
        priorities: List of strings indicating the priority order of attributes (e.g., ['fare', 'dist']).

    Returns:
        A dictionary with 'cost' (tuple of accumulated attributes) and 'path' (list of Route objects),
        or None if no path exists.
    """
    all_priorities = ["fare", "transfers", "dist"]
    for p in all_priorities:
        if p not in priorities:
            priorities.append(p)

    dist, prev = find_all_shortest_paths(flights, start, priorities, valid)
    end_tuple = dist[end.location_id]
    sorted_dist = {
        "fare": end_tuple[priorities.index("fare")],
        "dist": end_tuple[priorities.index("dist")],
        "transfers": end_tuple[priorities.index("transfers")],
    }
    print()
    return sorted_dist, reconstruct(end, prev)


def find_all_shortest_paths(flights: Flights, start: Location, priorities: List[str], valid=lambda *args: True) -> tuple[dict[list[int]], dict[int, tuple[Location, Route]]]:
    """
    Finds the shortest path between start and end locations using Dijkstra's algorithm,
    prioritizing the given attributes in order.

    Args:
        flights: The Flights object containing all locations and routes.
        start: The starting Location.
        end: The target Location.
        priorities: List of strings indicating the priority order of attributes (e.g., ['fare', 'dist']).

    Returns:
        A dictionary with 'cost' (tuple of accumulated attributes) and 'path' (list of Route objects),
        or None if no path exists.
    """
    # Initialize the priority queue
    pq = PriorityQueue()

    # Determine the initial cost based on priorities
    all_priorities = ["fare", "transfers", "dist"]
    for p in all_priorities:
        if p not in priorities:
            priorities.append(p)

    initial_cost = (0, 0, 0)
    pq.put((initial_cost, start.location_id, start))

    distances: dict[int, list[int]] = defaultdict(list)
    distances[start.location_id] = initial_cost

    predecessors: dict[int, tuple[Location, Route]] = defaultdict(None)
    predecessors[start.location_id] =  None

    while not pq.empty():
        current_cost, current_id, current_loc = pq.get()
        if current_cost > distances.get(current_id, tuple([float('inf')] * len(priorities))):
            continue
        
        for route in flights.flight_routes[current_loc]:
            if not valid(route):
                continue

            neighbor = route.arrival_loc

            new_cost = list(current_cost)
            for i, attr in enumerate(priorities):
                if attr == 'fare':
                    new_cost[i] += route.fare
                elif attr == 'dist':
                    new_cost[i] += route.dist
                elif attr == 'transfers':
                    new_cost[i] += 1
            new_cost = tuple(new_cost)

            existing_cost = distances.get(neighbor.location_id, tuple([float('inf')] * len(priorities)))
            if new_cost < existing_cost:
                distances[neighbor.location_id] = new_cost
                predecessors[neighbor.location_id] = (current_loc, route)
                pq.put((new_cost, neighbor.location_id, neighbor))

    return distances, predecessors


def reconstruct(end_location: Location, predecessors: dict[int, tuple[Location, Route]]) -> list[Route]:
    path_routes = []
    current_id = end_location.location_id
    while True:
        prev_info = predecessors.get(current_id)
        if prev_info is None:
            break
        prev_loc, route = prev_info
        path_routes.append(route)
        current_id = prev_loc.location_id

    return list(reversed(path_routes))

def plot_all_routes(flights: Flights, title: str, valid = lambda *args: True):
    coords = {l.location_id: l.geo_loc for l in flights.cities}
    labels = [list(l.airport_codes) for l in flights.cities]
    lons = [coords[airport][1] for airport in coords]
    lats = [coords[airport][0] for airport in coords]

    if not lons or not lats:
        print("Map plotting skipped: missing coordinates for some airports.")
        return

    line_lons = []
    line_lats = []
    done = set()

    fig = go.Figure(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers+text',
        text=labels,
        textposition="top center",
        line=dict(width=2, color='blue'),
        marker=dict(size=6),
        name=""
    ))

    number_of_routes = 0
    for location in flights.flight_routes:
        for route in flights.flight_routes[location]:
            if not valid(route):
                continue
            number_of_routes += 1
            dep_city = route.depart_loc.location_id
            arr_city = route.arrival_loc.location_id
            if (dep_city, arr_city) in done:
                continue

            done.add((dep_city, arr_city))
            if dep_city in coords and arr_city in coords:
                line_lons.extend([coords[dep_city][1], coords[arr_city][1], None])
                line_lats.extend([coords[dep_city][0], coords[arr_city][0], None])



    if line_lons and line_lats:
        fig.add_trace(go.Scattergeo(
            lon=line_lons,
            lat=line_lats,
            mode='lines',
            line=dict(width=0.2, color='rgba(255, 0, 0, 0.2)'),
            name=""
        ))


    fig.update_layout(
        title=f"{title} ({number_of_routes} Total Routes)",
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(217, 217, 217)',
            countrycolor='rgb(255, 255, 255)'
        )
    )

    return fig

def get_default_map(flights: Flights | None = None) -> go.Figure:
    if not flights:
        flights = Flights()
    coords = {l.location_id: l.geo_loc for l in flights.cities}
    labels = [f"[{' '.join(list(l.airport_codes))}]" for l in flights.cities]
    city_names = [l.city_name for l in flights.cities]  # New line for city names

    lons = [coords[airport][1] for airport in coords]
    lats = [coords[airport][0] for airport in coords]

    fig = go.Figure(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers',
        hoverinfo="text",  # Show only custom hover info
        hovertemplate=(
            "<b>City:</b> %{customdata[0]}<br>"
            "<b>Airports:</b> %{text}<br>"
            "<b>Lat:</b> %{lat:.2f}°<br>"
            "<b>Lon:</b> %{lon:.2f}°"
            "<extra></extra>"  # Removes trace name
        ),
        text=labels,
        customdata=list(zip(city_names)), 
        textposition="top center",
        line=dict(width=2, color='blue'),
        marker=dict(size=6),
    ))

    fig.update_layout(
        title="Airports",
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(217, 217, 217)',
            countrycolor='rgb(255, 255, 255)'
        )
    )
    return fig

def plot_route(flights: Flights, total_route: list[Route]) -> go.Figure:
    route_cities = [total_route[0].depart_loc]
    labels = [total_route[0].depart_airport]
    city_names = [total_route[0].depart_loc.city_name]

    for route in total_route:
        route_cities.append(route.arrival_loc)
        labels.append(route.arrival_airport)
        city_names.append(route.arrival_loc.city_name)

    coords = {l.location_id: l.geo_loc for l in route_cities}

    lons = [coords[airport][1] for airport in coords]
    lats = [coords[airport][0] for airport in coords]

    fig = go.Figure(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode="markers+text",
        text=city_names,
        customdata=labels, 
        hovertemplate=(
            "<b>City:</b> %{text}<br>"
            "<b>Airports:</b> %{customdata}<br>"
            "<b>Lat:</b> %{lat:.2f}°<br>"
            "<b>Lon:</b> %{lon:.2f}°"
            "<extra></extra>"
        ),
        textposition="top center",
        line=dict(width=2, color='blue'),
        marker=dict(size=6),
    ))

    # Collect line segments for all flight routes
    line_lons = []
    line_lats = []
    for route in total_route:
        dep_city = route.depart_loc.location_id
        arr_city = route.arrival_loc.location_id

        # Add departure and arrival coordinates, separated by None
        line_lons.extend([coords[dep_city][1], coords[arr_city][1], None])
        line_lats.extend([coords[dep_city][0], coords[arr_city][0], None])

        # Add all flight routes as lines
    if line_lons and line_lats:
        fig.add_trace(go.Scattergeo(
            lon=line_lons,
            lat=line_lats,
            mode='lines',
            hoverinfo='skip',
            line=dict(width=1, color='red'),
            name='Flight Routes'
        ))

    fig.update_layout(
        title="Airports and Flight Routes",
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(217, 217, 217)',
            countrycolor='rgb(255, 255, 255)'
        )
    )

    return fig


if __name__ == "__main__":
    test = Flights("Final.csv")
    cities = list(test.cities)
    start, end = cities[0], cities[1]
    dist, routes = find_shortest_path(test, start, end, ["fare"])
    plot_route(test, routes)
