from flights import Flights, Location, Route
from queue import PriorityQueue
from typing import Optional, Dict, List
import plotly.graph_objects as go

def find_shortest_path(flights: Flights, start: Location, end: Location, priorities: List[str], valid = lambda *args: True) -> tuple[list[int], list[Route]]:
    """
    Finds the shortest path between start and end locations using Dijkstra's algorithm,
    prioritizing the given attributes in order.
    
    Args:
        flights: The Flights object containing all locations and routes.
        start: The starting Location.
        end: The target Location.
        priorities: List of strings indicating the priority order of attributes (e.g., ['fare', 'dist']).
        valid: A function Route -> Bool: A filter to exclude specific routes.

    Returns:
        A dictionary with 'cost' (tuple of accumulated attributes) and 'path' (list of Route objects),
        or None if no path exists.
    """
    dist, prev = find_all_shortest_paths(flights, start, priorities, valid)
    return dist[end.location_id], reconstruct(end, prev)

def find_all_shortest_paths(flights: Flights, start: Location, priorities: List[str], valid = lambda *args: True) -> Optional[Dict]:
    """
    Finds the shortest path between start and end locations using Dijkstra's algorithm,
    prioritizing the given attributes in order.
    
    Args:
        flights: The Flights object containing all locations and routes.
        start: The starting Location.
        end: The target Location.
        priorities: List of strings indicating the priority order of attributes (e.g., ['fare', 'dist']).
        valid: A function Route -> Bool: A filter to exclude specific routes.
    
    Returns:
        A tuple of the final distances and all paths taken, which can be reconstructed 
    """
    pq = PriorityQueue()
    
    initial_cost = []
    for attr in priorities:
        if attr == 'fare':
            initial_cost.append(0.0)
        elif attr == 'dist':
            initial_cost.append(0.0)
        elif attr == 'transfers':
            initial_cost.append(1) 
    initial_cost = tuple(initial_cost)
    
    pq.put((initial_cost, start.location_id, start))
    
    distances: dict[int, list[int]] = {start.location_id: initial_cost}
    
    predecessors: dict[int, tuple[Location, Route]] = {start.location_id: None}
    
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

    return path_routes

def plot_map(flights: Flights):
    coords = {
        l.location_id: l.geo_loc for l in flights.cities
    }

    lons = [coords[airport][1] for airport in coords]
    lats = [coords[airport][0] for airport in coords] 
    labels = list(coords.keys())

    print(lons)
    print(lats)

    if not lons or not lats:
        print("Map plotting skipped: missing coordinates for some airports.")
        return

    fig = go.Figure(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers+text',
        text=labels,
        textposition="top center",
        line=dict(width=2, color='blue'),
        marker=dict(size=6),
    ))

        # Collect line segments for all flight routes
    line_lons = []
    line_lats = []
    done = set()
    for location in flights.flight_routes:
        for route in flights.flight_routes[location]:
            dep_city = route.depart_loc.location_id
            arr_city = route.arrival_loc.location_id
            if (dep_city, arr_city) in done:
                continue
            done.add((dep_city, arr_city))
            if dep_city in coords and arr_city in coords:
                # Add departure and arrival coordinates, separated by None
                line_lons.extend([coords[dep_city][1], coords[arr_city][1], None])
                line_lats.extend([coords[dep_city][0], coords[arr_city][0], None])

        # Add all flight routes as lines
    if line_lons and line_lats:
        fig.add_trace(go.Scattergeo(
            lon=line_lons,
            lat=line_lats,
            mode='lines',
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

    fig.show()

def plot_route(final_route: list[Route]) -> None:
    coords = {}
    for route in final_route:
        assert isinstance(route, Route)
        arrival_code = route.arrival_loc.location_id
        arrival_loc = route.arrival_loc.geo_loc
        departure_code = route.depart_loc.location_id
        departure_loc = route.depart_loc.geo_loc
        coords[arrival_code] = arrival_loc
        coords[departure_code] = departure_loc

    lons = [coords[airport][1] for airport in coords]
    lats = [coords[airport][0] for airport in coords] 
    labels = list(coords.keys())

    # Collect line segments for all flight routes
    line_lons = []
    line_lats = []

    for route in final_route:
        assert isinstance(route, Route)
        arrival_code = route.arrival_loc.location_id
        departure_code = route.depart_loc.location_id
        if arrival_code in coords and departure_code in coords:
            # Add departure and arrival coordinates, separated by None
            line_lons.extend([coords[departure_code][1], coords[arrival_code][1], None])
            line_lats.extend([coords[departure_code][0], coords[arrival_code][0], None])


    fig = go.Figure(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers+text',
        text=labels,
        textposition="top center",
        line=dict(width=2, color='blue'),
        marker=dict(size=6),
    ))

    if line_lons and line_lats:
        fig.add_trace(go.Scattergeo(
            lon=line_lons,
            lat=line_lats,
            mode='lines',
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

    fig.show()

def test_search(flights: Flights):
    start = flights.get_location_from_airport_code("LGB")
    end = flights.get_location_from_airport_code("LGA")

    def filter_flights(r: Route) -> bool:
        return 2018 <= r.year <= 2020

    dist, final_route = find_shortest_path(flights, start, end, priorities=["fare"])
    print(f"DIST: {dist}")
    print(f"Route: {final_route}")
    plot_route(final_route=final_route)

    dist, final_route = find_shortest_path(flights, start, end, priorities=["fare"], valid=filter_flights)
    print(f"DIST: {dist}")
    print(f"Route: {final_route}")
    plot_route(final_route=final_route)


if __name__ == "__main__":
    test = Flights("Final.csv")
    
    test_search(test)