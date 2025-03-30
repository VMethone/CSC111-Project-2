from flights import Flights, Location, Route
from queue import PriorityQueue
from typing import Optional, Dict, List
import plotly.graph_objects as go
from collections import defaultdict
from itertools import count
import heapq

from collections import defaultdict
from itertools import count
import heapq

def find_shortest_path(flights, origin, destination, priorities):
    graph = defaultdict(list)
    for depart, routes in flights.flight_routes.items():
        for r in routes:
            graph[r.depart_loc.airport_code].append((r.arrival_loc.airport_code, r))

    def get_priority(path):
        fare = sum(r.fare for r in path)
        dist = sum(r.dist for r in path)
        transfers = len(path) - 1
        metrics = {
            "fare": fare,
            "dist": dist,
            "transfers": transfers
        }
        return tuple(metrics[p] for p in priorities)

    visited = set()
    heap = []
    tie_breaker = count()

    # 初始状态（空路径）
    heapq.heappush(heap, (get_priority([]), next(tie_breaker), origin.airport_code, []))

    while heap:
        _, _, current_code, path = heapq.heappop(heap)

        if current_code == destination.airport_code:
            total_fare = sum(r.fare for r in path)
            total_dist = sum(r.dist for r in path)
            transfers = len(path) - 1
            return {
                "path": path,
                "cost": (total_fare, total_dist, transfers)
            }

        state_id = (current_code, len(path))
        if state_id in visited:
            continue
        visited.add(state_id)

        for neighbor_code, route in graph[current_code]:
            if neighbor_code == current_code:
                continue
            if route in path:
                continue
            new_path = path + [route]
            heapq.heappush(heap, (get_priority(new_path), next(tie_breaker), neighbor_code, new_path))

    return None

def find_city_to_city_path(flights, origin_city: str, dest_city: str, priorities: List[str]) -> Optional[dict]:
    from collections import defaultdict

    city_to_airports = defaultdict(set)
    for loc in flights.cities:
        city_to_airports[loc.city_name.strip().lower()].add(loc.airport_code)

    origin_airports = list(city_to_airports.get(origin_city.strip().lower(), []))
    dest_airports = list(city_to_airports.get(dest_city.strip().lower(), []))

    if not origin_airports or not dest_airports:
        print(f"无法找到城市机场: {origin_city} 或 {dest_city}")
        return None

    best_result = None
    for o_code in origin_airports:
        for d_code in dest_airports:
            origin = flights.airport_to_city.get(o_code)
            dest = flights.airport_to_city.get(d_code)

            if not origin or not dest:
                continue

            result = find_shortest_path(flights, origin, dest, priorities)
            if result:
                if best_result is None or result['cost'] < best_result['cost']:
                    best_result = result

    return best_result


def create_route_map(flights: Flights, title: str = "Flight Routes"):
    coords = {
        l.airport_code: l.geo_loc for l in flights.cities
    }

    lons = [coords[airport][1] for airport in coords]
    lats = [coords[airport][0] for airport in coords]
    labels = list(coords.keys())

    fig = go.Figure(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers+text',
        text=labels,
        textposition="top center",
        marker=dict(size=6),
    ))

    line_lons = []
    line_lats = []
    done = set()
    for location in flights.flight_routes:
        for route in flights.flight_routes[location]:
            dep = route.depart_loc.airport_code
            arr = route.arrival_loc.airport_code
            key = tuple(sorted((dep, arr)))
            if key in done:
                continue
            done.add(key)
            if dep in coords and arr in coords:
                line_lons += [coords[dep][1], coords[arr][1], None]
                line_lats += [coords[dep][0], coords[arr][0], None]

    fig.add_trace(go.Scattergeo(
        lon=line_lons,
        lat=line_lats,
        mode='lines',
        line=dict(width=1, color='red'),
        name='Routes'
    ))

    fig.update_layout(
        title=title,
        geo=dict(scope='usa', projection_type='albers usa', showland=True)
    )

    return fig


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