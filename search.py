from flights import Flights, Location, Route
from queue import PriorityQueue
from typing import Optional, Dict, List

def find_shortest_path(flights: Flights, start: Location, end: Location, priorities: List[str]) -> Optional[Dict]:
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
    initial_cost = []
    for attr in priorities:
        if attr == 'fare':
            initial_cost.append(0.0)
        elif attr == 'dist':
            initial_cost.append(0.0)
        elif attr == 'number_of_nodes':
            initial_cost.append(1)  # Start with 1 node (the starting location)
    initial_cost = tuple(initial_cost)
    
    # Add the start location to the priority queue
    pq.put((initial_cost, start.location_id, start))
    
    # Dictionary to keep track of the best known cost for each location
    distances: dict[int, list[int]] = {start.location_id: initial_cost}
    
    # Dictionary to reconstruct the path {current_location: (previous_location, route_taken)}
    predecessors: dict[int, tuple[Location, Route]] = {start.location_id: None}
    
    while not pq.empty():
        current_cost, current_id, current_loc = pq.get()
        
        # Early termination if the target is reached
        if current_loc == end:
            break
        
        # Skip if a better path has already been found
        if current_cost > distances.get(current_id, tuple([float('inf')] * len(priorities))):
            continue
        
        # Explore all outgoing routes from the current location
        for route in flights.flight_routes[current_loc]:
            neighbor = route.arrival_loc
            
            # Calculate the new cost by accumulating each prioritized attribute
            new_cost = list(current_cost)
            for i, attr in enumerate(priorities):
                if attr == 'fare':
                    new_cost[i] += route.fare
                elif attr == 'dist':
                    new_cost[i] += route.dist
                elif attr == 'number_of_nodes':
                    new_cost[i] += 1  # Each route adds one node to the path
            new_cost = tuple(new_cost)
            
            # Check if this new cost is better than the existing
            existing_cost = distances.get(neighbor.location_id, tuple([float('inf')] * len(priorities)))
            if new_cost < existing_cost:
                distances[neighbor.location_id] = new_cost
                predecessors[neighbor.location_id] = (current_loc, route)
                # Add the neighbor to the priority queue
                pq.put((new_cost, neighbor.location_id, neighbor))
    
    # Check if the end location is unreachable
    if end.location_id not in distances:
        return None
    
    # Reconstruct the path from end to start
    path_routes = []
    current_id = end.location_id
    while True:
        prev_info = predecessors.get(current_id)
        if prev_info is None:
            break
        prev_loc, route = prev_info
        path_routes.append(route)
        current_id = prev_loc.location_id
    # Reverse to get the path from start to end
    path_routes.reverse()
    
    return {
        'cost': distances[end.location_id],
        'path': path_routes
    }


if __name__ == "__main__":
    test = Flights("mini.csv")
    start = test.cities.copy().pop()
    end = test.cities.copy().pop(0)
    find_shortest_path(test, start, end, ["fare"])