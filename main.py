import pandas as pd
import networkx as nx
import kagglehub
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict
from flights import Flights, Location, Route
import search

def download_dataset():
    print("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("bhavikjikadara/us-airline-flight-routes-and-fares-1993-2024")
    print("Download complete. Dataset path:", path)
    return f"{path}/US Airline Flight Routes and Fares 1993-2024.csv"

def load_and_filter_data(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    origin_col = "airport_1"
    dest_col = "airport_2"
    fare_col = "fare"
    distance_col = "nsmiles"
    required_columns = [origin_col, dest_col, fare_col, distance_col]

    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: {col}")

    def filter_by_period(start, end):
        return df[(df["Year"] >= start) & (df["Year"] <= end)].dropna(subset=required_columns)

    df_pre = filter_by_period(2018, 2020)
    df_during = filter_by_period(2021, 2022)
    df_post = filter_by_period(2023, 2024)

    return df_pre, df_during, df_post, origin_col, dest_col, fare_col, distance_col

def build_graph(df, origin_col, dest_col, weight_col):
    G = nx.Graph()
    for _, row in df.iterrows():
        origin = row[origin_col]
        dest = row[dest_col]
        weight = row[weight_col]
        G.add_edge(origin, dest, weight=weight)
    return G

def print_graph_summary(name, G):
    print(f"{name} Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

def show_shortest_path(G, origin, destination, max_cost=None, direct_only=False, unit="USD"):
    label = "$" if unit.upper() == "USD" else "miles"
    if direct_only:
        if G.has_edge(origin, destination):
            cost = G[origin][destination]["weight"]
            if max_cost is None or cost <= max_cost:
                print(f"Direct flight from {origin} to {destination}: {cost:.2f} {label}")
            else:
                print(f"Direct flight costs {cost:.2f} {label}, over your limit of {max_cost:.2f} {label}")
        else:
            print(f"No direct flight from {origin} to {destination}")
    else:
        try:
            path = nx.shortest_path(G, origin, destination, weight="weight")
            cost = nx.shortest_path_length(G, origin, destination, weight="weight")
            if max_cost is None or cost <= max_cost:
                print(f"Best path from {origin} to {destination}: {path}")
                print(f"Total {label}: {cost:.2f}")
                return path
            else:
                print(f"Best path costs {cost:.2f} {label}, over your limit of {max_cost:.2f} {label}")
        except nx.NetworkXNoPath:
            print(f"No route from {origin} to {destination}")
    return []

def fare_trend_analysis(df_pre, df_during, df_post):
    print("Average Fare Trends:")
    for label, df in [("Pre", df_pre), ("During", df_during), ("Post", df_post)]:
        avg_fare = df["fare"].mean()
        print(f"{label}-pandemic: ${avg_fare:.2f}")

def fare_distance_correlation(df, label):
    corr, _ = pearsonr(df["fare"], df["nsmiles"])
    print(f"Correlation between fare and distance ({label}-pandemic): {corr:.3f}")

def get_user_inputs():
    origin = input("Enter origin airport code (e.g., JFK): ").strip().upper()
    destination = input("Enter destination airport code (e.g., LAX): ").strip().upper()
    allow_transfers = input("Allow transfers? (yes/no): ").strip().lower()
    direct_only = (allow_transfers == "no")
    budget_input = input("Enter max budget in USD (or press Enter to skip): ").strip()
    max_budget = float(budget_input) if budget_input else None
    return origin, destination, direct_only, max_budget

def draw_network(G, title="Airline Route Graph"):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title(title)
    plt.show()


def main():
    csv_path = download_dataset()
    df_pre, df_during, df_post, origin_col, dest_col, fare_col, distance_col = load_and_filter_data(csv_path)

    G_post_fare = build_graph(df_post, origin_col, dest_col, fare_col)
    G_post_distance = build_graph(df_post, origin_col, dest_col, distance_col)

    print("Graph Stats:")
    print_graph_summary("Post-pandemic (fare)", G_post_fare)

    fare_trend_analysis(df_pre, df_during, df_post)
    fare_distance_correlation(df_post, "Post")

    origin, destination, direct_only, max_budget = get_user_inputs()

    print("Fare-based route:")
    fare_route = show_shortest_path(G_post_fare, origin, destination, max_cost=max_budget, direct_only=direct_only, unit="USD")

    print("Distance-based route:")
    dist_route = show_shortest_path(G_post_distance, origin, destination, direct_only=direct_only, unit="miles")

    draw_network(G_post_fare, title="Post-Pandemic Fare Network")

    # if fare_route:
    #     plot_route_on_map(fare_route, title="Optimal Fare Route on Map")

def check_city_mappings():
    graph = Flights()
    graph.load_from_cvs("mini.csv")
    for key, value in graph.flight_routes.keys():
        assert isinstance(key, Location), f"{key} is type {type(key)}"
        print(f"Departure - {key.city_name}: ")
        for route in value:
            assert isinstance(route, Route)
            print(f"    TO {route.arrival_loc.city_name} for ${route.fare} during {route.year} (Q{route.quarter})")

def find_transfer_over_price():
    test = Flights("Final.csv")
    
    locations = test.cities
    def valid(item: Route) -> bool:
        return item.year == 2020 and item.quarter == 2 
    
    for start in locations:
        dist1, pred1 = search.find_all_shortest_paths(test, start, ["fare", "transfers"], valid)
        dist2, pred2 = search.find_all_shortest_paths(test, start, ["transfers", "fare"], valid)
        for end in locations:
            if start == end:
                continue
            print(f"Testing {start} -> {end}")
            if end.location_id not in dist1:
                continue
            if end.location_id not in dist2:
                continue
            if sorted(dist1[end.location_id]) != sorted(dist2[end.location_id]):
                route1 = search.reconstruct(end, pred1)
                route2 = search.reconstruct(end, pred2)
                print("Route 1")
                for r in route1:
                    print(f"    {r}")
                print("Route 2")
                for r in route2:
                    print(f"    {r}")

if __name__ == "__main__":
    # main()
    check_city_mappings()
