import pandas as pd
import networkx as nx
import kagglehub
from scipy.stats import pearsonr

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
    label = "$" if unit == "USD" else "miles"
    if direct_only:
        if G.has_edge(origin, destination):
            cost = G[origin][destination]["weight"]
            if max_cost is None or cost <= max_cost:
                print(f"Direct flight from {origin} to {destination} for {cost:.2f} {label}")
            else:
                print(f"Direct flight costs {cost:.2f} {label}, over your budget of {max_cost:.2f} {label}")
        else:
            print(f"No direct flight from {origin} to {destination}")
    else:
        try:
            path = nx.shortest_path(G, origin, destination, weight="weight")
            cost = nx.shortest_path_length(G, origin, destination, weight="weight")
            if max_cost is None or cost <= max_cost:
                print(f"Best path from {origin} to {destination}: {path}")
                print(f"Total {unit.lower()}: {cost:.2f} {label}")
            else:
                print(f"Best path costs {cost:.2f} {label}, over your budget of {max_cost:.2f} {label}")
        except nx.NetworkXNoPath:
            print(f"No route from {origin} to {destination}")


def fare_trend_analysis(df_pre, df_during, df_post):
    print("Average Fare Trends:")
    for label, df in [("Pre", df_pre), ("During", df_during), ("Post", df_post)]:
        avg_fare = df["fare"].mean()
        print(f"- {label}-pandemic: ${avg_fare:.2f}")

def fare_distance_correlation(df, label):
    corr, _ = pearsonr(df["fare"], df["nsmiles"])
    print(f"Correlation between fare and distance ({label}-pandemic): {corr:.3f}")

def alternative_route_comparison(G_fare, G_distance, origin, destination):
    print("Fare-optimized route:")
    show_shortest_path(G_fare, origin, destination)

    print("Distance-optimized route:")
    show_shortest_path(G_distance, origin, destination)

def get_user_inputs():
    origin = input("Enter origin airport code (e.g., JFK): ").strip().upper()
    destination = input("Enter destination airport code (e.g., LAX): ").strip().upper()
    allow_transfers = input("Allow transfers? (yes/no): ").strip().lower()
    direct_only = (allow_transfers == "no")
    budget_input = input("Enter max budget in USD (or press Enter to skip): ").strip()
    max_budget = float(budget_input) if budget_input else None
    return origin, destination, direct_only, max_budget

def main():
    csv_path = download_dataset()
    df_pre, df_during, df_post, origin_col, dest_col, fare_col, distance_col = load_and_filter_data(csv_path)

    # Build graphs for all periods
    G_pre = build_graph(df_pre, origin_col, dest_col, fare_col)
    G_during = build_graph(df_during, origin_col, dest_col, fare_col)
    G_post_fare = build_graph(df_post, origin_col, dest_col, fare_col)
    G_post_distance = build_graph(df_post, origin_col, dest_col, distance_col)

    # Stats
    print("Graph Stats:")
    print_graph_summary("Pre-pandemic", G_pre)
    print_graph_summary("During-pandemic", G_during)
    print_graph_summary("Post-pandemic (fare)", G_post_fare)

    # Trend analysis
    fare_trend_analysis(df_pre, df_during, df_post)

    # Correlation analysis
    fare_distance_correlation(df_post, "Post")

    # User input
    print("Route Finder")
    origin, destination, direct_only, max_budget = get_user_inputs()

    # Route suggestions
    print("Fare-based route:")
    show_shortest_path(G_post_fare, origin, destination, max_cost=max_budget, direct_only=direct_only)

    print("Distance-based route:")
    show_shortest_path(G_post_distance, origin, destination, direct_only=direct_only)

    alternative_route_comparison(G_post_fare, G_post_distance, origin, destination)


if __name__ == "__main__":
    main()
