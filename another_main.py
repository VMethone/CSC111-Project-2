import pandas as pd
import networkx as nx
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def load_dataset(filepath="filled_geocoded_airline_data.csv"):
    """
    Load the unified airline dataset.
    """
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def filter_by_period(df, start_year, end_year):
    """
    Filter data to a specific time period.
    """
    return df[(df["Year"] >= start_year) & (df["Year"] <= end_year)].copy()

# === Graph Construction ===
def build_graph(df, origin_col, dest_col, weight_col, extra_cols=None):
    G = nx.Graph()
    extra_cols = extra_cols or []
    for _, row in df.iterrows():
        origin = row[origin_col]
        dest = row[dest_col]
        weight = row[weight_col]
        edge_data = {"weight": weight}
        for col in extra_cols:
            edge_data[col] = row.get(col, 0)
        G.add_edge(origin, dest, **edge_data)
    return G


def build_airport_labels(df, airport_col, city_col):
    """
    Create label map: 'JFK' -> 'JFK (New York)'
    """
    return {
        row[airport_col]: f"{row[airport_col]} ({row[city_col]})"
        for _, row in df[[airport_col, city_col]].dropna().iterrows()
    }

def load_airport_coordinates_from_us_airports(filepath="us-airports.csv"):
    """
    Load real coordinates using IATA codes, stripping leading 'K' if present.
    """
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["iata_code", "latitude_deg", "longitude_deg"])

    coord_map = {}
    for _, row in df.iterrows():
        code = row["iata_code"].strip().upper()
        if code.startswith("K") and len(code) == 4:
            code = code[1:]
        coord_map[code] = (row["longitude_deg"], row["latitude_deg"])
    return coord_map

def show_shortest_path(G, origin, dest, allow_transfers=True, max_cost=None, unit="USD"):
    """
    Show best route using Dijkstra.
    """
    label = "$" if unit == "USD" else "miles"
    if not allow_transfers:
        if G.has_edge(origin, dest):
            cost = G[origin][dest]["weight"]
            if max_cost is None or cost <= max_cost:
                print(f"Direct flight {origin} ➝ {dest}: {cost:.2f} {label}")
            else:
                print(f"Direct flight costs {cost:.2f} {label}, over your limit.")
            return [origin, dest]
        else:
            print(f"No direct flight from {origin} to {dest}")
            return []
    else:
        try:
            path = nx.shortest_path(G, origin, dest, weight="weight")
            cost = nx.shortest_path_length(G, origin, dest, weight="weight")
            if max_cost is None or cost <= max_cost:
                print(f"Best path: {path}, total {label}: {cost:.2f}")
                return path
            else:
                print(f"Path found but costs {cost:.2f} {label}, over budget.")
        except nx.NetworkXNoPath:
            print("No path found.")
    return []

def fare_trend_analysis(df_pre, df_during, df_post):
    """
    Print average fare per period.
    """
    print("\nAverage Fare Trends:")
    for label, df in [("Pre", df_pre), ("During", df_during), ("Post", df_post)]:
        avg = df["fare"].mean()
        print(f"{label}-pandemic: ${avg:.2f}")

def fare_distance_correlation(df, label):
    """
    Correlation between fare and distance.
    """
    corr, _ = pearsonr(df["fare"], df["nsmiles"])
    print(f"Fare–Distance correlation ({label}): {corr:.3f}")

# === Visualization ===
def draw_network(G, title="Route Graph", label_map=None):
    """
    Show networkx graph.
    """
    pos = nx.spring_layout(G, seed=42)
    labels = {n: label_map.get(n, n) for n in G.nodes()} if label_map else None
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=600,
            node_color='lightblue', font_size=8, edge_color='gray')
    plt.title(title)
    plt.show()

def plot_full_network_routes(G, coord_map, title="Full Predicted Routes"):
    import folium
    from streamlit_folium import st_folium

    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)  # Centered on USA
    for u, v in G.edges():
        if u in coord_map and v in coord_map:
            loc_u = coord_map[u]
            loc_v = coord_map[v]
            folium.PolyLine(locations=[loc_u, loc_v], weight=2, color="blue").add_to(m)
    st_folium(m, width=800, height=600)

def plot_route_on_map(route, coord_map, title="Route Map"):
    """
    Plot the route using Plotly and coordinates from coord_map.
    """
    missing = [code for code in route if code not in coord_map]
    if missing:
        print("Missing coordinates for:", missing)

    lons = [coord_map[code][0] for code in route if code in coord_map]
    lats = [coord_map[code][1] for code in route if code in coord_map]

    if len(lons) < 2:
        print("Not enough coordinates to map route.")
        return

    fig = go.Figure(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='lines+markers',
        line=dict(width=2, color='blue'),
        marker=dict(size=6),
        text=route
    ))

    fig.update_layout(
        title=title,
        geo=dict(scope='usa', projection_type='albers usa', showland=True)
    )

    fig.show()

# === User Input ===
def get_user_inputs():
    origin = input("Enter origin airport code (e.g., JFK): ").strip().upper()
    dest = input("Enter destination airport code (e.g., LAX): ").strip().upper()
    allow_transfers = input("Allow transfers? (yes/no): ").strip().lower() != "no"
    budget_input = input("Max budget (USD, optional): ").strip()
    max_budget = float(budget_input) if budget_input else None
    return origin, dest, allow_transfers, max_budget

# === Main ===
def main():
    df = load_dataset("filled_geocoded_airline_data.csv")
    coord_map = load_airport_coordinates_from_us_airports("us-airports.csv")

    # Filter by time periods
    df_pre = filter_by_period(df, 2018, 2020)
    df_during = filter_by_period(df, 2021, 2022)
    df_post = filter_by_period(df, 2023, 2024)

    label_map = build_airport_labels(df, "airport_1", "city1")
    G_fare = build_graph(df_post, "airport_1", "airport_2", "fare")
    G_distance = build_graph(df_post, "airport_1", "airport_2", "nsmiles")

    # Summary
    print(f"Post-pandemic graph: {G_fare.number_of_nodes()} nodes, {G_fare.number_of_edges()} edges")
    fare_trend_analysis(df_pre, df_during, df_post)
    fare_distance_correlation(df_post, "Post")

    # Route finder
    origin, dest, allow_transfers, max_budget = get_user_inputs()

    print("\nFare-based route:")
    fare_route = show_shortest_path(G_fare, origin, dest, allow_transfers, max_budget, unit="USD")

    print("\nDistance-based route:")
    show_shortest_path(G_distance, origin, dest, allow_transfers, unit="miles")

    draw_network(G_fare, "Post-Pandemic Fare Graph", label_map)

    if fare_route:
        plot_route_on_map(fare_route, coord_map, "Fare-Optimized Route")

if __name__ == "__main__":
    main()
