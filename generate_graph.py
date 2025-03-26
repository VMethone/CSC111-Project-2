import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_dataset(filepath="filled_geocoded_airline_data.csv"):
    """
    Load the dataset and add 'period' and 'phase' columns.
    """
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()

    df["period"] = list(zip(df["Year"], df["quarter"]))

    def tag_phase(year, quarter):
        if year < 2020 or (year == 2020 and quarter == 1):
            return "pre"
        elif 2020 <= year <= 2022:
            return "pandemic"
        else:
            return "post"

    df["phase"] = df.apply(lambda row: tag_phase(row["Year"], row["quarter"]), axis=1)
    return df

def build_graphs_by_quarter(df, origin_col="airport_1", dest_col="airport_2", weight_col="fare"):
    """
    Create a dictionary mapping (year, quarter) to a graph.
    """
    graphs = {}
    for (year, quarter), group in df.groupby(["Year", "quarter"]):
        G = nx.Graph()
        for _, row in group.iterrows():
            G.add_edge(row[origin_col], row[dest_col], weight=row[weight_col])
        graphs[(year, quarter)] = G
    return graphs

def split_graphs_by_phase(graphs):
    """
    Split graphs by phase (pre, pandemic, post).
    """
    phase_graphs = {"pre": {}, "pandemic": {}, "post": {}}
    for (year, quarter), G in graphs.items():
        if year < 2020 or (year == 2020 and quarter == 1):
            phase_graphs["pre"][(year, quarter)] = G
        elif 2020 <= year <= 2022:
            phase_graphs["pandemic"][(year, quarter)] = G
        else:
            phase_graphs["post"][(year, quarter)] = G
    return phase_graphs

def visualize_graph(G, title):
    """
    Visualize a NetworkX graph using matplotlib.
    """
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', edge_color='gray', font_size=8)
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_dataset()
    fare_graphs = build_graphs_by_quarter(df, weight_col="fare")
    distance_graphs = build_graphs_by_quarter(df, weight_col="nsmiles")

    fare_phase_graphs = split_graphs_by_phase(fare_graphs)
    distance_phase_graphs = split_graphs_by_phase(distance_graphs)

    for phase in ["pre", "pandemic", "post"]:
        print(f"\n=== Fare Graphs: {phase.upper()}-pandemic ===")
        for (year, quarter), G in sorted(fare_phase_graphs[phase].items()):
            print(f"Fare Graph - {year} Q{quarter}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            visualize_graph(G, f"Fare Graph - {year} Q{quarter} ({phase})")
