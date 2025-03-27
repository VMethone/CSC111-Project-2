import pandas as pd
import networkx as nx
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np

# === Load data and prepare graphs ===
def load_dataset(filepath="Final.csv"):
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    df["period"] = list(zip(df["year"], df["quarter"]))
    return df

def build_graphs_by_quarter(df, origin_col="airport_1", dest_col="airport_2", weight_col="fare"):
    graphs = {}
    for (year, quarter), group in df.groupby(["year", "quarter"]):
        G = nx.Graph()
        for _, row in group.iterrows():
            G.add_edge(row[origin_col], row[dest_col], weight=row[weight_col])
        graphs[(year, quarter)] = G
    return graphs

def compute_route_averages(df):
    route_avg = {}
    for _, row in df.iterrows():
        route = tuple(sorted((row["airport_1"], row["airport_2"])))
        if route not in route_avg:
            route_avg[route] = {"fares": [], "passengers": []}
        route_avg[route]["fares"].append(row["fare"])
        route_avg[route]["passengers"].append(row["passengers"])
    return {
        route: {
            "avg_fare": sum(vals["fares"]) / len(vals["fares"]),
            "avg_passengers": sum(vals["passengers"]) / len(vals["passengers"])
        }
        for route, vals in route_avg.items()
    }

def generate_features_for_prediction(G_prev, route_avg):
    features = []
    seen_routes = set()
    nodes = list(G_prev.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            route = tuple(sorted((u, v)))
            if route in seen_routes:
                continue
            seen_routes.add(route)
            degree_u = G_prev.degree(u)
            degree_v = G_prev.degree(v)
            avg_fare = route_avg.get(route, {}).get("avg_fare", 0)
            avg_pax = route_avg.get(route, {}).get("avg_passengers", 0)
            features.append({
                "route": route,
                "existed_last_q": int(G_prev.has_edge(u, v)),
                "degree_u": degree_u,
                "degree_v": degree_v,
                "sum_degree": degree_u + degree_v,
                "diff_degree": abs(degree_u - degree_v),
                "avg_fare": avg_fare,
                "avg_passengers": avg_pax
            })
    return pd.DataFrame(features)

def build_graph_from_predictions(route_df, fare_preds, passenger_preds):
    G = nx.Graph()
    max_passengers = 300
    for idx, row in route_df.iterrows():
        if row["predicted"] == 1:
            u, v = row["route"]
            fare = fare_preds[idx]
            pax = int(min(round(passenger_preds[idx]), max_passengers))
            G.add_edge(u, v, weight=fare, fare=fare, passengers=pax)
    return G

def export_graph_to_filled_format(G, period, df_original, filename):
    city_lookup = {}
    for _, row in df_original.iterrows():
        city_lookup[row["airport_1"]] = row["city1"]
        city_lookup[row["airport_2"]] = row["city2"]

    rows = []
    for u, v, data in G.edges(data=True):
        city1 = city_lookup.get(u, "")
        city2 = city_lookup.get(v, "")
        rows.append({
            "Year": period[0],
            "quarter": period[1],
            "city1": city1,
            "city2": city2,
            "airport_1": u,
            "airport_2": v,
            "fare": round(data.get("fare", 0), 2),
            "passengers": int(data.get("passengers", 0))
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(filename, index=False)
    print(f"Saved formatted prediction: {filename}")

if __name__ == "__main__":
    df = load_dataset()
    graphs = build_graphs_by_quarter(df)
    route_avg = compute_route_averages(df)

    feature_rows = []
    sorted_periods = sorted(graphs.keys())
    for i in range(1, len(sorted_periods)):
        curr_period = sorted_periods[i]
        prev_period = sorted_periods[i - 1]
        G_curr = graphs[curr_period]
        G_prev = graphs[prev_period]
        routes = generate_features_for_prediction(G_prev, route_avg)
        labels, fares, pax = [], [], []
        df_curr = df[df["period"] == curr_period].copy()
        df_curr.set_index(["airport_1", "airport_2"], inplace=True)
        for idx, row in routes.iterrows():
            u, v = row["route"]
            exists = G_curr.has_edge(u, v)
            labels.append(int(exists))
            if exists:
                key1 = (u, v)
                key2 = (v, u)
                if key1 in df_curr.index:
                    match = df_curr.loc[key1]
                elif key2 in df_curr.index:
                    match = df_curr.loc[key2]
                else:
                    match = pd.Series({"fare": None, "passengers": None})
                fares.append(match["fare"] if pd.notna(match["fare"]) else None)
                pax.append(match["passengers"] if pd.notna(match["passengers"]) else None)
            else:
                fares.append(None)
                pax.append(None)
        routes["label"] = labels
        routes["fare"] = fares
        routes["passengers"] = pax
        feature_rows.extend(routes.to_dict("records"))

    feature_df = pd.DataFrame(feature_rows).dropna()

    X_train_route = feature_df[["existed_last_q", "degree_u", "degree_v"]]
    y_train_route = feature_df["label"]

    X_train_predict = feature_df[["existed_last_q", "degree_u", "degree_v", "sum_degree", "diff_degree", "avg_fare", "avg_passengers"]]
    y_train_fare = feature_df["fare"]
    y_train_pax = np.log1p(feature_df["passengers"])  # log-transform passengers

    route_model = RandomForestClassifier(random_state=42, n_estimators=50)
    fare_model = Ridge()
    pax_model = Ridge()

    route_model.fit(X_train_route, y_train_route)
    fare_model.fit(X_train_predict, y_train_fare)
    pax_model.fit(X_train_predict, y_train_pax)

    future_graphs = {}
    last_known_period = max(graphs.keys())
    G_last = graphs[last_known_period]
    future_periods = [(2025, 1), (2025, 2), (2025, 3), (2025, 4)]

    for period in future_periods:
        features = generate_features_for_prediction(G_last, route_avg)
        X_route = features[["existed_last_q", "degree_u", "degree_v"]]
        X_pred = features[["existed_last_q", "degree_u", "degree_v", "sum_degree", "diff_degree", "avg_fare", "avg_passengers"]]
        features["predicted"] = route_model.predict(X_route)
        fare_preds = fare_model.predict(X_pred)
        pax_preds = np.expm1(pax_model.predict(X_pred))  # inverse log-transform
        G_next = build_graph_from_predictions(features, fare_preds, pax_preds)
        future_graphs[period] = G_next
        G_last = G_next

        export_graph_to_filled_format(G_next, period, df, f"predicted_graph_{period[0]}Q{period[1]}.csv")

        print(f"Predicted graph for {period}: {G_next.number_of_nodes()} nodes, {G_next.number_of_edges()} edges")
