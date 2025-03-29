import pandas as pd
import networkx as nx
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

MAJOR_HUBS = {'ATL', 'JFK', 'LAX', 'ORD', 'DFW', 'DEN', 'SFO'}
MIN_PASSENGERS = 50
FARE_PER_MILE = (0.10, 0.30)

def load_dataset(filepath="Final.csv"):
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df["passengers"].between(1, 500)]
    df = df.dropna(subset=["city1", "city2", "airport_1", "airport_2", "fare"])
    df["period"] = list(zip(df["year"], df["quarter"]))
    return df

def compute_route_stats(df):
    stats = defaultdict(lambda: {
        'fares': [],
        'passengers': [],
        'distance': [],
        'frequency': 0
    })

    for _, row in df.iterrows():
        route = tuple(sorted((row["airport_1"], row["airport_2"])))
        stats[route]['fares'].append(row["fare"])
        stats[route]['passengers'].append(row["passengers"])
        stats[route]['distance'].append(row.get("nsmiles", 0))
        stats[route]['frequency'] += 1

    return {
        route: {
            'avg_fare': np.mean(vals['fares']),
            'std_fare': np.std(vals['fares']),
            'avg_passengers': np.mean(vals['passengers']),
            'max_passengers': np.max(vals['passengers']),
            'min_passengers': np.min(vals['passengers']),
            'distance': np.mean(vals['distance']),
            'frequency': vals['frequency']
        }
        for route, vals in stats.items()
    }

def generate_features(G, route_stats):
    features = []
    nodes = list(G.nodes())
    pagerank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(G)

    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            route = tuple(sorted((u, v)))

            base_feature = {
                'route': route,
                'existed_last_q': int(G.has_edge(u, v)),
                'degree_u': G.degree(u),
                'degree_v': G.degree(v),
                'distance': route_stats.get(route, {}).get('distance', 0)
            }

            network_feature = {
                'pagerank_u': pagerank.get(u, 0),
                'pagerank_v': pagerank.get(v, 0),
                'betweenness_u': betweenness.get(u, 0),
                'betweenness_v': betweenness.get(v, 0),
                'hub_u': int(u in MAJOR_HUBS),
                'hub_v': int(v in MAJOR_HUBS)
            }

            stats = route_stats.get(route, {})
            stat_feature = {
                'avg_fare': stats.get('avg_fare', 0),
                'fare_variation': stats.get('std_fare', 0) / (stats.get('avg_fare', 1e-6) + 1e-6),
                'passenger_range': stats.get('max_passengers', 0) - stats.get('min_passengers', 0),
                'frequency': stats.get('frequency', 0)
            }

            features.append({**base_feature, **network_feature, **stat_feature})

    return pd.DataFrame(features)

def get_fare_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ))
    ])

def get_pax_pipeline():
    return Pipeline([
        ('log_transform', FunctionTransformer(np.log1p)),
        ('model', HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ))
    ])

def apply_business_rules(df):
    df['pred_passengers'] = np.where(
        df['pred_passengers'] < MIN_PASSENGERS,
        0,
        np.round(df['pred_passengers'])
    )

    min_fare = df['distance'] * FARE_PER_MILE[0]
    max_fare = df['distance'] * FARE_PER_MILE[1]
    df['pred_fare'] = np.clip(df['pred_fare'], min_fare, max_fare)

    df.loc[df['pred_passengers'] == 0, 'pred_fare'] = 0
    return df

def main():
    df = load_dataset()
    route_stats = compute_route_stats(df)

    graphs = {}
    for (year, quarter), group in df.groupby(['year', 'quarter']):
        G = nx.Graph()
        for _, row in group.iterrows():
            G.add_edge(row['airport_1'], row['airport_2'],
                      weight=row['fare'],
                      passengers=row['passengers'])
        graphs[(year, quarter)] = G

    feature_rows = []
    sorted_periods = sorted(graphs.keys())

    for i in range(1, len(sorted_periods)):
        prev_period = sorted_periods[i-1]
        curr_period = sorted_periods[i]

        features = generate_features(graphs[prev_period], route_stats)

        curr_edges = {(u, v): data for u, v, data in graphs[curr_period].edges(data=True)}
        features['exists'] = features['route'].apply(
            lambda r: int(r in curr_edges or tuple(reversed(r)) in curr_edges)
        )
        features['fare'] = features['route'].apply(
            lambda r: curr_edges.get(r, {}).get('weight', 0) or curr_edges.get(tuple(reversed(r)), {}).get('weight', 0)
        )
        features['passengers'] = features['route'].apply(
            lambda r: curr_edges.get(r, {}).get('passengers', 0) or curr_edges.get(tuple(reversed(r)), {}).get('passengers', 0)
        )

        feature_rows.append(features)

    full_features = pd.concat(feature_rows)

    X_route = full_features[['existed_last_q', 'degree_u', 'degree_v', 'hub_u', 'hub_v']]
    y_route = full_features['exists']

    route_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )
    route_model.fit(X_route, y_route)

    fare_features = full_features[full_features['exists'] == 1][[
        'distance', 'hub_u', 'hub_v', 'pagerank_u', 'pagerank_v',
        'avg_fare', 'fare_variation', 'frequency'
    ]]
    fare_pipeline = get_fare_pipeline()
    fare_pipeline.fit(fare_features, full_features[full_features['exists'] == 1]['fare'])

    pax_features = full_features[full_features['exists'] == 1][[
        'distance', 'hub_u', 'hub_v', 'passenger_range', 'frequency'
    ]]
    pax_pipeline = get_pax_pipeline()
    pax_pipeline.fit(pax_features, full_features[full_features['exists'] == 1]['passengers'])

    future_periods = [(2025, q) for q in range(1, 5)]
    current_graph = graphs[sorted_periods[-1]]

    for period in future_periods:
        pred_features = generate_features(current_graph, route_stats)

        X_route_pred = pred_features[['existed_last_q', 'degree_u', 'degree_v', 'hub_u', 'hub_v']]
        pred_features['exists_pred'] = route_model.predict(X_route_pred)

        active_routes = pred_features[pred_features['exists_pred'] == 1]

        fare_pred = fare_pipeline.predict(active_routes[
            ['distance', 'hub_u', 'hub_v', 'pagerank_u', 'pagerank_v',
             'avg_fare', 'fare_variation', 'frequency']
        ])

        pax_pred = pax_pipeline.predict(active_routes[
            ['distance', 'hub_u', 'hub_v', 'passenger_range', 'frequency']
        ])

        pred_features = pred_features.merge(
            pd.DataFrame({
                'route': active_routes['route'],
                'pred_fare': fare_pred,
                'pred_passengers': pax_pred
            }),
            on='route',
            how='left'
        ).fillna(0)

        final_pred = apply_business_rules(pred_features)

        G_pred = nx.Graph()
        for _, row in final_pred.iterrows():
            if row['pred_passengers'] > 0:
                u, v = row['route']
                G_pred.add_edge(u, v,
                              fare=round(row['pred_fare'], 2),
                              passengers=int(row['pred_passengers']))

        export_predictions(G_pred, period, df)
        current_graph = G_pred

def export_predictions(G, period, df_template):
    records = []
    city_map = df_template[['airport_1', 'city1']].drop_duplicates().set_index('airport_1')['city1'].to_dict()

    for u, v, data in G.edges(data=True):
        records.append({
            'Year': period[0],
            'quarter': period[1],
            'airport_1': u,
            'airport_2': v,
            'city1': city_map.get(u, ''),
            'city2': city_map.get(v, ''),
            'fare': data['fare'],
            'passengers': data['passengers'],
            'distance': data.get('distance', 0)
        })

    df_out = pd.DataFrame(records)

    df_out.replace('', pd.NA, inplace=True)
    df_out = df_out.dropna(subset=["airport_1", "airport_2", "fare", "passengers", "distance"])

    df_out.to_csv(f"prediction_{period[0]}Q{period[1]}.csv", index=False)


if __name__ == "__main__":
    main()
