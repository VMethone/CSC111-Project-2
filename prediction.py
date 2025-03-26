import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(filepath="filled_geocoded_airline_data.csv"):
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    df["period"] = list(zip(df["Year"], df["quarter"]))
    return df


def build_graphs_by_quarter(df, origin_col="airport_1", dest_col="airport_2", weight_col="fare"):
    graphs = {}
    for (year, quarter), group in df.groupby(["Year", "quarter"]):
        G = nx.Graph()
        for _, row in group.iterrows():
            G.add_edge(row[origin_col], row[dest_col], weight=row[weight_col])
        graphs[(year, quarter)] = G
    return graphs


def build_route_feature_table(graphs):
    feature_rows = []
    sorted_periods = sorted(graphs.keys())

    for i in range(1, len(sorted_periods)):
        curr_period = sorted_periods[i]
        prev_period = sorted_periods[i - 1]

        G_curr = graphs[curr_period]
        G_prev = graphs[prev_period]

        all_routes = set(G_curr.edges()) | set(G_prev.edges())

        for u, v in all_routes:
            u, v = sorted((u, v))
            row = {
                "period": curr_period,
                "route": f"{u}-{v}",
                "existed_last_q": int(G_prev.has_edge(u, v)),
                "existed_now": int(G_curr.has_edge(u, v)),
                "fare": G_curr[u][v]["weight"] if G_curr.has_edge(u, v) else None,
                "degree_u": G_curr.degree(u) if u in G_curr else 0,
                "degree_v": G_curr.degree(v) if v in G_curr else 0
            }
            feature_rows.append(row)

    return pd.DataFrame(feature_rows)


if __name__ == "__main__":
    df = load_dataset()
    graphs = build_graphs_by_quarter(df)
    feature_df = build_route_feature_table(graphs)

    feature_df["period_num"] = feature_df["period"].apply(lambda p: p[0] * 10 + p[1])

    train_df = feature_df[feature_df["period_num"] < 20234]
    test_df = feature_df[feature_df["period_num"] == 20234]

    X_train = train_df[["existed_last_q", "degree_u", "degree_v"]]
    y_train = train_df["existed_now"]
    X_test = test_df[["existed_last_q", "degree_u", "degree_v"]]
    y_test = test_df["existed_now"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Route existence prediction accuracy on 2023 Q4: {acc:.2f}")

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Exist", "Exist"])
    disp.plot(cmap='Blues')
    plt.title("Route Existence Prediction: Confusion Matrix")
    plt.show()

    importances = model.feature_importances_
    features = X_train.columns
    plt.figure(figsize=(6, 4))
    plt.barh(features, importances, color="skyblue")
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.show()

    test_df = test_df.copy()
    test_df["predicted"] = preds
    plt.figure(figsize=(6, 4))
    sns.countplot(data=test_df, x="predicted", hue="existed_now")
    plt.title("Prediction vs Actual (2023 Q4)")
    plt.xlabel("Predicted Existence")
    plt.ylabel("Count")
    plt.legend(title="Actual")
    plt.show()

    incorrect = test_df[test_df["predicted"] != y_test]
    print("\nIncorrect predictions (sample):")
    print(incorrect[["route", "predicted", "existed_now", "degree_u", "degree_v"]].head())
