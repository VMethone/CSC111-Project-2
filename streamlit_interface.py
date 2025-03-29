import streamlit as st
import pandas as pd
import networkx as nx
from main import (
    load_dataset, filter_by_period, build_graph, build_airport_labels,
    load_airport_coordinates_from_us_airports, show_shortest_path,
    draw_network, plot_route_on_map
)
from induction_prediction import main as run_prediction
import os

st.set_page_config(layout="wide")
st.title("Airline Route & Fare Explorer (2018–2025)")

tab1, tab2, tab3 = st.tabs(["Route Finder", "Fare Trends", "Future Predictions"])

@st.cache_data
def get_data():
    df = load_dataset("filled_geocoded_airline_data.csv")
    coords = load_airport_coordinates_from_us_airports("us-airports.csv")
    return df, coords

with st.spinner("Loading data..."):
    df, coord_map = get_data()

label_map = build_airport_labels(df, "airport_1", "city1")

# === 辅助函数：路径总结 ===
def summarize_route(G, route, label_map):
    total_cost = 0
    total_distance = 0
    segments = []

    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        edge = G.get_edge_data(u, v) or G.get_edge_data(v, u)
        if edge:
            total_cost += edge.get("weight", 0)
            total_distance += edge.get("distance", 0)
            segments.append((u, v, edge.get("weight", 0), edge.get("distance", 0)))
        else:
            segments.append((u, v, "N/A", "N/A"))

    names = [label_map.get(code, code) for code in route]
    return total_cost, total_distance, names, segments

# === Tab 1 ===
with tab1:
    st.subheader("Find the Best Route")
    col1, col2 = st.columns(2)

    with col1:
        origin = st.text_input("Origin Airport Code", "JFK").upper()
        dest = st.text_input("Destination Airport Code", "LAX").upper()

    with col2:
        allow_transfers = st.checkbox("Allow Transfers", value=True)
        max_budget = st.number_input("Max Budget (USD, optional)", min_value=0.0, value=500.0)

    period = st.selectbox("Select Time Period", ["Pre-pandemic (2018–2020)", "During-pandemic (2021–2022)", "Post-pandemic (2023–2024)"])
    period_map = {
        "Pre-pandemic (2018–2020)": (2018, 2020),
        "During-pandemic (2021–2022)": (2021, 2022),
        "Post-pandemic (2023–2024)": (2023, 2024)
    }

    if st.button("Find Route"):
        start_year, end_year = period_map[period]
        df_period = filter_by_period(df, start_year, end_year)
        G_fare = build_graph(df_period, "airport_1", "airport_2", "fare")

        st.markdown("### Fare-based Shortest Route")
        route = show_shortest_path(G_fare, origin, dest, allow_transfers, max_cost=max_budget, unit="USD")

        if route and len(route) >= 2:
            plot_route_on_map(route, coord_map, f"Fare-Optimized Route ({period})")
            cost, dist, name_list, segs = summarize_route(G_fare, route, label_map)
            st.markdown("#### Route Summary")
            st.write(" → ".join(name_list))
            st.write(f"**Total Fare:** ${cost:.2f}")
            st.write(f"**Total Distance:** {int(dist)} miles")
            st.write(f"**Stops:** {len(route) - 2 if len(route) > 2 else 0}")
            seg_df = pd.DataFrame(segs, columns=["From", "To", "Fare", "Distance"])
            st.dataframe(seg_df)
        else:
            st.warning("No route found for the selected airports and time period.")

        st.markdown("### Fare Graph Network")
        draw_network(G_fare, f"Fare Graph: {period}", label_map)

# === Tab 2 ===
with tab2:
    st.subheader("Fare Trend and Correlation Analysis")
    df_pre = filter_by_period(df, 2018, 2020)
    df_during = filter_by_period(df, 2021, 2022)
    df_post = filter_by_period(df, 2023, 2024)

    st.markdown("#### Average Fares:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pre-pandemic Avg Fare", f"${df_pre['fare'].mean():.2f}")
    col2.metric("During-pandemic Avg Fare", f"${df_during['fare'].mean():.2f}")
    col3.metric("Post-pandemic Avg Fare", f"${df_post['fare'].mean():.2f}")

    st.markdown("#### Correlation Between Fare and Distance")
    corr_pre = df_pre[["fare", "nsmiles"]].corr().iloc[0, 1]
    corr_post = df_post[["fare", "nsmiles"]].corr().iloc[0, 1]
    st.write(f"**Pre-pandemic correlation:** {corr_pre:.3f}")
    st.write(f"**Post-pandemic correlation:** {corr_post:.3f}")

# === Tab 3 ===
with tab3:
    st.subheader("Load Prediction for 2025")
    quarter = st.selectbox("Select 2025 Quarter", ["Q1", "Q2", "Q3", "Q4"])
    filename = f"prediction_2025{quarter}.csv"

    if "pred_df" not in st.session_state:
        st.session_state.pred_df = None
        st.session_state.G_pred = None

    if st.button("Load Prediction File"):
        if os.path.exists(filename):
            pred_df = pd.read_csv(filename)
            pred_df = pred_df[pred_df["fare"] > 0]

            G_pred = build_graph(pred_df, "airport_1", "airport_2", "fare")
            for _, row in pred_df.iterrows():
                u, v = row["airport_1"], row["airport_2"]
                if G_pred.has_edge(u, v):
                    G_pred[u][v]["distance"] = row.get("distance", 0)

            st.session_state.pred_df = pred_df
            st.session_state.G_pred = G_pred
            st.success(f"Loaded prediction data for 2025 {quarter}")
        else:
            st.error(f"Prediction file '{filename}' not found.")

    if st.session_state.pred_df is not None:
        pred_df = st.session_state.pred_df
        G_pred = st.session_state.G_pred

        st.dataframe(pred_df.head(50))
        label_map_pred = build_airport_labels(pred_df, "airport_1", "city1")

        st.markdown("#### Available Airports in Prediction Graph")
        st.text(", ".join(list(G_pred.nodes())[:30]) + " ...")

        st.markdown("### Visualize Prediction Graph")
        draw_network(G_pred, f"Predicted Fare Graph ({quarter})", label_map_pred)

        st.markdown("### Try Route Search on Prediction")
        pred_origin = st.text_input("Origin (Prediction)", "JFK", key="pred_origin").upper()
        pred_dest = st.text_input("Destination (Prediction)", "LAX", key="pred_dest").upper()

        if st.button("Show Predicted Route"):
            try:
                route = show_shortest_path(G_pred, pred_origin, pred_dest, allow_transfers=True, unit="USD")
                if route and len(route) >= 2:
                    plot_route_on_map(route, coord_map, f"Predicted Route ({quarter})")
                    cost, dist, name_list, segs = summarize_route(G_pred, route, label_map_pred)
                    st.markdown("#### Route Summary")
                    st.write(" → ".join(name_list))
                    st.write(f"**Total Fare:** ${cost:.2f}")
                    st.write(f"**Total Distance:** {int(dist)} miles")
                    st.write(f"**Stops:** {len(route) - 2 if len(route) > 2 else 0}")

                    if cost < 30 or dist == 0:
                        st.warning("Predicted fare or distance may be invalid. This route may not be realistic.")

                    seg_df = pd.DataFrame(segs, columns=["From", "To", "Fare", "Distance"])
                    st.dataframe(seg_df)
                else:
                    st.warning("No predicted route found between the selected airports.")
            except ValueError as e:
                st.error(f"Route search failed: {e}")
