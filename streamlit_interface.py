import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from another_main import (
    load_dataset, filter_by_period, build_graph, build_airport_labels,
    show_shortest_path, plot_full_network_routes
)
import os
from streamlit_folium import st_folium
import folium

st.set_page_config(layout="wide")
st.title("Airline Route & Fare Explorer (2018–2025)")

tab1, tab2, tab3 = st.tabs(["Route Finder", "Fare Trends", "Future Predictions"])

@st.cache_data
def get_data():
    df = load_dataset("Final.csv")
    df.columns = df.columns.str.strip()  # ✅ strip column names
    return df

with st.spinner("Loading data..."):
    df = get_data()

label_map = build_airport_labels(df, "airport_1", "city1")
airport_to_city = label_map
city_to_airport = {v: k for k, v in airport_to_city.items()}

def parse_geocoded(coord_str):
    try:
        lat_str, lon_str = coord_str.strip("()").split(",")
        return float(lat_str), float(lon_str)
    except:
        return None, None

def summarize_route(G, route, label_map):
    total_cost = 0
    total_distance = 0
    segments = []

    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        edge = G.get_edge_data(u, v) or G.get_edge_data(v, u)
        if edge:
            fare = edge.get("weight", 0)
            distance = edge.get("distance") or edge.get("nsmiles", 0)
            try:
                distance = float(distance)
            except:
                distance = 0.0
            total_cost += fare
            total_distance += distance
            segments.append((u, v, fare, distance))
        else:
            segments.append((u, v, "N/A", "N/A"))

    names = [label_map.get(code, code) for code in route]
    return total_cost, total_distance, names, segments

def plot_route_on_map(route, coord_map, title="Route Map"):
    missing = [code for code in route if code not in coord_map]
    if missing:
        print("Missing coordinates for:", missing)

    lats = [coord_map[code][0] for code in route if code in coord_map]
    lons = [coord_map[code][1] for code in route if code in coord_map]

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

    st.plotly_chart(fig)

# === Tab 1 ===
with tab1:
    st.subheader("Find the Best Route")
    col1, col2 = st.columns(2)

    with col1:
        origin_city = st.selectbox("Origin City", sorted(city_to_airport.keys()), index=0)
        dest_city = st.selectbox("Destination City", sorted(city_to_airport.keys()), index=10)
        origin = city_to_airport[origin_city]
        dest = city_to_airport[dest_city]

    with col2:
        allow_transfers = st.checkbox("Allow Transfers", value=True)
        max_budget = st.number_input("Max Budget (USD, optional)", min_value=0.0, value=500.0)

    period = st.selectbox("Select Time Period", [
        "Pre-pandemic (2018–2020)",
        "During-pandemic (2021–2022)",
        "Post-pandemic (2023–2024)"
    ])
    period_map = {
        "Pre-pandemic (2018–2020)": (2018, 2020),
        "During-pandemic (2021–2022)": (2021, 2022),
        "Post-pandemic (2023–2024)": (2023, 2024)
    }

    if st.button("Find Route"):
        start_year, end_year = period_map[period]
        df_period = filter_by_period(df, start_year, end_year)
        df_period.columns = df_period.columns.str.strip()

        G_fare = build_graph(df_period, "airport_1", "airport_2", "fare", extra_cols=["distance", "nsmiles"])

        coord_map_fixed = {}
        for _, row in df_period.iterrows():
            if pd.notna(row["Geocoded_City1"]):
                coord_map_fixed[row["airport_1"]] = parse_geocoded(row["Geocoded_City1"])
            if pd.notna(row["Geocoded_City2"]):
                coord_map_fixed[row["airport_2"]] = parse_geocoded(row["Geocoded_City2"])

        if origin not in G_fare or dest not in G_fare:
            st.warning(f"No flights between {origin_city} and {dest_city} in this period.")
        else:
            route = show_shortest_path(G_fare, origin, dest, allow_transfers, max_cost=max_budget, unit="USD")
            if route and len(route) >= 2:
                plot_route_on_map(route, coord_map_fixed, f"Fare-Optimized Route ({period})")
                cost, dist, name_list, segs = summarize_route(G_fare, route, label_map)
                st.markdown("#### Route Summary")
                st.write(" → ".join(name_list))
                st.write(f"**Total Fare:** ${cost:.2f}")
                st.write(f"**Total Distance:** {int(dist)} miles")
                st.write(f"**Stops:** {len(route) - 2 if len(route) > 2 else 0}")
                st.dataframe(pd.DataFrame(segs, columns=["From", "To", "Fare", "Distance"]))
            else:
                st.warning("No route found.")

with tab2:
    st.subheader("Fare Trends and Historical Networks")

    # === Part 1: Fare Trend Over Time ===
    st.markdown("#### 📈 Average Fare per Year")

    df['year'] = df['year'].astype(int)
    yearly_fare = df.groupby("year")["fare"].mean().reset_index()

    st.line_chart(yearly_fare.set_index("year"))

    periods = {
        "Pre-pandemic (2018–2020)": (2018, 2020),
        "During-pandemic (2021–2022)": (2021, 2022),
        "Post-pandemic (2023–2024)": (2023, 2024),
    }

    for label, (start, end) in periods.items():
        key_name = f"show_network_{label}"
        if st.button(f"Show Flight Network: {label}", key=key_name):
            st.session_state["active_network_period"] = label

    if "active_network_period" in st.session_state:
        label = st.session_state["active_network_period"]
        start, end = periods[label]

        df_period = filter_by_period(df, start, end)
        df_period.columns = df_period.columns.str.strip()

        G = build_graph(df_period, "airport_1", "airport_2", "fare", extra_cols=["distance"])

        coord_map = {}
        for _, row in df_period.iterrows():
            if pd.notna(row.get("Geocoded_City1")):
                coord_map[row["airport_1"]] = parse_geocoded(row["Geocoded_City1"])
            if pd.notna(row.get("Geocoded_City2")):
                coord_map[row["airport_2"]] = parse_geocoded(row["Geocoded_City2"])

        plot_full_network_routes(G, coord_map, title=f"{label} Flight Network")


# === Tab 3 ===
with tab3:
    st.subheader("Load Prediction for 2025")
    quarter = st.selectbox("Select 2025 Quarter", ["Q1", "Q2", "Q3", "Q4"])
    filename = f"prediction_2025{quarter}.csv"

    if st.button("Load Prediction File"):
        if os.path.exists(filename):
            pred_df = pd.read_csv(filename)
            pred_df.columns = pred_df.columns.str.strip()
            pred_df = pred_df[pred_df["fare"] > 0]
            G_pred = build_graph(pred_df, "airport_1", "airport_2", "fare", extra_cols=["distance"])
            st.session_state.pred_df = pred_df
            st.session_state.G_pred = G_pred
            st.success(f"Loaded prediction data for 2025 {quarter}")
        else:
            st.error(f"Prediction file '{filename}' not found.")

    if "pred_df" in st.session_state and st.session_state.pred_df is not None:
        pred_df = st.session_state.pred_df
        G_pred = st.session_state.G_pred

        label_map_pred = build_airport_labels(pred_df, "airport_1", "city1")
        city_to_airport_pred = {v: k for k, v in label_map_pred.items()}

        coord_map_pred = {}
        for _, row in pred_df.iterrows():
            if pd.notna(row["latitude1"]) and pd.notna(row["longitude1"]):
                coord_map_pred[row["airport_1"]] = (row["latitude1"], row["longitude1"])
            if pd.notna(row["latitude2"]) and pd.notna(row["longitude2"]):
                coord_map_pred[row["airport_2"]] = (row["latitude2"], row["longitude2"])

        pred_origin_city = st.selectbox("Origin City (Prediction)", sorted(city_to_airport_pred.keys()), key="pred_origin")
        pred_dest_city = st.selectbox("Destination City (Prediction)", sorted(city_to_airport_pred.keys()), key="pred_dest")
        pred_origin = city_to_airport_pred[pred_origin_city]
        pred_dest = city_to_airport_pred[pred_dest_city]

        if st.button("Show Predicted Route"):
            try:
                route = show_shortest_path(G_pred, pred_origin, pred_dest, allow_transfers=True, unit="USD")
                if route and len(route) >= 2:
                    plot_route_on_map(route, coord_map_pred, f"Predicted Route ({quarter})")
                    cost, dist, name_list, segs = summarize_route(G_pred, route, label_map_pred)
                    st.markdown("#### Route Summary")
                    st.write(" → ".join(name_list))
                    st.write(f"**Total Fare:** ${cost:.2f}")
                    st.write(f"**Total Distance:** {int(dist)} miles")
                    st.write(f"**Stops:** {len(route) - 2 if len(route) > 2 else 0}")
                    st.dataframe(pd.DataFrame(segs, columns=["From", "To", "Fare", "Distance"]))
                else:
                    st.warning("No predicted route found.")
            except ValueError as e:
                st.error(str(e))

    if st.button("Show Predicted Flight Network"):
        coord_map_pred = {}
        for _, row in pred_df.iterrows():
            if pd.notna(row["latitude1"]) and pd.notna(row["longitude1"]):
                coord_map_pred[row["airport_1"]] = (row["latitude1"], row["longitude1"])
            if pd.notna(row["latitude2"]) and pd.notna(row["longitude2"]):
                coord_map_pred[row["airport_2"]] = (row["latitude2"], row["longitude2"])

        plot_full_network_routes(G_pred, coord_map_pred, title=f"Predicted Flight Network ({quarter})")
