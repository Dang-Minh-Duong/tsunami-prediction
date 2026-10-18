from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from inference import predict_tsunami

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "earthquake_data_tsunami.csv"

st.set_page_config(
    page_title="Tsunami Risk Prediction",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Tsunami Risk Prediction")
st.caption(
    "Machine Learning Dashboard | Earthquake Risk Classification"
)

st.subheader("Historical Tsunami Events")
if not DATA_PATH.is_file():
    st.error(f"Historical event data was not found: {DATA_PATH}")
    st.stop()


@st.cache_data
def load_tsunami_events(path: str) -> pd.DataFrame:
    events = pd.read_csv(path)
    required_columns = {"tsunami", "latitude", "longitude", "magnitude", "Year"}
    missing_columns = required_columns.difference(events.columns)
    if missing_columns:
        raise ValueError(
            "The event dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    return events


events = load_tsunami_events(str(DATA_PATH))
tsunami_events = events.loc[
    (events["tsunami"] == 1)
    & events["latitude"].between(-90, 90)
    & events["longitude"].between(-180, 180)
].dropna(subset=["latitude", "longitude", "magnitude"])

if tsunami_events.empty:
    st.warning("No tsunami events with valid coordinates were found in the dataset.")
else:
    st.caption(
        f"{len(tsunami_events)} recorded tsunami cases from the local dataset. "
        "Select a marker to inspect its earthquake details."
    )
    hover_columns = {
        column: True
        for column in ("Year", "Month", "magnitude", "depth", "sig")
        if column in tsunami_events.columns
    }
    hover_columns["magnitude"] = ":.1f"
    if "depth" in hover_columns:
        hover_columns["depth"] = ":.1f"

    map_figure = px.scatter_geo(
        tsunami_events,
        lat="latitude",
        lon="longitude",
        size="magnitude",
        size_max=12,
        hover_data=hover_columns,
        labels={
            "latitude": "Latitude",
            "longitude": "Longitude",
            "magnitude": "Magnitude",
            "depth": "Depth (km)",
            "sig": "Significance",
            "Year": "Year",
            "Month": "Month",
        },
        color_discrete_sequence=["#0077b6"],
        projection="natural earth",
    )
    map_figure.update_traces(
        marker={"opacity": 0.75, "line": {"width": 0.5, "color": "white"}}
    )
    map_figure.update_geos(
        showland=True,
        landcolor="#f2f4f7",
        showocean=True,
        oceancolor="#e6f4fa",
        showcountries=True,
        countrycolor="#cbd5e1",
    )
    map_figure.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(map_figure, use_container_width=True)

st.subheader("Tsunami Cases by Year")
annual_events = events.loc[events["tsunami"] == 1, ["Year"]].copy()
annual_events["Year"] = pd.to_numeric(annual_events["Year"], errors="coerce")
annual_events = annual_events.dropna(subset=["Year"])

if annual_events.empty:
    st.warning("No tsunami cases with a valid year were found in the dataset.")
else:
    annual_counts = (
        annual_events.groupby("Year")
        .size()
        .rename("Cases")
        .reset_index()
        .sort_values("Year")
    )
    annual_counts["Year"] = annual_counts["Year"].astype(int).astype(str)
    annual_figure = px.bar(
        annual_counts,
        x="Year",
        y="Cases",
        text="Cases",
        labels={"Year": "Year", "Cases": "Number of tsunami cases"},
        color_discrete_sequence=["#0077b6"],
    )
    annual_figure.update_traces(textposition="outside", cliponaxis=False)
    annual_figure.update_layout(
        height=360,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        xaxis={"type": "category"},
        yaxis={"rangemode": "tozero"},
    )
    st.plotly_chart(annual_figure, use_container_width=True)

with st.sidebar:
    st.header("Earthquake Parameters")

    magnitude = st.number_input(
        "Magnitude", min_value=0.0,
        max_value=10.0, value=7.0
    )

    depth = st.number_input(
        "Depth (km)", min_value=0.0, value=20.0
    )

    latitude = st.number_input(
        "Latitude", min_value=-90.0,
        max_value=90.0, value=10.0
    )

    longitude = st.number_input(
        "Longitude", min_value=-180.0,
        max_value=180.0, value=110.0
    )

    cdi = st.number_input("CDI", min_value=0, value=4)
    mmi = st.number_input("MMI", min_value=0, value=4)
    sig = st.number_input("SIG", min_value=0, value=700)
    nst = st.number_input("NST", min_value=0, value=100)
    dmin = st.number_input("DMIN", min_value=0.0, value=1.0)
    gap = st.number_input("GAP", min_value=0.0, value=30.0)
    month = st.selectbox("Month", list(range(1, 13)), index=0)

    predict_button = st.button(
        "Predict Tsunami Risk",
        type="primary",
        use_container_width=True
    )

if predict_button:
    input_data = {
        "magnitude": magnitude,
        "cdi": cdi,
        "mmi": mmi,
        "sig": sig,
        "nst": nst,
        "dmin": dmin,
        "gap": gap,
        "depth": depth,
        "latitude": latitude,
        "longitude": longitude,
        "Month": month,
    }

    try:
        result = predict_tsunami(input_data)

        probability = result["probability"]

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Tsunami Probability",
            f"{probability:.1%}"
        )

        col2.metric(
            "Decision Threshold",
            f"{result['threshold']:.1%}"
        )

        col3.metric(
            "Classification",
            result["label"]
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "value": result["threshold"] * 100
                }
            }
        ))

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "This is a machine-learning classification result, "
            "not an official tsunami warning."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Enter earthquake parameters and click Predict.")