import streamlit as st
import pandas as pd
from pathlib import Path
import folium
import xml.etree.ElementTree as ET
from streamlit_folium import st_folium
from routing.algorithms.dijkstra_dinov2_reduced_both import (load_graph, dijkstra_dinov2)
from PIL import Image
from routing.weather import (get_current_weather, get_weather_description, check_weather_warnings)

# Project data folder
DATA_DIR = Path(__file__).parent / "data"
PHOTO_DIR = DATA_DIR / "entrance_photos"

KML_PATH = DATA_DIR / "sidewalk_network.kml"


def load_sidewalk_geometry():
    """Load sidewalk segment coordinates from the KML file."""

    tree = ET.parse(KML_PATH)
    root = tree.getroot()

    sidewalk_segments = {}

    for placemark in root.iter():
        if not placemark.tag.endswith("Placemark"):
            continue

        name_element = placemark.find(".//{*}name")

        if name_element is None or not name_element.text:
            continue

        segment_id = name_element.text.strip()

        coordinate_elements = placemark.findall(
            ".//{*}LineString/{*}coordinates"
        )

        if not coordinate_elements:
            continue

        coordinates = []

        for element in coordinate_elements:
            for point in element.text.strip().split():
                values = point.split(",")

                longitude = float(values[0])
                latitude = float(values[1])

                coordinates.append([latitude, longitude])

        if coordinates:
            sidewalk_segments[segment_id] = coordinates

    return sidewalk_segments

def load_node_coordinates(sidewalk_segments):
    """Map each routing node to its geographic coordinates."""

    segments_df = pd.read_csv(DATA_DIR / "segments.csv")
    node_coordinates = {}

    for _, row in segments_df.iterrows():
        segment_id = row["segment_id"]
        start_node = row["start_point"]
        end_node = row["end_point"]

        coordinates = sidewalk_segments.get(segment_id)

        if not coordinates:
            continue

        node_coordinates.setdefault(start_node, []).append(coordinates[0])
        node_coordinates.setdefault(end_node, []).append(coordinates[-1])

    # Average coordinates where multiple sidewalk segments meet
    return {
        node: [
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points)
        ]
        for node, points in node_coordinates.items()
    }

def create_route_map(
    sidewalk_segments,
    route_segments,
    node_coordinates,
    start_node_id,
    destination_node_id
):
    """Display the sidewalk network, selected route, and start/end markers."""

    all_coordinates = [
        coordinate
        for coordinates in sidewalk_segments.values()
        for coordinate in coordinates
    ]

    center_lat = sum(point[0] for point in all_coordinates) / len(all_coordinates)
    center_lon = sum(point[1] for point in all_coordinates) / len(all_coordinates)

    campus_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles="OpenStreetMap"
    )

    # Draw the entire sidewalk network in gray
    for segment_id, coordinates in sidewalk_segments.items():
        folium.PolyLine(
            coordinates,
            color="gray",
            weight=4,
            opacity=0.7,
            tooltip=segment_id
        ).add_to(campus_map)

    # Highlight the selected route in red
    for segment_id in route_segments:
        if segment_id in sidewalk_segments:
            folium.PolyLine(
                sidewalk_segments[segment_id],
                color="red",
                weight=7,
                opacity=1,
                tooltip=f"Selected route: {segment_id}"
            ).add_to(campus_map)

    # Add starting-point marker
    if start_node_id in node_coordinates:
        folium.Marker(
            node_coordinates[start_node_id],
            tooltip="Starting Point",
            popup="Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(campus_map)

    # Add destination marker
    if destination_node_id in node_coordinates:
        folium.Marker(
            node_coordinates[destination_node_id],
            tooltip="Destination",
            popup="Destination",
            icon=folium.Icon(color="red", icon="flag")
        ).add_to(campus_map)

    route_coordinates = [
        coordinate
        for segment_id in route_segments
        if segment_id in sidewalk_segments
        for coordinate in sidewalk_segments[segment_id]
    ]

    if route_coordinates:
        campus_map.fit_bounds(
            route_coordinates,
            padding=(40,40)
        )
    return campus_map

def show_entrance_preview(entrance_row, title):
    """Display a rotated preview of the selected entrance."""

    with st.expander(f"Preview {title}"):
        photo_path = PHOTO_DIR / entrance_row["photo_filename"]

        if photo_path.exists():
            image = Image.open(photo_path)
            if entrance_row["entrance_id"] == "E015":
                rotated_image = image.rotate(0, expand=True)
            else:
                rotated_image = image.rotate(-90, expand=True)

            st.image(rotated_image, width=350)
        else:
            st.warning("Entrance photo not found.")

        if entrance_row["entrance_accessible"] == "Yes":
            st.success("Marked accessible")
        else:
            st.warning("Not marked accessible")

st.set_page_config(
    page_title="AccessAI",
    page_icon="♿",
    layout="centered"
)

@st.cache_data
def load_data():
    buildings = pd.read_csv(DATA_DIR / "buildings.csv")
    entrances = pd.read_csv(DATA_DIR / "entrances.csv")
    nodes = pd.read_csv(DATA_DIR / "nodes.csv")
    return buildings, entrances, nodes

buildings, entrances, nodes = load_data()

st.title("AccessAI")
st.subheader("Accessible Campus Route Finder")
weather = get_current_weather()
weather_description = get_weather_description(weather["weather_code"])

st.caption(
    f"Current campus weather: {weather_description} | "
    f"{weather['temperature_2m']}°F"
)
st.write(
    "Choose your starting point and destination to find a route "
    "that better fits your accessibility needs."
)

st.header("Mobility Profile")
mobility_profile = st.selectbox(
    "Choose your mobility profile",
    [
        "Walking cane",
        "Walker",
        "Mobility scooter",
        "Manual wheelchair",
        "Motorized wheelchair"
    ]
)

if mobility_profile == "Walking cane":
    use_stairs = st.radio(
        "Are you comfortable using stairs?",
        ["Yes, stairs are okay", "No, avoid stairs"],
    ) == "Yes, stairs are okay"
else:
    use_stairs = False

@st.cache_resource
def get_graph():
    return load_graph()

graph = get_graph()

st.divider()

# Starting point
st.header("Starting Point")

start_building = st.selectbox(
    "Choose your starting building",
    buildings["building_name"].tolist(),
    key="start_building"
)

start_building_id = buildings.loc[
    buildings["building_name"] == start_building,
    "building_id"
].iloc[0]

start_entrances = entrances[
    entrances["building_id"] == start_building_id
].copy()

start_entrances["display_name"] = start_entrances.apply(
    lambda row: (
        row["entrance_name"]
        if row["entrance_accessible"] == "Yes"
        else f'{row["entrance_name"]} (Not marked accessible)'
    ),
    axis=1
)

start_entrance_name = st.selectbox(
    "Choose your starting entrance",
    start_entrances["display_name"].tolist(),
    key="start_entrance"
)

start_entrance_id = start_entrances.loc[
    start_entrances["display_name"] == start_entrance_name,
    "entrance_id"
].iloc[0]

selected_start_entrance = start_entrances.loc[
    start_entrances["display_name"] == start_entrance_name
].iloc[0]

show_entrance_preview(selected_start_entrance, "Starting Entrance")

# Destination
st.divider()
st.header("Destination")

destination_building = st.selectbox(
    "Choose your destination building",
    buildings["building_name"].tolist(),
    index=1 if len(buildings) > 1 else 0,
    key="destination_building"
)

destination_building_id = buildings.loc[
    buildings["building_name"] == destination_building,
    "building_id"
].iloc[0]

destination_entrances = entrances[
    entrances["building_id"] == destination_building_id
].copy()

destination_entrances["display_name"] = destination_entrances.apply(
    lambda row: (
        row["entrance_name"]
        if row["entrance_accessible"] == "Yes"
        else f'{row["entrance_name"]} (Not marked accessible)'
    ),
    axis=1
)

destination_entrance_name = st.selectbox(
    "Choose your destination entrance",
    destination_entrances["display_name"].tolist(),
    key="destination_entrance"
)

destination_entrance_id = destination_entrances.loc[
    destination_entrances["display_name"] == destination_entrance_name,
    "entrance_id"
].iloc[0]

selected_destination_entrance = destination_entrances.loc[
    destination_entrances["display_name"] == destination_entrance_name
].iloc[0]

show_entrance_preview(selected_destination_entrance, "Destination Entrance")

# Convert selected entrances to internal routing node IDs
start_node = nodes.loc[
    nodes["entrance_id"] == start_entrance_id,
    "node_id"
]

destination_node = nodes.loc[
    nodes["entrance_id"] == destination_entrance_id,
    "node_id"
]

st.divider()


if "route_result" not in st.session_state:
    st.session_state.route_result = None

if "route_info" not in st.session_state:
    st.session_state.route_info = None


if st.button("Find My Route", type="primary", use_container_width=True):

    if start_node.empty or destination_node.empty:
        st.session_state.route_result = None
        st.session_state.route_info = None
        st.error("Could not find a routing connection for one of your selected entrances.")

    elif start_node.iloc[0] == destination_node.iloc[0]:
        st.session_state.route_result = None
        st.session_state.route_info = None
        st.warning("Please select different starting and destination entrances.")

    else:
        start_node_id = start_node.iloc[0]
        destination_node_id = destination_node.iloc[0]

        with st.spinner("Finding your route..."):
            result = dijkstra_dinov2(
                graph,
                start_node_id,
                destination_node_id,
                mobility_profile,
                use_stairs
            )

        st.session_state.route_result = result

        if result is not None:
            st.session_state.route_info = {
                "mobility_profile": mobility_profile,
                "start_building": start_building,
                "start_entrance_name": start_entrance_name,
                "destination_building": destination_building,
                "destination_entrance_name": destination_entrance_name,
            }
        else:
            st.session_state.route_info = None


# Display the saved route OUTSIDE the button block
result = st.session_state.route_result
info = st.session_state.route_info

if result is not None and info is not None:
    segments_df = pd.read_csv(DATA_DIR / "segments.csv")

    route_surfaces = segments_df[
        segments_df["segment_id"].isin(result["segments"])
    ]["surface"].dropna().tolist()

    weather_warnings = check_weather_warnings(weather, route_surfaces)

    for warning in weather_warnings:
        st.warning(warning)

    st.divider()
    st.success("Personalized route found!")

    st.write("**Mobility profile:**", info["mobility_profile"])

    st.metric(
        "Total walking distance",
        f"{result['distance_m']:.2f} meters"
    )

    st.metric(
        "Personalized routing cost",
        f"{result['routing_cost']:.2f}"
    )

    st.write(
        "**Starting point:**",
        info["start_building"],
        "—",
        info["start_entrance_name"]
    )

    st.write(
        "**Destination:**",
        info["destination_building"],
        "—",
        info["destination_entrance_name"]
    )

    st.write("Your route has", len(result["segments"]), "sidewalk segments.")

    with st.expander("View Route Details"):
        st.write("**Node sequence:**")
        st.write(" → ".join(result["nodes"]))

        st.write("**Segment sequence:**")
        st.write(" → ".join(result["segments"]))

        st.write("**Individual segment distances:**")
        for segment_id in result["segments"]:
            segment = segments_df[
                segments_df["segment_id"] == segment_id
            ].iloc[0]

            st.write(
                f"{segment_id}: {segment['start_point']} → "
                f"{segment['end_point']} | {segment['distance_m']} m"
            )
    st.subheader("Your Route on the Campus Map")

    sidewalk_segments = load_sidewalk_geometry()

    node_coordinates = load_node_coordinates(sidewalk_segments)

    route_map = create_route_map(
        sidewalk_segments,
        result["segments"],
        node_coordinates,
        result["nodes"][0],
        result["nodes"][-1]
    )

    st_folium(
        route_map,
        width=750,
        height=550
    )

elif st.session_state.route_result is None and st.session_state.route_info is None:
    pass
