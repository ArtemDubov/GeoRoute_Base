"""Streamlit-фронтенд с построением оптимального маршрута.

Возможности:
- поиск адресов через Nominatim / OpenStreetMap;
- список из нескольких точек (первая — депо);
- авто-подбор радиуса графа по габаритам точек;
- загрузка дорожного графа OSMnx с кэшем;
- три критерия: расстояние, время, комфорт;
- настройка коэффициентов комфорта по типам дорог;
- кратчайшие пути через scipy.sparse.csgraph.dijkstra;
- TSP: полный перебор (N <= 7) или жадный алгоритм;
- отрисовка маршрута и связок к адресам на карте.

Управление — в боковой панели (st.sidebar), карта — в основной области.
"""

from __future__ import annotations

import time
from html import escape
from itertools import permutations

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import streamlit as st
from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from streamlit_folium import st_folium


ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.timeout = 60


APP_TITLE = "RouteOptima — построение оптимальных маршрутов"
COUNTRY_CODES = "ru"
DEFAULT_CENTER = [55.751244, 37.618423]
DEFAULT_ZOOM = 10
POINT_ZOOM = 13
SEARCH_LIMIT = 5
NOMINATIM_DELAY_SECONDS = 1.1

MIN_GRAPH_RADIUS = 2000
MAX_GRAPH_RADIUS = 20000
GRAPH_RADIUS_MARGIN = 1500

FULL_BRUTEFORCE_LIMIT = 7

DEFAULT_SPEED = {
    "motorway": 110, "trunk": 90, "primary": 60, "secondary": 50,
    "tertiary": 40, "residential": 30, "living_street": 20,
    "service": 20, "unclassified": 30,
}

COMFORT_COEF = {
    "motorway": 0.7, "trunk": 0.8, "primary": 0.85, "secondary": 0.9,
    "tertiary": 1.0, "residential": 1.2, "living_street": 1.4,
    "service": 1.5, "unclassified": 1.3,
}


def init_session_state() -> None:
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "points" not in st.session_state:
        st.session_state.points = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "route_data" not in st.session_state:
        st.session_state.route_data = None
    if "comfort_coefs" not in st.session_state:
        st.session_state.comfort_coefs = COMFORT_COEF.copy()


@st.cache_resource
def get_geolocator() -> Nominatim:
    return Nominatim(user_agent="student_route_planner_streamlit")


@st.cache_data(show_spinner=False, ttl=60 * 60)
def search_address(query: str) -> tuple[list[dict], str | None]:
    geolocator = get_geolocator()
    try:
        time.sleep(NOMINATIM_DELAY_SECONDS)
        locations = geolocator.geocode(
            query,
            exactly_one=False,
            limit=SEARCH_LIMIT,
            country_codes=COUNTRY_CODES,
            language="ru",
            addressdetails=True,
            timeout=10,
        )
    except (GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable) as error:
        return [], str(error)

    if not locations:
        return [], None

    results = []
    for location in locations:
        results.append({
            "address": location.address,
            "lat": float(location.latitude),
            "lon": float(location.longitude),
        })
    return results, None


def compute_center_and_radius(points: list) -> tuple[float, float, int]:
    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2

    max_dist = 0.0
    for p in points:
        dy = (p["lat"] - center_lat) * 111_320
        dx = (p["lon"] - center_lon) * 111_320 * np.cos(np.radians(center_lat))
        d = float(np.hypot(dx, dy))
        if d > max_dist:
            max_dist = d

    radius = int(max(MIN_GRAPH_RADIUS, min(MAX_GRAPH_RADIUS, max_dist + GRAPH_RADIUS_MARGIN)))
    return float(center_lat), float(center_lon), radius


@st.cache_resource(show_spinner=False)
def load_graph(center_lat: float, center_lon: float, dist: int):
    G = ox.graph_from_point(
        (center_lat, center_lon), dist=dist, network_type="drive", simplify=True,
    )
    G_undir = ox.convert.to_undirected(G)
    G_proj = ox.project_graph(G_undir)

    nodes = list(G_proj.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}

    return G, G_undir, G_proj, node_to_idx, idx_to_node


def parse_maxspeed(value):
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0]
    try:
        return float(str(value).split()[0])
    except (ValueError, IndexError):
        return None


def build_weight_matrices(G_proj: nx.MultiGraph, node_to_idx: dict, comfort_coefs: dict | None = None):
    nodes = list(G_proj.nodes())
    n = len(nodes)

    coefs = COMFORT_COEF.copy()
    if comfort_coefs:
        coefs.update(comfort_coefs)

    rows, cols = [], []
    dist_data, time_data, comfort_data = [], [], []

    for u, v, d in G_proj.edges(data=True):
        length = d.get("length", 0)
        if length <= 0:
            continue
        highway = d.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0]

        speed_kmh = parse_maxspeed(d.get("maxspeed"))
        if speed_kmh is None:
            speed_kmh = DEFAULT_SPEED.get(highway, 30)
        speed_ms = speed_kmh / 3.6

        coef = coefs.get(highway, 1.0)

        i, j = node_to_idx[u], node_to_idx[v]
        rows.extend([i, j])
        cols.extend([j, i])
        dist_data.extend([length, length])
        time_data.extend([length / speed_ms, length / speed_ms])
        comfort_data.extend([length * coef, length * coef])

    shape = (n, n)
    matrices = {
        "distance": coo_matrix((dist_data, (rows, cols)), shape=shape).tocsr(),
        "time":     coo_matrix((time_data, (rows, cols)), shape=shape).tocsr(),
        "comfort":  coo_matrix((comfort_data, (rows, cols)), shape=shape).tocsr(),
    }
    return matrices


def shortest_paths_matrix(A, point_indices: list):
    all_distances, all_predecessors = dijkstra(
        csgraph=A, directed=False, indices=point_indices, return_predecessors=True,
    )
    n_points = len(point_indices)
    dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            dist_matrix[i, j] = all_distances[i, point_indices[j]]
    return dist_matrix, all_predecessors


def reconstruct_path(predecessors_row, source: int, target: int) -> list:
    path = []
    current = target
    while current != -9999 and current != source:
        path.append(current)
        current = predecessors_row[current]
    if current == source:
        path.append(source)
        return path[::-1]
    return []


def tsp_bruteforce(dist_matrix: np.ndarray, depot: int = 0):
    n = len(dist_matrix)
    other = [i for i in range(n) if i != depot]
    best_order, best_length = None, float("inf")
    for perm in permutations(other):
        order = (depot,) + perm + (depot,)
        length = sum(dist_matrix[order[k], order[k + 1]] for k in range(len(order) - 1))
        if length < best_length:
            best_length = length
            best_order = order
    return best_order, best_length


def tsp_greedy(dist_matrix: np.ndarray, depot: int = 0):
    n = len(dist_matrix)
    visited = [depot]
    unvisited = set(range(n)) - {depot}
    current = depot
    while unvisited:
        next_node = min(unvisited, key=lambda j: dist_matrix[current, j])
        visited.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    visited.append(depot)
    length = sum(dist_matrix[visited[k], visited[k + 1]] for k in range(len(visited) - 1))
    return visited, length


def build_full_route(visit_order, point_indices, all_predecessors, idx_to_node):
    full_path = []
    for k in range(len(visit_order) - 1):
        i, j = visit_order[k], visit_order[k + 1]
        segment = reconstruct_path(all_predecessors[i], point_indices[i], point_indices[j])
        if not segment:
            continue
        if full_path and full_path[-1] == segment[0]:
            segment = segment[1:]
        full_path.extend(segment)
    return [idx_to_node[i] for i in full_path]


def create_map(points, route_coords=None, snap_segments=None):
    if points:
        center = [points[0]["lat"], points[0]["lon"]]
        zoom = POINT_ZOOM
    else:
        center = DEFAULT_CENTER
        zoom = DEFAULT_ZOOM

    map_object = folium.Map(
        location=center, zoom_start=zoom, tiles="OpenStreetMap",
        control_scale=True, attribution_control=False,
    )

    if route_coords:
        folium.PolyLine(
            locations=route_coords, color="blue", weight=4, opacity=0.85,
            tooltip="Маршрут",
        ).add_to(map_object)

    if snap_segments:
        for seg in snap_segments:
            folium.PolyLine(
                locations=seg, color="gray", weight=2, opacity=0.7,
                dash_array="5,5", tooltip="Связка к дороге",
            ).add_to(map_object)

    for i, point in enumerate(points):
        color = "green" if i == 0 else "red"
        popup_html = (
            f"<b>Точка {i+1}</b><br>"
            f"{escape(point['address'])}<br>"
            f"{point['lat']:.6f}, {point['lon']:.6f}"
        )
        folium.Marker(
            location=[point["lat"], point["lon"]],
            tooltip="Депо" if i == 0 else f"Точка {i+1}",
            popup=folium.Popup(popup_html, max_width=320),
            icon=folium.Icon(color=color, icon="map-marker"),
        ).add_to(map_object)

    return map_object


def format_result(result: dict) -> str:
    return f"{result['address']} | {result['lat']:.5f}, {result['lon']:.5f}"


def build_route(points, criterion, comfort_coefs=None, progress_cb=None):
    center_lat, center_lon, radius = compute_center_and_radius(points)
    if progress_cb:
        progress_cb(f"Загружаю граф: центр ({center_lat:.4f}, {center_lon:.4f}), радиус {radius} м")

    G, G_undir, G_proj, node_to_idx, idx_to_node = load_graph(center_lat, center_lon, radius)

    if progress_cb:
        progress_cb(f"Граф: {len(G_proj.nodes())} узлов, {len(G_proj.edges())} рёбер. Считаю матрицы...")

    matrices = build_weight_matrices(G_proj, node_to_idx, comfort_coefs=comfort_coefs)
    A = matrices[criterion]

    if progress_cb:
        progress_cb("Ищу ближайшие узлы к точкам...")

    transformer = Transformer.from_crs("EPSG:4326", G_proj.graph["crs"], always_xy=True)

    point_indices = []
    snap_segments = []
    for point in points:
        x_proj, y_proj = transformer.transform(point["lon"], point["lat"])
        node_osm = ox.nearest_nodes(G_proj, X=x_proj, Y=y_proj)
        point_indices.append(node_to_idx[node_osm])

        node_data = G_proj.nodes[node_osm]
        node_latlon = transformer.transform(node_data["x"], node_data["y"], direction="INVERSE")
        snap_segments.append([
            (point["lat"], point["lon"]),
            (node_latlon[1], node_latlon[0]),
        ])

    if progress_cb:
        progress_cb("Запускаю Дейкстру и TSP...")

    dist_matrix, all_predecessors = shortest_paths_matrix(A, point_indices)

    unreachable = []
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j and np.isinf(dist_matrix[i, j]):
                unreachable.append((i + 1, j + 1))

    n = len(points)
    if n <= FULL_BRUTEFORCE_LIMIT:
        order, length = tsp_bruteforce(dist_matrix, depot=0)
        method = "полный перебор"
    else:
        order, length = tsp_greedy(dist_matrix, depot=0)
        method = "жадный алгоритм"

    route_osm = build_full_route(order, point_indices, all_predecessors, idx_to_node)
    route_coords = [
        (G.nodes[node_id]["y"], G.nodes[node_id]["x"]) for node_id in route_osm
    ]

    dm_dist, _ = shortest_paths_matrix(matrices["distance"], point_indices)
    dm_time, _ = shortest_paths_matrix(matrices["time"], point_indices)
    total_dist_m = sum(dm_dist[order[k], order[k + 1]] for k in range(len(order) - 1))
    total_time_s = sum(dm_time[order[k], order[k + 1]] for k in range(len(order) - 1))

    return {
        "order": order,
        "length": length,
        "method": method,
        "route_coords": route_coords,
        "snap_segments": snap_segments,
        "criterion": criterion,
        "radius": radius,
        "graph_nodes": len(G_proj.nodes()),
        "graph_edges": len(G_proj.edges()),
        "unreachable": unreachable,
        "total_distance_m": total_dist_m,
        "total_time_s": total_time_s,
    }


def render_sidebar() -> None:
    """Всё управление в боковой панели."""
    with st.sidebar:
        st.header("Поиск адреса")

        with st.form("address_search_form", clear_on_submit=False):
            query = st.text_input("Адрес", placeholder="Например: Невский проспект")
            search_submitted = st.form_submit_button("Найти", use_container_width=True)

        if search_submitted:
            query = query.strip()
            st.session_state.last_query = query
            if not query:
                st.warning("Введите адрес для поиска.")
                st.session_state.search_results = []
            else:
                with st.spinner("Ищу адрес..."):
                    results, error = search_address(query)
                st.session_state.search_results = results
                if error:
                    st.error(f"Ошибка поиска: {error}")
                elif not results:
                    st.info("Ничего не найдено. Уточните адрес (лучше без номера дома).")

        if st.session_state.search_results:
            st.divider()
            st.subheader("Результаты")
            selected_index = st.selectbox(
                "Выберите адрес",
                options=range(len(st.session_state.search_results)),
                format_func=lambda index: format_result(st.session_state.search_results[index]),
            )
            if st.button("Добавить точку", type="primary", use_container_width=True):
                result = st.session_state.search_results[selected_index]
                st.session_state.points.append({
                    "address": result["address"],
                    "lat": result["lat"],
                    "lon": result["lon"],
                })
                st.session_state.route_data = None
                st.success(f"Добавлено. Всего точек: {len(st.session_state.points)}")

        st.divider()
        st.subheader("Список точек")

        if st.session_state.points:
            for i, point in enumerate(st.session_state.points):
                label = "Депо" if i == 0 else f"Точка {i+1}"
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    st.markdown(f"**{label}**  \n{point['address']}")
                    st.caption(f"{point['lat']:.5f}, {point['lon']:.5f}")
                with col2:
                    if st.button("✕", key=f"del_{i}"):
                        st.session_state.points.pop(i)
                        st.session_state.route_data = None
                        st.rerun()

            if st.button("Очистить всё", use_container_width=True):
                st.session_state.points = []
                st.session_state.route_data = None
                st.rerun()
        else:
            st.caption("Точки пока не добавлены.")

        st.divider()
        st.subheader("Построение маршрута")

        criterion = st.radio(
            "Критерий",
            ["distance", "time", "comfort"],
            format_func=lambda c: {"distance": "Расстояние",
                                   "time": "Время",
                                   "comfort": "Комфорт"}[c],
            horizontal=False,
        )

        # Блок коэффициентов комфорта
        if criterion == "comfort":
            with st.expander("Коэффициенты комфорта", expanded=False):
                coefs = {}
                for road_type, default_val in COMFORT_COEF.items():
                    coefs[road_type] = st.number_input(
                        label=road_type,
                        min_value=0.1, max_value=5.0,
                        value=float(st.session_state.comfort_coefs.get(road_type, default_val)),
                        step=0.1,
                        key=f"coef_{road_type}",
                    )
                if st.button("Применить коэффициенты", use_container_width=True):
                    st.session_state.comfort_coefs = coefs
                    st.session_state.route_data = None
                    st.success("Коэффициенты обновлены.")

        can_build = len(st.session_state.points) >= 2
        if not can_build:
            st.caption("Добавьте минимум две точки.")

        if st.button("Построить маршрут", type="primary",
                     disabled=not can_build, use_container_width=True):
            status = st.empty()
            def _cb(msg):
                status.info(msg)
            try:
                route_data = build_route(
                    st.session_state.points,
                    criterion,
                    comfort_coefs=st.session_state.get("comfort_coefs"),
                    progress_cb=_cb,
                )
                st.session_state.route_data = route_data
                status.success("Маршрут построен.")
            except Exception as e:
                status.error(f"Ошибка: {e}")


def render_main_area() -> None:
    """Карта и описание маршрута в основной области."""
    rd = st.session_state.route_data

    st.title(APP_TITLE)

    if rd:
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Длина", f"{rd['total_distance_m']/1000:.2f} км")
        with col_b:
            st.metric("Время в пути", f"{rd['total_time_s']/60:.1f} мин")
        with col_c:
            st.metric("Метод", rd["method"])
        with col_d:
            st.metric("Точек", len(st.session_state.points))

        st.caption(
            f"Критерий: {rd['criterion']}  ·  "
            f"Граф: {rd['graph_nodes']} узлов, {rd['graph_edges']} рёбер, радиус {rd['radius']} м"
        )

        if rd["unreachable"]:
            st.warning(f"Недостижимые пары точек: {rd['unreachable']}")

        order_labels = ["Депо" if idx == 0 else f"Точка {idx+1}" for idx in rd["order"]]
        st.write("**Порядок посещения:** " + " → ".join(order_labels))

    map_object = create_map(
        st.session_state.points,
        route_coords=rd["route_coords"] if rd else None,
        snap_segments=rd["snap_segments"] if rd else None,
    )
    st_folium(map_object, height=650, use_container_width=True, returned_objects=[])


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide",
                       initial_sidebar_state="expanded")
    init_session_state()

    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()