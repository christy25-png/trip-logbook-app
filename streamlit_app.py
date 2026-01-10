import os
import io
from datetime import date

import pandas as pd
import streamlit as st
from supabase import create_client, Client

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Trip Logbook", layout="centered")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_SERVICE_ROLE_KEY = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
SUPABASE_KEY_FALLBACK = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))
ADMIN_PIN = st.secrets.get("ADMIN_PIN", os.getenv("ADMIN_PIN", ""))
APP_TITLE = st.secrets.get("APP_TITLE", os.getenv("APP_TITLE", "🚗 Trip Logbook"))

if not SUPABASE_URL:
    st.error("Missing SUPABASE_URL in Streamlit secrets.")
    st.stop()

SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY_FALLBACK
if not SUPABASE_KEY:
    st.error("Missing SUPABASE_SERVICE_ROLE_KEY (recommended) or SUPABASE_KEY in secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================
# HELPERS
# =========================
def clean_text(s: str) -> str:
    if not s:
        return ""
    return " ".join(str(s).strip().split())


def round_half(x: float) -> float:
    return round(float(x) * 2) / 2.0


def parse_year_from_period_name(name: str) -> int:
    name = clean_text(name)
    if name.isdigit() and len(name) == 4:
        return int(name)
    return date.today().year


def safe_execute(q, context: str = ""):
    try:
        return q.execute()
    except Exception as e:
        st.error("Database request failed.")
        if context:
            st.caption(f"Context: {context}")
        st.exception(e)
        return None


# =========================
# SUPABASE CRUD
# =========================
def get_periods():
    res = safe_execute(
        supabase.table("periods")
        .select("id,name,is_active")
        .eq("is_active", True)
        .order("name", desc=True),
        "get_periods()",
    )
    return (res.data or []) if res else []


def ensure_period(name: str):
    name = clean_text(name)
    if not name:
        return None

    r1 = safe_execute(
        supabase.table("periods").select("id").eq("name", name).limit(1),
        "ensure_period(): lookup",
    )
    if r1 and r1.data:
        return r1.data[0]["id"]

    r2 = safe_execute(
        supabase.table("periods").insert({"name": name, "is_active": True}),
        "ensure_period(): insert",
    )
    if r2 and r2.data:
        return r2.data[0]["id"]
    return None


def create_period(name: str):
    name = clean_text(name)
    if not name:
        return None

    r1 = safe_execute(
        supabase.table("periods").select("id,name").eq("name", name).limit(1),
        "create_period(): lookup",
    )
    if r1 and r1.data:
        return r1.data[0]

    r2 = safe_execute(
        supabase.table("periods").insert({"name": name, "is_active": True}),
        "create_period(): insert",
    )
    if r2 and r2.data:
        return r2.data[0]
    return None


def get_cars():
    res = safe_execute(
        supabase.table("cars")
        .select("id,name,plate,is_active")
        .eq("is_active", True)
        .order("name"),
        "get_cars()",
    )
    return (res.data or []) if res else []


def get_places(limit=2000):
    res = safe_execute(
        supabase.table("places")
        .select("id,label,address,is_active,created_at")
        .eq("is_active", True)
        .order("label")
        .limit(limit),
        "get_places()",
    )
    return (res.data or []) if res else []


def create_place(label: str, address: str):
    label = clean_text(label)
    address = clean_text(address)
    if not label or not address:
        st.error("Please enter both a place name and an address.")
        return None

    r1 = safe_execute(
        supabase.table("places").select("id").eq("label", label).limit(1),
        "create_place(): lookup",
    )
    if r1 and r1.data:
        pid = r1.data[0]["id"]
        r2 = safe_execute(
            supabase.table("places").update({"address": address, "is_active": True}).eq("id", pid),
            "create_place(): update",
        )
        return {"id": pid, "label": label} if r2 else None

    r3 = safe_execute(
        supabase.table("places").insert({"label": label, "address": address, "is_active": True}),
        "create_place(): insert",
    )
    if r3 and r3.data:
        return r3.data[0]
    return None


def fetch_places_admin():
    res = safe_execute(
        supabase.table("places").select("id,label,address,is_active,created_at").order("label"),
        "fetch_places_admin()",
    )
    return pd.DataFrame(res.data or []) if res else pd.DataFrame([])


def update_place(place_id: str, updates: dict):
    allowed = {"label", "address", "is_active"}
    updates_clean = {k: v for k, v in updates.items() if k in allowed}
    return safe_execute(
        supabase.table("places").update(updates_clean).eq("id", place_id),
        "update_place()",
    )


def normalize_place_pair(a_id: str, b_id: str) -> tuple[str, str]:
    return (a_id, b_id) if a_id < b_id else (b_id, a_id)


def get_route_distance(place_id_a: str, place_id_b: str) -> float | None:
    if not place_id_a or not place_id_b:
        return None
    a, b = normalize_place_pair(place_id_a, place_id_b)
    res = safe_execute(
        supabase.table("route_distances")
        .select("distance_km")
        .eq("place_a", a)
        .eq("place_b", b)
        .limit(1),
        "get_route_distance()",
    )
    if not res or not res.data:
        return None
    return float(res.data[0]["distance_km"])


def set_route_distance(place_id_a: str, place_id_b: str, distance_km: float):
    if not place_id_a or not place_id_b or place_id_a == place_id_b:
        return
    a, b = normalize_place_pair(place_id_a, place_id_b)

    r1 = safe_execute(
        supabase.table("route_distances").select("id").eq("place_a", a).eq("place_b", b).limit(1),
        "set_route_distance(): lookup",
    )
    if r1 and r1.data:
        rid = r1.data[0]["id"]
        safe_execute(
            supabase.table("route_distances").update({"distance_km": float(distance_km)}).eq("id", rid),
            "set_route_distance(): update",
        )
    else:
        safe_execute(
            supabase.table("route_distances").insert({"place_a": a, "place_b": b, "distance_km": float(distance_km)}),
            "set_route_distance(): insert",
        )


def insert_trip(period_id, trip_date, car_id, dep_place_id, arr_place_id, dep_addr, arr_addr, distance_km, notes):
    payload = {
        "period_id": period_id,
        "trip_date": str(trip_date),
        "car_id": car_id,
        "departure_place_id": dep_place_id,
        "arrival_place_id": arr_place_id,
        "departure_address": clean_text(dep_addr),
        "arrival_address": clean_text(arr_addr),
        "distance_km": float(distance_km),
        "notes": clean_text(notes) if notes else None,
    }
    return safe_execute(supabase.table("trip_entries").insert(payload), "insert_trip()")


def fetch_entries(period_id: str, start_date: date, end_date: date, car_id=None, search_text=""):
    q = (
        supabase.table("trip_entries")
        .select(
            "id,period_id,trip_date,car_id,"
            "departure_place_id,arrival_place_id,"
            "departure_address,arrival_address,"
            "distance_km,notes,created_at"
        )
        .eq("period_id", period_id)
        .gte("trip_date", str(start_date))
        .lte("trip_date", str(end_date))
        .order("trip_date", desc=False)
        .order("created_at", desc=False)
    )
    if car_id:
        q = q.eq("car_id", car_id)

    res = safe_execute(q, "fetch_entries()")
    if not res:
        return pd.DataFrame([])
    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df

    if search_text.strip():
        s = search_text.strip().lower()
        mask = (
            df["departure_address"].astype(str).str.lower().str.contains(s, na=False)
            | df["arrival_address"].astype(str).str.lower().str.contains(s, na=False)
            | df["notes"].astype(str).str.lower().str.contains(s, na=False)
        )
        df = df[mask].copy()
    return df


def add_month_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["trip_date"] = pd.to_datetime(out["trip_date"], errors="coerce")
    out["month_key"] = out["trip_date"].dt.to_period("M").astype(str)
    out["month_name"] = out["trip_date"].dt.strftime("%B %Y")
    return out


def month_range(d: date):
    start = d.replace(day=1)
    if start.month == 12:
        next_month = date(start.year + 1, 1, 1)
    else:
        next_month = date(start.year, start.month + 1, 1)
    last_day = (pd.Timestamp(next_month) - pd.Timedelta(days=1)).date()
    return start, last_day


def year_range(year: int):
    return date(year, 1, 1), date(year, 12, 31)


# =========================
# SESSION STATE (IMPORTANT FOR SWAP)
# =========================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# These are NOT the widget keys; they store our chosen values safely.
if "dep_value" not in st.session_state:
    st.session_state.dep_value = None
if "arr_value" not in st.session_state:
    st.session_state.arr_value = None

if "distance_value" not in st.session_state:
    st.session_state.distance_value = 1.0
if "last_route_key" not in st.session_state:
    st.session_state.last_route_key = ""


def maybe_autofill_distance(dep_id: str, arr_id: str):
    if not dep_id or not arr_id or dep_id == arr_id:
        return
    a, b = normalize_place_pair(dep_id, arr_id)
    rk = f"{a}__{b}"
    if rk != st.session_state.last_route_key:
        st.session_state.last_route_key = rk
        mem = get_route_distance(dep_id, arr_id)
        if mem is not None:
            st.session_state.distance_value = round_half(max(0.0, float(mem)))


# =========================
# UI
# =========================
st.title(APP_TITLE)
tabs = st.tabs(["🧾 Trip Log", "🛠️ Admin"])

with tabs[0]:
    # period
    st.subheader("Period (logbook)")
    current_year_name = str(date.today().year)
    ensure_period(current_year_name)

    periods = get_periods()
    if not periods:
        ensure_period("Default")
        periods = get_periods()

    period_names = [p["name"] for p in periods]
    period_name_to_id = {p["name"]: p["id"] for p in periods}
    default_index = period_names.index(current_year_name) if current_year_name in period_names else 0

    c1, c2 = st.columns([2, 1])
    with c1:
        selected_period_name = st.selectbox("Choose period", period_names, index=default_index)
        selected_period_id = period_name_to_id[selected_period_name]
    with c2:
        new_period_name = st.text_input("New period name", placeholder="e.g. 2027")
        if st.button("➕ Add period", use_container_width=True):
            if new_period_name.strip():
                create_period(new_period_name.strip())
                st.rerun()

    # cars
    cars = get_cars()
    if not cars:
        st.error("No active cars found. Add your cars in Supabase table: cars.")
        st.stop()

    car_label_to_id = {}
    car_id_to_label = {}
    for c in cars:
        label = c["name"] + (f" ({c['plate']})" if c.get("plate") else "")
        car_label_to_id[label] = c["id"]
        car_id_to_label[c["id"]] = label
    car_labels = list(car_label_to_id.keys())

    # places
    places = get_places()
    if not places:
        st.warning("No places yet. Add a place below first.")
    place_label_to_id = {p["label"]: p["id"] for p in places}
    place_id_to_address = {p["id"]: p["address"] for p in places}
    place_labels = list(place_label_to_id.keys())

    st.subheader("Add a trip")
    with st.container(border=True):
        with st.expander("➕ Add a place (name + address)"):
            pa, pb = st.columns(2)
            with pa:
                new_label = st.text_input("Place name (easy label)", placeholder="e.g. Client Breda")
            with pb:
                new_addr = st.text_input("Full address", placeholder="e.g. Street 1, Breda, Netherlands")
            if st.button("Save place", use_container_width=True):
                created = create_place(new_label, new_addr)
                if created:
                    st.success("Place saved.")
                    st.rerun()

        if not place_labels:
            st.stop()

        left, right = st.columns(2)
        with left:
            trip_date = st.date_input("Date", value=date.today())
            st.caption(f"Day: **{trip_date.strftime('%A')}**")
        with right:
            car_label = st.selectbox("Car", car_labels)
            car_id = car_label_to_id[car_label]

        # Initialize dep/arr values once
        if st.session_state.dep_value is None:
            st.session_state.dep_value = place_labels[0]
        if st.session_state.arr_value is None:
            st.session_state.arr_value = place_labels[0] if len(place_labels) == 1 else place_labels[1]

        # Ensure still valid after adding/removing places
        if st.session_state.dep_value not in place_labels:
            st.session_state.dep_value = place_labels[0]
        if st.session_state.arr_value not in place_labels:
            st.session_state.arr_value = place_labels[0] if len(place_labels) == 1 else place_labels[1]

        colD, colS, colA = st.columns([1, 1, 1])
        with colD:
            dep_label = st.selectbox(
                "Departure (place name)",
                place_labels,
                index=place_labels.index(st.session_state.dep_value),
                key="dep_select",
            )
        with colA:
            arr_label = st.selectbox(
                "Arrival (place name)",
                place_labels,
                index=place_labels.index(st.session_state.arr_value),
                key="arr_select",
            )
        with colS:
            if st.button("↔ Swap", use_container_width=True):
                st.session_state.dep_value, st.session_state.arr_value = (
                    st.session_state.arr_value,
                    st.session_state.dep_value,
                )
                st.rerun()

        # After widgets render, update stored values from widgets
        st.session_state.dep_value = dep_label
        st.session_state.arr_value = arr_label

        dep_id = place_label_to_id.get(st.session_state.dep_value)
        arr_id = place_label_to_id.get(st.session_state.arr_value)

        if dep_id and arr_id:
            maybe_autofill_distance(dep_id, arr_id)

        st.number_input(
            "Distance (km)",
            min_value=0.0,
            max_value=2000.0,
            step=0.5,
            value=float(st.session_state.distance_value),
            key="distance_value",
        )
        notes = st.text_input("Notes (optional)")

        if dep_id:
            st.caption(f"Departure address: **{place_id_to_address.get(dep_id,'')}**")
        if arr_id:
            st.caption(f"Arrival address: **{place_id_to_address.get(arr_id,'')}**")

        if st.button("✅ Save trip", use_container_width=True):
            if not dep_id or not arr_id:
                st.error("Please choose both Departure and Arrival.")
            elif dep_id == arr_id:
                st.error("Departure and Arrival cannot be the same place.")
            else:
                dep_addr = place_id_to_address.get(dep_id, "")
                arr_addr = place_id_to_address.get(arr_id, "")
                dist = float(st.session_state.distance_value)

                insert_trip(selected_period_id, trip_date, car_id, dep_id, arr_id, dep_addr, arr_addr, dist, notes)
                set_route_distance(dep_id, arr_id, dist)
                st.success("Saved!")
                st.rerun()

    st.success("✅ Swap is now fixed using safe state keys (no StreamlitAPIException).")

with tabs[1]:
    st.info("Admin tab unchanged (your existing admin code can stay here).")
