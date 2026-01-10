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
def show_api_error(e: Exception):
    st.error("Database request failed.")
    st.write("Details (for debugging):")
    st.exception(e)


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


# =========================
# SUPABASE CRUD
# =========================
def get_periods():
    try:
        res = (
            supabase.table("periods")
            .select("id,name,is_active")
            .eq("is_active", True)
            .order("name", desc=True)
            .execute()
        )
        return res.data or []
    except Exception as e:
        show_api_error(e)
        return []


def ensure_period(name: str):
    name = clean_text(name)
    if not name:
        return None
    try:
        existing = supabase.table("periods").select("id").eq("name", name).limit(1).execute().data
        if existing:
            return existing[0]["id"]
        created = supabase.table("periods").insert({"name": name, "is_active": True}).execute().data
        if created:
            return created[0]["id"]
    except Exception as e:
        show_api_error(e)
    return None


def create_period(name: str):
    name = clean_text(name)
    if not name:
        return None
    try:
        existing = supabase.table("periods").select("id,name").eq("name", name).limit(1).execute().data
        if existing:
            return existing[0]
        res = supabase.table("periods").insert({"name": name, "is_active": True}).execute()
        rows = res.data or []
        return rows[0] if rows else None
    except Exception as e:
        show_api_error(e)
        return None


def get_cars():
    try:
        res = (
            supabase.table("cars")
            .select("id,name,plate,is_active")
            .eq("is_active", True)
            .order("name")
            .execute()
        )
        return res.data or []
    except Exception as e:
        show_api_error(e)
        return []


def get_places(limit=2000):
    """
    Returns list of dict: {id, label, address, is_active}
    """
    try:
        res = (
            supabase.table("places")
            .select("id,label,address,is_active,created_at")
            .eq("is_active", True)
            .order("label")
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        show_api_error(e)
        return []


def create_place(label: str, address: str):
    label = clean_text(label)
    address = clean_text(address)
    if not label or not address:
        return None
    try:
        existing = supabase.table("places").select("id,label").eq("label", label).limit(1).execute().data
        if existing:
            # If already exists, update address (nice for beginners)
            pid = existing[0]["id"]
            supabase.table("places").update({"address": address, "is_active": True}).eq("id", pid).execute()
            return {"id": pid, "label": label}
        res = supabase.table("places").insert({"label": label, "address": address, "is_active": True}).execute()
        rows = res.data or []
        return rows[0] if rows else None
    except Exception as e:
        show_api_error(e)
        return None


def fetch_places_admin():
    try:
        res = supabase.table("places").select("id,label,address,is_active,created_at").order("label").execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        show_api_error(e)
        return pd.DataFrame([])


def update_place(place_id: str, updates: dict):
    allowed = {"label", "address", "is_active"}
    updates_clean = {k: v for k, v in updates.items() if k in allowed}
    try:
        return supabase.table("places").update(updates_clean).eq("id", place_id).execute()
    except Exception as e:
        show_api_error(e)
        return None


def normalize_place_pair(a_id: str, b_id: str) -> tuple[str, str]:
    # UUIDs are strings; lexicographic compare works consistently here for ordering
    return (a_id, b_id) if a_id < b_id else (b_id, a_id)


def get_route_distance(place_id_a: str, place_id_b: str) -> float | None:
    if not place_id_a or not place_id_b:
        return None
    a, b = normalize_place_pair(place_id_a, place_id_b)
    try:
        res = (
            supabase.table("route_distances")
            .select("distance_km")
            .eq("place_a", a)
            .eq("place_b", b)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return None
        return float(rows[0]["distance_km"])
    except Exception:
        return None


def set_route_distance(place_id_a: str, place_id_b: str, distance_km: float):
    if not place_id_a or not place_id_b:
        return
    a, b = normalize_place_pair(place_id_a, place_id_b)
    try:
        existing = (
            supabase.table("route_distances")
            .select("id")
            .eq("place_a", a)
            .eq("place_b", b)
            .limit(1)
            .execute()
        ).data

        if existing:
            rid = existing[0]["id"]
            supabase.table("route_distances").update({"distance_km": float(distance_km)}).eq("id", rid).execute()
        else:
            supabase.table("route_distances").insert({"place_a": a, "place_b": b, "distance_km": float(distance_km)}).execute()
    except Exception:
        pass


def insert_trip(
    period_id: str,
    trip_date: date,
    car_id: str,
    departure_place_id: str,
    arrival_place_id: str,
    departure_address: str,
    arrival_address: str,
    distance_km: float,
    notes: str,
):
    payload = {
        "period_id": period_id,
        "trip_date": str(trip_date),
        "car_id": car_id,
        "departure_place_id": departure_place_id,
        "arrival_place_id": arrival_place_id,
        "departure_address": clean_text(departure_address),
        "arrival_address": clean_text(arrival_address),
        "distance_km": float(distance_km),
        "notes": clean_text(notes) if notes else None,
    }
    try:
        return supabase.table("trip_entries").insert(payload).execute()
    except Exception as e:
        show_api_error(e)
        return None


def fetch_entries(period_id: str, start_date: date, end_date: date, car_id=None, search_text=""):
    try:
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

        res = q.execute()
        rows = res.data or []
        df = pd.DataFrame(rows)

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
    except Exception as e:
        show_api_error(e)
        return pd.DataFrame([])


def update_trip(trip_id: str, updates: dict):
    allowed = {
        "trip_date", "car_id",
        "departure_place_id", "arrival_place_id",
        "departure_address", "arrival_address",
        "distance_km", "notes"
    }
    updates_clean = {k: v for k, v in updates.items() if k in allowed}
    try:
        return supabase.table("trip_entries").update(updates_clean).eq("id", trip_id).execute()
    except Exception as e:
        show_api_error(e)
        return None


def delete_trips(trip_ids: list[str]):
    if not trip_ids:
        return
    try:
        return supabase.table("trip_entries").delete().in_("id", trip_ids).execute()
    except Exception as e:
        show_api_error(e)
        return None


# =========================
# DATE RANGE + GROUPING
# =========================
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


def add_month_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["trip_date"] = pd.to_datetime(out["trip_date"], errors="coerce")
    out["month_key"] = out["trip_date"].dt.to_period("M").astype(str)
    out["month_name"] = out["trip_date"].dt.strftime("%B %Y")
    return out


def monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Month", "Total distance (km)"])
    d = add_month_columns(df)
    d["distance_km"] = pd.to_numeric(d["distance_km"], errors="coerce").fillna(0.0)
    totals = (
        d.groupby(["month_key", "month_name"], as_index=False)["distance_km"]
        .sum()
        .sort_values("month_key")
    )
    totals = totals.rename(columns={"month_name": "Month", "distance_km": "Total distance (km)"})
    totals["Total distance (km)"] = totals["Total distance (km)"].round(1)
    return totals[["Month", "Total distance (km)"]]


# =========================
# EXPORT
# =========================
def make_export_df(df: pd.DataFrame, car_id_to_label: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Car"] = out["car_id"].map(car_id_to_label).fillna("")
    out["Date"] = pd.to_datetime(out["trip_date"]).dt.date.astype(str)

    out = out.rename(columns={
        "departure_address": "Departure address",
        "arrival_address": "Arrival address",
        "distance_km": "Distance (km)",
        "notes": "Notes",
    })

    out["Distance (km)"] = pd.to_numeric(out["Distance (km)"], errors="coerce").fillna(0.0).round(1)
    out["Notes"] = out["Notes"].fillna("")
    out = out[["Date", "Car", "Departure address", "Arrival address", "Distance (km)", "Notes"]]
    return out


def export_csv_bytes(df: pd.DataFrame) -> bytes:
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8")


def _autosize_worksheet(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                val = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(val))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max_len + 2, 55)


def export_xlsx_bytes_grouped(df_export: pd.DataFrame, df_raw: pd.DataFrame, title: str) -> bytes:
    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = "Summary"

    ws_summary["A1"] = title
    ws_summary["A1"].font = Font(bold=True, size=14)
    ws_summary["A2"] = "Monthly totals"
    ws_summary["A2"].font = Font(bold=True, size=12)

    totals = monthly_totals(df_raw)
    if totals.empty:
        ws_summary["A4"] = "No trips in this selection."
    else:
        start_row = 4
        for r_idx, row in enumerate(dataframe_to_rows(totals, index=False, header=True), start_row):
            for c_idx, value in enumerate(row, 1):
                ws_summary.cell(row=r_idx, column=c_idx, value=value)
                if r_idx == start_row:
                    ws_summary.cell(row=r_idx, column=c_idx).font = Font(bold=True)
        ws_summary.freeze_panes = "A5"
        _autosize_worksheet(ws_summary)

    if not df_raw.empty and not df_export.empty:
        d = add_month_columns(df_raw)
        work = df_export.copy()
        work["month_key"] = d["month_key"].values
        work["month_name"] = d["month_name"].values

        month_keys = sorted(work["month_key"].dropna().unique().tolist())
        for mk in month_keys:
            block = work[work["month_key"] == mk].copy()
            month_name = block["month_name"].iloc[0] if not block.empty else mk

            dt = pd.Period(mk).to_timestamp()
            sheet_name = dt.strftime("%b")
            if sheet_name in wb.sheetnames:
                n = 2
                while f"{sheet_name}{n}" in wb.sheetnames:
                    n += 1
                sheet_name = f"{sheet_name}{n}"

            ws = wb.create_sheet(title=sheet_name)
            ws["A1"] = month_name
            ws["A1"].font = Font(bold=True, size=13)
            ws["A2"] = f"Monthly total: {block['Distance (km)'].sum():.1f} km"
            ws["A2"].font = Font(bold=True)

            start_row = 4
            for r_idx, row in enumerate(dataframe_to_rows(block.drop(columns=["month_key", "month_name"]), index=False, header=True), start_row):
                for c_idx, value in enumerate(row, 1):
                    ws.cell(row=r_idx, column=c_idx, value=value)
                    if r_idx == start_row:
                        ws.cell(row=r_idx, column=c_idx).font = Font(bold=True)

            ws.freeze_panes = "A5"
            _autosize_worksheet(ws)

    buff = io.BytesIO()
    wb.save(buff)
    return buff.getvalue()


def export_pdf_bytes_grouped(df_export: pd.DataFrame, df_raw: pd.DataFrame, title: str) -> bytes:
    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=A4)

    _, height = A4
    left = 40
    y = height - 50

    def new_page():
        c.showPage()
        return height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, y, title)
    y -= 24

    totals = monthly_totals(df_raw)
    c.setFont("Helvetica", 11)
    if totals.empty:
        c.drawString(left, y, "No trips in this selection.")
        c.showPage()
        c.save()
        return buff.getvalue()

    overall_total = float(pd.to_numeric(df_raw["distance_km"], errors="coerce").fillna(0.0).sum())
    c.drawString(left, y, f"Total distance: {overall_total:.1f} km")
    y = new_page()

    d = add_month_columns(df_raw)
    work = df_export.copy()
    work["month_key"] = d["month_key"].values
    work["month_name"] = d["month_name"].values

    month_keys = sorted(work["month_key"].dropna().unique().tolist())
    for mk in month_keys:
        block = work[work["month_key"] == mk].copy()
        month_name = block["month_name"].iloc[0]
        month_total = float(block["Distance (km)"].sum())

        c.setFont("Helvetica-Bold", 14)
        c.drawString(left, y, month_name)
        y -= 16

        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, f"Monthly total: {month_total:.1f} km")
        y -= 18

        headers = ["Date", "Car", "Departure address", "Arrival address", "Km", "Notes"]
        col_widths = [70, 70, 145, 145, 35, 70]

        c.setFont("Helvetica-Bold", 9)
        xx = left
        for h, cw in zip(headers, col_widths):
            c.drawString(xx, y, h)
            xx += cw
        y -= 12

        c.setFont("Helvetica", 9)
        for _, row in block.iterrows():
            if y < 70:
                y = new_page()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(left, y, month_name + " (cont.)")
                y -= 16
                c.setFont("Helvetica-Bold", 11)
                c.drawString(left, y, f"Monthly total: {month_total:.1f} km")
                y -= 18
                c.setFont("Helvetica-Bold", 9)
                xx = left
                for h, cw in zip(headers, col_widths):
                    c.drawString(xx, y, h)
                    xx += cw
                y -= 12
                c.setFont("Helvetica", 9)

            values = [
                str(row.get("Date", ""))[:12],
                str(row.get("Car", ""))[:12],
                str(row.get("Departure address", ""))[:32],
                str(row.get("Arrival address", ""))[:32],
                f"{float(row.get('Distance (km)', 0.0)):.1f}",
                str(row.get("Notes", ""))[:18],
            ]
            xx = left
            for v, cw in zip(values, col_widths):
                c.drawString(xx, y, v)
                xx += cw
            y -= 12

        y -= 14
        if mk != month_keys[-1]:
            y = new_page()

    c.showPage()
    c.save()
    return buff.getvalue()


# =========================
# SESSION STATE
# =========================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if "departure_label" not in st.session_state:
    st.session_state.departure_label = ""
if "arrival_label" not in st.session_state:
    st.session_state.arrival_label = ""

if "distance_value" not in st.session_state:
    st.session_state.distance_value = 1.0

if "last_route_key" not in st.session_state:
    st.session_state.last_route_key = ""


def maybe_autofill_distance(dep_place_id: str, arr_place_id: str):
    if not dep_place_id or not arr_place_id or dep_place_id == arr_place_id:
        return
    a, b = normalize_place_pair(dep_place_id, arr_place_id)
    rk = f"{a}__{b}"
    if rk != st.session_state.last_route_key:
        st.session_state.last_route_key = rk
        mem = get_route_distance(dep_place_id, arr_place_id)
        if mem is not None:
            st.session_state.distance_value = round_half(max(0.0, float(mem)))


# =========================
# UI
# =========================
st.title(APP_TITLE)
tabs = st.tabs(["🧾 Trip Log", "🛠️ Admin"])

# ---------- TRIP LOG ----------
with tabs[0]:
    with st.expander("🔐 Admin mode"):
        if not ADMIN_PIN:
            st.info("Set ADMIN_PIN in Streamlit secrets to enable Admin features.")
        pin = st.text_input("Enter admin PIN", type="password", key="admin_pin_input")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Unlock admin", use_container_width=True, key="unlock_admin_btn"):
                if ADMIN_PIN and pin == ADMIN_PIN:
                    st.session_state.is_admin = True
                    st.success("Admin mode enabled.")
                else:
                    st.error("Wrong PIN.")
        with c2:
            if st.button("Lock admin", use_container_width=True, key="lock_admin_btn"):
                st.session_state.is_admin = False
                st.info("Admin mode disabled.")

    # Period selection (default current year)
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

    colp1, colp2 = st.columns([2, 1])
    with colp1:
        selected_period_name = st.selectbox("Choose period", period_names, index=default_index, key="period_select")
        selected_period_id = period_name_to_id[selected_period_name]
    with colp2:
        new_period_name = st.text_input("New period name", placeholder="e.g. 2027", key="new_period_name")
        if st.button("➕ Add period", use_container_width=True, key="add_period_btn"):
            if new_period_name.strip():
                create_period(new_period_name.strip())
                st.success("Period added.")
                st.rerun()
            else:
                st.error("Type a name first (e.g. 2027).")

    # Cars
    cars = get_cars()
    if not cars:
        st.error("No active cars found. Add your cars in the 'cars' table.")
        st.stop()

    car_label_to_id = {}
    car_id_to_label = {}
    for c in cars:
        label = c["name"] + (f" ({c['plate']})" if c.get("plate") else "")
        car_label_to_id[label] = c["id"]
        car_id_to_label[c["id"]] = label
    car_labels = list(car_label_to_id.keys())

    # Places
    places = get_places()
    if not places:
        st.warning("No places yet. Add a place below (name + address) and it will appear in the dropdowns.")

    place_label_to_id = {p["label"]: p["id"] for p in places}
    place_id_to_label = {p["id"]: p["label"] for p in places}
    place_id_to_address = {p["id"]: p["address"] for p in places}
    place_labels = list(place_label_to_id.keys())

    # Add place (inside trip tab for dad)
    st.subheader("Add a trip")
    with st.container(border=True):
        # Add place inline
        with st.expander("➕ Add a place (name + address)"):
            cA, cB = st.columns(2)
            with cA:
                new_place_label = st.text_input("Place name (easy label)", placeholder="e.g. Client Breda")
            with cB:
                new_place_address = st.text_input("Full address", placeholder="e.g. Street 1, 4811 AA Breda, Netherlands")

            if st.button("Save place", use_container_width=True):
                if not clean_text(new_place_label) or not clean_text(new_place_address):
                    st.error("Please enter both a place name and an address.")
                else:
                    create_place(new_place_label, new_place_address)
                    st.success("Place saved.")
                    st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            trip_date = st.date_input("Date", value=date.today(), key="trip_date_input")
            st.caption(f"Day: **{trip_date.strftime('%A')}**")
        with col2:
            car_label = st.selectbox("Car", car_labels, key="car_select")
            car_id = car_label_to_id[car_label]

        # Departure / Arrival as dropdowns (no typing)
        cD, cS, cA = st.columns([1, 1, 1])

        with cD:
            dep_label = st.selectbox("Departure (place name)", place_labels, key="dep_label_select")
        with cA:
            arr_label = st.selectbox("Arrival (place name)", place_labels, key="arr_label_select")
        with cS:
            if st.button("↔ Swap", use_container_width=True, key="swap_btn"):
                st.session_state.dep_label_select, st.session_state.arr_label_select = (
                    st.session_state.arr_label_select,
                    st.session_state.dep_label_select,
                )
                st.rerun()

        dep_id = place_label_to_id.get(st.session_state.dep_label_select)
        arr_id = place_label_to_id.get(st.session_state.arr_label_select)

        if dep_id and arr_id:
            maybe_autofill_distance(dep_id, arr_id)

        col3, col4 = st.columns(2)
        with col3:
            distance = st.number_input(
                "Distance (km)",
                min_value=0.0,
                max_value=2000.0,
                step=0.5,
                value=float(st.session_state.distance_value),
                key="distance_value",
                help="You can type the distance (0.5 increments).",
            )
        with col4:
            notes = st.text_input("Notes (optional)", key="notes_input")

        # Show addresses (so dad can confirm)
        if dep_id:
            st.caption(f"Departure address: **{place_id_to_address.get(dep_id,'')}**")
        if arr_id:
            st.caption(f"Arrival address: **{place_id_to_address.get(arr_id,'')}**")

        if st.button("✅ Save trip", use_container_width=True, key="save_trip_btn"):
            if not dep_id or not arr_id:
                st.error("Please choose both Departure and Arrival.")
            elif dep_id == arr_id:
                st.error("Departure and Arrival cannot be the same place.")
            else:
                dep_addr = place_id_to_address.get(dep_id, "")
                arr_addr = place_id_to_address.get(arr_id, "")

                distance_to_save = float(st.session_state.distance_value)

                insert_trip(
                    selected_period_id,
                    trip_date,
                    car_id,
                    dep_id,
                    arr_id,
                    dep_addr,
                    arr_addr,
                    distance_to_save,
                    notes,
                )

                # Update distance memory (works both directions)
                set_route_distance(dep_id, arr_id, distance_to_save)

                st.success("Saved!")
                st.rerun()

    st.divider()

    # ----- View mode (DEFAULT: All months year) -----
    st.subheader("View trips")
    view_mode = st.radio(
        "View mode",
        ["One month", "All months (year)", "Custom range"],
        horizontal=True,
        index=1,  # ✅ default to All months (year)
        key="view_mode_radio",
    )

    year_for_period = parse_year_from_period_name(selected_period_name)

    if view_mode == "One month":
        pick = st.date_input("Pick a day in the month", value=date.today(), key="view_month_pick")
        start_date, end_date = month_range(pick)
    elif view_mode == "All months (year)":
        start_date, end_date = year_range(year_for_period)
        st.caption(f"Showing: **{year_for_period} (January → December)**")
    else:
        cA, cB = st.columns(2)
        with cA:
            start_date = st.date_input("Start", value=date(year_for_period, 1, 1), key="range_start")
        with cB:
            end_date = st.date_input("End", value=date.today(), key="range_end")

    colf1, colf2 = st.columns(2)
    with colf1:
        filter_car = st.selectbox("Car filter", ["All cars"] + car_labels, key="car_filter_select")
    with colf2:
        search_text = st.text_input(
            "Search (addresses/notes)",
            placeholder="type to search...",
            key="search_text"
        )

    filter_car_id = None if filter_car == "All cars" else car_label_to_id[filter_car]

    df = fetch_entries(selected_period_id, start_date, end_date, car_id=filter_car_id, search_text=search_text)

    total_km = 0.0 if df.empty else float(pd.to_numeric(df["distance_km"], errors="coerce").fillna(0).sum())
    st.metric("Total distance (selected)", f"{total_km:.1f} km")

    if view_mode == "All months (year)" and not df.empty:
        st.write("### Total distance per month")
        st.dataframe(monthly_totals(df), use_container_width=True, hide_index=True)

    st.write("### Trips")
    df_export = make_export_df(df, car_id_to_label)

    if df.empty:
        st.info("No trips found.")
    else:
        if view_mode == "All months (year)":
            d = add_month_columns(df)
            df_export_block = df_export.copy()
            df_export_block["month_key"] = d["month_key"].values
            df_export_block["month_name"] = d["month_name"].values

            for mk in sorted(df_export_block["month_key"].unique().tolist()):
                block = df_export_block[df_export_block["month_key"] == mk].copy()
                month_name = block["month_name"].iloc[0]
                month_total = float(block["Distance (km)"].sum())

                st.subheader(month_name)
                st.caption(f"Monthly total: **{month_total:.1f} km**")
                st.dataframe(
                    block.drop(columns=["month_key", "month_name"]),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.dataframe(df_export, use_container_width=True, hide_index=True)

    st.divider()

    # Manage trips (edit / delete)
    st.subheader("Manage trips (edit / delete)")
    if df.empty:
        st.info("Nothing to manage for this selection.")
    else:
        manage_df = df.copy()
        manage_df["trip_date"] = pd.to_datetime(manage_df["trip_date"], errors="coerce").dt.date
        manage_df["distance_km"] = pd.to_numeric(manage_df["distance_km"], errors="coerce").fillna(0.0).astype(float)
        manage_df["car_label"] = manage_df["car_id"].map(car_id_to_label).fillna("")
        manage_df["departure_label"] = manage_df["departure_place_id"].map(place_id_to_label).fillna("")
        manage_df["arrival_label"] = manage_df["arrival_place_id"].map(place_id_to_label).fillna("")
        manage_df["DELETE"] = False

        manage_df = manage_df[[
            "DELETE", "trip_date", "car_label", "departure_label", "arrival_label", "distance_km", "notes", "id"
        ]].copy()

        for col in ["notes", "id", "car_label", "departure_label", "arrival_label"]:
            manage_df[col] = manage_df[col].fillna("").astype(str)

        edited = st.data_editor(
            manage_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "DELETE": st.column_config.CheckboxColumn("Delete?"),
                "trip_date": st.column_config.DateColumn("Date"),
                "car_label": st.column_config.SelectboxColumn("Car", options=car_labels),
                "departure_label": st.column_config.SelectboxColumn("Departure", options=place_labels),
                "arrival_label": st.column_config.SelectboxColumn("Arrival", options=place_labels),
                "distance_km": st.column_config.NumberColumn("Distance (km)", min_value=0.0, step=0.5),
                "notes": st.column_config.TextColumn("Notes"),
                "id": st.column_config.TextColumn("ID", disabled=True),
            },
            disabled=["id"],
            key="manage_editor",
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save edits", use_container_width=True, key="save_edits_btn"):
                changes = 0
                for i in range(len(edited)):
                    new_row = edited.iloc[i]
                    old_row = manage_df.iloc[i]
                    trip_id = str(new_row["id"]).strip()

                    if bool(new_row["DELETE"]):
                        continue

                    updates = {}

                    # date
                    if str(new_row["trip_date"]) != str(old_row["trip_date"]):
                        updates["trip_date"] = str(pd.to_datetime(new_row["trip_date"]).date())

                    # car
                    if str(new_row["car_label"]) != str(old_row["car_label"]):
                        updates["car_id"] = car_label_to_id.get(str(new_row["car_label"]))

                    # places
                    if str(new_row["departure_label"]) != str(old_row["departure_label"]):
                        pid = place_label_to_id.get(str(new_row["departure_label"]))
                        if pid:
                            updates["departure_place_id"] = pid
                            updates["departure_address"] = place_id_to_address.get(pid, "")

                    if str(new_row["arrival_label"]) != str(old_row["arrival_label"]):
                        pid = place_label_to_id.get(str(new_row["arrival_label"]))
                        if pid:
                            updates["arrival_place_id"] = pid
                            updates["arrival_address"] = place_id_to_address.get(pid, "")

                    # notes
                    if clean_text(new_row["notes"]) != clean_text(old_row["notes"]):
                        updates["notes"] = clean_text(new_row["notes"])

                    # distance
                    if float(new_row["distance_km"]) != float(old_row["distance_km"]):
                        updates["distance_km"] = float(new_row["distance_km"])

                    if updates:
                        update_trip(trip_id, updates)

                        # update route memory if we have both places + distance
                        dep_pid = updates.get("departure_place_id") or place_label_to_id.get(str(new_row["departure_label"]))
                        arr_pid = updates.get("arrival_place_id") or place_label_to_id.get(str(new_row["arrival_label"]))
                        dist_now = float(updates.get("distance_km", float(new_row["distance_km"])))
                        if dep_pid and arr_pid and dep_pid != arr_pid:
                            set_route_distance(dep_pid, arr_pid, dist_now)

                        changes += 1

                st.success(f"Saved edits on {changes} trip(s).")
                st.rerun()

        with b2:
            if st.button("🗑️ Delete selected", use_container_width=True, key="delete_selected_btn"):
                to_delete = edited.loc[edited["DELETE"] == True, "id"].astype(str).tolist()
                if not to_delete:
                    st.info("No trips selected.")
                else:
                    delete_trips(to_delete)
                    st.success(f"Deleted {len(to_delete)} trip(s).")
                    st.rerun()

    # Export
    st.divider()
    st.subheader("Export (bottom)")

    default_name = f"{selected_period_name}_{start_date}_to_{end_date}"
    file_base = st.text_input("File name", value=default_name, help="Used for CSV / XLSX / PDF.", key="export_name")
    file_base = (file_base or "trips").strip().replace("/", "-")

    export_df = df_export if not df_export.empty else pd.DataFrame(
        columns=["Date", "Car", "Departure address", "Arrival address", "Distance (km)", "Notes"]
    )

    csv_bytes = export_csv_bytes(export_df)
    title = f"Trip Logbook — {selected_period_name}"
    xlsx_bytes = export_xlsx_bytes_grouped(export_df, df, title)
    pdf_bytes = export_pdf_bytes_grouped(export_df, df, title)

    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("⬇️ CSV", csv_bytes, f"{file_base}.csv", "text/csv", use_container_width=True)
    with e2:
        st.download_button(
            "⬇️ XLSX",
            xlsx_bytes,
            f"{file_base}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with e3:
        st.download_button("⬇️ PDF", pdf_bytes, f"{file_base}.pdf", "application/pdf", use_container_width=True)


# ---------- ADMIN ----------
with tabs[1]:
    if not st.session_state.is_admin:
        st.info("Admin is locked. Unlock it in the Trip Log tab.")
    else:
        st.header("Admin Panel")
        st.write("### Places (edit name + address)")

        places_df = fetch_places_admin()
        if places_df.empty:
            st.info("No places found yet.")
        else:
            places_df = places_df[["id", "label", "address", "is_active", "created_at"]].copy()
            places_df["label"] = places_df["label"].fillna("").astype(str)
            places_df["address"] = places_df["address"].fillna("").astype(str)

            edited_places = st.data_editor(
                places_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "id": st.column_config.TextColumn("ID", disabled=True),
                    "label": st.column_config.TextColumn("Place name (label)"),
                    "address": st.column_config.TextColumn("Address"),
                    "is_active": st.column_config.CheckboxColumn("Active"),
                    "created_at": st.column_config.TextColumn("Created", disabled=True),
                },
                disabled=["id", "created_at"],
                key="places_editor",
            )

            if st.button("💾 Save place changes", use_container_width=True, key="admin_save_places_btn"):
                changed = 0
                for i in range(len(edited_places)):
                    n = edited_places.iloc[i]
                    o = places_df.iloc[i]
                    pid = str(n["id"])

                    updates = {}
                    if clean_text(n["label"]) != clean_text(o["label"]):
                        updates["label"] = clean_text(n["label"])
                    if clean_text(n["address"]) != clean_text(o["address"]):
                        updates["address"] = clean_text(n["address"])
                    if bool(n["is_active"]) != bool(o["is_active"]):
                        updates["is_active"] = bool(n["is_active"])

                    if updates:
                        update_place(pid, updates)
                        changed += 1

                st.success(f"Updated {changed} place(s).")
                st.rerun()
