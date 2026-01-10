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
    q = (
        supabase.table("periods")
        .select("id,name,is_active")
        .eq("is_active", True)
        .order("name", desc=True)
    )
    res = safe_execute(q, "get_periods()")
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
    q = (
        supabase.table("cars")
        .select("id,name,plate,is_active")
        .eq("is_active", True)
        .order("name")
    )
    res = safe_execute(q, "get_cars()")
    return (res.data or []) if res else []


def get_places(limit=2000):
    q = (
        supabase.table("places")
        .select("id,label,address,is_active,created_at")
        .eq("is_active", True)
        .order("label")
        .limit(limit)
    )
    res = safe_execute(q, "get_places()")
    return (res.data or []) if res else []


def create_place(label: str, address: str):
    label = clean_text(label)
    address = clean_text(address)
    if not label or not address:
        st.error("Please enter both a place name and an address.")
        return None

    # if label exists -> update address (easy for dad)
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
        supabase.table("route_distances").select("distance_km").eq("place_a", a).eq("place_b", b).limit(1),
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


def update_trip(trip_id: str, updates: dict):
    allowed = {
        "trip_date", "car_id",
        "departure_place_id", "arrival_place_id",
        "departure_address", "arrival_address",
        "distance_km", "notes"
    }
    updates_clean = {k: v for k, v in updates.items() if k in allowed}
    return safe_execute(
        supabase.table("trip_entries").update(updates_clean).eq("id", trip_id),
        "update_trip()",
    )


def delete_trips(trip_ids: list[str]):
    if not trip_ids:
        return None
    return safe_execute(
        supabase.table("trip_entries").delete().in_("id", trip_ids),
        "delete_trips()",
    )


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
# EXPORTS
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
    return out[["Date", "Car", "Departure address", "Arrival address", "Distance (km)", "Notes"]]


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
        ws.column_dimensions[col_letter].width = min(max_len + 2, 60)


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


# ---------- Trip Log ----------
with tabs[0]:
    with st.expander("🔐 Admin mode"):
        if not ADMIN_PIN:
            st.info("Set ADMIN_PIN in Streamlit secrets to enable Admin features.")
        pin = st.text_input("Enter admin PIN", type="password", key="admin_pin_input")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Unlock admin", use_container_width=True):
                if ADMIN_PIN and pin == ADMIN_PIN:
                    st.session_state.is_admin = True
                    st.success("Admin mode enabled.")
                else:
                    st.error("Wrong PIN.")
        with c2:
            if st.button("Lock admin", use_container_width=True):
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
        selected_period_name = st.selectbox("Choose period", period_names, index=default_index)
        selected_period_id = period_name_to_id[selected_period_name]
    with colp2:
        new_period_name = st.text_input("New period name", placeholder="e.g. 2027")
        if st.button("➕ Add period", use_container_width=True):
            if new_period_name.strip():
                create_period(new_period_name.strip())
                st.success("Period added.")
                st.rerun()
            else:
                st.error("Type a name first (e.g. 2027).")

    # Cars
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

    # Places
    places = get_places()
    place_label_to_id = {p["label"]: p["id"] for p in places}
    place_id_to_label = {p["id"]: p["label"] for p in places}
    place_id_to_address = {p["id"]: p["address"] for p in places}
    place_labels = list(place_label_to_id.keys())

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
                created = create_place(new_place_label, new_place_address)
                if created:
                    st.success("Place saved.")
                    st.rerun()

        if not place_labels:
            st.warning("Add at least one place first (open the expander above).")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            trip_date = st.date_input("Date", value=date.today())
            st.caption(f"Day: **{trip_date.strftime('%A')}**")
        with col2:
            car_label = st.selectbox("Car", car_labels)
            car_id = car_label_to_id[car_label]

        cD, cS, cA = st.columns([1, 1, 1])
        with cD:
            dep_label = st.selectbox("Departure (place name)", place_labels, key="dep_label")
        with cA:
            arr_label = st.selectbox("Arrival (place name)", place_labels, key="arr_label")
        with cS:
            if st.button("↔ Swap", use_container_width=True):
                st.session_state.dep_label, st.session_state.arr_label = st.session_state.arr_label, st.session_state.dep_label
                st.rerun()

        dep_id = place_label_to_id.get(st.session_state.dep_label)
        arr_id = place_label_to_id.get(st.session_state.arr_label)

        if dep_id and arr_id:
            maybe_autofill_distance(dep_id, arr_id)

        col3, col4 = st.columns(2)
        with col3:
            st.number_input(
                "Distance (km)",
                min_value=0.0,
                max_value=2000.0,
                step=0.5,
                value=float(st.session_state.distance_value),
                key="distance_value",
            )
        with col4:
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

    st.divider()

    # View mode (DEFAULT all months)
    st.subheader("View trips")
    view_mode = st.radio(
        "View mode",
        ["One month", "All months (year)", "Custom range"],
        horizontal=True,
        index=1,  # default: All months
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
        filter_car = st.selectbox("Car filter", ["All cars"] + car_labels, key="car_filter")
    with colf2:
        search_text = st.text_input("Search (addresses/notes)", placeholder="type to search...", key="search_text")

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
                st.dataframe(block.drop(columns=["month_key", "month_name"]), use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_export, use_container_width=True, hide_index=True)

    st.divider()

    # Export at bottom
    st.subheader("Export")
    default_name = f"{selected_period_name}_{start_date}_to_{end_date}"
    file_base = st.text_input("File name", value=default_name, key="export_name")
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
        st.download_button("⬇️ XLSX", xlsx_bytes, f"{file_base}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with e3:
        st.download_button("⬇️ PDF", pdf_bytes, f"{file_base}.pdf", "application/pdf", use_container_width=True)


# ---------- Admin ----------
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

            if st.button("💾 Save place changes", use_container_width=True):
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
