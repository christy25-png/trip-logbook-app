import os
import io
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st
from supabase import create_client, Client

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================
# Config
# =========================
st.set_page_config(page_title="Trip Logbook", layout="centered")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_SERVICE_ROLE_KEY = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
SUPABASE_KEY_FALLBACK = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))  # optional
ADMIN_PIN = st.secrets.get("ADMIN_PIN", os.getenv("ADMIN_PIN", ""))
APP_TITLE = st.secrets.get("APP_TITLE", os.getenv("APP_TITLE", "🚗 Trip Logbook"))
GOOGLE_MAPS_API_KEY = st.secrets.get("GOOGLE_MAPS_API_KEY", os.getenv("GOOGLE_MAPS_API_KEY", ""))

if not SUPABASE_URL:
    st.error("Missing SUPABASE_URL in Streamlit secrets.")
    st.stop()

# Prefer service role key to avoid RLS/policy issues
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY_FALLBACK
if not SUPABASE_KEY:
    st.error("Missing SUPABASE_SERVICE_ROLE_KEY (recommended) or SUPABASE_KEY in secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

PLACE_TYPE_NEW = "✍️ Type a new place..."


# =========================
# Error helper (shows useful message)
# =========================
def show_api_error(e: Exception):
    st.error("Database request failed.")
    st.write("Details (for debugging):")
    st.exception(e)


# =========================
# Supabase Helpers
# =========================
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


def create_period(name: str):
    name = (name or "").strip()
    if not name:
        return None
    try:
        res = supabase.table("periods").upsert({"name": name, "is_active": True}).execute()
        data = res.data or []
        return data[0] if data else None
    except Exception as e:
        show_api_error(e)
        return None


def get_places(limit=600):
    """
    Returns list of display strings for dropdown + mapping dict display->row.
    Display prefers nickname; falls back to name/formatted_address.
    """
    try:
        res = (
            supabase.table("places")
            .select("id,name,nickname,formatted_address,google_place_id,is_active")
            .eq("is_active", True)
            .order("nickname", desc=False)
            .order("name", desc=False)
            .limit(limit)
            .execute()
        )
        rows = res.data or []
        mapping = {}
        display_list = []
        for r in rows:
            nick = (r.get("nickname") or "").strip()
            nm = (r.get("name") or "").strip()
            addr = (r.get("formatted_address") or "").strip()
            # Make a friendly label
            if nick and addr:
                label = f"{nick} — {addr}"
            elif nick:
                label = nick
            elif nm and addr:
                label = f"{nm} — {addr}"
            else:
                label = nm or addr or "Unnamed place"

            # Avoid duplicates by adding a tiny suffix if needed
            base = label
            k = 1
            while label in mapping:
                k += 1
                label = f"{base} ({k})"

            mapping[label] = r
            display_list.append(label)

        return display_list, mapping
    except Exception as e:
        show_api_error(e)
        return [], {}


def upsert_place(name: str, nickname: str = None, formatted_address: str = None, google_place_id: str = None):
    name = (name or "").strip()
    if not name and not formatted_address and not google_place_id:
        return

    payload = {
        "name": name or (formatted_address or google_place_id or "Place").strip(),
        "is_active": True,
    }
    if nickname:
        payload["nickname"] = nickname.strip()
    if formatted_address:
        payload["formatted_address"] = formatted_address.strip()
    if google_place_id:
        payload["google_place_id"] = google_place_id.strip()

    try:
        supabase.table("places").upsert(payload).execute()
    except Exception as e:
        show_api_error(e)


def insert_trip(period_id: str, trip_date, car_id, departure, arrival, distance_km, notes=""):
    payload = {
        "period_id": period_id,
        "trip_date": str(trip_date),
        "car_id": car_id,
        "departure": departure.strip(),
        "arrival": arrival.strip(),
        "distance_km": float(distance_km),
        "notes": notes.strip() if notes else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
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
            .select("id,period_id,trip_date,car_id,departure,arrival,distance_km,notes,created_at,updated_at")
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
                df["departure"].astype(str).str.lower().str.contains(s, na=False)
                | df["arrival"].astype(str).str.lower().str.contains(s, na=False)
                | df["notes"].astype(str).str.lower().str.contains(s, na=False)
            )
            df = df[mask].copy()

        return df
    except Exception as e:
        show_api_error(e)
        return pd.DataFrame([])


def update_trip(trip_id: str, updates: dict):
    updates = dict(updates)
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        return supabase.table("trip_entries").update(updates).eq("id", trip_id).execute()
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


def fetch_places_admin():
    try:
        res = supabase.table("places").select("id,name,nickname,formatted_address,google_place_id,is_active,created_at").order("name").execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        show_api_error(e)
        return pd.DataFrame([])


def update_place(place_id: str, updates: dict):
    try:
        return supabase.table("places").update(updates).eq("id", place_id).execute()
    except Exception as e:
        show_api_error(e)
        return None




# =========================
# UI Helpers
# =========================
def place_picker(label: str, display_places: list[str], mapping: dict, key_prefix: str) -> str:
    choice = st.selectbox(label, options=(display_places + [PLACE_TYPE_NEW]), key=f"{key_prefix}_choice")
    if choice == PLACE_TYPE_NEW:
        return st.text_input(f"{label} (type)", key=f"{key_prefix}_typed").strip()
    # use stored formatted label directly for trip text (keeps it readable)
    return (choice or "").strip()


def month_range(d: date):
    start = d.replace(day=1)
    if start.month == 12:
        next_month = date(start.year + 1, 1, 1)
    else:
        next_month = date(start.year, start.month + 1, 1)
    last_day = (pd.Timestamp(next_month) - pd.Timedelta(days=1)).date()
    return start, last_day


def make_export_df(df: pd.DataFrame, car_id_to_label: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Car"] = out["car_id"].map(car_id_to_label).fillna("")
    out = out.rename(columns={
        "trip_date": "Date",
        "departure": "Departure",
        "arrival": "Arrival",
        "distance_km": "Distance (km)",
        "notes": "Notes",
    })
    out = out[["Date", "Car", "Departure", "Arrival", "Distance (km)", "Notes"]]
    out["Date"] = pd.to_datetime(out["Date"]).dt.date.astype(str)
    return out


def export_csv_bytes(df: pd.DataFrame) -> bytes:
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8")


def export_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Trips", index=False)
    return buff.getvalue()


def export_pdf_bytes(df: pd.DataFrame, title: str, total_km: float) -> bytes:
    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=A4)
    w, h = A4

    x = 40
    y = h - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)

    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(x, y, f"Total distance: {total_km:.1f} km")
    y -= 25

    if df.empty:
        c.setFont("Helvetica", 10)
        c.drawString(x, y, "No trips found.")
        c.showPage()
        c.save()
        return buff.getvalue()

    headers = list(df.columns)
    col_widths = [75, 70, 135, 135, 75, 120]

    c.setFont("Helvetica-Bold", 9)
    xx = x
    for name, cw in zip(headers, col_widths):
        c.drawString(xx, y, str(name)[:18])
        xx += cw

    y -= 12
    c.setFont("Helvetica", 9)

    for _, row in df.iterrows():
        if y < 60:
            c.showPage()
            y = h - 50
            c.setFont("Helvetica", 9)

        xx = x
        values = [row.get(col, "") for col in headers]
        values = [("" if pd.isna(v) else str(v)) for v in values]
        for v, cw in zip(values, col_widths):
            c.drawString(xx, y, v[:28])
            xx += cw
        y -= 12

    c.showPage()
    c.save()
    return buff.getvalue()


# =========================
# Session State
# =========================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False


# =========================
# Header
# =========================
st.title(APP_TITLE)

tabs = st.tabs(["🧾 Trip Log", "🛠️ Admin"])

# =========================
# Trip Log Tab
# =========================
with tabs[0]:
    # Admin unlock (still accessible from Trip tab)
    with st.expander("🔐 Admin mode"):
        if not ADMIN_PIN:
            st.info("Set ADMIN_PIN in Streamlit secrets to enable Admin features.")
        pin = st.text_input("Enter admin PIN", type="password")
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

    # Periods (logbooks)
    st.subheader("Period (logbook)")

    periods = get_periods()
    if not periods:
        create_period("Default")
        periods = get_periods()

    period_name_to_id = {p["name"]: p["id"] for p in periods}
    period_names = list(period_name_to_id.keys())

    colp1, colp2 = st.columns([2, 1])
    with colp1:
        selected_period_name = st.selectbox("Choose period", period_names)
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
        st.error("No active cars found. Add your cars in Supabase table 'cars'.")
        st.stop()

    car_label_to_id = {}
    car_id_to_label = {}
    for c in cars:
        label = c["name"] + (f" ({c['plate']})" if c.get("plate") else "")
        car_label_to_id[label] = c["id"]
        car_id_to_label[c["id"]] = label
    car_labels = list(car_label_to_id.keys())

    # Places history
    display_places, place_map = get_places()

    st.subheader("Add a trip")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            trip_date = st.date_input("Date", value=date.today())
        with col2:
            car_label = st.selectbox("Car", car_labels)
            car_id = car_label_to_id[car_label]

        departure = place_picker("Departure", display_places, place_map, "dep")
        arrival = place_picker("Arrival", display_places, place_map, "arr")

        col3, col4 = st.columns(2)
        with col3:
            distance = st.number_input("Distance (km)", min_value=0.0, step=0.1, format="%.1f")
        with col4:
            notes = st.text_input("Notes (optional)")

        if st.button("✅ Save trip", use_container_width=True):
            if not departure or not arrival:
                st.error("Please fill in Departure and Arrival.")
            else:
                insert_trip(selected_period_id, trip_date, car_id, departure, arrival, distance, notes)
                upsert_place(departure)
                upsert_place(arrival)
                st.success("Saved!")
                st.rerun()
    st.divider()

    # Filters
    st.subheader("Filters")
    mode = st.radio("Range", ["This month", "Custom range"], horizontal=True)
    if mode == "This month":
        pick = st.date_input("Pick any day in the month", value=date.today())
        start_date, end_date = month_range(pick)
    else:
        cA, cB = st.columns(2)
        with cA:
            start_date = st.date_input("Start date", value=date(date.today().year, 1, 1))
        with cB:
            end_date = st.date_input("End date", value=date.today())

    f1, f2 = st.columns(2)
    with f1:
        filter_car = st.selectbox("Car filter", ["All cars"] + car_labels)
    with f2:
        search_text = st.text_input("Search (departure/arrival/notes)", placeholder="type to filter...")

    filter_car_id = None if filter_car == "All cars" else car_label_to_id[filter_car]

    df = fetch_entries(selected_period_id, start_date, end_date, car_id=filter_car_id, search_text=search_text)
    total_km = 0.0 if df.empty else float(pd.to_numeric(df["distance_km"], errors="coerce").fillna(0).sum())

    st.metric("Total distance (selected)", f"{total_km:.1f} km")

    df_export = make_export_df(df, car_id_to_label)

    st.write("### Trips")
    if df_export.empty:
        st.info("No trips found.")
    else:
        st.dataframe(df_export, use_container_width=True, hide_index=True)

    # Export
    st.write("### Export")
    csv_bytes = export_csv_bytes(df_export)
    xlsx_bytes = export_xlsx_bytes(df_export)
    pdf_bytes = export_pdf_bytes(
        df_export,
        title=f"Trip Logbook — {selected_period_name} ({start_date} to {end_date})",
        total_km=total_km
    )

    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("⬇️ CSV", csv_bytes,
                           f"trips_{selected_period_name}_{start_date}_to_{end_date}.csv",
                           "text/csv", use_container_width=True)
    with e2:
        st.download_button("⬇️ XLSX", xlsx_bytes,
                           f"trips_{selected_period_name}_{start_date}_to_{end_date}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with e3:
        st.download_button("⬇️ PDF", pdf_bytes,
                           f"trips_{selected_period_name}_{start_date}_to_{end_date}.pdf",
                           "application/pdf", use_container_width=True)

    st.divider()

    # Manage trips merge edit/delete
    st.subheader("Manage trips (edit / delete)")
    if df.empty:
        st.info("Nothing to manage for this selection.")
    else:
        manage_df = df.copy()
        manage_df["car_label"] = manage_df["car_id"].map(car_id_to_label).fillna("")
        manage_df["DELETE"] = False
        manage_df = manage_df[["DELETE", "trip_date", "car_label", "departure", "arrival", "distance_km", "notes", "id"]].copy()

        edited = st.data_editor(
            manage_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "DELETE": st.column_config.CheckboxColumn("Delete?"),
                "trip_date": st.column_config.DateColumn("Date"),
                "car_label": st.column_config.SelectboxColumn("Car", options=car_labels),
                "distance_km": st.column_config.NumberColumn("Distance (km)", min_value=0.0, step=0.1),
                "notes": st.column_config.TextColumn("Notes"),
                "id": st.column_config.TextColumn("ID", disabled=True),
            },
            disabled=["id"],
            key="manage_editor",
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save edits", use_container_width=True):
                changes = 0
                for i in range(len(edited)):
                    new_row = edited.iloc[i]
                    old_row = manage_df.iloc[i]
                    trip_id = str(new_row["id"])

                    if bool(new_row["DELETE"]):
                        continue

                    updates = {}
                    if str(new_row["trip_date"]) != str(old_row["trip_date"]):
                        updates["trip_date"] = str(pd.to_datetime(new_row["trip_date"]).date())

                    if str(new_row["car_label"]) != str(old_row["car_label"]):
                        updates["car_id"] = car_label_to_id.get(str(new_row["car_label"]))

                    for field in ["departure", "arrival", "notes"]:
                        if str(new_row[field]) != str(old_row[field]):
                            updates[field] = ("" if pd.isna(new_row[field]) else str(new_row[field]).strip())

                    if str(new_row["distance_km"]) != str(old_row["distance_km"]):
                        updates["distance_km"] = float(new_row["distance_km"])

                    if updates:
                        update_trip(trip_id, updates)
                        upsert_place(str(new_row["departure"]))
                        upsert_place(str(new_row["arrival"]))
                        changes += 1

                st.success(f"Saved edits on {changes} row(s).")
                st.rerun()

        with b2:
            if st.button("🗑️ Delete selected", use_container_width=True):
                to_delete = edited.loc[edited["DELETE"] == True, "id"].astype(str).tolist()
                if not to_delete:
                    st.info("No rows selected.")
                else:
                    delete_trips(to_delete)
                    st.success(f"Deleted {len(to_delete)} row(s).")
                    st.rerun()


# =========================
# Admin Tab (its own space)
# =========================
with tabs[1]:
    if not st.session_state.is_admin:
        st.info("Admin is locked. Unlock it in the Trip Log tab.")
    else:
        st.header("Admin Panel")

        st.write("### Places manager (edit / deactivate)")
        places_df = fetch_places_admin()

        if places_df.empty:
            st.info("No places yet.")
        else:
            places_df = places_df[["id", "nickname", "name", "formatted_address", "google_place_id", "is_active", "created_at"]].copy()

            edited_places = st.data_editor(
                places_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "id": st.column_config.TextColumn("ID", disabled=True),
                    "nickname": st.column_config.TextColumn("Nickname (what Dad sees)"),
                    "name": st.column_config.TextColumn("Name"),
                    "formatted_address": st.column_config.TextColumn("Address"),
                    "google_place_id": st.column_config.TextColumn("Google Place ID"),
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
                    for col in ["nickname", "name", "formatted_address", "google_place_id"]:
                        if str(n[col]) != str(o[col]):
                            updates[col] = None if pd.isna(n[col]) else str(n[col]).strip()

                    if bool(n["is_active"]) != bool(o["is_active"]):
                        updates["is_active"] = bool(n["is_active"])

                    if updates:
                        update_place(pid, updates)
                        changed += 1

                st.success(f"Updated {changed} place(s).")
                st.rerun()
