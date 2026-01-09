import os
import io
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from supabase import create_client, Client

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Trip Logbook", layout="centered")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_SERVICE_ROLE_KEY = st.secrets.get(
    "SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)
SUPABASE_KEY_FALLBACK = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))  # optional
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

PLACE_TYPE_NEW = "✍️ Type a new place..."


# =========================
# ERROR DISPLAY
# =========================
def show_api_error(e: Exception):
    st.error("Database request failed.")
    st.write("Details (for debugging):")
    st.exception(e)


# =========================
# SUPABASE HELPERS
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


def create_period(name: str):
    name = (name or "").strip()
    if not name:
        return None
    try:
        res = supabase.table("periods").upsert({"name": name, "is_active": True}).execute()
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


def get_places(limit=600):
    try:
        res = (
            supabase.table("places")
            .select("name,is_active")
            .eq("is_active", True)
            .order("name")
            .limit(limit)
            .execute()
        )
        rows = res.data or []
        return [r["name"] for r in rows if r.get("name")]
    except Exception as e:
        show_api_error(e)
        return []


def upsert_place(name: str):
    name = (name or "").strip()
    if not name:
        return
    try:
        supabase.table("places").upsert({"name": name, "is_active": True}).execute()
    except Exception as e:
        show_api_error(e)


def insert_trip(period_id: str, trip_date: date, car_id: str, departure: str, arrival: str, distance_km: float, notes: str):
    payload = {
        "period_id": period_id,
        "trip_date": str(trip_date),
        "car_id": car_id,
        "departure": departure.strip(),
        "arrival": arrival.strip(),
        "distance_km": float(distance_km),
        "notes": notes.strip() if notes else None,
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
            .select("id,period_id,trip_date,car_id,departure,arrival,distance_km,notes,created_at")
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
    allowed = {"trip_date", "car_id", "departure", "arrival", "distance_km", "notes"}
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


def fetch_places_admin():
    try:
        res = supabase.table("places").select("id,name,is_active,created_at").order("name").execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        show_api_error(e)
        return pd.DataFrame([])


def update_place(place_id: str, updates: dict):
    allowed = {"name", "is_active"}
    updates_clean = {k: v for k, v in updates.items() if k in allowed}
    try:
        return supabase.table("places").update(updates_clean).eq("id", place_id).execute()
    except Exception as e:
        show_api_error(e)
        return None


# =========================
# UI HELPERS
# =========================
def place_picker(label: str, places: list[str], key_prefix: str) -> str:
    choice = st.selectbox(label, options=(places + [PLACE_TYPE_NEW]), key=f"{key_prefix}_choice")
    if choice == PLACE_TYPE_NEW:
        return st.text_input(f"{label} (type)", key=f"{key_prefix}_typed").strip()
    return (choice or "").strip()


def month_range(d: date):
    start = d.replace(day=1)
    if start.month == 12:
        next_month = date(start.year + 1, 1, 1)
    else:
        next_month = date(start.year, start.month + 1, 1)
    last_day = (pd.Timestamp(next_month) - pd.Timedelta(days=1)).date()
    return start, last_day


def make_export_df_from_raw(df_raw: pd.DataFrame, car_id_to_label: dict) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw
    out = df_raw.copy()
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

    x = 40
    y = A4[1] - 50
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
            y = A4[1] - 50
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


def build_grouped_editor_df(df_trips: pd.DataFrame, car_id_to_label: dict) -> pd.DataFrame:
    if df_trips.empty:
        return df_trips

    d = df_trips.copy()
    d["trip_date"] = pd.to_datetime(d["trip_date"], errors="coerce").dt.date
    d["distance_km"] = pd.to_numeric(d["distance_km"], errors="coerce").fillna(0.0).astype(float)
    d["notes"] = d["notes"].fillna("").astype(str)
    d["departure"] = d["departure"].fillna("").astype(str)
    d["arrival"] = d["arrival"].fillna("").astype(str)
    d["car_label"] = d["car_id"].map(car_id_to_label).fillna("")

    if "created_at" in d.columns:
        d = d.sort_values(["trip_date", "created_at"], ascending=[True, True])
    else:
        d = d.sort_values(["trip_date"], ascending=[True])

    rows = []
    for trip_date_val, chunk in d.groupby("trip_date", sort=True):
        rows.append({
            "RowType": "DATE",
            "Delete?": False,
            "Date": trip_date_val,
            "Car": "",
            "Departure": f"—— {trip_date_val.isoformat()} ({trip_date_val.strftime('%A')}) ——",
            "Arrival": "",
            "Distance (km)": 0.0,
            "Notes": "",
            "ID": "",
        })

        for _, r in chunk.iterrows():
            rows.append({
                "RowType": "TRIP",
                "Delete?": False,
                "Date": r["trip_date"],
                "Car": r["car_label"],
                "Departure": r["departure"],
                "Arrival": r["arrival"],
                "Distance (km)": float(r["distance_km"]),
                "Notes": r["notes"],
                "ID": str(r["id"]),
            })

    out = pd.DataFrame(rows)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date
    out["Distance (km)"] = pd.to_numeric(out["Distance (km)"], errors="coerce").fillna(0.0).astype(float)
    for col in ["Car", "Departure", "Arrival", "Notes", "ID", "RowType"]:
        out[col] = out[col].fillna("").astype(str)

    return out


# =========================
# SESSION STATE
# =========================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# This is the “current working date” for Add Trip + Skip
if "entry_date" not in st.session_state:
    st.session_state.entry_date = date.today()


# =========================
# MAIN UI
# =========================
st.title(APP_TITLE)
tabs = st.tabs(["🧾 Trip Log", "🛠️ Admin"])


# ---------- TRIP LOG TAB ----------
with tabs[0]:
    # Admin unlock
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

    # Period selection
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
        st.error("No active cars found. Add Mercedes / Volkswagen in the 'cars' table.")
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

    # Add trip + Skip date + weekday display
    st.subheader("Add a trip")

    with st.container(border=True):
        # show weekday for current entry_date
        weekday = st.session_state.entry_date.strftime("%A")
        st.caption(f"Selected date: **{st.session_state.entry_date.isoformat()}** ({weekday})")

        # allow manual change too
        st.session_state.entry_date = st.date_input(
            "Date",
            value=st.session_state.entry_date,
            key="entry_date_input"
        )

        colA, colB = st.columns([2, 1])
        with colA:
            car_label = st.selectbox("Car", car_labels)
            car_id = car_label_to_id[car_label]
        with colB:
            # Skip date button
            if st.button("⏭️ Skip date", use_container_width=True):
                st.session_state.entry_date = st.session_state.entry_date + timedelta(days=1)
                st.rerun()

        departure = place_picker("Departure", places, "dep")
        arrival = place_picker("Arrival", places, "arr")

        col3, col4 = st.columns(2)
        with col3:
            distance = st.number_input("Distance (km)", min_value=0.0, step=0.1, format="%.1f")
        with col4:
            notes = st.text_input("Notes (optional)")

        if st.button("✅ Save trip", use_container_width=True):
            if not departure or not arrival:
                st.error("Please fill in Departure and Arrival.")
            else:
                insert_trip(selected_period_id, st.session_state.entry_date, car_id, departure, arrival, distance, notes)
                upsert_place(departure)
                upsert_place(arrival)

                # Progress date after saving
                st.session_state.entry_date = st.session_state.entry_date + timedelta(days=1)

                st.success("Saved! Moving to next day.")
                st.rerun()

    st.divider()

    # Simple range + table (kept from polished version)
    colr1, colr2 = st.columns([1, 2])
    with colr1:
        range_mode = st.radio("Range", ["This month", "Custom"], horizontal=False)
    with colr2:
        if range_mode == "This month":
            pick = st.date_input("Month", value=date.today())
            start_date, end_date = month_range(pick)
        else:
            cA, cB = st.columns(2)
            with cA:
                start_date = st.date_input("Start", value=date(date.today().year, 1, 1))
            with cB:
                end_date = st.date_input("End", value=date.today())

    colf1, colf2 = st.columns(2)
    with colf1:
        filter_car = st.selectbox("Car", ["All cars"] + car_labels)
    with colf2:
        search_text = st.text_input("Search", placeholder="type to search...")

    filter_car_id = None if filter_car == "All cars" else car_label_to_id[filter_car]

    df = fetch_entries(selected_period_id, start_date, end_date, car_id=filter_car_id, search_text=search_text)

    total_km = 0.0 if df.empty else float(pd.to_numeric(df["distance_km"], errors="coerce").fillna(0).sum())
    st.metric("Total distance", f"{total_km:.1f} km")

    st.subheader("Trips")
    if df.empty:
        st.info("No trips found for this range.")
    else:
        grouped_editor_df = build_grouped_editor_df(df, car_id_to_label)

        edited = st.data_editor(
            grouped_editor_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "RowType": st.column_config.TextColumn("Type", disabled=True),
                "Delete?": st.column_config.CheckboxColumn("Delete?"),
                "Date": st.column_config.DateColumn("Date"),
                "Car": st.column_config.SelectboxColumn("Car", options=[""] + car_labels),
                "Departure": st.column_config.TextColumn("Departure"),
                "Arrival": st.column_config.TextColumn("Arrival"),
                "Distance (km)": st.column_config.NumberColumn("Distance (km)", min_value=0.0, step=0.1),
                "Notes": st.column_config.TextColumn("Notes"),
                "ID": st.column_config.TextColumn("ID", disabled=True),
            },
            disabled=["ID", "RowType"],
            key="one_table_editor",
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save edits", use_container_width=True):
                changes = 0

                original_trips = grouped_editor_df[grouped_editor_df["RowType"] == "TRIP"].copy()
                original_by_id = {row["ID"]: row for _, row in original_trips.iterrows() if row["ID"]}

                for _, row_new in edited.iterrows():
                    if row_new.get("RowType") != "TRIP":
                        continue

                    trip_id = str(row_new.get("ID", "")).strip()
                    if not trip_id:
                        continue

                    row_old = original_by_id.get(trip_id)
                    if row_old is None:
                        continue

                    if bool(row_new.get("Delete?", False)):
                        continue

                    updates = {}

                    if str(row_new["Date"]) != str(row_old["Date"]):
                        updates["trip_date"] = str(pd.to_datetime(row_new["Date"]).date())

                    if str(row_new["Car"]) != str(row_old["Car"]):
                        car_label_new = str(row_new["Car"]).strip()
                        if car_label_new in car_label_to_id:
                            updates["car_id"] = car_label_to_id[car_label_new]

                    for field_ui, field_db in [("Departure", "departure"), ("Arrival", "arrival"), ("Notes", "notes")]:
                        nv = "" if pd.isna(row_new[field_ui]) else str(row_new[field_ui]).strip()
                        ov = "" if pd.isna(row_old[field_ui]) else str(row_old[field_ui]).strip()
                        if nv != ov:
                            updates[field_db] = nv

                    nv_dist = float(row_new["Distance (km)"]) if str(row_new["Distance (km)"]).strip() else 0.0
                    ov_dist = float(row_old["Distance (km)"]) if str(row_old["Distance (km)"]).strip() else 0.0
                    if nv_dist != ov_dist:
                        updates["distance_km"] = nv_dist

                    if updates:
                        update_trip(trip_id, updates)
                        if "departure" in updates:
                            upsert_place(updates["departure"])
                        if "arrival" in updates:
                            upsert_place(updates["arrival"])
                        changes += 1

                st.success(f"Saved {changes} edited trip(s).")
                st.rerun()

        with b2:
            if st.button("🗑️ Delete selected", use_container_width=True):
                to_delete = []
                for _, r in edited.iterrows():
                    if r.get("RowType") == "TRIP" and bool(r.get("Delete?", False)):
                        tid = str(r.get("ID", "")).strip()
                        if tid:
                            to_delete.append(tid)

                if not to_delete:
                    st.info("No trips selected for deletion.")
                else:
                    delete_trips(to_delete)
                    st.success(f"Deleted {len(to_delete)} trip(s).")
                    st.rerun()

    st.divider()
    st.subheader("Export")

    default_name = f"{selected_period_name}_{start_date}_to_{end_date}"
    file_base = st.text_input("File name", value=default_name, help="Used for CSV / XLSX / PDF.")
    file_base = (file_base or "trips").strip().replace("/", "-")

    df_export = make_export_df_from_raw(df, car_id_to_label) if not df.empty else pd.DataFrame(
        columns=["Date", "Car", "Departure", "Arrival", "Distance (km)", "Notes"]
    )

    csv_bytes = export_csv_bytes(df_export)
    xlsx_bytes = export_xlsx_bytes(df_export)
    pdf_bytes = export_pdf_bytes(df_export, f"Trip Logbook — {selected_period_name}", total_km)

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


# ---------- ADMIN TAB ----------
with tabs[1]:
    if not st.session_state.is_admin:
        st.info("Admin is locked. Unlock it in the Trip Log tab.")
    else:
        st.header("Admin Panel")

        st.write("### Places manager (rename / deactivate)")
        places_df = fetch_places_admin()
        if places_df.empty:
            st.info("No places yet. They appear automatically when trips are saved.")
        else:
            places_df = places_df[["id", "name", "is_active", "created_at"]].copy()
            for col in ["id", "name"]:
                places_df[col] = places_df[col].fillna("").astype(str)

            edited_places = st.data_editor(
                places_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "id": st.column_config.TextColumn("ID", disabled=True),
                    "name": st.column_config.TextColumn("Place name"),
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
                    if str(n["name"]).strip() != str(o["name"]).strip():
                        updates["name"] = str(n["name"]).strip()
                    if bool(n["is_active"]) != bool(o["is_active"]):
                        updates["is_active"] = bool(n["is_active"])

                    if updates:
                        update_place(pid, updates)
                        changed += 1

                st.success(f"Updated {changed} place(s).")
                st.rerun()
