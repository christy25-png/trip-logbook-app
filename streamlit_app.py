import os
import io
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st
from supabase import create_client, Client

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================
# App Config
# =========================
st.set_page_config(page_title="Trip Logbook", layout="centered")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))
ADMIN_PIN = st.secrets.get("ADMIN_PIN", os.getenv("ADMIN_PIN", ""))
APP_TITLE = st.secrets.get("APP_TITLE", os.getenv("APP_TITLE", "🚗 Trip Logbook"))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL / SUPABASE_KEY in Streamlit secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================
# Supabase Helpers
# =========================
def sb_select(table: str, columns: str):
    return supabase.table(table).select(columns)


def get_cars():
    res = (
        sb_select("cars", "id,name,plate,is_active")
        .eq("is_active", True)
        .order("name")
        .execute()
    )
    return res.data or []


def get_places(limit=400):
    res = (
        sb_select("places", "name,is_active")
        .eq("is_active", True)
        .order("name")
        .limit(limit)
        .execute()
    )
    return [r["name"] for r in (res.data or [])]


def upsert_place(name: str):
    name = (name or "").strip()
    if not name:
        return
    # 'name' is unique; upsert will insert if new, or update is_active if exists.
    supabase.table("places").upsert({"name": name, "is_active": True}).execute()


def insert_trip(trip_date, car_id, departure, arrival, distance_km, notes=""):
    payload = {
        "trip_date": str(trip_date),
        "car_id": car_id,
        "departure": departure.strip(),
        "arrival": arrival.strip(),
        "distance_km": float(distance_km),
        "notes": notes.strip() if notes else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return supabase.table("trip_entries").insert(payload).execute()


def fetch_entries(start_date: date, end_date: date, car_id=None, search_text=""):
    q = (
        supabase.table("trip_entries")
        .select("id,trip_date,departure,arrival,distance_km,notes,created_at,updated_at,car_id,cars(name)")
        .gte("trip_date", str(start_date))
        .lte("trip_date", str(end_date))
        .order("trip_date", desc=False)
        .order("created_at", desc=False)
    )
    if car_id:
        q = q.eq("car_id", car_id)

    res = q.execute()
    rows = res.data or []

    # Flatten cars(name) join
    for r in rows:
        r["car_name"] = (r.get("cars") or {}).get("name", "")
        r.pop("cars", None)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # optional search filter
    if search_text.strip():
        s = search_text.strip().lower()
        mask = (
            df["departure"].astype(str).str.lower().str.contains(s, na=False)
            | df["arrival"].astype(str).str.lower().str.contains(s, na=False)
            | df["notes"].astype(str).str.lower().str.contains(s, na=False)
        )
        df = df[mask].copy()

    return df


def update_trip(trip_id: str, updates: dict):
    updates = dict(updates)
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    return supabase.table("trip_entries").update(updates).eq("id", trip_id).execute()


def delete_trips(trip_ids: list[str]):
    if not trip_ids:
        return
    return supabase.table("trip_entries").delete().in_("id", trip_ids).execute()


def fetch_places_admin():
    res = sb_select("places", "id,name,is_active,created_at").order("name").execute()
    return pd.DataFrame(res.data or [])


def update_place(place_id: str, updates: dict):
    return supabase.table("places").update(updates).eq("id", place_id).execute()


# =========================
# Period Helpers
# =========================
def period_selector():
    st.write("### Period")
    mode = st.radio("Choose period type", ["Year", "Custom range"], horizontal=True)
    if mode == "Year":
        current_year = date.today().year
        years = list(range(current_year - 5, current_year + 6))
        year = st.selectbox("Year", years, index=years.index(current_year))
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        label = f"{year}"
        return start, end, label
    else:
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Start date", value=date(date.today().year, 1, 1))
        with c2:
            end = st.date_input("End date", value=date.today())
        if end < start:
            st.error("End date must be after start date.")
        label = f"{start.isoformat()}_to_{end.isoformat()}"
        return start, end, label


# =========================
# Export Helpers
# =========================
def make_export_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    cols = ["trip_date", "car_name", "departure", "arrival", "distance_km", "notes"]
    out = out[cols]
    out = out.rename(columns={
        "trip_date": "Date",
        "car_name": "Car",
        "departure": "Departure",
        "arrival": "Arrival",
        "distance_km": "Distance (km)",
        "notes": "Notes",
    })
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
        c.drawString(x, y, "No trips found for this period.")
        c.showPage()
        c.save()
        return buff.getvalue()

    # Table layout
    headers = list(df.columns)
    col_widths = [75, 65, 135, 135, 70, 120]  # fits A4 reasonably
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
# UI Helpers (Place picker)
# =========================
PLACE_TYPE_NEW = "✍️ Type a new place..."

def place_picker(label: str, places: list[str], key_prefix: str) -> str:
    choice = st.selectbox(label, options=(places + [PLACE_TYPE_NEW]), key=f"{key_prefix}_choice")
    if choice == PLACE_TYPE_NEW:
        return st.text_input(f"{label} (type)", key=f"{key_prefix}_typed").strip()
    return choice.strip()


# =========================
# Session State
# =========================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False


# =========================
# App Header
# =========================
st.title(APP_TITLE)

with st.expander("🔐 Admin mode", expanded=False):
    if not ADMIN_PIN:
        st.info("Set ADMIN_PIN in your Streamlit secrets to enable Admin mode.")
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


# =========================
# Load cars / maps
# =========================
cars = get_cars()
if not cars:
    st.error("No cars found. Add cars in Supabase table 'cars' or run the SQL starter insert.")
    st.stop()

car_label_to_id = {}
car_id_to_label = {}

for c in cars:
    label = c["name"] + (f" ({c['plate']})" if c.get("plate") else "")
    car_label_to_id[label] = c["id"]
    car_id_to_label[c["id"]] = label

car_labels = list(car_label_to_id.keys())


# =========================
# Add Trip (Dad-friendly)
# =========================
st.subheader("Add a trip")

places = get_places()

with st.container(border=True):
    col1, col2 = st.columns([1, 1])
    with col1:
        trip_date = st.date_input("Date", value=date.today())
    with col2:
        car_label = st.selectbox("Car", car_labels)
        car_id = car_label_to_id[car_label]

    departure = place_picker("Departure", places, "dep")
    arrival = place_picker("Arrival", places, "arr")

    col3, col4 = st.columns([1, 1])
    with col3:
        distance = st.number_input("Distance (km)", min_value=0.0, step=0.1, format="%.1f")
    with col4:
        notes = st.text_input("Notes (optional)")

    if st.button("✅ Save trip", use_container_width=True):
        if not departure or not arrival:
            st.error("Please fill in Departure and Arrival.")
        else:
            insert_trip(trip_date, car_id, departure, arrival, distance, notes)
            upsert_place(departure)
            upsert_place(arrival)
            st.success("Saved!")
            st.rerun()


st.divider()

# =========================
# Overview + Period + Filters
# =========================
period_start, period_end, period_label = period_selector()

st.write("### Filters")
f1, f2 = st.columns([1, 1])
with f1:
    filter_car = st.selectbox("Car filter", ["All cars"] + car_labels)
with f2:
    search_text = st.text_input("Search (departure/arrival/notes)", placeholder="type to filter...")

filter_car_id = None if filter_car == "All cars" else car_label_to_id[filter_car]

df = fetch_entries(period_start, period_end, car_id=filter_car_id, search_text=search_text)
df_export = make_export_df(df)

total_km = 0.0
if not df.empty:
    total_km = float(pd.to_numeric(df["distance_km"], errors="coerce").fillna(0).sum())

st.metric(
    label=f"Total distance ({period_label})" + ("" if filter_car == "All cars" else f" • {filter_car}"),
    value=f"{total_km:.1f} km",
)

st.write("### Trips")
if df_export.empty:
    st.info("No trips found for this period/filter.")
else:
    st.dataframe(df_export, use_container_width=True, hide_index=True)

# =========================
# Export
# =========================
st.write("### Export")
csv_bytes = export_csv_bytes(df_export)
xlsx_bytes = export_xlsx_bytes(df_export)
pdf_bytes = export_pdf_bytes(df_export, f"Trip Logbook — {period_label}", total_km)

e1, e2, e3 = st.columns(3)
with e1:
    st.download_button(
        "⬇️ CSV",
        data=csv_bytes,
        file_name=f"trips_{period_label}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with e2:
    st.download_button(
        "⬇️ XLSX",
        data=xlsx_bytes,
        file_name=f"trips_{period_label}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with e3:
    st.download_button(
        "⬇️ PDF",
        data=pdf_bytes,
        file_name=f"trips_{period_label}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

st.divider()

# =========================
# Manage Trips (Edit/Delete) - Dad can do it
# =========================
st.subheader("Manage trips (edit / delete)")

if df.empty:
    st.info("Nothing to manage for this period.")
else:
    manage_df = df.copy()

    # Friendly car label column for editing
    manage_df["car_label"] = manage_df["car_id"].map(car_id_to_label).fillna("")

    # Add delete checkbox
    manage_df["DELETE"] = False

    # Keep only what we want to edit + id
    manage_df = manage_df[[
        "DELETE",
        "trip_date",
        "car_label",
        "departure",
        "arrival",
        "distance_km",
        "notes",
        "id",
    ]].copy()

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

                # Compare fields
                if str(new_row["trip_date"]) != str(old_row["trip_date"]):
                    updates["trip_date"] = str(pd.to_datetime(new_row["trip_date"]).date())

                if str(new_row["car_label"]) != str(old_row["car_label"]):
                    updates["car_id"] = car_label_to_id.get(str(new_row["car_label"]), None)

                for field in ["departure", "arrival", "notes"]:
                    if str(new_row[field]) != str(old_row[field]):
                        updates[field] = ("" if pd.isna(new_row[field]) else str(new_row[field]).strip())

                if str(new_row["distance_km"]) != str(old_row["distance_km"]):
                    updates["distance_km"] = float(new_row["distance_km"])

                # Clean invalid updates
                if updates.get("car_id") is None and "car_id" in updates:
                    updates.pop("car_id", None)

                if updates:
                    update_trip(trip_id, updates)
                    # Keep place history updated
                    upsert_place(str(new_row["departure"]))
                    upsert_place(str(new_row["arrival"]))
                    changes += 1

            st.success(f"Saved edits on {changes} row(s).")
            st.rerun()

    with b2:
        if st.button("🗑️ Delete selected", use_container_width=True):
            to_delete = edited.loc[edited["DELETE"] == True, "id"].astype(str).tolist()
            if not to_delete:
                st.info("No rows selected for deletion.")
            else:
                delete_trips(to_delete)
                st.success(f"Deleted {len(to_delete)} row(s).")
                st.rerun()

# =========================
# Admin Panel (Extra features)
# =========================
if st.session_state.is_admin:
    st.divider()
    st.header("Admin Panel")

    st.write("### Places manager")
    places_df = fetch_places_admin()
    if places_df.empty:
        st.info("No places yet. They will appear automatically when trips are saved.")
    else:
        # Add a column to toggle active/inactive and allow renaming
        places_df = places_df[["id", "name", "is_active", "created_at"]].copy()
        places_df = places_df.sort_values("name")

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
            # Compare row-by-row (same order)
            for i in range(len(edited_places)):
                n = edited_places.iloc[i]
                o = places_df.iloc[i]
                pid = str(n["id"])

                updates = {}
                if str(n["name"]) != str(o["name"]):
                    updates["name"] = str(n["name"]).strip()
                if bool(n["is_active"]) != bool(o["is_active"]):
                    updates["is_active"] = bool(n["is_active"])

                if updates:
                    try:
                        update_place(pid, updates)
                        changed += 1
                    except Exception as e:
                        st.error(f"Could not update place '{o['name']}': {e}")

            st.success(f"Updated {changed} place(s).")
            st.rerun()

    st.write("### Admin tips")
    st.caption(
        "If you want the app private/secure later, we can enable RLS and add login or a shared access PIN. "
        "Right now, Admin mode protects admin tools inside the app, but database security depends on RLS settings."
    )
