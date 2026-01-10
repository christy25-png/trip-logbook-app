import os
import io
from datetime import date

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


def ensure_period(name: str) -> str | None:
    name = (name or "").strip()
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
    name = (name or "").strip()
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
        existing = supabase.table("places").select("id").eq("name", name).limit(1).execute().data
        if existing:
            pid = existing[0]["id"]
            supabase.table("places").update({"is_active": True}).eq("id", pid).execute()
        else:
            supabase.table("places").insert({"name": name, "is_active": True}).execute()
    except Exception:
        pass


def normalize_pair(a: str, b: str) -> tuple[str, str]:
    a = (a or "").strip()
    b = (b or "").strip()
    return (a, b) if a.lower() <= b.lower() else (b, a)


def get_route_distance(departure: str, arrival: str) -> float | None:
    dep = (departure or "").strip()
    arr = (arrival or "").strip()
    if not dep or not arr:
        return None
    a, b = normalize_pair(dep, arr)
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


def set_route_distance(departure: str, arrival: str, distance_km: float):
    dep = (departure or "").strip()
    arr = (arrival or "").strip()
    if not dep or not arr:
        return
    a, b = normalize_pair(dep, arr)
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
            supabase.table("route_distances").insert(
                {"place_a": a, "place_b": b, "distance_km": float(distance_km)}
            ).execute()
    except Exception:
        pass


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


# =========================
# SESSION STATE
# =========================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# Keys used by widgets (do NOT assign widget return values into these keys)
if "dep_choice" not in st.session_state:
    st.session_state.dep_choice = ""
if "arr_choice" not in st.session_state:
    st.session_state.arr_choice = ""
if "dep_typed" not in st.session_state:
    st.session_state.dep_typed = ""
if "arr_typed" not in st.session_state:
    st.session_state.arr_typed = ""

if "distance_value" not in st.session_state:
    st.session_state.distance_value = 0.0
if "distance_manual" not in st.session_state:
    st.session_state.distance_manual = False


def get_dep_value(places_list: list[str]) -> str:
    if st.session_state.dep_choice == PLACE_TYPE_NEW:
        return (st.session_state.dep_typed or "").strip()
    if st.session_state.dep_choice in places_list:
        return (st.session_state.dep_choice or "").strip()
    return (st.session_state.dep_typed or "").strip()


def get_arr_value(places_list: list[str]) -> str:
    if st.session_state.arr_choice == PLACE_TYPE_NEW:
        return (st.session_state.arr_typed or "").strip()
    if st.session_state.arr_choice in places_list:
        return (st.session_state.arr_choice or "").strip()
    return (st.session_state.arr_typed or "").strip()


def autofill_distance_if_possible(places_list: list[str]):
    st.session_state.distance_manual = False
    dep = get_dep_value(places_list)
    arr = get_arr_value(places_list)
    d = get_route_distance(dep, arr)
    if d is not None:
        st.session_state.distance_value = float(d)


def on_distance_change():
    st.session_state.distance_manual = True


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

    # Period selection (default to current year)
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
        st.error("No active cars found. Add your cars in the 'cars' table.")
        st.stop()

    car_label_to_id = {}
    car_id_to_label = {}
    for c in cars:
        label = c["name"] + (f" ({c['plate']})" if c.get("plate") else "")
        car_label_to_id[label] = c["id"]
        car_id_to_label[c["id"]] = label
    car_labels = list(car_label_to_id.keys())

    places = get_places()
    dep_options = places + [PLACE_TYPE_NEW]
    arr_options = places + [PLACE_TYPE_NEW]

    # keep current selections valid
    if st.session_state.dep_choice and st.session_state.dep_choice not in dep_options:
        st.session_state.dep_choice = PLACE_TYPE_NEW
    if st.session_state.arr_choice and st.session_state.arr_choice not in arr_options:
        st.session_state.arr_choice = PLACE_TYPE_NEW

    # ----- Add a trip -----
    st.subheader("Add a trip")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            trip_date = st.date_input("Date", value=date.today())
            st.caption(f"Day: **{trip_date.strftime('%A')}**")
        with col2:
            car_label = st.selectbox("Car", car_labels)
            car_id = car_label_to_id[car_label]

        # Swap button (works)
        swap_col1, swap_col2 = st.columns([1, 3])
        with swap_col1:
            if st.button("↔ Swap", use_container_width=True):
                st.session_state.dep_choice, st.session_state.arr_choice = (
                    st.session_state.arr_choice,
                    st.session_state.dep_choice,
                )
                st.session_state.dep_typed, st.session_state.arr_typed = (
                    st.session_state.arr_typed,
                    st.session_state.dep_typed,
                )
                autofill_distance_if_possible(places)
                st.rerun()
        with swap_col2:
            st.caption("Swap Departure and Arrival")

        # IMPORTANT FIX:
        # Do NOT assign widget return values into st.session_state keys.
        st.selectbox(
            "Departure",
            dep_options,
            index=dep_options.index(st.session_state.dep_choice) if st.session_state.dep_choice in dep_options else 0,
            key="dep_choice",
            on_change=lambda: autofill_distance_if_possible(places),
        )
        if st.session_state.dep_choice == PLACE_TYPE_NEW:
            st.text_input(
                "Departure (type)",
                value=st.session_state.dep_typed,
                key="dep_typed",
                on_change=lambda: autofill_distance_if_possible(places),
            )

        st.selectbox(
            "Arrival",
            arr_options,
            index=arr_options.index(st.session_state.arr_choice) if st.session_state.arr_choice in arr_options else 0,
            key="arr_choice",
            on_change=lambda: autofill_distance_if_possible(places),
        )
        if st.session_state.arr_choice == PLACE_TYPE_NEW:
            st.text_input(
                "Arrival (type)",
                value=st.session_state.arr_typed,
                key="arr_typed",
                on_change=lambda: autofill_distance_if_possible(places),
            )

        # auto-fill distance if possible (only when user hasn't manually edited)
        if not st.session_state.distance_manual:
            dep_now = get_dep_value(places)
            arr_now = get_arr_value(places)
            if dep_now and arr_now:
                mem = get_route_distance(dep_now, arr_now)
                if mem is not None:
                    st.session_state.distance_value = float(mem)

        col3, col4 = st.columns(2)
        with col3:
            distance = st.number_input(
                "Distance (km)",
                min_value=0.0,
                step=0.1,
                format="%.1f",
                value=float(st.session_state.distance_value),
                key="distance_input",
                on_change=on_distance_change,
            )
        with col4:
            notes = st.text_input("Notes (optional)")

        st.session_state.distance_value = float(distance)

        if st.button("✅ Save trip", use_container_width=True):
            departure = get_dep_value(places)
            arrival = get_arr_value(places)

            if not departure or not arrival:
                st.error("Please fill in Departure and Arrival.")
            else:
                insert_trip(selected_period_id, trip_date, car_id, departure, arrival, float(distance), notes)
                upsert_place(departure)
                upsert_place(arrival)
                set_route_distance(departure, arrival, float(distance))

                # after saving, allow next route to autofill again
                st.session_state.distance_manual = False

                st.success("Saved!")
                st.rerun()

    # ======= The rest of your app (Trips, Manage, Export, Admin) remains the same style =======

    st.divider()

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
    df_export = make_export_df(df, car_id_to_label)
    if df_export.empty:
        st.info("No trips found.")
    else:
        st.dataframe(df_export, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Manage trips (edit / delete)")
    if df.empty:
        st.info("Nothing to manage for this selection.")
    else:
        manage_df = df.copy()
        manage_df["trip_date"] = pd.to_datetime(manage_df["trip_date"], errors="coerce").dt.date
        manage_df["distance_km"] = pd.to_numeric(manage_df["distance_km"], errors="coerce").fillna(0.0).astype(float)
        manage_df["car_label"] = manage_df["car_id"].map(car_id_to_label).fillna("")
        manage_df["DELETE"] = False

        manage_df = manage_df[[
            "DELETE", "trip_date", "car_label", "departure", "arrival", "distance_km", "notes", "id"
        ]].copy()

        for col in ["departure", "arrival", "notes", "car_label", "id"]:
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
                    trip_id = str(new_row["id"]).strip()

                    if bool(new_row["DELETE"]):
                        continue

                    updates = {}

                    if str(new_row["trip_date"]) != str(old_row["trip_date"]):
                        updates["trip_date"] = str(pd.to_datetime(new_row["trip_date"]).date())

                    if str(new_row["car_label"]) != str(old_row["car_label"]):
                        updates["car_id"] = car_label_to_id.get(str(new_row["car_label"]))

                    for field in ["departure", "arrival", "notes"]:
                        nv = "" if pd.isna(new_row[field]) else str(new_row[field]).strip()
                        ov = "" if pd.isna(old_row[field]) else str(old_row[field]).strip()
                        if nv != ov:
                            updates[field] = nv

                    if float(new_row["distance_km"]) != float(old_row["distance_km"]):
                        updates["distance_km"] = float(new_row["distance_km"])

                    if updates:
                        update_trip(trip_id, updates)

                        dep_now = updates.get("departure", str(old_row["departure"]))
                        arr_now = updates.get("arrival", str(old_row["arrival"]))
                        dist_now = updates.get("distance_km", float(old_row["distance_km"]))

                        upsert_place(dep_now)
                        upsert_place(arr_now)
                        set_route_distance(dep_now, arr_now, float(dist_now))

                        changes += 1

                st.success(f"Saved edits on {changes} trip(s).")
                st.rerun()

        with b2:
            if st.button("🗑️ Delete selected", use_container_width=True):
                to_delete = edited.loc[edited["DELETE"] == True, "id"].astype(str).tolist()
                if not to_delete:
                    st.info("No trips selected.")
                else:
                    delete_trips(to_delete)
                    st.success(f"Deleted {len(to_delete)} trip(s).")
                    st.rerun()

    st.divider()
    st.subheader("Export")

    default_name = f"{selected_period_name}_{start_date}_to_{end_date}"
    file_base = st.text_input("File name", value=default_name, help="Used for CSV / XLSX / PDF.")
    file_base = (file_base or "trips").strip().replace("/", "-")

    if df_export.empty:
        export_df = pd.DataFrame(columns=["Date", "Car", "Departure", "Arrival", "Distance (km)", "Notes"])
    else:
        export_df = df_export

    csv_bytes = export_csv_bytes(export_df)
    xlsx_bytes = export_xlsx_bytes(export_df)
    pdf_bytes = export_pdf_bytes(export_df, f"Trip Logbook — {selected_period_name}", total_km)

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
