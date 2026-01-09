import os
import io
from datetime import date
import pandas as pd
import streamlit as st
from supabase import create_client, Client
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Trip Logbook", layout="centered")
st.title("🚗 Trip Logbook")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL / SUPABASE_KEY in secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_cars():
    res = supabase.table("cars").select("id,name,plate").eq("is_active", True).order("name").execute()
    return res.data or []

def insert_trip(trip_date, car_id, departure, arrival, distance_km):
    payload = {
        "trip_date": str(trip_date),
        "car_id": car_id,
        "departure": departure.strip(),
        "arrival": arrival.strip(),
        "distance_km": float(distance_km),
    }
    return supabase.table("trip_entries").insert(payload).execute()

def month_range(d: date):
    start = d.replace(day=1)
    if start.month == 12:
        next_month = start.replace(year=start.year + 1, month=1, day=1)
    else:
        next_month = start.replace(month=start.month + 1, day=1)
    last_day = (pd.Timestamp(next_month) - pd.Timedelta(days=1)).date()
    return start, last_day

def fetch_entries(month_start, month_end, car_id=None):
    q = (
        supabase.table("trip_entries")
        .select("trip_date,departure,arrival,distance_km,car_id,cars(name)")
        .gte("trip_date", str(month_start))
        .lte("trip_date", str(month_end))
        .order("trip_date", desc=False)
    )
    if car_id:
        q = q.eq("car_id", car_id)

    res = q.execute()
    rows = res.data or []
    for r in rows:
        r["car_name"] = (r.get("cars") or {}).get("name", "")
        r.pop("cars", None)

    df = pd.DataFrame(rows)
    return df

def make_export_df(df):
    if df.empty:
        return df
    out = df.copy()
    out = out[["trip_date", "car_name", "departure", "arrival", "distance_km"]]
    out.columns = ["Date", "Car", "Departure", "Arrival", "Distance (km)"]
    out["Date"] = pd.to_datetime(out["Date"]).dt.date.astype(str)
    return out

def export_pdf_bytes(df, title):
    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=A4)
    w, h = A4

    x = 40
    y = h - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 30

    c.setFont("Helvetica", 10)
    if df.empty:
        c.drawString(x, y, "No trips found.")
        c.showPage()
        c.save()
        return buff.getvalue()

    col_widths = [80, 70, 150, 150, 80]
    headers = list(df.columns)

    c.setFont("Helvetica-Bold", 9)
    xx = x
    for name, cw in zip(headers, col_widths):
        c.drawString(xx, y, name)
        xx += cw
    y -= 14
    c.setFont("Helvetica", 9)

    for _, row in df.iterrows():
        if y < 60:
            c.showPage()
            y = h - 50
            c.setFont("Helvetica", 9)

        xx = x
        vals = [str(row[col]) for col in headers]
        for v, cw in zip(vals, col_widths):
            c.drawString(xx, y, v[:28])
            xx += cw
        y -= 12

    c.showPage()
    c.save()
    return buff.getvalue()

cars = get_cars()
if not cars:
    st.warning("No cars found. Add cars in Supabase table 'cars'.")
    st.stop()

car_labels = []
car_lookup = {}
for c in cars:
    label = c["name"] + (f" ({c['plate']})" if c.get("plate") else "")
    car_labels.append(label)
    car_lookup[label] = c["id"]

st.subheader("Add a trip")
trip_date = st.date_input("Date", value=date.today())
car_label = st.selectbox("Car", car_labels)
car_id = car_lookup[car_label]

departure = st.text_input("Departure")
arrival = st.text_input("Arrival")
distance = st.number_input("Distance (km)", min_value=0.0, step=0.1, format="%.1f")

if st.button("✅ Save trip", use_container_width=True):
    if not departure.strip() or not arrival.strip():
        st.error("Please fill in Departure and Arrival.")
    else:
        insert_trip(trip_date, car_id, departure, arrival, distance)
        st.success("Saved!")
        st.rerun()

st.divider()
st.subheader("Monthly overview")

month_pick = st.date_input("Pick a day in the month", value=date.today(), key="monthpick")
month_start, month_end = month_range(month_pick)

filter_car = st.selectbox("Filter car (optional)", ["All cars"] + car_labels)
filter_car_id = None if filter_car == "All cars" else car_lookup[filter_car]

df = fetch_entries(month_start, month_end, filter_car_id)
df_export = make_export_df(df)

total_km = 0.0 if df.empty else float(pd.to_numeric(df["distance_km"], errors="coerce").fillna(0).sum())
st.metric(f"Total distance for {month_start.strftime('%B %Y')}", f"{total_km:.1f} km")

if df_export.empty:
    st.info("No trips yet for this month.")
else:
    st.dataframe(df_export, use_container_width=True, hide_index=True)

st.subheader("Export")
csv_bytes = df_export.to_csv(index=False).encode("utf-8")
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
    df_export.to_excel(writer, index=False, sheet_name="Trips")
xlsx_bytes = xlsx_buf.getvalue()

pdf_bytes = export_pdf_bytes(df_export, f"Trip Logbook — {month_start.strftime('%B %Y')}")

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("⬇️ CSV", csv_bytes, f"trips_{month_start.strftime('%Y_%m')}.csv", "text/csv", use_container_width=True)
with c2:
    st.download_button("⬇️ XLSX", xlsx_bytes, f"trips_{month_start.strftime('%Y_%m')}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
with c3:
    st.download_button("⬇️ PDF", pdf_bytes, f"trips_{month_start.strftime('%Y_%m')}.pdf", "application/pdf", use_container_width=True)
