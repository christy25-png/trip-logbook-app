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
# DEBUG + SAFETY HELPERS
# =========================
def clean_text(s: str) -> str:
    if not s:
        return ""
    return " ".join(str(s).strip().split())


def safe_execute(query, context: str = ""):
    """
    Executes a supabase query and prints FULL error details if it fails.
    """
    try:
        res = query.execute()
        return res
    except Exception as e:
        st.error("Database request failed.")
        if context:
            st.caption(f"Context: {context}")

        # Show *everything* we can (this is what you need right now)
        st.write("### Details (full error):")
        st.exception(e)

        # Some supabase/postgrest errors contain a dict in e.args[0]
        try:
            if hasattr(e, "args") and e.args:
                st.write("### Raw args:")
                st.write(e.args)
        except Exception:
            pass

        return None


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

    q1 = supabase.table("periods").select("id").eq("name", name).limit(1)
    r1 = safe_execute(q1, "ensure_period(): lookup")
    if r1 and r1.data:
        return r1.data[0]["id"]

    q2 = supabase.table("periods").insert({"name": name, "is_active": True})
    r2 = safe_execute(q2, "ensure_period(): insert")
    if r2 and r2.data:
        return r2.data[0]["id"]
    return None


def create_period(name: str):
    name = clean_text(name)
    if not name:
        return None

    q1 = supabase.table("periods").select("id,name").eq("name", name).limit(1)
    r1 = safe_execute(q1, "create_period(): lookup")
    if r1 and r1.data:
        return r1.data[0]

    q2 = supabase.table("periods").insert({"name": name, "is_active": True})
    r2 = safe_execute(q2, "create_period(): insert")
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
        st.error("Please fill both label and address.")
        return None

    # Upsert-ish: if label exists, update address
    q1 = supabase.table("places").select("id,label").eq("label", label).limit(1)
    r1 = safe_execute(q1, "create_place(): lookup")
    if r1 and r1.data:
        pid = r1.data[0]["id"]
        q_upd = supabase.table("places").update({"address": address, "is_active": True}).eq("id", pid)
        r_upd = safe_execute(q_upd, "create_place(): update existing")
        return {"id": pid, "label": label} if r_upd else None

    q2 = supabase.table("places").insert({"label": label, "address": address, "is_active": True})
    r2 = safe_execute(q2, "create_place(): insert")
    if r2 and r2.data:
        return r2.data[0]
    return None


# =========================
# UI
# =========================
st.title(APP_TITLE)

with st.expander("🧪 Debug / Connection test"):
    st.caption("This helps diagnose whether it’s RLS, wrong keys, or missing tables.")
    if st.button("Test read: SELECT from places"):
        res = safe_execute(supabase.table("places").select("*").limit(5), "connection test: select places")
        if res:
            st.success("Read works.")
            st.write(res.data)

    if st.button("Test insert: add a dummy place"):
        res = safe_execute(
            supabase.table("places").insert({"label": "TEST_PLACE", "address": "TEST_ADDRESS", "is_active": True}),
            "connection test: insert places",
        )
        if res:
            st.success("Insert works.")
            st.write(res.data)

st.subheader("Add a place (name + address)")
with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        new_label = st.text_input("Place name (easy label)", value="", key="new_place_label")
    with c2:
        new_address = st.text_input("Full address", value="", key="new_place_address")

    if st.button("Save place", use_container_width=True):
        created = create_place(new_label, new_address)
        if created:
            st.success("Place saved!")
            st.rerun()
