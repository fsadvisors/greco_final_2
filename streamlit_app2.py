import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from difflib import SequenceMatcher

st.set_page_config(page_title="Greco.AI Reconciliation", page_icon="🧾", layout="wide")
st.sidebar.title("Greco.AI")
st.sidebar.markdown("Automated file fetch from a public Google Sheet containing direct file links.")

sheet_url = st.sidebar.text_input("Paste your public Google Sheet URL here:")

HEADERS = [
    "Party",
    "GSTIN",
    "Invoice Date",
    "Invoice No",
    "Invoice Value",
    "Taxable Value",
    "CGST",
    "SGST",
    "IGST",
    "CESS",
]

def get_gsheet_data(sheet_url):
    try:
        key = sheet_url.split("/d/")[1].split("/")[0]
        export_url = f"https://docs.google.com/spreadsheets/d/{key}/export?format=csv"
        df = pd.read_csv(export_url)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Could not load Google Sheet: {e}")
        return None

def get_latest_link(df, file_type):
    filtered = df[df["type"].str.lower() == file_type]
    if len(filtered) > 0:
        return filtered.iloc[-1]["link"]
    return None

def read_file_from_link(link):
    try:
        if link.endswith(".csv") or "format=csv" in link:
            return pd.read_csv(link)
        elif link.endswith(".xlsx") or "format=xlsx" in link:
            return pd.read_excel(link)
        else:
            st.error("Unknown file format. Make sure the link ends with .csv or .xlsx or contains ?format=csv/xlsx")
            return None
    except Exception as e:
        st.error(f"Could not read the file from link: {e}")
        return None

def get_suffix(filename: str) -> str:
    fn = filename.lower()
    if "portal" in fn or "gst" in fn:
        return "_portal"
    if "books" in fn:
        return "_books"
    return ""

def clean_df(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna(subset=[f"Invoice No{suffix}", f"GSTIN{suffix}"])
    vals = [f"{c}{suffix}" for c in ["Invoice Value","Taxable Value","IGST","CGST","SGST"]]
    return df.dropna(subset=vals, how="all").reset_index(drop=True)

def make_remark_logic(row, gst_sfx, books_sfx, amount_tol, date_tol):
    def getval(r, col):
        v = r.get(col, "")
        return "" if pd.isna(v) or str(v).strip()=="" else v
    def norm_id(s):  return re.sub(r"[\W_]+","",str(s)).lower()
    def strip_ws(s): return re.sub(r"\s+","",str(s)).lower()
    def sim(a,b):    return SequenceMatcher(None,a,b).ratio()
    gst_cols   = [f+gst_sfx for f in HEADERS]
    books_cols = [f+books_sfx for f in HEADERS]
    if all(getval(row,c)=="" for c in gst_cols):
        return "❌ Not in 2B"
    if all(getval(row,c)=="" for c in books_cols):
        return "❌ Not in books"
    mismatches = []
    trivial = False
    b_date = row.get(f"Invoice Date{books_sfx}")
    g_date = row.get(f"Invoice Date{gst_sfx}")
    if pd.notna(b_date) and pd.notna(g_date):
        try:
            bd = pd.to_datetime(b_date)
            gd = pd.to_datetime(g_date)
            delta_days = abs((bd - gd).days)
            if delta_days == 0:
                pass
            elif delta_days <= date_tol:
                trivial = True
            else:
                mismatches.append("⚠️ Mismatch of Invoice Date")
        except:
            pass
    b_no = getval(row, f"Invoice No{books_sfx}")
    g_no = getval(row, f"Invoice No{gst_sfx}")
    if norm_id(b_no) != norm_id(g_no):
        mismatches.append("⚠️ Mismatch of Invoice No")
    elif strip_ws(b_no) != strip_ws(g_no):
        trivial = True
    b_g = str(getval(row, f"GSTIN{books_sfx}")).lower()
    g_g = str(getval(row, f"GSTIN{gst_sfx}")).lower()
    if b_g and g_g and b_g != g_g:
        mismatches.append("⚠️ Mismatch of GSTIN")
    for fld in ["Invoice Value","Taxable Value","IGST","CGST","SGST","CESS"]:
        bv = row.get(f"{fld}{books_sfx}", 0) or 0
        gv = row.get(f"{fld}{gst_sfx}",   0) or 0
        try:
            diff = abs(float(bv) - float(gv))
            if diff > amount_tol:
                mismatches.append(f"⚠️ Mismatch of {fld}")
            elif diff > 0:
                trivial = True
        except:
            pass
    bp = str(getval(row, f"Party{books_sfx}"))
    gp = str(getval(row, f"Party{gst_sfx}"))
    s = sim(re.sub(r"[^\w\s]","",bp).lower(), re.sub(r"[^\w\s]","",gp).lower())
    if s < 0.8:
        mismatches.append("⚠️ Mismatch of Party")
    elif s < 1.0:
        trivial = True
    if mismatches:
        return " & ".join(dict.fromkeys(mismatches))
    if trivial:
        return "✅ Matched, trivial error"
    return "✅ Matched"

gst_file_link = books_file_link = ""
df_gst = df_books = None

if sheet_url:
    gsheet_df = get_gsheet_data(sheet_url)
    if gsheet_df is not None:
        st.success("Loaded Google Sheet!")
        with st.expander("Show Sheet Data"):
            st.dataframe(gsheet_df)
        gst_file_link = get_latest_link(gsheet_df, "gst")
        books_file_link = get_latest_link(gsheet_df, "books")

        if gst_file_link:
            df_gst = read_file_from_link(gst_file_link)
            if df_gst is not None:
                st.success("GST file loaded.")

        if books_file_link:
            df_books = read_file_from_link(books_file_link)
            if df_books is not None:
                st.success("Books file loaded.")

        if not gst_file_link or not df_gst is not None or not books_file_link or not df_books is not None:
            st.warning("Did not find both GST and Books links in the sheet or failed to load files.")

with st.expander("⚙️ Threshold Settings", expanded=True):
    amt_threshold  = st.selectbox(
        "Amount difference threshold for mismatch",
        [0.01, 0.1, 1, 10, 100],
        index=0
    )
    date_threshold = st.selectbox(
        "Date difference threshold (days)",
        [1, 2, 3, 4, 5, 6],
        index=4
    )

if df_gst is not None and df_books is not None:
    gst_sfx   = get_suffix(gst_file_link or "portal.csv")
    books_sfx = get_suffix(books_file_link or "books.csv")

    df_gst_ren   = df_gst.rename(columns={col: f"{col}{gst_sfx}" for col in HEADERS if col in df_gst.columns})
    df_books_ren = df_books.rename(columns={col: f"{col}{books_sfx}" for col in HEADERS if col in df_books.columns})

    df_gst_cl   = clean_df(df_gst_ren, gst_sfx)
    df_books_cl = clean_df(df_books_ren, books_sfx)

    df_gst_cl["key"]   = df_gst_cl[f"Invoice No{gst_sfx}"].astype(str)   + "_" + df_gst_cl[f"GSTIN{gst_sfx}"]
    df_books_cl["key"] = df_books_cl[f"Invoice No{books_sfx}"].astype(str) + "_" + df_books_cl[f"GSTIN{books_sfx}"]
    merged = pd.merge(df_gst_cl, df_books_cl, on="key", how="outer")

    merged["Remarks"] = merged.apply(
        lambda r: make_remark_logic(r, gst_sfx, books_sfx, amt_threshold, date_threshold),
        axis=1
    )

    st.success("✅ Reconciliation Complete!")
    st.session_state.merged = merged

if "merged" in st.session_state:
    df = st.session_state.merged
    st.subheader("📊 Summary")
    counts = {
        "matched":  int(df.Remarks.eq("✅ Matched").sum()),
        "trivial":  int(df.Remarks.str.contains("trivial").sum()),
        "mismatch": int(df.Remarks.str.contains("⚠️").sum()),
        "missing":  int(df.Remarks.str.contains("❌").sum()),
    }
    c1, c2, c3, c4 = st.columns(4)
    if c1.button(f"✅ Matched\n{counts['matched']}"):    st.session_state.filter="matched"
    if c2.button(f"✅ Trivial\n{counts['trivial']}"):   st.session_state.filter="trivial"
    if c3.button(f"⚠️ Mismatch\n{counts['mismatch']}"): st.session_state.filter="mismatch"
    if c4.button(f"❌ Missing\n{counts['missing']}"):   st.session_state.filter="missing"

    flt = st.session_state.get("filter", None)
    def filter_df(df, cat):
        if cat=="matched":   return df[df.Remarks=="✅ Matched"]
        if cat=="trivial":   return df[df.Remarks.str.contains("trivial")]
        if cat=="mismatch":  return df[df.Remarks.str.contains("⚠️")]
        if cat=="missing":   return df[df.Remarks.str.contains("❌")]
        return df

    sub = filter_df(df, flt)
    sub_no_key = sub.drop(columns=["key"], errors="ignore")

    if sub_no_key.empty:
        st.info("No records in this category.")
    else:
        page_size = 30
        total = len(sub_no_key)
        pages = (total - 1) // page_size + 1
        page = st.number_input("Page", 1, pages, value=1)
        st.dataframe(
            sub_no_key.iloc[(page-1)*page_size : page*page_size],
            height=400
        )

        buf = io.BytesIO()
        sub_no_key.to_excel(buf, index=False)
        buf.seek(0)
        st.download_button(
            "Download Filtered Report",
            data=buf,
            file_name="filtered_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if df_gst is not None:
    st.subheader("GST Data (Preview)")
    st.dataframe(df_gst.head())

if df_books is not None:
    st.subheader("Books Data (Preview)")
    st.dataframe(df_books.head())
