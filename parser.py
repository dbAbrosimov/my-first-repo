import streamlit as st
import io
from io import BytesIO
from lxml import etree
from datetime import datetime
import pandas as pd


def load_data(file_bytes):
    """Parse Apple Health XML bytes into DataFrame."""
    records = []
    for _, elem in etree.iterparse(BytesIO(file_bytes), tag="Record"):
        record_type = elem.get("type")
        start = elem.get("startDate")
        if not record_type or not start:
            elem.clear()
            continue
        date = datetime.fromisoformat(start).date()
        if record_type.endswith("SleepAnalysis"):
            if elem.get("value") != "HKCategoryValueSleepAnalysisAsleepDeep":
                elem.clear()
                continue
            duration = (
                datetime.fromisoformat(elem.get("endDate"))
                - datetime.fromisoformat(start)
            ).total_seconds() / 60
            records.append({"date": date, "metric": record_type, "value": duration})
            dt_start = datetime.fromisoformat(start)
            bedtime_val = dt_start.hour * 60 + dt_start.minute
            records.append(
                {"date": dt_start.date(), "metric": "Bedtime", "value": bedtime_val}
            )
            dt_end = datetime.fromisoformat(elem.get("endDate"))
            wake_val = dt_end.hour * 60 + dt_end.minute
            records.append(
                {"date": dt_end.date(), "metric": "WakeTime", "value": wake_val}
            )
        else:
            try:
                val = float(elem.get("value") or 0)
            except ValueError:
                val = 0
            records.append({"date": date, "metric": record_type, "value": val})
        elem.clear()
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def get_types(file_bytes):
    """Return sorted list of unique Record types."""
    types = set()
    for _, elem in etree.iterparse(BytesIO(file_bytes), tag="Record"):
        record_type = elem.get("type")
        if record_type:
            types.add(record_type)
        elem.clear()
    return sorted(types)
