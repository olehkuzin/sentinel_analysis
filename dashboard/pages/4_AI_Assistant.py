"""
Page 4 — AI Assistant
Urban climate analysis chatbot powered by Claude (Anthropic API) with tool use.
Claude can fetch NDVI and LST satellite data on demand using defined tools.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import math
import os

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv

from dashboard.modules import (
    geography,
    ingestion,
    ndvi as ndvi_mod,
    lst as lst_mod,
    preprocessing,
)

load_dotenv()

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
st.title("AI Urban Climate Assistant")

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4.5"

SYSTEM_PROMPT = (
    "You are an urban climate analyst for Czech Republic with access to real "
    "Copernicus satellite data. When asked about NDVI or LST for any region or "
    "city, always fetch the data first using your tools before answering. "
    "Be concise and insightful.\n\n"
    f"Available regions: {list(geography.REGIONS.keys())}\n"
    f"Available cities: {list(geography.CITIES.keys())}"
)

# ---------------------------------------------------------------------------
# Tool definitions (sent to Claude)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_ndvi_data",
            "description": (
                "Fetch NDVI (vegetation index) satellite data for a Czech region or city "
                "for a specific year and month. Returns mean, min, max, std and coverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region_or_city": {
                        "type": "string",
                        "description": "Name of a Czech NUTS-3 region or major city.",
                    },
                    "year": {"type": "integer", "description": "Year (2020–2024)."},
                    "month": {"type": "integer", "description": "Month number (1–12)."},
                },
                "required": ["region_or_city", "year", "month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lst_data",
            "description": (
                "Fetch LST (Land Surface Temperature in °C) satellite data for a Czech "
                "region or city for a specific year and month. Returns mean, min, max, "
                "std and coverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region_or_city": {
                        "type": "string",
                        "description": "Name of a Czech NUTS-3 region or major city.",
                    },
                    "year": {"type": "integer", "description": "Year (2020–2024)."},
                    "month": {"type": "integer", "description": "Month number (1–12)."},
                },
                "required": ["region_or_city", "year", "month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_regions",
            "description": (
                "Fetch and compare NDVI and LST for two Czech regions or cities "
                "for the same year and month."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region1": {
                        "type": "string",
                        "description": "Name of the first region or city.",
                    },
                    "region2": {
                        "type": "string",
                        "description": "Name of the second region or city.",
                    },
                    "year": {"type": "integer", "description": "Year (2020–2024)."},
                    "month": {"type": "integer", "description": "Month number (1–12)."},
                },
                "required": ["region1", "region2", "year", "month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_regions",
            "description": "List all available Czech NUTS-3 regions.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_cities",
            "description": "List all available Czech cities.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ---------------------------------------------------------------------------
# Raster + stats helpers
# ---------------------------------------------------------------------------

def _raster_size(bbox: list[float], target_res_m: int = 500) -> tuple[int, int]:
    lon_span = bbox[2] - bbox[0]
    lat_span = bbox[3] - bbox[1]
    centre_lat = (bbox[1] + bbox[3]) / 2
    lon_m = lon_span * 111_320 * math.cos(math.radians(centre_lat))
    lat_m = lat_span * 111_000
    return (max(32, int(lon_m / target_res_m)), max(32, int(lat_m / target_res_m)))


def _resolve_bbox(name: str) -> list[float]:
    if name in geography.REGIONS:
        return geography.get_region_bbox(name)
    if name in geography.CITIES:
        return geography.get_city_bbox(name)
    raise ValueError(
        f"Unknown location '{name}'. "
        f"Valid regions: {list(geography.REGIONS.keys())}. "
        f"Valid cities: {list(geography.CITIES.keys())}."
    )


@st.cache_data(persist="disk")
def _fetch_ndvi_stats(name: str, year: int, month: int) -> dict:
    bbox = _resolve_bbox(name)
    arr = ingestion.fetch_ndvi_raster(bbox, _raster_size(bbox), year, month)
    arr = preprocessing.mask_invalid(arr, -1.0, 1.0)
    s = ndvi_mod.compute_stats(arr)
    return {
        "location": name,
        "period": f"{MONTH_NAMES[month - 1]} {year}",
        "ndvi_mean": round(s["mean"], 4) if not np.isnan(s["mean"]) else None,
        "ndvi_min": round(s["min"], 4) if not np.isnan(s["min"]) else None,
        "ndvi_max": round(s["max"], 4) if not np.isnan(s["max"]) else None,
        "ndvi_std": round(s["std"], 4) if not np.isnan(s["std"]) else None,
        "coverage_pct": s["coverage_pct"],
        "vegetation_class": ndvi_mod.interpret_value(s["mean"]),
    }


@st.cache_data(persist="disk")
def _fetch_lst_stats(name: str, year: int, month: int) -> dict:
    bbox = _resolve_bbox(name)
    arr = ingestion.fetch_lst_raster(bbox, _raster_size(bbox), year, month)
    arr = preprocessing.mask_invalid(arr, -20.0, 60.0)
    s = lst_mod.compute_stats(arr)
    return {
        "location": name,
        "period": f"{MONTH_NAMES[month - 1]} {year}",
        "lst_mean_c": round(s["mean"], 2) if not np.isnan(s["mean"]) else None,
        "lst_min_c": round(s["min"], 2) if not np.isnan(s["min"]) else None,
        "lst_max_c": round(s["max"], 2) if not np.isnan(s["max"]) else None,
        "lst_std_c": round(s["std"], 2) if not np.isnan(s["std"]) else None,
        "coverage_pct": s["coverage_pct"],
        "temperature_class": lst_mod.interpret_value(s["mean"]),
    }


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, tool_input: dict) -> str:
    try:
        if tool_name == "list_available_regions":
            return json.dumps({"regions": list(geography.REGIONS.keys())})

        if tool_name == "list_available_cities":
            return json.dumps({"cities": list(geography.CITIES.keys())})

        if tool_name == "get_ndvi_data":
            result = _fetch_ndvi_stats(
                tool_input["region_or_city"],
                tool_input["year"],
                tool_input["month"],
            )
            return json.dumps(result)

        if tool_name == "get_lst_data":
            result = _fetch_lst_stats(
                tool_input["region_or_city"],
                tool_input["year"],
                tool_input["month"],
            )
            return json.dumps(result)

        if tool_name == "compare_regions":
            r1, r2 = tool_input["region1"], tool_input["region2"]
            year, month = tool_input["year"], tool_input["month"]
            ndvi1 = _fetch_ndvi_stats(r1, year, month)
            lst1 = _fetch_lst_stats(r1, year, month)
            ndvi2 = _fetch_ndvi_stats(r2, year, month)
            lst2 = _fetch_lst_stats(r2, year, month)
            return json.dumps({
                r1: {**ndvi1, **{k: v for k, v in lst1.items() if k not in ndvi1}},
                r2: {**ndvi2, **{k: v for k, v in lst2.items() if k not in ndvi2}},
            })

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# OpenRouter API call with tool-use loop (OpenAI-compatible format)
# ---------------------------------------------------------------------------

def _call_claude(messages: list[dict]) -> tuple[str, list[dict]]:
    """
    Send messages to Claude via OpenRouter, handle tool_calls in a loop,
    and return the final text reply plus the updated messages list.

    Uses OpenAI-compatible format:
    - tools: [{type: "function", function: {...}}]
    - tool calls: choices[0].message.tool_calls[].function.{name, arguments}
    - tool results: {role: "tool", tool_call_id, content}
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "No API key found. Set OPENROUTER_API_KEY in your .env file.", messages

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Prepend system message (OpenAI format — system goes in messages, not a top-level key)
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    while True:
        payload = {
            "model": MODEL,
            "max_tokens": 1024,
            "tools": TOOLS,
            "messages": full_messages,
        }

        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            return f"API error {exc.response.status_code}: {exc.response.text}", messages
        except Exception as exc:
            return f"Request failed: {exc}", messages

        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")

        # Append assistant turn to full_messages
        full_messages = full_messages + [message]

        if finish_reason == "tool_calls":
            tool_calls = message.get("tool_calls", [])
            tool_result_messages = []
            for tc in tool_calls:
                tool_id = tc["id"]
                tool_name = tc["function"]["name"]
                tool_input = json.loads(tc["function"]["arguments"])

                label = f"{tool_name}({', '.join(str(v) for v in tool_input.values())})"
                with st.spinner(f"🛰️ Fetching satellite data — {label}…"):
                    result_str = execute_tool(tool_name, tool_input)

                tool_result_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_str,
                })

            full_messages = full_messages + tool_result_messages
            continue  # loop back to send tool results to Claude

        # finish_reason == "stop" or anything else — return text
        text = message.get("content") or ""
        # Rebuild history without the system prefix so it stays clean for next call
        updated_messages = [m for m in full_messages if m.get("role") != "system"]
        return text, updated_messages


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content} for Anthropic API
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # list of {role: str, text: str} for UI

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
if not st.session_state.display_messages:
    st.info(
        "Ask anything — Claude will fetch satellite data automatically. Try:\n"
        "- *\"What is the NDVI in Praha for July 2023?\"*\n"
        "- *\"Compare vegetation in Brno and Ostrava in summer 2022\"*\n"
        "- *\"Which region had the highest LST in August 2021?\"*\n"
        "- *\"What are your recommendations for Praha's urban heat?\"*"
    )

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

user_input = st.chat_input("Ask about NDVI, LST, urban heat, vegetation…")

if user_input:
    # Show user message immediately
    st.session_state.display_messages.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Append to Anthropic message history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call Claude (handles tool loop internally)
    with st.chat_message("assistant"):
        reply, updated_messages = _call_claude(st.session_state.messages)
        st.markdown(reply)

    st.session_state.messages = updated_messages
    st.session_state.display_messages.append({"role": "assistant", "text": reply})

# Sidebar clear button
if st.session_state.display_messages:
    st.sidebar.header("Chat")
    if st.sidebar.button("Clear conversation", key="ai_clear"):
        st.session_state.messages = []
        st.session_state.display_messages = []
        st.rerun()