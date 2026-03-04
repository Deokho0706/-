# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ─────────────────────────────
# 고정 스펙
# ─────────────────────────────
SCENARIO_COUNT = 5000
BLOCK_MIN = 6
BLOCK_MAX = 12
MAX_HISTORY_START = pd.Timestamp("1950-01-01")
EOK = 1e8

CASHLIKE_TICKERS = {"SGOV", "BIL", "SHV", "ICSH", "TFLO", "USFR"}

# ─────────────────────────────
# 스타일 (모바일/라이트+다크)
# ─────────────────────────────

def apply_light_css() -> None:
    st.markdown(
        """
        <style>
        /* 배경 및 전체 레이아웃 */
        .stApp { 
            background: linear-gradient(135deg, #f8f9fc 0%, #f1f4f9 100%);
        }
        html, body, [class*="css"] { 
            line-height: 1.4 !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif !important;
        }
        
        /* 메트릭 카드 */
        div[data-testid="stMetricContainer"] {
            background: white;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(0, 0, 0, 0.03);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetricContainer"]:hover {
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(0, 0, 0, 0.08);
        }
        div[data-testid="stMetricValue"] { 
            font-size: 1.8rem;
            font-weight: 700;
            color: #2563eb;
        }
        div[data-testid="stMetricLabel"] { 
            font-size: 0.9rem;
            color: #0f172a;
            font-weight: 600;
        }
        
        /* 컨테이너 및 섹션 */
        [data-testid="stContainer"] {
            border-radius: 12px;
        }
        [data-testid="stVerticalBlockContainer"] {
            gap: 1rem;
        }
        
        /* 버튼 스타일 */
        button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            padding: 0.8rem 1.5rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
            color: white !important;
        }
        button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
            transform: translateY(-2px);
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        }
        button[kind="secondary"],
        button[kind="tertiary"] {
            color: #2563eb !important;
            font-weight: 600 !important;
        }
        
        /* 제목 스타일 */
        h1 {
            color: #000000 !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px !important;
        }
        h2 {
            color: #000000 !important;
            font-weight: 700 !important;
        }
        h3 {
            color: #000000 !important;
            font-weight: 700 !important;
        }
        
        /* 캡션 및 텍스트 */
        p, [data-testid="stText"] {
            color: #0f172a !important;
            font-size: 0.95rem !important;
        }
        [data-testid="stCaption"] {
            color: #000000 !important;
            font-weight: 800 !important;
            font-size: 0.95rem !important;
            margin-bottom: 0.5rem;
        }
        
        /* 입력 필드 */
        input, textarea, select {
            border-radius: 8px !important;
            border: 1.5px solid #cbd5e1 !important;
            background: white !important;
            transition: all 0.2s ease !important;
            font-size: 0.95rem !important;
            color: #000000 !important;
        }
        input::placeholder, textarea::placeholder {
            color: #94a3b8 !important;
        }
        input:focus, textarea:focus, select:focus {
            border: 1.5px solid #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
            background: #ffffff !important;
        }
        label {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        /* 슬라이더 */
        .stSlider {
            padding: 0.5rem 0 !important;
        }
        
        /* 사이드바 */
        [data-testid="stSidebar"] {
            background: white;
            border-right: 1px solid #e2e8f0;
        }
        [data-testid="stSidebar"] h2 {
            color: #000000 !important;
            margin-top: 1.5rem !important;
            font-weight: 800;
        }
        [data-testid="stSidebar"] .stSubheader {
            color: #000000 !important;
        }
        
        /* 사이드바 텍스트 - 더 진한 색 */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stCaption {
            color: #000000 !important;
            font-weight: 600;
        }
        
        /* 사이드바 컨테이너 - 고급 스타일 */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div > div > [data-testid="stContainer"] {
            background: #f0f9ff;
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 12px;
            padding: 1.2rem !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
            transition: all 0.3s ease;
            margin-bottom: 1rem;
        }
        
        [data-testid="stSidebar"] [data-testid="stContainer"] > label {
            color: #1a202c !important;
        }
        
        /* Expander */
        [data-testid="stExpander"] {
            background: white;
            border-radius: 8px;
            border: 2px solid #e0e7ff;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
        }
        [data-testid="stExpanderDetails"] {
            padding: 1rem !important;
        }
        [data-testid="stExpander"] button,
        [data-testid="stExpander"] summary {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        /* Info, Warning, Error 박스 */
        [data-testid="stAlert"] {
            border-radius: 8px !important;
            border-left: 4px solid !important;
            background: white !important;
            padding: 1rem !important;
        }
        [data-testid="stAlert"] p {
            color: #000000 !important;
            font-weight: 500 !important;
        }
        
        /* 컬럼 레이아웃 */
        [data-testid="stHorizontalBlock"] {
            gap: 1.2rem !important;
        }
        
        /* 디바이더 */
        hr {
            margin: 1.5rem 0 !important;
            border: none !important;
            border-top: 1px solid #e2e8f0 !important;
        }

        /* ✅ 사이드바 캡션/라벨 진하게 표시 */
        [data-testid="stSidebar"] [data-testid="stCaption"] {
            color: #0b1220 !important;
            font-weight: 800 !important;
            opacity: 1 !important;
            font-size: 0.95rem !important;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #0b1220 !important;
            opacity: 1 !important;
        }

        /* ✅ BaseWeb 슬라이더/체크박스 텍스트 진하게 */
        [data-baseweb] * {
            opacity: 1 !important;
        }
        [data-baseweb="slider"] * {
            color: #0b1220 !important;
            opacity: 1 !important;
        }
        [data-baseweb="slider"] [role="slider"] {
            background: #2563eb !important;
            border: 2px solid #ffffff !important;
        }
                /* 사이드바 캡션을 '입력' 텍스트 색과 동일하게 */
                [data-testid="stSidebar"] [data-testid="stCaption"]{
                    color:#0f172a !important;
                    font-weight:700 !important;
                    opacity:1 !important;
                }

                /* 혹시 다른 캡션 구조가 섞일 경우 대비 */
                [data-testid="stSidebar"] small{
                    color:#0f172a !important;
                    opacity:1 !important;
                }

                /* BaseWeb 위젯 내부 텍스트도 동일하게 */
                [data-testid="stSidebar"] [data-baseweb]{
                    color:#0f172a !important;
                }

                /* 모바일 토큰: 폰에서 글자/간격 줄이기 */
                @media (max-width: 640px) {
                    h1 { font-size: 1.6rem !important; }
                    h2 { font-size: 1.2rem !important; }
                    div[data-testid="stMetricValue"]{ font-size: 1.4rem !important; }
                    div[data-testid="stMetricContainer"]{ padding: 0.85rem !important; }
                    [data-testid="stHorizontalBlock"]{ gap: 0.7rem !important; }
                    /* 컨테이너 패딩 축소 */
                    div[data-testid="stContainer"]{ padding: 0.85rem !important; }
                    /* 메트릭 라벨/값 폰트 축소 */
                    div[data-testid="stMetricLabel"]{ font-size: 0.85rem !important; }
                }
                </style>
        """,
        unsafe_allow_html=True,
    )


def apply_dark_css() -> None:
    st.markdown("""
        <style>
        /* Dark theme tokens */
        :root {
            --bg: #0b1220;
            --card: #0f1726;
            --muted: #a9b4c7; /* 요청된 muted */
            --text: #e8eef9;  /* 요청된 텍스트 */
            --accent: #0284c7;
            --accent-strong: #2563eb;
            --glass: rgba(255,255,255,0.03);
        }

        .stApp { background: var(--bg) !important; color: var(--text) !important; }
        html, body, [class*="css"] { color: var(--text) !important; font-smooth: always; }

        /* Cards & panels */
        div[data-testid="stMetricContainer"],
        [data-testid="stSidebar"] .element-container,
        [data-testid="stExpander"],
        [data-testid="stAlert"],
        [data-testid="stContainer"] {
            background: linear-gradient(180deg, var(--card), #0b1220) !important;
            border: 1px solid rgba(255,255,255,0.04) !important;
            color: var(--text) !important;
            border-radius: 12px !important;
            box-shadow: 0 6px 20px rgba(2,132,199,0.06) !important;
        }

        /* Text tokens */
        h1,h2,h3,p,label,[data-testid="stText"],[data-testid="stMarkdown"] { color: var(--text) !important; }
        [data-testid="stCaption"] { color: var(--text) !important; opacity: 1 !important; font-weight: 800 !important; }
        .muted, [data-testid="stMetricLabel"] { color: var(--muted) !important; }

        /* Buttons */
        button[kind="primary"] { background: linear-gradient(135deg,var(--accent-strong),#1d4ed8) !important; color: #fff !important; border-radius: 8px !important; }

        /* BaseWeb specific overrides (slider / checkbox / radio) */
        /* Slider track */
        [data-baseweb="slider"] .rc-slider-rail,
        [data-baseweb="slider"] .rc-slider-track {
            background: rgba(255,255,255,0.06) !important;
            height: 8px !important;
            border-radius: 999px !important;
            box-shadow: none !important;
        }
        /* Slider handle (thumb) */
        [data-baseweb="slider"] .rc-slider-handle {
            background: var(--accent) !important;
            border: 2px solid #fff !important;
            box-shadow: 0 6px 18px rgba(2,132,199,0.18) !important;
            width: 16px !important; height: 16px !important; margin-top: -4px !important;
            opacity: 1 !important;
        }
        /* Checkbox / Radio */
        [data-baseweb="checkbox"] .checkmark,
        [data-baseweb="radio"] .radio {
            background: linear-gradient(180deg,var(--accent),var(--accent-strong)) !important;
            border: none !important;
            box-shadow: 0 6px 18px rgba(2,132,199,0.12) !important;
        }
        [data-baseweb="checkbox"] label, [data-baseweb="radio"] label { color: var(--text) !important; opacity: 1 !important; }

        /* force contrast for streamlit widgets */
        .stSlider, .stCheckbox, .stRadio, .stSelectbox { color: var(--text) !important; }

        /* Plotly dark defaults */
        .js-plotly-plot .plotly .main-svg { background: transparent !important; }
        .js-plotly-plot .g .xtick text, .js-plotly-plot .g .ytick text, .js-plotly-plot .g .axis-title { fill: var(--text) !important; }
        .js-plotly-plot .legendtext { fill: var(--text) !important; }

        /* Reduce opacity issues caused by BaseWeb */
        [data-baseweb] { opacity: 1 !important; }

        </style>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────
# 포맷
# ─────────────────────────────
def krw_compact(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e12:
        return f"{sign}{a/1e12:.2f}조"
    if a >= 1e8:
        return f"{sign}{a/1e8:.2f}억"
    if a >= 1e4:
        return f"{sign}{a/1e4:.1f}만"
    return f"{sign}{int(round(a)):,}원"

def format_krw_readable(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    return f"{int(round(v)):,}원 (약 {krw_compact(v)})"

def pct_str01(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x*100:.1f}%"

# ─────────────────────────────
# 유틸
# ─────────────────────────────
def _stable_hash(payload: dict) -> str:
    txt = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(txt.encode("utf-8")).hexdigest()

def _split_tokens(text: str) -> List[str]:
    t = (text or "").replace(",", "\n").replace(";", "\n")
    out: List[str] = []
    for line in t.splitlines():
        s = line.strip()
        if s:
            out.append(s)
    return out

def _is_krw_ticker(ticker: str) -> bool:
    t = (ticker or "").upper()
    return t.endswith(".KS") or t.endswith(".KQ")

def _to_daily_calendar_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    idx = idx.normalize()
    return pd.date_range(start=idx.min(), end=idx.max(), freq="D")

def _extract_field(download_df: pd.DataFrame, field: str, tickers: List[str]) -> pd.DataFrame:
    if download_df is None or download_df.empty:
        raise ValueError("yfinance 다운로드 결과가 비어 있습니다.")

    df = download_df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)

        if field in lv0:
            out = df[field].copy()
        elif field in lv1:
            out = df.xs(field, axis=1, level=1).copy()
        else:
            raise ValueError(f"다운로드 데이터에서 '{field}' 컬럼을 찾을 수 없습니다.")

        out.columns = [str(c) for c in out.columns]
        want = [t for t in tickers if t in out.columns]
        out = out.loc[:, want]
        if len(out.columns) != len(tickers):
            missing = [t for t in tickers if t not in out.columns]
            raise ValueError(f"'{field}' 데이터가 없는 티커가 있습니다: {missing}")
        return out

    if field not in df.columns:
        raise ValueError(f"다운로드 데이터에서 '{field}' 컬럼을 찾을 수 없습니다.")
    t = tickers[0]
    out = df[[field]].copy()
    out.columns = [t]
    return out

# ─────────────────────────────
# 현실모드 유틸
# ─────────────────────────────
def geo_annual_from_monthly(df_ret: pd.DataFrame) -> pd.Series:
    lr = np.log1p(df_ret.astype(float))
    mu_m = lr.mean(axis=0)
    return np.expm1(mu_m * 12.0)

def winsorize_monthly_returns(df_ret: pd.DataFrame, pct: float) -> pd.DataFrame:
    p = float(pct)
    if p <= 0 or df_ret.empty:
        return df_ret
    q = p / 100.0
    lo = df_ret.quantile(q)
    hi = df_ret.quantile(1.0 - q)
    return df_ret.clip(lower=lo, upper=hi, axis=1)

def slice_recent_years(df_ret: pd.DataFrame, lookback_years: int) -> pd.DataFrame:
    y = int(lookback_years)
    if y <= 0 or df_ret.empty:
        return df_ret
    end = df_ret.index.max()
    cutoff = end - pd.DateOffset(years=y)
    out = df_ret.loc[df_ret.index >= cutoff].copy()
    return out if not out.empty else df_ret

def parse_target_rates_input(text: str, tickers: List[str]) -> Dict[str, float]:
    toks = _split_tokens(text)
    if len(toks) != len(tickers):
        raise ValueError(f"기대수익 입력 개수({len(toks)})가 티커 개수({len(tickers)})와 다릅니다.")
    out: Dict[str, float] = {}
    for t, s in zip(tickers, toks):
        v = float(s.replace("%", "").strip()) / 100.0
        out[t] = v
    return out

def build_targets_auto(
    df_ret_used: pd.DataFrame,
    tickers: List[str],
    cash_target_pp: float,
    risky_haircut_pp: float,
    risky_cap_pp: float,
    risky_floor_pp: float,
) -> Dict[str, float]:
    hist_geo = geo_annual_from_monthly(df_ret_used[tickers])
    cap = float(risky_cap_pp) / 100.0
    floor = float(risky_floor_pp) / 100.0
    haircut = float(risky_haircut_pp) / 100.0
    cash_target = float(cash_target_pp) / 100.0

    targets: Dict[str, float] = {}
    for t in tickers:
        if t.upper() in CASHLIKE_TICKERS:
            targets[t] = cash_target
        else:
            r = float(hist_geo.get(t, 0.0)) - haircut
            r = min(max(r, floor), cap)
            targets[t] = r
    return targets

def adjust_returns_to_targets(df_ret_used: pd.DataFrame, targets_annual: Dict[str, float], annual_drag_pp: float) -> pd.DataFrame:
    lr = np.log1p(df_ret_used.astype(float))
    mu = lr.mean(axis=0)
    drag = float(annual_drag_pp) / 100.0

    lr_adj = lr.copy()
    for c in lr.columns:
        if c in targets_annual and targets_annual[c] is not None:
            mu_target_m = np.log1p(float(targets_annual[c])) / 12.0
            lr_adj[c] = lr[c] - mu[c] + mu_target_m
        if drag != 0.0:
            lr_adj[c] = lr_adj[c] - (drag / 12.0)

    return np.expm1(lr_adj)

# ─────────────────────────────
# 데이터 메타
# ─────────────────────────────
@dataclass
class DataMeta:
    tickers: List[str]
    has_usd_asset: bool
    fx_ticker_used: str
    start_month: str
    end_month: str
    months_used: int
    dropped_months: int

# ─────────────────────────────
# 데이터 준비
# ─────────────────────────────
@st.cache_data(show_spinner=False)
def prepare_monthly_returns(
    tickers: Tuple[str, ...],
    dividend_reinvest: bool,
) -> Tuple[pd.DataFrame, DataMeta]:
    if not tickers:
        raise ValueError("티커가 비어 있습니다.")

    ticker_list = list(tickers)
    has_usd = any(not _is_krw_ticker(t) for t in ticker_list)

    end_dt = pd.Timestamp(date.today())
    start_dt = MAX_HISTORY_START

    dl = yf.download(
        tickers=ticker_list,
        start=start_dt.date(),
        end=end_dt.date(),
        auto_adjust=False,
        actions=True,
        group_by="column",
        progress=False,
    )

    close_raw = _extract_field(dl, "Close", ticker_list)
    bad = [t for t in ticker_list if close_raw[t].dropna().empty]
    if bad:
        raise ValueError(f"가격 데이터가 비어있는 티커가 있습니다: {bad}")

    try:
        div_raw = _extract_field(dl, "Dividends", ticker_list)
    except Exception:
        div_raw = pd.DataFrame(index=close_raw.index, columns=ticker_list, data=0.0)

    close_raw.index = pd.to_datetime(close_raw.index)
    div_raw.index = pd.to_datetime(div_raw.index)

    full_idx = _to_daily_calendar_index(close_raw.index)
    close_re = close_raw.reindex(full_idx).ffill()
    div_re = div_raw.reindex(full_idx).fillna(0.0)

    close_m = close_re.resample("M").last()
    div_m = div_re.resample("M").sum()

    fx_used = "N/A (USD 없음)"
    if has_usd:
        fx_candidates = ["USDKRW=X", "KRW=X"]
        fx_series = None
        last_err = None
        for fx_ticker in fx_candidates:
            try:
                fx_dl = yf.download(
                    tickers=fx_ticker,
                    start=start_dt.date(),
                    end=end_dt.date(),
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                )
                if fx_dl is None or fx_dl.empty or "Close" not in fx_dl.columns:
                    raise ValueError("FX 데이터가 비어있거나 Close가 없습니다.")
                fx_close = fx_dl["Close"].copy().dropna()
                if fx_close.empty:
                    raise ValueError("FX Close가 전부 NaN입니다.")

                med = float(fx_close.median())
                if med < 10.0:
                    fx_close = 1.0 / fx_close

                fx_close.index = pd.to_datetime(fx_close.index)
                fx_full = _to_daily_calendar_index(fx_close.index)
                fx_m = fx_close.reindex(fx_full).ffill().resample("M").last().ffill()

                fx_series = fx_m
                fx_used = fx_ticker
                break
            except Exception as e:
                last_err = e
                continue

        if fx_series is None:
            raise ValueError(f"USD 자산이 포함되어 FX가 필요하지만 다운로드 실패: {last_err}")

        fx_mtx = pd.DataFrame(index=close_m.index, columns=ticker_list, dtype=float)
        fx_aligned = fx_series.reindex(close_m.index).ffill()
        for t in ticker_list:
            fx_mtx[t] = 1.0 if _is_krw_ticker(t) else fx_aligned.astype(float)
    else:
        fx_mtx = pd.DataFrame(index=close_m.index, columns=ticker_list, data=1.0, dtype=float)

    prev_close = close_m.shift(1)
    prev_fx = fx_mtx.shift(1)

    if dividend_reinvest:
        numer = (close_m + div_m) * fx_mtx
    else:
        numer = close_m * fx_mtx

    denom = prev_close * prev_fx
    growth = numer / denom
    ret = (growth - 1.0).replace([np.inf, -np.inf], np.nan)

    before = int(ret.shape[0])
    ret_clean = ret.dropna(how="any")
    after = int(ret_clean.shape[0])
    dropped = before - after

    if ret_clean.empty:
        raise ValueError("전처리 후 월별 수익률 데이터가 비었습니다. 티커/기간을 확인하세요.")
    if len(ret_clean) < (BLOCK_MAX + 1):
        raise ValueError(f"월별 수익률이 너무 짧습니다({len(ret_clean)}개월). 최소 {BLOCK_MAX + 1}개월 이상 필요합니다.")

    meta = DataMeta(
        tickers=ticker_list,
        has_usd_asset=bool(has_usd),
        fx_ticker_used=fx_used,
        start_month=ret_clean.index.min().strftime("%Y-%m"),
        end_month=ret_clean.index.max().strftime("%Y-%m"),
        months_used=int(len(ret_clean)),
        dropped_months=int(dropped),
    )
    return ret_clean, meta

# ─────────────────────────────
# 부트스트랩 인덱스
# ─────────────────────────────
def generate_bootstrap_indices(
    rng: np.random.Generator,
    hist_len: int,
    horizon_months: int,
    scenario_count: int,
    block_mode: str,
) -> np.ndarray:
    if hist_len <= 0:
        raise ValueError("hist_len이 0 이하입니다.")
    if horizon_months <= 0:
        raise ValueError("horizon_months가 0 이하입니다.")

    if block_mode == "random":
        lengths = list(range(BLOCK_MIN, BLOCK_MAX + 1))
    else:
        fixed = int(block_mode.split("_")[1])
        lengths = [fixed]

    aranges = {L: np.arange(L, dtype=np.int32) for L in lengths}
    idx = np.empty((scenario_count, horizon_months), dtype=np.int32)

    for s in range(scenario_count):
        pos = 0
        while pos < horizon_months:
            L = int(rng.choice(lengths))
            start = int(rng.integers(0, hist_len))
            block = (start + aranges[L]) % hist_len
            take = min(L, horizon_months - pos)
            idx[s, pos:pos + take] = block[:take]
            pos += take
    return idx

# ─────────────────────────────
# 복구 기간 계산
# ─────────────────────────────
def calculate_recovery_periods(value_path: np.ndarray) -> Tuple[float, float, float]:
    """
    복구 기간 계산 (월 단위)
    (평균 복구기간, 최단 복구기간, 최장 복구기간)
    """
    S = value_path.shape[1]
    recovery_periods = []
    
    for s in range(S):
        path = value_path[:, s]
        peak = np.maximum.accumulate(path)
        
        for t in range(1, len(path)):
            if path[t] > peak[t-1]:
                recovery = t
                recovery_periods.append(recovery)
    
    if not recovery_periods:
        return 0.0, 0.0, 0.0
    
    avg_recovery = float(np.mean(recovery_periods))
    min_recovery = float(np.min(recovery_periods))
    max_recovery = float(np.max(recovery_periods))
    
    return avg_recovery, min_recovery, max_recovery

# ─────────────────────────────
# 시뮬 (고정배분)
# ─────────────────────────────
def run_simulation_fixed_allocation(
    hist_returns: np.ndarray,        # (N, A)
    weights: np.ndarray,             # (A,)
    years: int,
    initial_capital: float,
    monthly_contribution: float,
    seed: int,
    block_mode: str,
    scenario_count: int = SCENARIO_COUNT,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = int(years) * 12
    S = int(scenario_count)
    N, A = hist_returns.shape
    if A != len(weights):
        raise ValueError("자산 개수(A)와 weights 길이가 다릅니다.")
    if N < (BLOCK_MAX + 1):
        raise ValueError("히스토리 월 수가 너무 짧습니다.")

    rng = np.random.default_rng(int(seed))
    indices = generate_bootstrap_indices(rng, N, T, S, block_mode)

    w = weights.astype(np.float64, copy=False)
    values = np.zeros((S, A), dtype=np.float64)
    if float(initial_capital) > 0:
        values[:] = float(initial_capital) * w[None, :]

    value_path = np.zeros((T + 1, S), dtype=np.float64)
    value_path[0, :] = values.sum(axis=1)

    prog = st.progress(0, text="계산 중...") if show_progress else None

    for t in range(1, T + 1):
        r = hist_returns[indices[:, t - 1], :]
        values *= (1.0 + r)
        if float(monthly_contribution) > 0:
            values += float(monthly_contribution) * w[None, :]
        value_path[t, :] = values.sum(axis=1)

        if prog is not None and (t % max(1, T // 30) == 0 or t == T):
            prog.progress(int(t / T * 100), text=f"계산 {t}/{T}개월")

    if prog is not None:
        prog.empty()

    terminal = value_path[-1, :].copy()

    peak = np.maximum.accumulate(value_path, axis=0)
    drawdown = (peak - value_path) / np.maximum(peak, 1e-12)
    mdd = np.max(drawdown, axis=0)

    return value_path, terminal, mdd

# ─────────────────────────────
# Plotly
# ─────────────────────────────
def _apply_light_plotly(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000", size=13, family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", weight="bold"),
        xaxis=dict(
            showgrid=True, 
            gridcolor="#e5e7eb", 
            zeroline=False, 
            title_font=dict(color="#000000", size=13),
            tickfont=dict(color="#000000", size=12),
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor="#e5e7eb", 
            zeroline=False, 
            title_font=dict(color="#000000", size=13),
            tickfont=dict(color="#000000", size=12),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
    )

def _apply_dark_plotly(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#e6edf7", size=13),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
    )

def make_path_fanchart_mobile(value_path: np.ndarray) -> go.Figure:
    T_plus_1, S = value_path.shape
    paths = value_path

    # 모든 시나리오 경로를 흐리게 표시 (최대 2000개로 제한)
    fig = go.Figure()
    for s in range(min(S, 2000)):
        fig.add_trace(go.Scatter(
            x=np.arange(T_plus_1),
            y=paths[:, s] / EOK,
            mode="lines",
            line=dict(width=0.5, color="rgba(37,99,235,0.02)"),
            hoverinfo="skip",
            showlegend=False,
        ))
    
    pcts = np.percentile(paths, [5, 50, 95], axis=1)
    p5, p50, p95 = pcts[0] / EOK, pcts[1] / EOK, pcts[2] / EOK

    x = np.arange(T_plus_1)
    tick_vals = list(range(0, T_plus_1, 12))
    tick_text = [f"{v//12}년" for v in tick_vals]

    # 밴드 영역 (단색 2D 느낌)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.04)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="범위(p5~p95)",
    ))
    
    # 중앙값 라인
    fig.add_trace(go.Scatter(
        x=x, y=p50, mode="lines",
        line=dict(width=2, color="#000000"),
        name="보통(p50)",
    ))

    _apply_light_plotly(fig)
    fig.update_layout(
        height=360,
        xaxis=dict(title="시간", tickvals=tick_vals, ticktext=tick_text, automargin=True),
        yaxis=dict(title="자산(억원)", ticksuffix="억", tickformat=",.0f", automargin=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#000000")),
    )
    return fig

def make_terminal_hist_mobile(terminal: np.ndarray, goal: float) -> go.Figure:
    x = (terminal / EOK).astype(float)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x, nbinsx=50,
        marker=dict(color="rgba(37,99,235,0.35)"),
        name="분포",
    ))
    if float(goal) > 0:
        goal_e = float(goal) / EOK
        fig.add_shape(
            type="line", x0=goal_e, x1=goal_e,
            y0=0, y1=1, yref="paper",
            line=dict(color="#ef4444", width=3, dash="dash"),
        )
    _apply_light_plotly(fig)
    fig.update_layout(
        height=300,
        xaxis=dict(title="만기 잔고(억원)", ticksuffix="억", tickformat=",.0f", automargin=True),
        yaxis=dict(title="개수(상대)", automargin=True),
        showlegend=False,
    )
    return fig


def make_path_fanchart_dark(value_path: np.ndarray) -> go.Figure:
    T_plus_1, S = value_path.shape
    paths = value_path
    pcts = np.percentile(paths, [5, 50, 95], axis=1)
    p5, p50, p95 = pcts[0]/EOK, pcts[1]/EOK, pcts[2]/EOK

    x = np.arange(T_plus_1)
    tick_vals = list(range(0, T_plus_1, 12))
    tick_text = [f"{v//12}년" for v in tick_vals]

    fig = go.Figure()
    
    # 모든 개별 경로를 흐리게 표시
    sample_indices = np.random.choice(S, min(150, S), replace=False)
    for idx in sample_indices:
        scenario_path = paths[:, idx] / EOK
        fig.add_trace(go.Scatter(
            x=x, y=scenario_path,
            mode="lines",
            line=dict(width=0.5, color="rgba(100, 150, 220, 0.08)"),
            hoverinfo="skip",
            showlegend=False,
        ))
    
    # 밴드
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself",
        fillcolor="rgba(59,130,246,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="p5‑p95",
    ))
    # 중앙값
    fig.add_trace(go.Scatter(
        x=x, y=p50, mode="lines",
        line=dict(width=4, color="#3b82f6"),
        name="중앙값",
        hovertemplate="중앙값: %{y:.2f}억<br>%{x}개월<extra></extra>",
    ))

    _apply_dark_plotly(fig)
    fig.update_layout(
        height=360,
        xaxis=dict(title="시간", tickvals=tick_vals, ticktext=tick_text, automargin=True),
        yaxis=dict(title="자산(억원)", ticksuffix="억", tickformat=",.0f", automargin=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_terminal_hist_dark(terminal: np.ndarray, goal: float) -> go.Figure:
    x = (terminal / EOK).astype(float)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x, nbinsx=50,
        marker=dict(color="rgba(59,130,246,0.5)"),
        name="분포",
    ))
    if float(goal) > 0:
        goal_e = float(goal) / EOK
        fig.add_shape(
            type="line", x0=goal_e, x1=goal_e,
            y0=0, y1=1, yref="paper",
            line=dict(color="#ef4444", width=3, dash="dash"),
        )
    _apply_dark_plotly(fig)
    fig.update_layout(
        height=300,
        xaxis=dict(title="만기 잔고(억원)", ticksuffix="억", tickformat=",.0f", automargin=True),
        yaxis=dict(title="개수(상대)", automargin=True),
        showlegend=False,
    )
    return fig

# ─────────────────────────────
# 세션
# ─────────────────────────────
def init_session_state() -> None:
    defaults = {
        "data_loaded": False,
        "sim_completed": False,
        "df_monthly_return": None,
        "df_monthly_return_used": None,
        "targets_used": None,
        "meta": None,
        "value_path": None,
        "terminal_wealth": None,
        "mdd": None,
        "tickers_input": "SGOV",
        "weights_input": "100",
        "last_error": None,
        "target_returns_input": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_sim_state() -> None:
    st.session_state["sim_completed"] = False
    st.session_state["value_path"] = None
    st.session_state["terminal_wealth"] = None
    st.session_state["mdd"] = None

def reset_data_state() -> None:
    st.session_state["data_loaded"] = False
    st.session_state["df_monthly_return"] = None
    st.session_state["df_monthly_return_used"] = None
    st.session_state["targets_used"] = None
    st.session_state["meta"] = None
    reset_sim_state()

# ─────────────────────────────
# 메인
# ─────────────────────────────
def main() -> None:
    st.set_page_config(page_title="포트폴리오 시뮬레이터", layout="centered", initial_sidebar_state="collapsed")
    apply_light_css()
    init_session_state()

    st.title("📊 포트폴리오 시뮬레이터")
    st.info("📈 과거 데이터를 바탕으로 '가능한 범위'를 보여주는 도구예요. 확정 예측이 아니며, 투자 결정의 참고 자료로만 사용해주세요.")

    # ── 사이드바(입력 최소)
    with st.sidebar:
        st.subheader("⚙️ 입력")

        with st.container(border=True):
            st.caption("📅 투자 기간")
            years = st.slider("기간(년)", 5, 40, 30, label_visibility="collapsed")
            dividend_reinvest = st.checkbox("배당 재투자", value=True)

        with st.container(border=True):
            st.caption("💰 자금 설정")
            initial_capital = st.number_input("초기금(원)", 0, 1_000_000_000, 0, 100_000)
            monthly_contribution = st.number_input("월 납입(원)", 0, 10_000_000, 500_000, 50_000)
            goal_amount = st.number_input("목표(원)", 0, 10_000_000_000, 100_000_000, 1_000_000)

        with st.container(border=True):
            st.caption("💼 자산 구성(최대 5개)")
            st.text_area("티커", key="tickers_input", height=80, help="예: SGOV, QQQM, VOO, 005930.KS", label_visibility="collapsed")
            st.text_area("비중(%)", key="weights_input", height=80, help="티커 순서대로. 합이 100이어야 합니다.", label_visibility="collapsed")

        run_btn = st.button("🧮 계산하기", use_container_width=True)

        with st.expander("🔧 고급 설정", expanded=False):
            block_label = st.selectbox("블록(월)", ["랜덤(6~12)", "6개월 고정", "9개월 고정", "12개월 고정"], index=0)
            block_mode = "random"
            if block_label == "6개월 고정":
                block_mode = "fixed_6"
            elif block_label == "9개월 고정":
                block_mode = "fixed_9"
            elif block_label == "12개월 고정":
                block_mode = "fixed_12"

            seed = st.number_input("시드", 0, 1_000_000_000, 42)

            st.divider()
            st.subheader("🎯 현실모드(추천)")
            realistic_mode = st.checkbox("현실모드 사용", value=True)
            lookback_years = st.slider("최근 N년만 사용", 3, 50, 20)
            winsor_pct = st.slider("이상치 완화(%)", 0.0, 2.0, 0.5, 0.1)
            annual_drag_pp = st.slider("연 드래그(%p)", 0.0, 5.0, 0.8, 0.1)
            cash_target_pp = st.slider("현금성(예: SGOV) 타겟(연,%)", 0.0, 8.0, 3.5, 0.1)
            risky_haircut_pp = st.slider("리스크 헤어컷(연,%p)", 0.0, 6.0, 2.0, 0.1)
            risky_cap_pp = st.slider("리스크 상한(연,%)", 0.0, 20.0, 10.0, 0.5)
            risky_floor_pp = st.slider("리스크 하한(연,%)", -10.0, 10.0, 0.0, 0.5)

            manual_targets = st.checkbox("기대수익 직접 입력", value=False)
            if manual_targets:
                st.text_area("기대수익(연,%)", key="target_returns_input", height=90)

    # ── 입력 파싱
    tickers = [t.strip().upper() for t in _split_tokens(st.session_state["tickers_input"])]
    if len(tickers) == 0:
        st.info("👈 왼쪽 패널에서 **티커와 비중**을 입력하고 **🚀 계산하기**를 눌러주세요!")
        return
    if len(tickers) > 5:
        st.error("티커는 최대 5개까지만 가능해요.")
        return
    if len(set(tickers)) != len(tickers):
        st.error("같은 티커가 중복됐어요. 중복 제거 부탁!")
        return

    w_tokens = _split_tokens(st.session_state["weights_input"])
    w_clean = [w.strip().replace("%", "") for w in w_tokens if w.strip()]
    try:
        weights_raw = np.array([float(x) for x in w_clean], dtype=float)
    except Exception:
        st.error("비중(%)은 숫자만 입력해줘요. 예: 50 또는 50%")
        return
    if len(weights_raw) != len(tickers):
        st.error(f"티커 개수({len(tickers)})와 비중 개수({len(weights_raw)})가 달라요.")
        return
    weight_sum = float(weights_raw.sum())
    if abs(weight_sum - 100.0) > 0.01:
        st.warning(f"비중 합계가 100이 아닙니다. 현재: {weight_sum:.2f}%")
        return
    weights = (weights_raw / 100.0).astype(np.float64)

    # ── 실행
    if run_btn:
        try:
            with st.spinner("데이터 불러오는 중..."):
                df_ret, meta = prepare_monthly_returns(tuple(tickers), bool(dividend_reinvest))
            st.session_state["df_monthly_return"] = df_ret
            st.session_state["meta"] = meta
            st.session_state["data_loaded"] = True

            df_base = df_ret[tickers].copy()
            targets_used: Optional[Dict[str, float]] = None
            df_used = df_base.copy()

            if bool(realistic_mode):
                df_src = slice_recent_years(df_base, int(lookback_years))
                df_src = winsorize_monthly_returns(df_src, float(winsor_pct))
                if df_src.empty or len(df_src) < (BLOCK_MAX + 1):
                    raise ValueError("현실모드 적용 후 데이터가 부족합니다. 최근 분석 기간을 늘려주세요.")

                if bool(manual_targets):
                    targets_used = parse_target_rates_input(st.session_state.get("target_returns_input", ""), tickers)
                else:
                    targets_used = build_targets_auto(
                        df_ret_used=df_src,
                        tickers=tickers,
                        cash_target_pp=float(cash_target_pp),
                        risky_haircut_pp=float(risky_haircut_pp),
                        risky_cap_pp=float(risky_cap_pp),
                        risky_floor_pp=float(risky_floor_pp),
                    )

                df_used = adjust_returns_to_targets(
                    df_ret_used=df_src,
                    targets_annual=targets_used,
                    annual_drag_pp=float(annual_drag_pp),
                )

            st.session_state["df_monthly_return_used"] = df_used
            st.session_state["targets_used"] = targets_used

            hist = df_used.to_numpy(dtype=np.float64)

            with st.spinner("시뮬레이션 계산 중..."):
                value_path, terminal, mdd = run_simulation_fixed_allocation(
                    hist_returns=hist,
                    weights=weights,
                    years=int(years),
                    initial_capital=float(initial_capital),
                    monthly_contribution=float(monthly_contribution),
                    seed=int(seed),
                    block_mode=block_mode,
                    scenario_count=SCENARIO_COUNT,
                    show_progress=True,
                )

            st.session_state["value_path"] = value_path
            st.session_state["terminal_wealth"] = terminal
            st.session_state["mdd"] = mdd
            st.session_state["sim_completed"] = True
            st.session_state["last_error"] = None
        except Exception as e:
            reset_data_state()
            st.session_state["last_error"] = str(e)

    # ── 오류 표시
    if st.session_state.get("last_error"):
        st.error(f"실행 실패: {st.session_state['last_error']}")

    # ── 결과 표시
    if not st.session_state["sim_completed"]:
        st.info("👈 왼쪽 패널에서 입력하고 **🚀 계산하기**를 눌러주세요!")
        return

    terminal = st.session_state["terminal_wealth"]
    value_path = st.session_state["value_path"]

    p5, p50, p95 = np.percentile(terminal, [5, 50, 95])
    goal_prob = float(np.mean(terminal >= float(goal_amount))) if float(goal_amount) > 0 else np.nan
    total_principal = float(initial_capital) + float(monthly_contribution) * (int(years) * 12)

    # 카드 요약 (2x2 그리드)
    st.subheader("🎮 시뮬레이션 결과")
    row1_col1, row1_col2 = st.columns([1, 1], gap="medium")
    row1_col1.metric("평균값", krw_compact(p50))
    row1_col2.metric("총 납입액", krw_compact(total_principal))

    row2_col1, row2_col2 = st.columns([1, 1], gap="medium")
    row2_col1.metric("보수적(p5)", krw_compact(p5))
    row2_col2.metric("낙관적(p95)", krw_compact(p95))

    if float(goal_amount) > 0:
        st.metric("🎯 목표 달성 확률", f"{goal_prob*100:.1f}%")
        st.progress(goal_prob)

    tab1, tab2 = st.tabs(["자산 경로", "만기 분포"])
    with tab1:
        fig1 = make_path_fanchart_mobile(value_path)
        try:
            fig1.update_layout(height=280, showlegend=False, font=dict(size=10))
            fig1.update_xaxes(tickfont=dict(size=10))
            fig1.update_yaxes(tickfont=dict(size=10))
        except Exception:
            pass
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
    with tab2:
        fig2 = make_terminal_hist_mobile(terminal, float(goal_amount))
        try:
            fig2.update_layout(height=250, showlegend=False, font=dict(size=10))
            fig2.update_xaxes(tickfont=dict(size=10))
            fig2.update_yaxes(tickfont=dict(size=10))
        except Exception:
            pass
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # 상세 분석/다운로드 섹션
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    with st.expander("🛠 상세 분석 (고급)", expanded=False):
        # 복구 기간 계산
        avg_recovery, min_recovery, max_recovery = calculate_recovery_periods(value_path)
        
        # 📊 리스크 분석
        st.subheader("📊 리스크 분석")
        r1, r2 = st.columns(2)
        r1.metric("최대 낙폭(MDD)", f"{float(np.median(st.session_state['mdd']))*100:.2f}%")
        r2.metric("MDD 범위", f"{float(np.min(st.session_state['mdd']))*100:.1f}% ~ {float(np.max(st.session_state['mdd']))*100:.1f}%")

        st.divider()

        # ⏱️ 복구 기간 분석
        st.subheader("⏱️ 복구 기간 분석")
        rec1, rec2, rec3 = st.columns(3)
        rec1.metric("평균 복구기간", f"{int(avg_recovery)}개월")
        rec2.metric("최단 복구기간", f"{int(min_recovery)}개월")
        rec3.metric("최장 복구기간", f"{int(max_recovery)}개월")

        st.divider()

        # 🎯 현실모드 타겟
        st.subheader("🎯 현실모드 타겟(연 수익률)")
        targets_used = st.session_state.get("targets_used", None)
        if targets_used:
            t_cols = st.columns(len(targets_used))
            for (ticker, rate), col in zip(targets_used.items(), t_cols):
                col.metric(ticker, f"{rate*100:.2f}%")
        else:
            st.metric("SGOV", "3.50%")
        
        df_out = pd.DataFrame({
            "scenario": np.arange(len(terminal)),
            "terminal_wealth": terminal,
            "mdd": st.session_state["mdd"],
        })
        st.download_button(
            label="📥 결과 CSV 다운로드",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name=f"sim_{years}y_seed.csv",
            mime="text/csv",
            use_container_width=True,
        )

if __name__ == "__main__":
    main()