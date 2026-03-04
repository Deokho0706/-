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
# 스타일 (모바일/라이트)
# ─────────────────────────────
def apply_light_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f6f7fb; }
        html, body, [class*="css"] { line-height: 1.35 !important; }
        .element-container { margin-bottom: 0.55rem; }
        div[data-testid="stMetricValue"] { font-size: 1.55rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.95rem; }
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
        font=dict(color="#111827", size=13),
        xaxis=dict(showgrid=True, gridcolor="#e5e7eb", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb", zeroline=False),
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
    )

def make_path_fanchart_mobile(value_path: np.ndarray) -> go.Figure:
    T_plus_1, _ = value_path.shape
    paths = value_path

    pcts = np.percentile(paths, [5, 50, 95], axis=1)
    p5, p50, p95 = pcts[0] / EOK, pcts[1] / EOK, pcts[2] / EOK

    x = np.arange(T_plus_1)
    tick_vals = list(range(0, T_plus_1, 12))
    tick_text = [f"{v//12}년" for v in tick_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="범위(p5~p95)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p50, mode="lines",
        line=dict(width=4, color="#2563eb"),
        name="보통(p50)",
    ))

    _apply_light_plotly(fig)
    fig.update_layout(
        height=360,
        xaxis=dict(title="시간", tickvals=tick_vals, ticktext=tick_text, automargin=True),
        yaxis=dict(title="자산(억원)", ticksuffix="억", tickformat=",.0f", automargin=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
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
    st.set_page_config(page_title="포트폴리오 시뮬레이터", layout="centered")
    apply_light_css()
    init_session_state()

    st.title("📱포트폴리오 시뮬레이터")
    st.caption("과거 데이터를 바탕으로 ‘가능한 범위’를 보여주는 도구예요. 확정 예측이 아닙니다.")

    # ── 사이드바(입력 최소)
    with st.sidebar:
        st.subheader("입력")

        years = st.slider("기간(년)", 5, 40, 30)
        dividend_reinvest = st.checkbox("배당 재투자", value=True)

        initial_capital = st.number_input("초기금(원)", 0, 1_000_000_000, 0, 100_000)
        monthly_contribution = st.number_input("월 납입(원)", 0, 10_000_000, 500_000, 50_000)
        goal_amount = st.number_input("목표(원)", 0, 10_000_000_000, 100_000_000, 1_000_000)

        st.divider()
        st.subheader("자산(최대 5개)")
        st.text_area("티커", key="tickers_input", height=90, help="예: SGOV, QQQM, VOO, 005930.KS")
        st.text_area("비중(%)", key="weights_input", height=90, help="티커 순서대로. 합이 100이어야 합니다.")

        run_btn = st.button("🚀 계산하기", use_container_width=True)

        with st.expander("고급 설정", expanded=False):
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
            st.subheader("현실모드(추천)")
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
        st.info("왼쪽(또는 아래)에서 티커를 입력해줘 🙂")
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
                    raise ValueError("현실모드 적용 후 데이터가 너무 짧아요. 최근 N년을 늘려주세요.")

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

    # ── 결과 표시(모바일: 금액 먼저)
    if not st.session_state["sim_completed"]:
        st.info("왼쪽에서 입력하고 **🚀 계산하기**를 눌러줘 🙂")
        return

    terminal = st.session_state["terminal_wealth"]
    value_path = st.session_state["value_path"]

    p5, p50, p95 = np.percentile(terminal, [5, 50, 95])
    goal_prob = float(np.mean(terminal >= float(goal_amount))) if float(goal_amount) > 0 else np.nan
    total_principal = float(initial_capital) + float(monthly_contribution) * (int(years) * 12)

    # 1티어 카드(금액)
    st.subheader("✅ 결과 요약")
    st.metric("보통은(중앙값)", format_krw_readable(p50))
    c1, c2 = st.columns(2)
    c1.metric("보수적으로(p5)", format_krw_readable(p5))
    c2.metric("낙관적으로(p95)", format_krw_readable(p95))

    st.metric("총 납입액", format_krw_readable(total_principal))

    # 2티어: 목표확률(작게)
    if float(goal_amount) > 0:
        st.caption(f"목표 {format_krw_readable(goal_amount)} **만기 달성 확률**: **{pct_str01(goal_prob)}**")

    # 미니 메시지
    st.write(
        f"👉 {years}년 동안 월 {krw_compact(monthly_contribution)} 기준으로, "
        f"보통은 **{krw_compact(p50)}**, 보수적으로 **{krw_compact(p5)}**, 낙관적으로 **{krw_compact(p95)}** 정도 범위예요."
    )

    # 그래프는 접기(신뢰 보강)
    with st.expander("📊 그래프 보기(신뢰 보강용)", expanded=False):
        st.plotly_chart(make_path_fanchart_mobile(value_path), use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(make_terminal_hist_mobile(terminal, float(goal_amount)), use_container_width=True, config={"displayModeBar": False})

    # 상세/다운로드도 접기
    with st.expander("🔍 상세(고급)", expanded=False):
        # 리스크(낙폭)
        mdd_med = float(np.median(st.session_state["mdd"]))
        st.caption(f"최대 낙폭(MDD) 중앙값: {mdd_med*100:.1f}% (참고)")

        targets_used = st.session_state.get("targets_used", None)
        if targets_used is not None:
            st.write("현실모드 타겟(연)")
            st.write({k: f"{v*100:.2f}%" for k, v in targets_used.items()})

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