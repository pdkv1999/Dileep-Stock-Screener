import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI()
_pool = ThreadPoolExecutor(max_workers=8)

# Nifty 100 universe (Nifty 50 + Nifty Next 50)
STOCKS = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL",
    "KOTAKBANK","ITC","LT","AXISBANK","BAJFINANCE","MARUTI","HCLTECH","SUNPHARMA",
    "TITAN","WIPRO","ULTRACEMCO","ONGC","NTPC","POWERGRID","TATAMOTORS","BAJAJFINSV",
    "TECHM","JSWSTEEL","TATASTEEL","COALINDIA","INDUSINDBK","DRREDDY","NESTLEIND",
    "GRASIM","HINDALCO","CIPLA","BRITANNIA","BPCL","DIVISLAB","EICHERMOT","HEROMOTOCO",
    "BAJAJ-AUTO","HDFCLIFE","SBILIFE","M&M","LTIM","ADANIPORTS","APOLLOHOSP",
    "ASIANPAINT","TATACONSUM","UPL","ZOMATO",
    # Nifty Next 50
    "ABB","ADANIENT","AMBUJACEM","BANDHANBNK","BANKBARODA","BEL","BERGEPAINT",
    "BOSCHLTD","CANBK","CHOLAFIN","COLPAL","DABUR","DLF","FEDERALBNK","GODREJCP",
    "GODREJPROP","HAVELLS","ICICIGI","ICICIPRULI","IDFCFIRSTB","INDHOTEL","INDUSTOWER",
    "IRCTC","MARICO","MUTHOOTFIN","NAUKRI","PAGEIND","PERSISTENT","PIDILITIND","PNB",
    "POLYCAB","RECLTD","SAIL","SBICARD","SIEMENS","TORNTPHARM","TRENT","TVSMOTOR",
    "VBL","VEDL","ZYDUSLIFE","MCDOWELL-N","OFSS","DMART","ALKEM","AUROPHARMA",
    "SHREECEM","BAJAJHLDNG","TATAPOWER","YESBANK",
]

SECTORS = {
    "Banking":     ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","SBIN","INDUSINDBK",
                    "BANDHANBNK","BANKBARODA","FEDERALBNK","IDFCFIRSTB","YESBANK","PNB","CANBK"],
    "IT":          ["TCS","INFY","HCLTECH","WIPRO","TECHM","LTIM","PERSISTENT","OFSS"],
    "Auto":        ["MARUTI","TATAMOTORS","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT","TVSMOTOR","M&M"],
    "Pharma":      ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","APOLLOHOSP","ALKEM","AUROPHARMA",
                    "TORNTPHARM","ZYDUSLIFE"],
    "FMCG":        ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO","COLPAL",
                    "TATACONSUM","MCDOWELL-N","VBL"],
    "Infra":       ["LT","ABB","SIEMENS","BEL","INDUSTOWER","HAVELLS","POLYCAB"],
    "Energy":      ["ONGC","BPCL","TATAPOWER","ADANIPORTS","VEDL","COALINDIA","RECLTD"],
    "Metals":      ["TATASTEEL","JSWSTEEL","HINDALCO","SAIL"],
    "Cement":      ["ULTRACEMCO","GRASIM","AMBUJACEM","SHREECEM"],
    "NBFC":        ["BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","SBICARD"],
    "Consumer":    ["TITAN","ASIANPAINT","BERGEPAINT","PIDILITIND","PAGEIND","BOSCHLTD"],
    "Insurance":   ["HDFCLIFE","SBILIFE","ICICIGI","ICICIPRULI"],
    "Realty":      ["DLF","GODREJPROP","GODREJCP"],
    "Power":       ["NTPC","POWERGRID"],
    "Telecom":     ["BHARTIARTL"],
    "Hospitality": ["INDHOTEL"],
    "Other":       ["RELIANCE","ADANIENT","BAJAJHLDNG","IRCTC","NAUKRI","ZOMATO",
                    "TRENT","DMART","UPL"],
}

_STOCK_SECTOR = {sym: sec for sec, syms in SECTORS.items() for sym in syms}

SCAN_META = {
    "volume_surge":    "Volume Surge",
    "rsi_momentum":    "RSI Momentum",
    "macd_bullish":    "MACD Bullish Cross",
    "golden_cross":    "Golden Cross",
    "near_52w_high":   "Near 52-Week High",
    "strong_momentum": "Strong Momentum ⚡",
}


def _fetch_all() -> dict[str, pd.DataFrame]:
    tickers = " ".join(f"{s}.NS" for s in STOCKS)
    raw = yf.download(
        tickers, period="6mo", interval="1d",
        progress=False, auto_adjust=True, group_by="ticker",
    )
    result = {}
    for sym in STOCKS:
        try:
            df = raw[f"{sym}.NS"].dropna()
            if len(df) >= 50:
                result[sym] = df
        except Exception:
            pass
    return result


def _technicals(df: pd.DataFrame) -> dict:
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    open_  = df["Open"].squeeze()
    high_  = df["High"].squeeze()
    low_   = df["Low"].squeeze()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss))

    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    macd_s = macd.ewm(span=9, adjust=False).mean()

    price      = float(close.iloc[-1])
    prev       = float(close.iloc[-2])
    open_today = float(open_.iloc[-1])
    high_today = float(high_.iloc[-1])
    low_today  = float(low_.iloc[-1])
    vol_today  = float(volume.iloc[-1])
    vol_avg    = float(volume.rolling(20).mean().iloc[-1])
    high_52w   = float(close.rolling(min(252, len(close))).max().iloc[-1])
    day_range  = high_today - low_today

    return {
        "price":          round(price, 2),
        "change":         round((price - prev) / prev * 100, 2),
        "volume_ratio":   round(vol_today / vol_avg, 2) if vol_avg else 0,
        "rsi":            round(float(rsi.iloc[-1]), 1),
        "ema20":          float(ema20.iloc[-1]),
        "ema50":          float(ema50.iloc[-1]),
        "ema20_prev":     float(ema20.iloc[-2]),
        "ema50_prev":     float(ema50.iloc[-2]),
        "macd":           float(macd.iloc[-1]),
        "macd_sig":       float(macd_s.iloc[-1]),
        "macd_prev":      float(macd.iloc[-2]),
        "msig_prev":      float(macd_s.iloc[-2]),
        "high_52w":       round(high_52w, 2),
        "gap_pct":        round((open_today - prev) / prev * 100, 2),
        "close_in_range": round((price - low_today) / day_range, 2) if day_range > 0 else 0.5,
    }


def _run_scan(scan: str) -> list[dict]:
    data = _fetch_all()
    rows = []

    if scan == "strong_momentum":
        # compute all technicals upfront so we can calculate sector averages
        all_tech = {}
        for sym, df in data.items():
            try:
                all_tech[sym] = _technicals(df)
            except Exception:
                pass

        # sector average daily change
        sector_changes: dict[str, list] = {}
        for sym, t in all_tech.items():
            sec = _STOCK_SECTOR.get(sym, "Other")
            sector_changes.setdefault(sec, []).append(t["change"])
        sector_avg = {sec: sum(v) / len(v) for sec, v in sector_changes.items() if v}

        for sym, t in all_tech.items():
            sec     = _STOCK_SECTOR.get(sym, "Other")
            sec_chg = sector_avg.get(sec, 0)

            checks = [
                t["change"] > 0,                     # 1. up on the day
                t["volume_ratio"] >= 1.5,             # 2. above-average volume
                55 <= t["rsi"] <= 75,                 # 3. RSI sweet spot
                t["macd"] > t["macd_sig"],            # 4. MACD bullish
                t["ema20"] > t["ema50"],              # 5. uptrend
                t["price"] >= t["high_52w"] * 0.90,  # 6. near 52W high
                t["gap_pct"] >= 0.3,                  # 7. gap-up open
                sec_chg >= 0.5,                       # 8. sector tailwind
                t["close_in_range"] >= 0.75,          # 9. closing near day's high (delivery proxy)
            ]
            score = sum(checks)
            if score >= 5:
                rows.append({
                    "symbol":         sym,
                    "price":          t["price"],
                    "change":         t["change"],
                    "volume_ratio":   t["volume_ratio"],
                    "rsi":            t["rsi"],
                    "signal":         "Bullish" if t["ema20"] > t["ema50"] else "Bearish",
                    "high_52w":       t["high_52w"],
                    "score":          score,
                    "sector":         sec,
                    "sector_change":  round(sec_chg, 2),
                    "gap_pct":        t["gap_pct"],
                })
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows

    # ── all other scans ──────────────────────────────────────────────────────
    for sym, df in data.items():
        try:
            t = _technicals(df)
        except Exception:
            continue

        match = False
        if scan == "volume_surge"  and t["volume_ratio"] >= 2.0:
            match = True
        elif scan == "rsi_momentum" and 55 <= t["rsi"] <= 75:
            match = True
        elif scan == "macd_bullish" and t["macd"] > t["macd_sig"] and t["macd_prev"] <= t["msig_prev"]:
            match = True
        elif scan == "golden_cross" and t["ema20"] > t["ema50"] and t["ema20_prev"] <= t["ema50_prev"]:
            match = True
        elif scan == "near_52w_high" and t["price"] >= t["high_52w"] * 0.95:
            match = True

        if match:
            rows.append({
                "symbol":       sym,
                "price":        t["price"],
                "change":       t["change"],
                "volume_ratio": t["volume_ratio"],
                "rsi":          t["rsi"],
                "signal":       "Bullish" if t["ema20"] > t["ema50"] else "Bearish",
                "high_52w":     t["high_52w"],
            })

    rows.sort(key=lambda x: x["volume_ratio"], reverse=True)
    return rows


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "web_static" / "index.html").read_text()


@app.get("/api/scans")
async def list_scans():
    return [{"key": k, "label": v} for k, v in SCAN_META.items()]


@app.get("/api/scan/{scan_key}")
async def run_scan(scan_key: str):
    if scan_key not in SCAN_META:
        raise HTTPException(status_code=400, detail="Unknown scan")
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(_pool, _run_scan, scan_key)
    return {"scan": SCAN_META[scan_key], "count": len(rows), "results": rows}
