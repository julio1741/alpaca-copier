#!/usr/bin/env python3
"""Backtest de estrategias candidatas para ingreso diario sobre SPY (capital $100k, sin apalancamiento).

Estrategias (todas cerradas antes de un nuevo día o con regla explícita):
  bh         comprar y mantener (referencia)
  overnight  comprar al cierre, vender en la apertura siguiente
  intraday   comprar en la apertura, vender al cierre
  ibs        mean reversion: comprar al cierre si IBS=(c-l)/(h-l) < 0.2, vender al cierre siguiente
  rsi2       comprar al cierre si RSI(2) < 10 y c > SMA200, vender al cierre si RSI(2) > 70 o 5 días
  im         intraday momentum: a las 15:30 ir en dirección del retorno 09:30-10:00; cerrar a las 16:00
  on_ibs     overnight solo cuando IBS < 0.5 (filtro de reversión)

Entrenamiento 2016-2023 · Validación 2024-09/2026. Métricas por día de calendario de bolsa.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache" / "daily"
CAP = 100_000
SPREAD = 0.01  # costo por acción por operación (medio spread SPY), sin comisiones


def load_daily(sym: str) -> pd.DataFrame:
    b = json.load(open(CACHE / f"{sym}.json"))
    df = pd.DataFrame(b, columns=["t", "o", "h", "l", "c", "v"])
    df["d"] = pd.to_datetime(df.t.str[:10])
    return df.set_index("d").drop(columns="t")


def cost(px: float) -> float:
    """Fracción de costo por ida y vuelta."""
    return 2 * SPREAD / px


def strat_returns(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, l, c = df.o, df.h, df.l, df.c
    prev_c = c.shift(1)
    out = {}
    out["bh"] = c.pct_change()
    out["overnight"] = (o / prev_c - 1) - cost(o)
    out["intraday"] = (c / o - 1) - cost(o)
    ibs = ((c - l) / (h - l)).replace([np.inf, -np.inf], np.nan)
    sig = (ibs.shift(1) < 0.2)
    out["ibs"] = np.where(sig, c / prev_c - 1 - cost(c), 0.0)
    out["ibs"] = pd.Series(out["ibs"], index=df.index)
    # RSI(2)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    sma = c.rolling(200).mean()
    pos = np.zeros(len(df)); held = 0; days = 0
    for i in range(1, len(df)):
        if held:
            days += 1
            pos[i] = 1
            if rsi.iloc[i] > 70 or days >= 5:
                held = 0
        if not held and rsi.iloc[i] < 10 and c.iloc[i] > sma.iloc[i]:
            held, days = 1, 0
    pos = pd.Series(pos, index=df.index)
    r = c.pct_change()
    trades = pos.diff().abs().fillna(0)
    out["rsi2"] = pos.shift(1).fillna(0) * r - trades * SPREAD / c
    on_sig = (ibs < 0.5).shift(1)
    out["on_ibs"] = np.where(on_sig.shift(-1).fillna(False) if False else (ibs.shift(1) < 0.5), (o / prev_c - 1) - cost(o), 0.0)
    out["on_ibs"] = pd.Series(out["on_ibs"], index=df.index)
    return out


def intraday_momentum(path: Path) -> pd.Series:
    b = json.load(open(path))
    df = pd.DataFrame(b, columns=["t", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df.t).dt.tz_convert("America/New_York")
    df["d"] = df.ts.dt.date
    df["hm"] = df.ts.dt.strftime("%H:%M")
    rows = []
    for d, g in df.groupby("d"):
        g = g.set_index("hm")
        if not {"09:30", "15:30"} <= set(g.index):
            continue
        first = g.loc["10:00", "o"] / g.loc["09:30", "o"] - 1 if "10:00" in g.index else g.loc["09:30", "c"] / g.loc["09:30", "o"] - 1
        o_last, c_last = g.loc["15:30", "o"], g.loc["15:30", "c"]
        last = c_last / o_last - 1
        r = np.sign(first) * last - cost(o_last) if first != 0 else 0.0
        rows.append((pd.Timestamp(d), r))
    return pd.Series(dict(rows)).sort_index()


def metrics(r: pd.Series, cap: float = CAP) -> dict:
    r = r.dropna()
    if r.empty:
        return {}
    eq = cap * (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    daily_usd = r * cap  # P&L diario sobre capital fijo (sin reinvertir) para leer en dólares
    active = r != 0
    yrs = len(r) / 252
    return {
        "años": round(yrs, 1), "días": len(r), "activo%": round(active.mean() * 100),
        "total%": round((eq.iloc[-1] / cap - 1) * 100, 1), "anual%": round(((eq.iloc[-1] / cap) ** (1 / yrs) - 1) * 100, 1),
        "$/día": round(daily_usd.mean(), 1), "$/día med": round(daily_usd[active].median(), 1) if active.any() else 0,
        "días+%": round((daily_usd[active] > 0).mean() * 100) if active.any() else 0,
        "peor día $": round(daily_usd.min()), "std $": round(daily_usd.std()),
        "sharpe": round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else 0, "maxDD%": round(dd * 100, 1),
    }


def split(r: pd.Series):
    return r[r.index < "2024-01-01"], r[r.index >= "2024-01-01"]


if __name__ == "__main__":
    spy = load_daily("SPY")
    rets = strat_returns(spy)
    rets["im"] = intraday_momentum(CACHE / "SPY_30min.json")
    pd.set_option("display.width", 250)
    for name, period in (("ENTRENAMIENTO 2016-2023", 0), ("VALIDACIÓN 2024-sep/2026", 1)):
        print(f"\n== {name}")
        rows = {k: metrics(split(v)[period]) for k, v in rets.items()}
        print(pd.DataFrame(rows).T.to_string())
