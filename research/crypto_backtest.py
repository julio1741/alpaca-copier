#!/usr/bin/env python3
"""Backtest de estrategias de tendencia/momentum en cripto (diario, Binance 2017-2026).

Señal calculada al cierre del día t, posición tomada desde el cierre de t (retorno t -> t+1).
Costos: comisión por cambio de posición (fee por lado) y, en modo perp, funding diario sobre la posición.
Entrenamiento: hasta 2022-12-31. Validación: 2023-01-01 en adelante.

Datos: research/cache/binance/{SYM}USDT-1d-YYYY-MM.zip (data.binance.vision).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache"
SPLIT = "2023-01-01"


def load(sym: str) -> pd.DataFrame:
    frames = []
    for zp in sorted((CACHE / "binance").glob(f"{sym}USDT-1d-*.zip")):
        with zipfile.ZipFile(zp) as z:
            raw = z.read(z.namelist()[0])
        df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 1, 2, 3, 4, 5],
                         names=["t", "o", "h", "l", "c", "v"])
        df = df[pd.to_numeric(df.t, errors="coerce").notna()]
        t = df.t.astype("int64")
        t = np.where(t > 10 ** 14, t // 1000, t)  # desde 2025 vienen en microsegundos
        df["d"] = pd.to_datetime(t, unit="ms")
        frames.append(df.drop(columns="t"))
    return pd.concat(frames).drop_duplicates("d").set_index("d").sort_index().astype(float)


# ---------- señales: devuelven posición objetivo en [-1, 1] ----------

def sig_sma(c: pd.Series, n: int, short: bool = False) -> pd.Series:
    up = c > c.rolling(n).mean()
    return up.astype(float) - ((~up).astype(float) if short else 0)


def sig_mom(c: pd.Series, n: int, short: bool = False) -> pd.Series:
    up = c.pct_change(n) > 0
    return up.astype(float) - ((~up).astype(float) if short else 0)


def sig_donchian(df: pd.DataFrame, n_in: int, n_out: int) -> pd.Series:
    hi = df.h.rolling(n_in).max().shift(1)
    lo = df.l.rolling(n_out).min().shift(1)
    pos = np.zeros(len(df))
    held = 0
    for i in range(len(df)):
        if not held and df.c.iloc[i] > hi.iloc[i]:
            held = 1
        elif held and df.c.iloc[i] < lo.iloc[i]:
            held = 0
        pos[i] = held
    return pd.Series(pos, index=df.index)


def vol_target(pos: pd.Series, c: pd.Series, target_ann: float, max_lev: float, look: int = 30) -> pd.Series:
    vol = c.pct_change().rolling(look).std() * np.sqrt(365)
    return (pos * (target_ann / vol).clip(upper=max_lev)).fillna(0)


def returns(pos: pd.Series, c: pd.Series, fee: float, funding_ann: float = 0.0) -> pd.Series:
    """Retorno diario de la estrategia. `pos` decidido al cierre de t se aplica al retorno t->t+1."""
    r = c.pct_change().shift(-1)
    turn = pos.diff().abs().fillna(pos.abs())
    out = pos * r - turn * fee - pos.clip(lower=0) * funding_ann / 365 + (-pos).clip(lower=0) * funding_ann / 365
    return out.shift(1).dropna()  # fechado en el día en que se realiza


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 30:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    active = r != 0
    return {"CAGR%": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
            "Sharpe": round(r.mean() / r.std() * np.sqrt(365), 2) if r.std() > 0 else 0,
            "maxDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
            "días+%": round((r[active] > 0).mean() * 100) if active.any() else 0,
            "en mercado%": round(active.mean() * 100),
            "peor día%": round(r.min() * 100, 1)}


def split(r: pd.Series):
    return r[r.index < SPLIT], r[r.index >= SPLIT]


def row(name: str, r: pd.Series) -> dict:
    a, b = split(r)
    ma, mb = metrics(a), metrics(b)
    return {"estrategia": name, **{f"TR {k}": v for k, v in ma.items() if k in ("CAGR%", "Sharpe", "maxDD%")},
            **{f"TE {k}": v for k, v in mb.items()}}


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    data = {s: load(s) for s in ("BTC", "ETH", "SOL")}
    for s, df in data.items():
        print(s, len(df), df.index[0].date(), "->", df.index[-1].date(), "último", df.c.iloc[-1])
    FEE = 0.0025  # Alpaca spot (taker)
    for s in ("BTC", "ETH"):
        df = data[s]
        c = df.c
        rows = [row("comprar y mantener", returns(pd.Series(1.0, index=c.index), c, 0))]
        for n in (20, 50, 100, 150, 200):
            rows.append(row(f"SMA{n} largo/fuera", returns(sig_sma(c, n), c, FEE)))
        for n in (7, 14, 30, 60, 90):
            rows.append(row(f"momentum {n}d largo/fuera", returns(sig_mom(c, n), c, FEE)))
        for a, b in ((20, 10), (55, 20)):
            rows.append(row(f"Donchian {a}/{b}", returns(sig_donchian(df, a, b), c, FEE)))
        print(f"\n=== {s} — spot (comisión {FEE*100:.2f}% por lado) ===")
        print(pd.DataFrame(rows).set_index("estrategia").to_string())
