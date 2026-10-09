#!/usr/bin/env python3
"""Revisión de la estrategia ORB contra reglas publicadas (sin ajustar parámetros nuestros).

Variantes, todas con las reglas tal como las publicaron sus autores:
  base   nuestra regla actual: rango 30 min, compra al romper el máximo, stop en el mínimo, objetivo 1R.
  zar5   Zarattini & Aziz (2023): vela de 5 min; si cierra al alza, compra a las 09:35 (si a la baja, vende en
         corto); stop en el extremo opuesto de esa vela; objetivo 10R o cierre del día.
  noise  Zarattini, Aziz & Barbon (2024): "área de ruido" = movimiento medio desde la apertura a esa misma hora
         en los 14 días previos; cada media hora compra sobre el borde superior / vende bajo el inferior; salida
         con stop móvil max(borde, VWAP) o al cierre. Tamaño por volatilidad (2% diario, tope 4x).
  *-L    solo compras.

Se mide en tres tramos: 2019-2022 (donde elegimos la regla base), 2023-2026, y desde jun-2024 (posterior a la
publicación de ambos papers: ahí ninguna regla pudo ajustarse a esos datos).
Topes de apalancamiento: 4x (cuenta con margen >= US$2.000) y 1x (la cuenta actual de US$1.000).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache" / "intraday"
N = 390  # minutos de 09:30 a 15:59


def load(sym: str):
    df = pd.read_pickle(CACHE / f"{sym}_1min.pkl").between_time("09:30", "15:59")
    days, arrs = [], []
    for d, g in df.groupby(df.index.date):
        if len(g) < 300:
            continue
        mins = (g.index.hour * 60 + g.index.minute - 570).to_numpy()
        a = np.full((N, 5), np.nan)
        a[mins] = g[["o", "h", "l", "c", "v"]].to_numpy()
        a = pd.DataFrame(a).ffill().bfill().to_numpy()
        days.append(pd.Timestamp(d))
        arrs.append(a)
    return pd.DatetimeIndex(days), np.stack(arrs)


def walk(bars, i0, side, entry, stop, tgt):
    """Recorre desde la vela i0; devuelve precio de salida (stop conservador si stop y objetivo en la misma vela)."""
    for j in range(i0, N):
        h, l = bars[j, 1], bars[j, 2]
        if (l <= stop) if side == 1 else (h >= stop):
            return stop
        if tgt is not None and ((h >= tgt) if side == 1 else (l <= tgt)):
            return tgt
    return bars[-1, 3]


def orb_base(bars, long_only=True, n_or=30, k=1.0):
    hi, lo = bars[:n_or, 1].max(), bars[:n_or, 2].min()
    for i in range(n_or, N):
        o, h, l = bars[i, :3]
        if h >= hi and l <= lo:
            return None
        if h >= hi:
            entry = max(o, hi)
            risk = entry - lo
            return [(1, entry, walk(bars, i, 1, entry, lo, entry + k * risk if k else None), risk / entry)]
        if l <= lo and not long_only:
            entry = min(o, lo)
            risk = hi - entry
            return [(-1, entry, walk(bars, i, -1, entry, hi, entry - k * risk), risk / entry)]
    return None


def orb_zar5(bars, long_only=False):
    o0, c4 = bars[0, 0], bars[4, 3]
    hi, lo = bars[:5, 1].max(), bars[:5, 2].min()
    side = 1 if c4 > o0 else (-1 if c4 < o0 else 0)
    if side == 0 or (long_only and side == -1):
        return None
    entry = bars[5, 0]
    stop = lo if side == 1 else hi
    risk = abs(entry - stop)
    if risk <= 0 or side * (entry - stop) <= 0:
        return None
    return [(side, entry, walk(bars, 5, side, entry, stop, entry + side * 10 * risk), risk / entry)]


def run_orb(days, X, fn, lev_cap, cost, **kw):
    pnl = np.zeros(len(days))
    for t in range(len(days)):
        trades = fn(X[t], **kw)
        if not trades:
            continue
        for side, entry, exit_px, risk_pct in trades:
            lev = min(lev_cap, 0.01 / risk_pct)
            pnl[t] += lev * (side * (exit_px / entry - 1) - cost)
    return pd.Series(pnl, index=days)


def run_noise(days, X, lev_cap, cost, long_only=False, lookback=14, target_vol=0.02):
    opens, closes = X[:, 0, 0], X[:, -1, 3]
    move = np.abs(X[:, :, 3] / opens[:, None] - 1)            # |cierre del minuto / apertura - 1|
    daily_ret = np.r_[np.nan, closes[1:] / closes[:-1] - 1]
    pv = X[:, :, 3] * X[:, :, 4]
    vwap = np.cumsum(pv, axis=1) / np.maximum(np.cumsum(X[:, :, 4], axis=1), 1)
    checks = range(30, 361, 30)  # 10:00, 10:30, ..., 15:30 (precio = cierre del minuto anterior)
    pnl = np.full(len(days), np.nan)
    for t in range(lookback + 1, len(days)):
        sig = move[t - lookback:t].mean(axis=0)
        ub = max(opens[t], closes[t - 1]) * (1 + sig)
        lb = min(opens[t], closes[t - 1]) * (1 - sig)
        vol = np.nanstd(daily_ret[t - lookback:t])  # 14 retornos diarios hasta ayer
        lev = min(lev_cap, target_vol / vol) if vol and vol > 0 else 0
        pos, entry, r = 0, 0.0, 0.0
        for k in checks:
            p = X[t, k - 1, 3]
            if pos == 1 and p < max(ub[k - 1], vwap[t, k - 1]):
                r += p / entry - 1 - cost; pos = 0
            elif pos == -1 and p > min(lb[k - 1], vwap[t, k - 1]):
                r += 1 - p / entry - cost; pos = 0
            if pos == 0:
                if p > ub[k - 1]:
                    pos, entry = 1, p
                elif p < lb[k - 1] and not long_only:
                    pos, entry = -1, p
        if pos:
            p = closes[t]
            r += (p / entry - 1 if pos == 1 else 1 - p / entry) - cost
        pnl[t] = lev * r
    return pd.Series(pnl, index=days).dropna()


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return {"anual%": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
            "Sharpe": round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else 0.0,
            "maxDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
            "US$/día c/1000": round(r.mean() * 1000, 2)}


SLICES = {"2019-22": ("2019", "2022-12-31"), "2023-26": ("2023", "2026-12-31"), "post-paper": ("2024-06", "2026-12-31")}

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for sym in ("QQQ", "SPY"):
        days, X = load(sym)
        for cost in (0.0001, 0.0002):
            rows = []
            for lev in (4.0, 1.0):
                variants = {
                    "base (OR30 1R largos)": run_orb(days, X, orb_base, lev, cost),
                    "zar5 largo/corto": run_orb(days, X, orb_zar5, lev, cost),
                    "zar5-L": run_orb(days, X, orb_zar5, lev, cost, long_only=True),
                    "noise largo/corto": run_noise(days, X, lev, cost),
                    "noise-L": run_noise(days, X, lev, cost, long_only=True),
                }
                for name, r in variants.items():
                    row = {"regla": name, "tope": f"{lev:g}x"}
                    for s, (a, b) in SLICES.items():
                        for k, v in stats(r[a:b]).items():
                            if k in ("anual%", "Sharpe") or s == "post-paper":
                                row[f"{s} {k}"] = v
                    rows.append(row)
            print(f"\n=== {sym} · costo {cost*100:.2f}% ida y vuelta ===")
            print(pd.DataFrame(rows).set_index(["regla", "tope"]).to_string())
