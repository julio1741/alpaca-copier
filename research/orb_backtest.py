#!/usr/bin/env python3
"""Ruptura del rango de apertura (ORB), velas de 1 minuto, SPY y QQQ 2019-2026.

Regla por día (una operación como máximo):
  1. Rango de apertura = máximo y mínimo de los primeros N minutos desde las 09:30 NY.
  2. Después, la primera vez que el precio supera el máximo -> compra en el máximo (orden stop);
     si primero perfora el mínimo -> venta corta en el mínimo (o nada, en modo "solo largos").
  3. Stop = el otro extremo del rango (riesgo R = ancho del rango).
     Objetivo = k x R, o "fin del día" (cierre a las 15:59).
  4. Si en una misma vela se tocan stop y objetivo, se asume el stop (conservador).
Costo: COST_RT de ida y vuelta sobre el nocional (spread + deslizamiento).
Tamaño: se arriesga RISK del capital por operación, con apalancamiento máximo MAX_LEV (como en futuros).

Entrenamiento 2019-2022, validación 2023-2026.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache" / "intraday"
COST_RT = 0.0001   # 0,01% ida y vuelta (≈5 centavos en SPY de $500)
RISK = 0.01        # 1% del capital por operación
MAX_LEV = 4.0
SPLIT = "2023-01-01"


def load_days(sym: str) -> list[tuple[pd.Timestamp, np.ndarray]]:
    df = pd.read_pickle(CACHE / f"{sym}_1min.pkl").between_time("09:30", "15:59")
    out = []
    for d, g in df.groupby(df.index.date):
        if len(g) < 300:  # días incompletos (cierres tempranos incluidos se descartan)
            continue
        out.append((pd.Timestamp(d), g[["o", "h", "l", "c"]].to_numpy()))
    return out


def trade_day(bars: np.ndarray, n_or: int, target: float | None, shorts: bool):
    """Devuelve (retorno sobre el nocional, R múltiplo, ancho del rango %) o None si no operó."""
    hi, lo = bars[:n_or, 1].max(), bars[:n_or, 2].min()
    rng = hi - lo
    if rng <= 0:
        return None
    for i in range(n_or, len(bars)):
        o, h, l, c = bars[i]
        side = 1 if h >= hi else (-1 if (shorts and l <= lo) else 0)
        if h >= hi and l <= lo:  # rompe ambos en la misma vela: no se sabe cuál primero -> se omite
            return None
        if side == 0:
            continue
        entry = max(o, hi) if side == 1 else min(o, lo)   # gap por encima: se llena a la apertura de la vela
        stop = lo if side == 1 else hi
        risk = abs(entry - stop)
        tgt = entry + side * target * risk if target else None
        for j in range(i, len(bars)):
            _, hj, lj, cj = bars[j]
            hit_stop = (lj <= stop) if side == 1 else (hj >= stop)
            hit_tgt = tgt is not None and ((hj >= tgt) if side == 1 else (lj <= tgt))
            if hit_stop and not (j == i and side * (stop - entry) > 0):
                exit_px = stop
                break
            if hit_tgt:
                exit_px = tgt
                break
        else:
            exit_px = bars[-1, 3]
        ret = side * (exit_px - entry) / entry - COST_RT
        return ret, ret * entry / risk, risk / entry
    return None


def run(days, n_or, target, shorts):
    rows = []
    for d, bars in days:
        t = trade_day(bars, n_or, target, shorts)
        if t is None:
            rows.append((d, 0.0, np.nan, 0.0))
            continue
        ret, r_mult, risk_pct = t
        lev = min(MAX_LEV, RISK / risk_pct)
        rows.append((d, ret * lev, r_mult, lev))
    return pd.DataFrame(rows, columns=["d", "pnl", "R", "lev"]).set_index("d")


def metrics(df: pd.DataFrame) -> dict:
    r = df.pnl
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    tr = df.R.dropna()
    return {"op/día%": round(len(tr) / len(r) * 100), "R medio": round(tr.mean(), 3),
            "aciertos%": round((tr > 0).mean() * 100), "anual%": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
            "Sharpe": round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else 0,
            "maxDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
            "días+%": round((r[r != 0] > 0).mean() * 100), "lev medio": round(df.lev[df.lev > 0].mean(), 1)}


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for sym in ("SPY", "QQQ"):
        days = load_days(sym)
        rows = []
        for n_or, target, shorts in itertools.product((5, 15, 30), (1.0, 2.0, None), (True, False)):
            df = run(days, n_or, target, shorts)
            a, b = metrics(df[df.index < SPLIT]), metrics(df[df.index >= SPLIT])
            name = f"OR {n_or}m · obj {'fin del día' if target is None else f'{target:g}R'} · {'largo/corto' if shorts else 'solo largos'}"
            rows.append({"regla": name, **{f"TR {k}": v for k, v in a.items() if k in ('R medio', 'anual%', 'Sharpe', 'maxDD%')},
                         **{f"TE {k}": v for k, v in b.items()}})
        print(f"\n=== {sym} ({len(days)} días) — riesgo 1%/operación, apalancamiento máx {MAX_LEV:g}x, costo {COST_RT*100:.2f}% i/v ===")
        print(pd.DataFrame(rows).set_index("regla").sort_values("TR Sharpe", ascending=False).to_string())
