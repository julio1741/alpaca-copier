#!/usr/bin/env python3
"""Backtest de copiar compras de ejecutivos (Formulario 4).

Entrada: apertura del día hábil SIGUIENTE a la presentación. Salida: cierre tras H sesiones.
Exceso = retorno - retorno de SPY en la misma ventana. Entrenamiento 2024-2025, validación 2026-Q1.
Requiere: cache/insider_events.pkl (insiders.py) y cache/bars_ins.json.
"""
from __future__ import annotations

import bisect
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache"


class Px:
    def __init__(self):
        raw = json.load(open(CACHE / "bars_ins.json"))
        self.d = {}
        for s, b in raw.items():
            b.sort()
            dates = [x[0] for x in b]
            self.d[s] = (dates, {x[0]: i for i, x in enumerate(b)},
                         np.array([x[1] for x in b]), np.array([x[2] for x in b]), np.array([x[3] for x in b]))
        self.cal = self.d["SPY"][0]

    def next_session(self, day: str) -> str | None:
        i = bisect.bisect_right(self.cal, day)
        return self.cal[i] if i < len(self.cal) else None


def build_trades(ev: pd.DataFrame, px: Px, holds=(20, 60, 120)) -> pd.DataFrame:
    spy_dates, spy_idx, spy_o, spy_c, _ = px.d["SPY"]
    rows = []
    for r in ev.itertuples():
        if r.ticker not in px.d:
            continue
        entry_day = px.next_session(r.filed.strftime("%Y-%m-%d"))
        dates, idx, o, c, v = px.d[r.ticker]
        if entry_day not in idx or entry_day not in spy_idx:
            continue
        i, si = idx[entry_day], spy_idx[entry_day]
        if i < 20:
            continue
        dollar_vol = float(np.mean(c[i - 20:i] * v[i - 20:i]))
        rec = {"acc": r.ACCESSION_NUMBER, "ticker": r.ticker, "cik": r.ISSUERCIK, "filed": r.filed,
               "entry_day": entry_day, "entry": o[i], "dollar_vol": dollar_vol, "value": r.value,
               "officer": r.is_officer, "director": r.is_director, "ten": r.is_10pct, "ceo_cfo": r.is_ceo_cfo,
               "cluster": r.cluster, "pct_inc": r.pct_increase, "lag": (r.filed - r.trans_date).days}
        for h in holds:
            sj = si + h - 1
            rec[f"ret{h}"] = rec[f"ex{h}"] = np.nan
            if sj >= len(spy_c):
                continue
            j = bisect.bisect_right(dates, spy_dates[sj]) - 1  # último cierre del ticker <= día de salida
            if j <= i or spy_dates[sj] > dates[-1]:
                continue
            ret = c[j] / o[i] - 1
            rec[f"ret{h}"], rec[f"ex{h}"] = ret, ret - (spy_c[sj] / spy_o[si] - 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def stats(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    if n < 2:
        return {"n": n, "mean": np.nan, "med": np.nan, "hit": np.nan, "t": np.nan}
    return {"n": n, "mean": x.mean(), "med": x.median(), "hit": (x > 0).mean(), "t": x.mean() / (x.std() / n ** .5)}


FILTERS = {
    "rol": {"todos": lambda d: d.index == d.index, "officer": lambda d: d.officer,
            "ceo_cfo": lambda d: d.ceo_cfo, "no_solo_10%": lambda d: ~(d.ten & ~d.officer & ~d.director)},
    "monto": {">=25k": 25e3, ">=100k": 1e5, ">=500k": 5e5},
    "grupo": {">=1": 1, ">=2": 2, ">=3": 3},
}


def grid(tr: pd.DataFrame, h: int) -> pd.DataFrame:
    out = []
    for (rn, rf), (mn, mv), (gn, gv) in itertools.product(FILTERS["rol"].items(), FILTERS["monto"].items(),
                                                          FILTERS["grupo"].items()):
        m = rf(tr) & (tr.value >= mv) & (tr.cluster >= gv)
        out.append({"rol": rn, "monto": mn, "grupo": gn, "h": h, **stats(tr.loc[m, f"ex{h}"])})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ev = pd.read_pickle(CACHE / "insider_events.pkl")
    px = Px()
    tr = build_trades(ev, px)
    tr.to_pickle(CACHE / "insider_trades.pkl")
    base = tr[(tr.entry >= 5) & (tr.dollar_vol >= 1e6)]
    print(f"trades {len(tr)}; con precio>=$5 y volumen>=$1M/día: {len(base)}")
    train = base[base.filed < "2026-01-01"]
    test = base[base.filed >= "2026-01-01"]
    for h in (20, 60, 120):
        print(f"TODOS h={h:3d}  train {stats(train[f'ex{h}'])}  test {stats(test[f'ex{h}'])}")
