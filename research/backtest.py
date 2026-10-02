#!/usr/bin/env python3
"""Backtest: ¿qué tan bien le habría ido al bot copiando a cada congresista de la Cámara?

Simula exactamente las reglas del bot (copier.decide):
  - Compra: se detecta el día de la declaración y se compra en la APERTURA del siguiente día hábil.
  - Venta:  cuando el mismo congresista declara la venta de ese ticker, se vende al CIERRE del día de
            esa declaración (o del siguiente hábil). Sin venta declarada -> posición abierta, valorizada
            al último cierre disponible.
  - Cada compra es un lote independiente; se compara contra SPY en la misma ventana (exceso).

Entrenamiento = compras declaradas en 2024-2025; validación = declaradas en 2026.

Datos (research/cache): ptr_index.tsv, all_tx.pkl (transacciones parseadas), bars.json (Alpaca 1Day).
"""
from __future__ import annotations

import bisect
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from copier import decide  # noqa: E402  mismas reglas que el bot

CACHE = ROOT / "cache"


@dataclass
class Lot:
    member: str
    ticker: str
    filed: date
    entry_day: str
    entry: float
    exit_day: str
    exit: float
    closed: bool
    spy_ret: float
    pct: float
    managed: bool

    @property
    def ret(self) -> float:
        return self.exit / self.entry - 1

    @property
    def excess(self) -> float:
        return self.ret - self.spy_ret

    @property
    def days(self) -> int:
        return (date.fromisoformat(self.exit_day) - date.fromisoformat(self.entry_day)).days


class Prices:
    def __init__(self, bars: dict):
        self.days = {s: [b[0] for b in v] for s, v in bars.items()}
        self.bars = {s: {b[0]: (b[1], b[2]) for b in v} for s, v in bars.items()}
        self.cal = self.days["SPY"]

    def next_session(self, d: date, strictly_after: bool) -> str | None:
        key = d.isoformat()
        i = bisect.bisect_right(self.cal, key) if strictly_after else bisect.bisect_left(self.cal, key)
        return self.cal[i] if i < len(self.cal) else None

    def get(self, sym: str, day: str, field: int) -> float | None:
        b = self.bars.get(sym, {}).get(day)
        return b[field] if b else None

    def last(self, sym: str) -> tuple[str, float] | None:
        if sym not in self.days:
            return None
        d = self.days[sym][-1]
        return d, self.bars[sym][d][1]


def load():
    cfg = json.loads((ROOT.parent / "config.json").read_text())
    rows = pickle.load(open(CACHE / "all_tx.pkl", "rb"))
    prices = Prices(json.load(open(CACHE / "bars.json")))
    return cfg, rows, prices


def member_key(last: str, dist: str) -> str:
    """El índice escribe el mismo nombre de varias formas ("Marjorie Taylor Mrs Greene"); apellido+distrito es estable."""
    return f"{last.strip()} ({dist.strip()})"


def build_lots(cfg, rows, prices, hold_cap_days: int | None = None) -> tuple[list[Lot], dict]:
    # (miembro, ticker) -> fechas de declaración de ventas
    sales = defaultdict(list)
    buys = []
    stats = defaultdict(int)
    for r, t in rows:
        y, doc, last, first, dist, fdate = r
        member = member_key(last, dist)
        filed = datetime.strptime(fdate, "%m/%d/%Y").date()
        d = decide(t, cfg)
        if d.action == "close":
            sales[(member, t.ticker)].append(filed)
        elif d.action == "buy":
            managed = "managed" in f"{t.description} {t.asset}".lower()
            buys.append((member, t.ticker, filed, d.pct, managed))
    for v in sales.values():
        v.sort()

    lots = []
    for member, sym, filed, pct, managed in buys:
        entry_day = prices.next_session(filed, strictly_after=True)
        entry = prices.get(sym, entry_day, 0) if entry_day else None
        if not entry:
            stats["sin_precio"] += 1
            continue
        exit_day = None
        for s in sales.get((member, sym), []):
            if s.isoformat() >= entry_day:
                exit_day = prices.next_session(s, strictly_after=False)
                break
        if hold_cap_days is not None:
            cap = prices.next_session(date.fromisoformat(entry_day).fromordinal(
                date.fromisoformat(entry_day).toordinal() + hold_cap_days), strictly_after=False)
            if cap and (exit_day is None or cap < exit_day):
                exit_day = cap
        closed = exit_day is not None
        exit_px = prices.get(sym, exit_day, 1) if exit_day else None
        if exit_px is None:
            lastb = prices.last(sym)
            if not lastb:
                stats["sin_precio"] += 1
                continue
            exit_day, exit_px = lastb
            closed = False
        spy_in = prices.get("SPY", entry_day, 0)
        spy_out = prices.get("SPY", exit_day, 1) or prices.last("SPY")[1]
        lots.append(Lot(member, sym, filed, entry_day, entry, exit_day, exit_px, closed,
                        spy_out / spy_in - 1, pct, managed))
        stats["lotes"] += 1
    return lots, stats


def summarize(lots: list[Lot]) -> dict:
    ex = [l.excess for l in lots]
    n = len(ex)
    sd = pstdev(ex) if n > 1 else 0.0
    w = sum(l.pct for l in lots) or 1
    return {
        "n": n,
        "ret": mean(l.ret for l in lots) if n else 0,
        "excess": mean(ex) if n else 0,
        "excess_w": sum(l.excess * l.pct for l in lots) / w if n else 0,
        "median": median(ex) if n else 0,
        "hit": sum(e > 0 for e in ex) / n if n else 0,
        "t": (mean(ex) / (sd / n ** 0.5)) if n > 1 and sd > 0 else 0,
        "closed": sum(l.closed for l in lots) / n if n else 0,
        "days": mean(l.days for l in lots) if n else 0,
        "managed": sum(l.managed for l in lots) / n if n else 0,
        "tickers": len({l.ticker for l in lots}),
    }


def split(lots):
    train = [l for l in lots if l.filed.year in (2024, 2025)]
    test = [l for l in lots if l.filed.year == 2026]
    return train, test


def ranking(lots, min_train=10):
    by = defaultdict(list)
    for l in lots:
        by[l.member].append(l)
    out = []
    for m, ls in by.items():
        tr, te = split(ls)
        if len(tr) < min_train:
            continue
        out.append((m, summarize(tr), summarize(te), summarize(ls)))
    return out


def fmt(s):
    return (f"n={s['n']:4d} exceso={s['excess']*100:+6.1f}% (pond {s['excess_w']*100:+6.1f}%) "
            f"med={s['median']*100:+6.1f}% acierto={s['hit']*100:4.0f}% t={s['t']:+5.2f} "
            f"cerr={s['closed']*100:3.0f}% días={s['days']:4.0f}")


def simulate(lots: list[Lot], prices: Prices, cfg: dict, start: str, end: str, capital: float = 100_000,
             park_spy: bool = False, hold_days: int | None = None, rotate: bool = False) -> dict:
    """Cartera diaria con las reglas del bot: % del equity por compra, tope por ticker, solo efectivo.
    Compra en la apertura del día de entrada; vende al cierre del día de salida (venta declarada).
    park_spy: el efectivo sin usar queda en SPY (se vende SPY para financiar cada compra).
    hold_days: vende al cierre cuando pasan N días corridos desde la última compra copiada del ticker.
    rotate: si no alcanza la plata para una compra, vende a la apertura la posición comprada hace más tiempo."""
    cal = [d for d in prices.cal if start <= d <= end]
    ins = defaultdict(list)
    outs = defaultdict(set)
    for l in lots:
        if start <= l.entry_day <= end:
            ins[l.entry_day].append(l)
            if l.closed:
                outs[l.exit_day].add(l.ticker)
    cash, pos = capital, {}  # ticker -> acciones
    last_px: dict[str, float] = {}
    last_buy: dict[str, str] = {}
    curve, n_buys, buy_days, skipped = [], 0, set(), 0

    def equity_at(day, field):
        tot = cash
        for t, q in pos.items():
            px = prices.get(t, day, field) or last_px.get(t, 0)
            tot += q * px
        return tot

    for day in cal:
        if park_spy:  # liquidar SPY a la apertura para tener todo como "efectivo disponible"
            px = prices.get("SPY", day, 0)
            cash += pos.pop("SPY", 0) * px
        eq_open = equity_at(day, 0)
        for l in ins.get(day, []):
            px = prices.get(l.ticker, day, 0)
            if not px:
                continue
            held = pos.get(l.ticker, 0) * px
            want = min(eq_open * l.pct / 100, eq_open * cfg["max_position_pct"] / 100 - held)
            while rotate and cash < want:
                olds = sorted((d, t) for t, d in last_buy.items() if t in pos and t != l.ticker
                              and prices.get(t, day, 0))
                if not olds:
                    break
                t_old = olds[0][1]
                cash += pos.pop(t_old) * prices.get(t_old, day, 0)
                last_buy.pop(t_old, None)
            notional = min(want, cash)
            if notional < cfg["min_order_usd"]:
                skipped += 1
                continue
            pos[l.ticker] = pos.get(l.ticker, 0) + notional / px
            cash -= notional
            last_buy[l.ticker] = day
            n_buys += 1
            buy_days.add(day)
        for t in list(pos):
            if t == "SPY" and park_spy:
                continue
            px = prices.get(t, day, 1)
            if px:
                last_px[t] = px
            expired = (hold_days is not None and t in last_buy and
                       (date.fromisoformat(day) - date.fromisoformat(last_buy[t])).days >= hold_days)
            if (t in outs.get(day, ()) or expired) and px:
                cash += pos.pop(t) * px
                last_buy.pop(t, None)
        if park_spy:  # recomprar SPY con lo que sobró, a la misma apertura (sin costo)
            px = prices.get("SPY", day, 0)
            pos["SPY"] = pos.get("SPY", 0) + cash / px
            cash = 0.0
        curve.append((day, equity_at(day, 1)))
    spy0, spy1 = prices.get("SPY", cal[0], 0), prices.get("SPY", cal[-1], 1)
    eq = [v for _, v in curve]
    peak, dd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        dd = min(dd, v / peak - 1)
    return {"ret": eq[-1] / capital - 1, "spy": spy1 / spy0 - 1, "maxdd": dd, "buys": n_buys,
            "buy_days": len(buy_days), "sessions": len(cal), "skipped": skipped,
            "invested_end": 1 - (cash + pos.get("SPY", 0) * (prices.get("SPY", cal[-1], 1) or 0)) / eq[-1],
            "curve": curve}


if __name__ == "__main__":
    cfg, rows, prices = load()
    lots, stats = build_lots(cfg, rows, prices)
    print("lotes:", dict(stats))
    tr, te = split(lots)
    print("TODOS  train", fmt(summarize(tr)))
    print("TODOS  test ", fmt(summarize(te)))
    rk = sorted(ranking(lots), key=lambda x: -x[1]["excess"])
    print(f"\n{len(rk)} congresistas con >=10 compras copiables en 2024-25. Orden por exceso en entrenamiento:\n")
    for m, a, b, _ in rk:
        print(f"{m[:34]:34} TRAIN {fmt(a)}\n{'':34} TEST  {fmt(b)}")
