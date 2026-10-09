#!/usr/bin/env python3
"""Bot 4: "área de ruido" (Zarattini, Aziz y Barbon, 2024) en QQQ. Experimento paralelo al bot ORB.

Regla (tal como la publicaron; contrastada en research/orb_variants.py, QQQ 2019-2026):
  - σ(HH:MM) = promedio, en los 14 días previos, de |cierre del minuto HH:MM / apertura del día - 1|:
    cuánto suele haberse movido QQQ desde la apertura a esa hora. Es el "ruido" normal.
  - Bordes del día: superior = max(apertura, cierre de ayer) × (1 + σ); inferior = min(...) × (1 - σ).
  - Solo a las HH:00 y HH:30 (desde las 10:00 hasta 30 min antes del cierre):
      sin posición: compra si el precio está sobre el borde superior; vende en corto si está bajo el inferior.
      comprado: sale si el precio cae bajo max(borde superior, VWAP) (y si quedó bajo el inferior, gira a corto).
      vendido: sale si el precio sube sobre min(borde inferior, VWAP) (y si quedó sobre el superior, gira a largo).
  - Todo cerrado 2 minutos antes del cierre.
  - Tamaño: equity de ayer × min(4, 2% / volatilidad diaria de 14 días), hasta el poder de compra de la cuenta.
    Compras fraccionarias; los cortos solo en acciones enteras (Alpaca no admite cortos fraccionarios ni cortos con
    menos de US$2.000 de equity: en ese caso el corto se omite y queda anotado).
Cuenta compartida: con NOISE_CAPITAL_USD el bot opera como si tuviera ese capital propio (más su P&L acumulado)
dentro de una cuenta ajena (la del copiador), cierra solo lo que él abrió y marca sus órdenes con el prefijo "nz-",
que el copiador descuenta de su caja y sus posiciones.
Datos: SIP (como el backtest) para todo lo que tenga más de 15 minutos; IEX para el tramo reciente.
Telemetría: cada orden se compara con el precio que vio la regla al decidir (deslizamiento).

Uso: noise_bot.py status | levels [--date AAAA-MM-DD] | serve
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from alpaca_client import Alpaca, load_env

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", ROOT / "noise_data"))
STATE = DATA / "state.json"
JOURNAL = DATA / "trades.csv"
NY = ZoneInfo("America/New_York")
DATA_URL = "https://data.alpaca.markets/v2"
log = logging.getLogger("noise")

CFG = {
    "symbol": "QQQ",
    "lookback": 14,
    "target_vol": 0.02,
    "max_lev": 4.0,
    "check_every": 30,           # minutos
    "flat_minutes_before_close": 2,
    "stale_check_minutes": 5,    # una revisión atrasada más que esto se omite (la regla mira el precio a esa hora)
    "allow_short": True,
    "capital_usd": float(os.environ["NOISE_CAPITAL_USD"]) if os.environ.get("NOISE_CAPITAL_USD") else None,
}

FIELDS = ["date", "time", "action", "side", "qty", "theo_px", "fill_px", "slip_pct", "pnl_usd", "upper", "lower",
          "vwap", "note"]


# ---------- estado y bitácora ----------

def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"day": None, "runs": {}}


def save_state(st: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, default=str))
    tmp.replace(STATE)


def journal(row: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    new = not JOURNAL.exists()
    with JOURNAL.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def journal_rows() -> list[dict]:
    if not JOURNAL.exists():
        return []
    with JOURNAL.open() as f:
        return list(csv.DictReader(f))


# ---------- datos ----------

def fetch_bars(api: Alpaca, feed: str, start: datetime, end: datetime, timeframe: str = "1Min") -> list[dict]:
    if end <= start:
        return []
    out, token = [], None
    while True:
        params = {"symbols": CFG["symbol"], "timeframe": timeframe, "feed": feed, "limit": 10000, "adjustment": "raw",
                  "start": start.isoformat(), "end": (end - timedelta(seconds=1)).isoformat()}
        if token:
            params["page_token"] = token
        d = api._req("GET", f"{DATA_URL}/stocks/bars", params=params)
        out += d["bars"].get(CFG["symbol"], [])
        token = d.get("next_page_token")
        if not token:
            return out


def ny_time(b: dict) -> datetime:
    return datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(NY)


def minute_index(b: dict) -> int:
    t = ny_time(b)
    return t.hour * 60 + t.minute - 570


def recent_bars(api: Alpaca, start: datetime, end: datetime) -> list[dict]:
    """Velas de 1 min: SIP hasta hace 16 min (lo que permite el plan gratuito) e IEX desde ahí.
    Las de IEX llevan su volumen escalado por la razón SIP/IEX observada hoy, para que el VWAP no se deforme."""
    cut = min(end, datetime.now(NY) - timedelta(minutes=16))
    sip = fetch_bars(api, "sip", start, cut)
    if cut >= end:
        return sip
    iex = fetch_bars(api, "iex", start, end)
    if sip:
        last = ny_time(sip[-1])
        iex_overlap = sum(b["v"] for b in iex if ny_time(b) <= last)
        scale = sum(b["v"] for b in sip) / iex_overlap if iex_overlap else 30.0
        tail = [dict(b, v=b["v"] * scale) for b in iex if ny_time(b) > last]
        return sip + tail
    return [dict(b, v=b["v"] * 30.0) for b in iex]


def last_price(api: Alpaca) -> float:
    return float(api._req("GET", f"{DATA_URL}/stocks/{CFG['symbol']}/trades/latest", params={"feed": "iex"})["trade"]["p"])


# ---------- niveles del día ----------

def sessions_before(api: Alpaca, day: date, n: int) -> list[dict]:
    cal = api.calendar((day - timedelta(days=3 * n + 10)).isoformat(), (day - timedelta(days=1)).isoformat())
    return cal[-n:]


def day_levels(api: Alpaca, day: date) -> dict:
    """σ por minuto, cierre de ayer, volatilidad diaria y apertura de hoy (necesita que haya pasado 09:46 NY)."""
    n = CFG["lookback"]
    prev = sessions_before(api, day, n + 1)          # 15 sesiones: 14 para σ y retornos + 1 para el primer retorno
    first = datetime.combine(date.fromisoformat(prev[0]["date"]), datetime.min.time(), NY)
    daily = fetch_bars(api, "sip", first, datetime.combine(day, datetime.min.time(), NY), "1Day")
    closes = [b["c"] for b in daily]
    if len(closes) < n + 1:
        raise RuntimeError(f"solo {len(closes)} cierres diarios")
    rets = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - n, len(closes))]
    mu = sum(rets) / n
    vol = math.sqrt(sum((r - mu) ** 2 for r in rets) / n)
    sigma_sum, days_used = [0.0] * 390, 0
    for s in prev[-n:]:
        d = date.fromisoformat(s["date"])
        op = datetime.combine(d, datetime.strptime(s["open"], "%H:%M").time(), NY)
        cl = datetime.combine(d, datetime.strptime(s["close"], "%H:%M").time(), NY)
        bars = fetch_bars(api, "sip", op, cl)
        if not bars:
            continue
        o = bars[0]["o"]
        closes_by_min, last_c = {minute_index(b): b["c"] for b in bars}, bars[0]["c"]
        for k in range(390):                          # minutos sin operaciones heredan el último cierre
            last_c = closes_by_min.get(k, last_c)
            sigma_sum[k] += abs(last_c / o - 1)
        days_used += 1
    if days_used < n - 2:
        raise RuntimeError(f"solo {days_used} días de velas para σ")
    op_today = datetime.combine(day, datetime.strptime("09:30", "%H:%M").time(), NY)
    first_bar = fetch_bars(api, "sip", op_today, op_today + timedelta(minutes=5))
    if not first_bar:
        raise RuntimeError("sin la vela de apertura de hoy")
    return {"date": str(day), "open": first_bar[0]["o"], "prev_close": closes[-1], "vol": vol,
            "lev": min(CFG["max_lev"], CFG["target_vol"] / vol) if vol > 0 else 0.0,
            "sigma": [x / days_used for x in sigma_sum]}


def bands(lv: dict, k: int) -> tuple[float, float]:
    """Bordes para la decisión de las 09:30 + k minutos (usa σ del minuto anterior, como el backtest)."""
    s = lv["sigma"][k - 1]
    return max(lv["open"], lv["prev_close"]) * (1 + s), min(lv["open"], lv["prev_close"]) * (1 - s)


def vwap_now(api: Alpaca, day: date) -> float:
    op = datetime.combine(day, datetime.strptime("09:30", "%H:%M").time(), NY)
    bars = recent_bars(api, op, datetime.now(NY))
    v = sum(b["v"] for b in bars)
    return sum(b.get("vw", (b["h"] + b["l"] + b["c"]) / 3) * b["v"] for b in bars) / v if v else float("nan")


# ---------- órdenes ----------

def get_order(api: Alpaca, oid: str) -> dict | None:
    return api._req("GET", f"{api.base}/orders/{oid}")


def wait_fill(api: Alpaca, oid: str, seconds: int = 30) -> dict:
    for _ in range(seconds * 2):
        o = get_order(api, oid)
        if o and o["status"] == "filled":
            return o
        if o and o["status"] in ("canceled", "expired", "rejected"):
            return o
        time.sleep(0.5)
    raise RuntimeError(f"la orden {oid} no terminó en {seconds}s")


def bot_capital(acct: dict) -> float:
    """Capital con que se dimensiona: el virtual (si la cuenta es compartida) más el P&L acumulado, o el equity de ayer."""
    if CFG["capital_usd"] is None:
        return float(acct["last_equity"])
    return CFG["capital_usd"] + sum(float(r["pnl_usd"]) for r in journal_rows() if r["action"] == "close" and r["pnl_usd"])


def open_position(api: Alpaca, st: dict, side: int, theo: float, ctx: dict) -> None:
    day = st["day"]
    if api.position(CFG["symbol"]):
        log.warning("La cuenta ya tiene %s de otro origen; no abro para no mezclar posiciones.", CFG["symbol"])
        journal({**ctx, "action": "skip", "side": side, "theo_px": theo, "note": "posición ajena en la cuenta"})
        return
    acct = api.account()
    capital, bp = bot_capital(acct), float(acct["buying_power"])
    notional = min(capital * day["lev"], bp * 0.98)
    if side == 1:
        qty = math.floor(notional / theo * 1e4) / 1e4
    else:
        qty = math.floor(notional / theo)
        if float(acct["equity"]) < 2000:
            qty = 0
    if qty <= 0:
        why = "corto omitido: cuenta bajo US$2.000 o sin capital para 1 acción" if side == -1 else "sin poder de compra"
        log.info("Señal %s a %.2f, pero %s", "LARGO" if side == 1 else "CORTO", theo, why)
        journal({**ctx, "action": "skip", "side": side, "theo_px": theo, "note": why})
        return
    o = api.submit_order(symbol=CFG["symbol"], side="buy" if side == 1 else "sell", type="market", qty=f"{qty}",
                         time_in_force="day", client_order_id=f"nz-{day['date']}-{ctx['time'].replace(':', '')}-open")
    o = wait_fill(api, o["id"])
    if o["status"] != "filled" and float(o.get("filled_qty") or 0) == 0:
        log.warning("Orden de apertura %s: %s", o["status"], o.get("id"))
        journal({**ctx, "action": "rejected", "side": side, "qty": qty, "theo_px": theo, "note": o["status"]})
        return
    fill, q = float(o["filled_avg_price"]), float(o["filled_qty"])
    slip = side * (fill / theo - 1) * 100       # positivo = peor que el precio de la regla
    day.update(pos=side, entry_px=fill, qty=q)
    save_state(st)
    journal({**ctx, "action": "open", "side": side, "qty": q, "theo_px": theo, "fill_px": fill, "slip_pct": round(slip, 4)})
    log.info("ABRE %s %.4f a %.2f (regla %.2f, desliz. %+.4f%%)", "LARGO" if side == 1 else "CORTO", q, fill, theo, slip)


def close_position(api: Alpaca, st: dict, theo: float | None, ctx: dict, reason: str) -> None:
    day = st["day"]
    side = day.get("pos", 0)
    held = float((api.position(CFG["symbol"]) or {}).get("qty", 0))
    qty = min(day["qty"], abs(held)) if held * side > 0 else 0   # solo lo propio y solo si sigue ahí
    if qty <= 0:
        log.warning("No había posición propia que cerrar (%s).", reason)
        day.update(pos=0)
        save_state(st)
        return
    o = api.submit_order(symbol=CFG["symbol"], side="sell" if side == 1 else "buy", type="market", qty=f"{qty:g}",
                         time_in_force="day", client_order_id=f"nz-{day['date']}-{ctx['time'].replace(':', '')}-close")
    o = wait_fill(api, o["id"])
    fill = float(o["filled_avg_price"])
    pnl = side * (fill - day["entry_px"]) * day["qty"] if side else 0.0
    slip = (-side * (fill / theo - 1) * 100) if theo and side else ""
    day.update(pos=0, pnl=day.get("pnl", 0.0) + pnl)
    save_state(st)
    journal({**ctx, "action": "close", "side": side, "qty": day["qty"], "theo_px": theo or "", "fill_px": fill,
             "slip_pct": round(slip, 4) if slip != "" else "", "pnl_usd": round(pnl, 2), "note": reason})
    log.info("CIERRA (%s) a %.2f -> P&L %+.2f USD", reason, fill, pnl)


# ---------- fases ----------

def phase_check(api: Alpaca, st: dict, k: int) -> None:
    today = date.today()
    if (st.get("day") or {}).get("date") != str(today):
        st["day"] = {**day_levels(api, today), "pos": 0, "pnl": 0.0}
        lv = st["day"]
        log.info("Niveles del día: apertura %.2f, cierre de ayer %.2f, volatilidad diaria %.2f%% -> apalancamiento %.2fx",
                 lv["open"], lv["prev_close"], lv["vol"] * 100, lv["lev"])
        save_state(st)
    day = st["day"]
    ub, lb = bands(day, k)
    px, vw = last_price(api), vwap_now(api, today)
    hhmm = (datetime.combine(today, datetime.strptime("09:30", "%H:%M").time()) + timedelta(minutes=k)).strftime("%H:%M")
    ctx = {"date": str(today), "time": hhmm, "upper": round(ub, 2), "lower": round(lb, 2), "vwap": round(vw, 2)}
    log.info("%s precio %.2f | borde sup %.2f, inf %.2f, VWAP %.2f | posición %s", hhmm, px, ub, lb, vw, day["pos"])
    if day["pos"] == 1 and px < max(ub, vw):
        close_position(api, st, px, ctx, "stop móvil")
    elif day["pos"] == -1 and px > min(lb, vw):
        close_position(api, st, px, ctx, "stop móvil")
    if day["pos"] == 0:
        if px > ub:
            open_position(api, st, 1, px, ctx)
        elif px < lb and CFG["allow_short"]:
            open_position(api, st, -1, px, ctx)


def phase_flat(api: Alpaca, st: dict) -> None:
    day = st.get("day") or {}
    ctx = {"date": str(date.today()), "time": datetime.now(NY).strftime("%H:%M")}
    if day.get("date") == str(date.today()) and day.get("pos"):
        close_position(api, st, last_price(api), ctx, "fin del día")
    if CFG["capital_usd"] is None and api.position(CFG["symbol"]):  # cuenta propia: al cierre, sin posición
        log.warning("Quedaba una posición abierta al cierre; la cierro.")
        wait_fill(api, api.close_position(CFG["symbol"])["id"])


def phase_report(api: Alpaca, st: dict) -> None:
    rows = journal_rows()
    closes = [r for r in rows if r["action"] == "close"]
    today = [r for r in closes if r["date"] == str(date.today())]
    slips = [float(r["slip_pct"]) for r in rows if r.get("slip_pct") not in ("", None)]
    if not closes:
        log.info("Telemetría: aún no hay operaciones cerradas.")
        return
    pnls = [float(r["pnl_usd"]) for r in closes]
    log.info("Hoy: %d operaciones, P&L %+.2f USD. Acumulado: %d operaciones, P&L %+.2f USD, %d%% ganadoras",
             len(today), sum(float(r["pnl_usd"]) for r in today), len(closes), sum(pnls),
             round(100 * sum(p > 0 for p in pnls) / len(pnls)))
    if slips:
        log.info("Telemetría: deslizamiento medio por orden %+.4f%% -> ~%.4f%% ida y vuelta (backtest supuso 0,01%%)",
                 sum(slips) / len(slips), 2 * sum(slips) / len(slips))


def events(api: Alpaca, now: datetime) -> list[tuple[datetime, datetime, str]]:
    out = []
    for d in api.calendar(now.date().isoformat(), (now.date() + timedelta(days=10)).isoformat()):
        day = date.fromisoformat(d["date"])
        op = datetime.combine(day, datetime.strptime(d["open"], "%H:%M").time(), NY)
        cl = datetime.combine(day, datetime.strptime(d["close"], "%H:%M").time(), NY)
        k = CFG["check_every"]
        while op + timedelta(minutes=k) <= cl - timedelta(minutes=30):
            out.append((op + timedelta(minutes=k, seconds=5), cl, f"check:{k}"))
            k += CFG["check_every"]
        out.append((cl - timedelta(minutes=CFG["flat_minutes_before_close"]), cl, "flat"))
        out.append((cl + timedelta(minutes=5), cl, "report"))
    return sorted(out)


def run_event(api: Alpaca, when: datetime, phase: str) -> None:
    st = load_state()
    log.info("=== %s (%s NY) ===", phase, datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S"))
    try:
        if phase.startswith("check:"):
            if datetime.now(NY) - when > timedelta(minutes=CFG["stale_check_minutes"]):
                log.warning("Revisión %s atrasada; se omite.", phase)
            else:
                phase_check(api, st, int(phase.split(":")[1]))
        elif phase == "flat":
            phase_flat(api, st)
        else:
            phase_report(api, st)
    except Exception:  # noqa: BLE001
        log.exception("La fase %s falló", phase)
    st = load_state()
    st.setdefault("runs", {})[f"{when.date()}:{phase}"] = datetime.now(NY).isoformat(timespec="seconds")
    # conserva solo las marcas de los últimos días
    cutoff = str(date.today() - timedelta(days=7))
    st["runs"] = {k: v for k, v in st["runs"].items() if k[:10] >= cutoff}
    save_state(st)


def serve(api: Alpaca) -> None:
    log.info("Bot área de ruido iniciado: %s, revisiones cada %d min, volatilidad objetivo %.0f%%, tope %.0fx, capital %s",
             CFG["symbol"], CFG["check_every"], CFG["target_vol"] * 100, CFG["max_lev"],
             f"virtual US${CFG['capital_usd']:,.0f} (cuenta compartida)" if CFG["capital_usd"] else "toda la cuenta")
    announced = None
    while True:
        try:
            now = datetime.now(NY)
            done = load_state().get("runs", {})
            pending = [(t, cl, p) for t, cl, p in events(api, now)
                       if f"{t.date()}:{p}" not in done and now < cl + timedelta(minutes=30)]
            if pending and pending[0][0] <= now:
                run_event(api, pending[0][0], pending[0][2])
                continue
            if pending and announced != pending[0][0]:
                log.info("Próxima fase: %s el %s NY", pending[0][2], pending[0][0].strftime("%Y-%m-%d %H:%M"))
                announced = pending[0][0]
            wait = (pending[0][0] - now).total_seconds() if pending else 300
            time.sleep(max(1, min(60, wait)))
        except Exception:  # noqa: BLE001
            log.exception("Error en el ciclo; reintento en 10 s")
            time.sleep(10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    lv = sub.add_parser("levels", help="calcula los niveles de un día (sirve para validar contra el backtest)")
    lv.add_argument("--date", default=str(date.today()))
    sub.add_parser("serve")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(DATA / "noise.log")])
    load_env(ROOT / ".env")
    if not os.environ.get("APCA_API_KEY_ID"):
        while True:
            log.error("Faltan las variables APCA_*; esperando...")
            time.sleep(600)
    api = Alpaca()
    if not api.is_paper:
        sys.exit("Este bot solo opera contra cuentas paper.")
    if args.cmd == "status":
        a = api.account()
        print(f"PAPER {a['account_number']} equity=${float(a['equity']):,.2f} | día: "
              f"{ {k: v for k, v in (load_state().get('day') or {}).items() if k != 'sigma'} }")
        phase_report(api, load_state())
    elif args.cmd == "levels":
        d = date.fromisoformat(args.date)
        lv = day_levels(api, d)
        for k in (30, 60, 120, 240, 360):
            ub, lb = bands(lv, k)
            print(f"{d} +{k:3d}min  borde sup {ub:.3f}  inf {lb:.3f}")
        print(f"apertura {lv['open']} cierre ayer {lv['prev_close']} vol {lv['vol']*100:.3f}% lev {lv['lev']:.2f}x")
    else:
        serve(api)


if __name__ == "__main__":
    main()
