#!/usr/bin/env python3
"""Bot 3: ruptura del rango de apertura (ORB) en QQQ.

Regla (validada en research/orb_backtest.py: elegida 2019-2022, validada 2023-2026):
  - Rango de apertura = máximo y mínimo de QQQ entre las 09:30 y las 10:00 NY (velas de 1 min, feed IEX).
  - 10:00: orden de compra STOP en el máximo del rango (fraccionaria, válida el día).
  - Al llenarse: stop de protección en el mínimo del rango (queda puesto en Alpaca aunque el bot se caiga)
    y objetivo = entrada + (entrada - mínimo), es decir 1 a 1. El objetivo lo vigila el bot: si el precio lo
    toca, cancela el stop y vende a mercado.
  - Solo compras, una operación por día, todo cerrado 2 minutos antes del cierre.
  - Tamaño: arriesga RISK del equity hasta el poder de compra de la cuenta (sin margen = 1x).
Rango: SIP (mercado consolidado, como el backtest) hasta donde lo permite el plan gratuito (datos de más de 15 min)
y IEX para el resto; a las 10:15 se recalcula entero con SIP y se corrigen las órdenes si el rango era más ancho.
Salidas: siempre cerrando la posición real de la cuenta (cuenta dedicada a este bot), nunca una cantidad supuesta;
así una ejecución parcial (el paper las simula al azar) no deja acciones abiertas de un día para otro.
Telemetría (lo que decide si la estrategia aguanta): deslizamiento de entrada y salida contra los precios
teóricos, P&L por operación, y su comparación con el presupuesto de costos del backtest (0,01% ida y vuelta).

Uso: orb_bot.py status | run --phase setup|refine|flat|report [--dry-run] | serve
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
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from alpaca_client import Alpaca, load_env

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", ROOT / "orb_data"))
STATE = DATA / "state.json"
JOURNAL = DATA / "journal.csv"
NY = ZoneInfo("America/New_York")
DATA_URL = "https://data.alpaca.markets/v2"
log = logging.getLogger("orb")

CFG = {
    "symbol": "QQQ",
    "or_minutes": 30,
    "target_r": 1.0,
    "risk": 0.01,
    "max_lev": 4.0,          # tope teórico; la cuenta real limita con su poder de compra
    "cost_budget_rt": 0.0001,  # costo ida y vuelta supuesto en el backtest
    "poll_seconds": 3,
    "flat_minutes_before_close": 2,
}

FIELDS = ["date", "mode", "or_high", "or_low", "range_pct", "equity", "qty", "entry_order", "entry_px", "entry_slip_pct",
          "stop_order", "target_px", "exit_px", "exit_reason", "exit_slip_pct", "pnl_usd", "r_mult", "status", "note"]


# ---------- estado y bitácora ----------

def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"day": None, "runs": {}}


def save_state(st: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, default=str))
    tmp.replace(STATE)


def journal_rows() -> list[dict]:
    if not JOURNAL.exists():
        return []
    with JOURNAL.open() as f:
        return list(csv.DictReader(f))


def journal_upsert(row: dict) -> None:
    rows = journal_rows()
    for i, r in enumerate(rows):
        if r["date"] == row["date"]:
            rows[i] = {**r, **{k: v for k, v in row.items() if v is not None}}
            break
    else:
        rows.append(row)
    DATA.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


# ---------- mercado ----------

def fetch_bars(api: Alpaca, feed: str, start: datetime, end: datetime) -> list[dict]:
    if end <= start:
        return []
    d = api._req("GET", f"{DATA_URL}/stocks/bars", params={
        "symbols": CFG["symbol"], "timeframe": "1Min", "feed": feed, "limit": 1000,
        "start": start.isoformat(), "end": (end - timedelta(seconds=1)).isoformat()})
    return d["bars"].get(CFG["symbol"], [])


def opening_range(api: Alpaca, day: date) -> tuple[float, float, str]:
    """Máximo y mínimo 09:30-10:00. SIP (lo que usó el backtest) para lo que tenga más de 15 min; IEX para el resto.
    IEX es una parte del mercado y ve el rango ~0,01-0,02% más estrecho, del orden del costo que decide la estrategia."""
    start = datetime.combine(day, datetime.strptime("09:30", "%H:%M").time(), NY)
    end = start + timedelta(minutes=CFG["or_minutes"])
    sip_end = min(end, datetime.now(NY) - timedelta(minutes=16))
    sip = fetch_bars(api, "sip", start, sip_end)
    iex = fetch_bars(api, "iex", start, end) if sip_end < end else []
    bars = sip + iex
    if not bars:
        raise RuntimeError("sin velas del rango de apertura")
    return max(b["h"] for b in bars), min(b["l"] for b in bars), f"{len(sip)} velas SIP + {len(iex)} IEX"


def last_price(api: Alpaca) -> float:
    return float(api._req("GET", f"{DATA_URL}/stocks/{CFG['symbol']}/trades/latest", params={"feed": "iex"})["trade"]["p"])


def get_order(api: Alpaca, oid: str) -> dict | None:
    return api._req("GET", f"{api.base}/orders/{oid}") if oid else None


def cancel_order(api: Alpaca, oid: str) -> dict | None:
    """Cancela y espera el estado final: la cancelación es asíncrona y la orden aún puede ejecutarse mientras tanto.
    Devuelve la orden final para que quien llama mire filled_qty antes de mandar otra orden."""
    try:
        api._req("DELETE", f"{api.base}/orders/{oid}")
    except RuntimeError as e:
        log.warning("No se pudo cancelar %s: %s", oid, e)
    for _ in range(10):
        o = get_order(api, oid)
        if not o or o["status"] in ("canceled", "filled", "expired", "rejected"):
            return o
        time.sleep(0.5)
    return get_order(api, oid)


# ---------- fases ----------

def phase_setup(api: Alpaca, st: dict, dry_run: bool) -> None:
    today = date.today()
    if st.get("day", {}) and st["day"].get("date") == str(today):
        log.info("El día ya está preparado.")
        return
    hi, lo, src = opening_range(api, today)
    rng = hi - lo
    acct = api.account()
    equity, bp = float(acct["equity"]), float(acct["buying_power"])
    row = {"date": str(today), "mode": "dry-run" if dry_run else "paper", "or_high": hi, "or_low": lo,
           "range_pct": round(rng / hi * 100, 3), "equity": round(equity, 2)}
    if rng <= 0:
        journal_upsert({**row, "status": "skipped", "note": "rango nulo"})
        return
    notional = min(equity * CFG["risk"] / (rng / hi), equity * CFG["max_lev"], bp * 0.98)
    qty = math.floor(notional / hi * 1e4) / 1e4
    log.info("Rango de apertura %s (%s): máximo %.2f, mínimo %.2f (%.2f%%). Compra STOP %.4f acc en %.2f (~$%.0f)",
             CFG["symbol"], src, hi, lo, rng / hi * 100, qty, hi, qty * hi)
    if qty <= 0:
        journal_upsert({**row, "status": "skipped", "note": "sin poder de compra"})
        return
    if dry_run:
        journal_upsert({**row, "qty": qty, "status": "dry-run"})
        return
    o = api.submit_order(symbol=CFG["symbol"], side="buy", type="stop", stop_price=f"{hi:.2f}", qty=f"{qty}",
                         time_in_force="day", client_order_id=f"orb-entry-{today}")
    st["day"] = {"date": str(today), "or_high": hi, "or_low": lo, "qty": qty, "entry_order": o["id"],
                 "stage": "waiting_breakout"}
    save_state(st)
    journal_upsert({**row, "qty": qty, "entry_order": o["id"], "status": "waiting_breakout"})


def manage_tick(api: Alpaca, st: dict) -> None:
    """Se llama cada pocos segundos durante la sesión. Pasa por: waiting_breakout -> in_trade -> done."""
    day = st.get("day") or {}
    if day.get("date") != str(date.today()) or day.get("stage") in (None, "done"):
        return
    if day["stage"] == "waiting_breakout":
        o = get_order(api, day["entry_order"])
        if not o or o["status"] != "filled":
            return
        entry = float(o["filled_avg_price"])
        qty = float(o["filled_qty"])
        target = entry + CFG["target_r"] * (entry - day["or_low"])
        slip = (entry / day["or_high"] - 1) * 100
        stop = api.submit_order(symbol=CFG["symbol"], side="sell", type="stop", stop_price=f"{day['or_low']:.2f}",
                                qty=f"{qty}", time_in_force="day", client_order_id=f"orb-stop-{day['date']}")
        day.update(stage="in_trade", entry_px=entry, filled_qty=qty, target_px=target, stop_order=stop["id"])
        save_state(st)
        journal_upsert({"date": day["date"], "entry_px": entry, "entry_slip_pct": round(slip, 4), "stop_order": stop["id"],
                        "target_px": round(target, 2), "status": "in_trade"})
        log.info("ENTRADA llenada a %.2f (ruptura en %.2f, deslizamiento %+.3f%%). Stop %.2f, objetivo %.2f",
                 entry, day["or_high"], slip, day["or_low"], target)
        return
    if day["stage"] == "in_trade":
        so = get_order(api, day["stop_order"])
        if so and so["status"] == "filled":
            finish(api, st, float(so["filled_avg_price"]), "stop", day["or_low"])
            return
        px = last_price(api)
        if px >= day["target_px"]:
            log.info("OBJETIVO tocado (%.2f >= %.2f): cancelo el stop y vendo a mercado", px, day["target_px"])
            so = cancel_order(api, day["stop_order"])
            if so and so["status"] == "filled":  # el stop se llenó justo antes de cancelarlo
                finish(api, st, float(so["filled_avg_price"]), "stop", day["or_low"])
                return
            close_position(api, st, "objetivo", day["target_px"])


def wait_fill(api: Alpaca, oid: str, seconds: int = 30) -> float:
    for _ in range(seconds):
        o = get_order(api, oid)
        if o and o["status"] == "filled":
            return float(o["filled_avg_price"])
        time.sleep(1)
    raise RuntimeError(f"la orden {oid} no se llenó en {seconds}s")


def close_position(api: Alpaca, st: dict, reason: str, theo_px: float | None) -> None:
    """Vende a mercado toda la posición que la cuenta tenga de verdad (no la cantidad que el bot cree tener)."""
    if not api.position(CFG["symbol"]):
        log.warning("No hay posición abierta; nada que vender (%s).", reason)
        st["day"]["stage"] = "done"
        save_state(st)
        return
    o = api.close_position(CFG["symbol"])
    finish(api, st, wait_fill(api, o["id"]), reason, theo_px)


def refine_range(api: Alpaca, st: dict, dry_run: bool) -> None:
    """10:15: el rango completo ya está disponible en SIP. Si es más ancho que el usado a las 10:00, corrige la orden
    de entrada (máximo) o el stop de protección y el objetivo (mínimo)."""
    day = st.get("day") or {}
    if day.get("date") != str(date.today()) or day.get("stage") not in ("waiting_breakout", "in_trade"):
        return
    start = datetime.combine(date.today(), datetime.strptime("09:30", "%H:%M").time(), NY)
    bars = fetch_bars(api, "sip", start, start + timedelta(minutes=CFG["or_minutes"]))
    if not bars:
        log.warning("Sin velas SIP para recalcular el rango.")
        return
    hi, lo = max(b["h"] for b in bars), min(b["l"] for b in bars)
    log.info("Rango SIP completo: máximo %.2f (usado %.2f), mínimo %.2f (usado %.2f)", hi, day["or_high"], lo, day["or_low"])
    if dry_run:
        return
    if day["stage"] == "waiting_breakout" and round(hi, 2) > round(day["or_high"], 2):
        o = cancel_order(api, day["entry_order"])
        if o and float(o.get("filled_qty") or 0) > 0:
            log.info("La entrada ya se había ejecutado; sigo con el rango anterior.")
            return
        new = api.submit_order(symbol=CFG["symbol"], side="buy", type="stop", stop_price=f"{hi:.2f}", qty=f"{day['qty']}",
                               time_in_force="day", client_order_id=f"orb-entry-{day['date']}-sip")
        day.update(or_high=hi, entry_order=new["id"])
        journal_upsert({"date": day["date"], "or_high": hi, "entry_order": new["id"], "note": "rango corregido con SIP"})
        log.info("Compra STOP movida a %.2f", hi)
    if round(lo, 2) < round(day["or_low"], 2):
        day["or_low"] = lo
        if day["stage"] == "in_trade" and day.get("stop_order"):
            so = cancel_order(api, day["stop_order"])
            if so and so["status"] == "filled":
                finish(api, st, float(so["filled_avg_price"]), "stop", float(so["stop_price"]))
                return
            pos = api.position(CFG["symbol"])
            if not pos:
                return
            new = api.submit_order(symbol=CFG["symbol"], side="sell", type="stop", stop_price=f"{lo:.2f}",
                                   qty=pos["qty"], time_in_force="day", client_order_id=f"orb-stop-{day['date']}-sip")
            day.update(stop_order=new["id"], target_px=day["entry_px"] + CFG["target_r"] * (day["entry_px"] - lo))
            journal_upsert({"date": day["date"], "stop_order": new["id"], "target_px": round(day["target_px"], 2),
                            "note": "rango corregido con SIP"})
            log.info("Stop movido a %.2f; objetivo %.2f", lo, day["target_px"])
        journal_upsert({"date": day["date"], "or_low": lo})
    save_state(st)


def finish(api: Alpaca, st: dict, exit_px: float, reason: str, theo_px: float | None) -> None:
    day = st["day"]
    pnl = (exit_px - day["entry_px"]) * day["filled_qty"]
    r = (exit_px - day["entry_px"]) / (day["entry_px"] - day["or_low"]) if day["entry_px"] > day["or_low"] else 0
    slip = (exit_px / theo_px - 1) * 100 if theo_px else ""
    day["stage"] = "done"
    save_state(st)
    journal_upsert({"date": day["date"], "exit_px": exit_px, "exit_reason": reason,
                    "exit_slip_pct": round(slip, 4) if slip != "" else "", "pnl_usd": round(pnl, 2),
                    "r_mult": round(r, 3), "status": "closed"})
    log.info("SALIDA por %s a %.2f -> P&L %+.2f USD (%+.2f R)", reason, exit_px, pnl, r)


def phase_flat(api: Alpaca, st: dict, dry_run: bool) -> None:
    day = st.get("day") or {}
    if dry_run:
        return
    if day.get("date") == str(date.today()):
        flatten_day(api, st, day)
    if api.position(CFG["symbol"]):  # red de seguridad: la cuenta es del bot y al cierre debe quedar sin posición
        log.warning("Quedaba una posición abierta al cierre; la cierro.")
        wait_fill(api, api.close_position(CFG["symbol"])["id"])


def flatten_day(api: Alpaca, st: dict, day: dict) -> None:
    if day.get("stage") == "waiting_breakout":
        o = cancel_order(api, day["entry_order"])
        if o and float(o.get("filled_qty") or 0) > 0:  # se llenó (o en parte) en el último momento: se cierra abajo
            day["stage"] = "in_trade"
            day.update(entry_px=float(o["filled_avg_price"]), filled_qty=float(o["filled_qty"]), stop_order="")
        else:
            day["stage"] = "done"
            save_state(st)
            journal_upsert({"date": day["date"], "status": "no-breakout"})
            log.info("Sin ruptura hoy: orden de entrada cancelada.")
            return
    if day.get("stage") == "in_trade":
        if day.get("stop_order"):
            so = cancel_order(api, day["stop_order"])
            if so and so["status"] == "filled":  # el stop se ejecutó mientras se cancelaba
                finish(api, st, float(so["filled_avg_price"]), "stop", day["or_low"])
                return
        close_position(api, st, "fin del día", None)


def phase_report(api: Alpaca, st: dict, dry_run: bool) -> None:
    rows = [r for r in journal_rows() if r.get("status") == "closed"]
    if not rows:
        log.info("Telemetría: aún no hay operaciones cerradas.")
        return
    pnls = [float(r["pnl_usd"]) for r in rows]
    eslip = [float(r["entry_slip_pct"]) for r in rows if r.get("entry_slip_pct")]
    xslip = [float(r["exit_slip_pct"]) for r in rows if r.get("exit_slip_pct")]
    avg_cost = (sum(eslip) / len(eslip) if eslip else 0) - (sum(xslip) / len(xslip) if xslip else 0)
    log.info("Telemetría %d operaciones: P&L %+.2f USD (%+.2f por operación), %d%% ganadoras, R medio %+.2f",
             len(rows), sum(pnls), sum(pnls) / len(pnls), round(100 * sum(p > 0 for p in pnls) / len(pnls)),
             sum(float(r["r_mult"]) for r in rows) / len(rows))
    log.info("Deslizamiento medio: entrada %+.4f%%, salida %+.4f%% -> costo efectivo ~%.4f%% ida y vuelta "
             "(el backtest supuso %.4f%%; con 0,03%% la ventaja casi desaparece)",
             sum(eslip) / len(eslip) if eslip else 0, sum(xslip) / len(xslip) if xslip else 0, avg_cost,
             CFG["cost_budget_rt"] * 100)


PHASES = {"setup": phase_setup, "refine": refine_range, "flat": phase_flat, "report": phase_report}


def events(api: Alpaca, now: datetime) -> list[tuple[datetime, datetime, str]]:
    """(cuándo, cierre de la sesión, fase)."""
    out = []
    for d in api.calendar(now.date().isoformat(), (now.date() + timedelta(days=10)).isoformat()):
        day = date.fromisoformat(d["date"])
        op = datetime.combine(day, datetime.strptime(d["open"], "%H:%M").time(), NY)
        cl = datetime.combine(day, datetime.strptime(d["close"], "%H:%M").time(), NY)
        out.append((op + timedelta(minutes=CFG["or_minutes"], seconds=20), cl, "setup"))
        out.append((op + timedelta(minutes=CFG["or_minutes"] + 16), cl, "refine"))
        out.append((cl - timedelta(minutes=CFG["flat_minutes_before_close"]), cl, "flat"))
        out.append((cl + timedelta(minutes=5), cl, "report"))
    return sorted(out)


def run_phase(api: Alpaca, phase: str, dry_run: bool) -> None:
    st = load_state()
    log.info("=== fase %s (%s NY) ===", phase, datetime.now(NY).strftime("%Y-%m-%d %H:%M"))
    try:
        PHASES[phase](api, st, dry_run)
    except Exception:  # noqa: BLE001
        log.exception("La fase %s falló", phase)
    if not dry_run:
        st = load_state()
        st.setdefault("runs", {})[f"{date.today()}:{phase}"] = datetime.now(NY).isoformat(timespec="seconds")
        save_state(st)


def serve(api: Alpaca) -> None:
    log.info("Bot ORB iniciado: %s, rango %d min, objetivo %.1fR, riesgo %.1f%%", CFG["symbol"], CFG["or_minutes"],
             CFG["target_r"], CFG["risk"] * 100)
    announced = None
    while True:
        try:
            now = datetime.now(NY)
            st = load_state()
            done = st.get("runs", {})
            pending = [(t, cl, p) for t, cl, p in events(api, now) if f"{t.date()}:{p}" not in done and now < cl + timedelta(minutes=30)]
            if pending and pending[0][0] <= now:
                run_phase(api, pending[0][2], dry_run=False)
                continue
            if (st.get("day") or {}).get("stage") in ("waiting_breakout", "in_trade"):
                manage_tick(api, st)
                time.sleep(CFG["poll_seconds"])
                continue
            if pending and announced != pending[0][:1]:
                log.info("Próxima fase: %s el %s NY", pending[0][2], pending[0][0].strftime("%Y-%m-%d %H:%M"))
                announced = pending[0][:1]
            wait = (pending[0][0] - now).total_seconds() if pending else 300
            time.sleep(max(1, min(60, wait)))
        except Exception:  # noqa: BLE001
            log.exception("Error en el ciclo; reintento en 10 s")
            time.sleep(10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=list(PHASES), required=True)
    r.add_argument("--dry-run", action="store_true")
    sub.add_parser("serve")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(DATA / "orb.log")])
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
        print(f"PAPER {a['account_number']} equity=${float(a['equity']):,.2f} | día: {load_state().get('day')}")
        phase_report(api, load_state(), True)
    elif args.cmd == "run":
        run_phase(api, args.phase, args.dry_run)
    else:
        serve(api)


if __name__ == "__main__":
    main()
