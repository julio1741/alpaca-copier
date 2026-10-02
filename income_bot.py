#!/usr/bin/env python3
"""Bot de ingreso diario: ventaja overnight de SPY con filtro de reversión (IBS).

Regla (validada en research/daily_income_backtest.py, 2016-2023 entrenamiento / 2024-2026 validación):
  - 3 min antes del cierre (12 en modo subasta): IBS = (precio - mínimo) / (máximo - mínimo) del día.
    Si IBS < 0.5 (cerró en la mitad baja de su rango) -> comprar SPY en la subasta de cierre (orden "cls").
  - Tamaño: exposición = min(1, objetivo_std / std de los últimos 20 retornos overnight) x freno.
    Mantiene la pérdida/ganancia diaria típica cerca del objetivo (0,3% del capital ≈ $300 en $100k).
  - 09:31: vender todo (modo "auction": 09:15 en la subasta de apertura "opg").
  - 09:45: conciliar fills, anotar P&L real en la bitácora y actualizar el aprendizaje:
      * métricas móviles 20/60 días (US$/día, % días positivos) vs lo esperado por el backtest;
      * freno: si los últimos 60 días operados son negativos, la exposición se reduce a la mitad
        hasta que vuelvan a ser positivos (control de régimen, no ajuste de parámetros).

Uso:
  income_bot.py signal           muestra la señal de hoy sin operar
  income_bot.py run --phase entry|exit|report [--dry-run]
  income_bot.py status
  income_bot.py serve            proceso permanente (Railway)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import statistics
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from alpaca_client import Alpaca, load_env

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", ROOT / "income_data"))
STATE = DATA / "state.json"
JOURNAL = DATA / "journal.csv"
NY = ZoneInfo("America/New_York")
log = logging.getLogger("income")

CFG = {
    "symbol": "SPY",
    "ibs_threshold": 0.5,
    "target_daily_std": 0.003,   # fracción del equity
    "vol_lookback": 20,
    "max_exposure": 1.0,
    "brake_lookback_trades": 60,
    "brake_factor": 0.5,
    # "fractional": órdenes de mercado por monto (sirve con cualquier capital; entra 3 min antes del cierre
    #               y sale 1 min después de la apertura). "auction": acciones enteras en las subastas (cls/opg).
    "order_mode": "fractional",
    "entry_minutes_before_close": 3,
    "exit_time": "09:31",
    "report_time": "09:45",
    "expected_usd_per_active_day_pct": 0.00045,  # backtest validación: ~$45 por día operado en $100k (vol-target)
}
CFG.update(json.loads((ROOT / "income_config.json").read_text())) if (ROOT / "income_config.json").exists() else None

JOURNAL_FIELDS = ["date", "mode", "ibs", "ibs_close", "vol20", "exposure", "brake", "equity", "qty", "buy_order",
                  "buy_px", "sell_order", "sell_px", "pnl_usd", "status", "note"]


# ---------- estado / bitácora ----------

def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"open": None, "brake": 1.0, "runs": {}}


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


def journal_write(rows: list[dict]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in JOURNAL_FIELDS})


def journal_upsert(row: dict) -> None:
    rows = journal_rows()
    for i, r in enumerate(rows):
        if r["date"] == row["date"]:
            rows[i] = {**r, **row}
            break
    else:
        rows.append(row)
    journal_write(rows)


# ---------- datos ----------

def today_bar(api: Alpaca, sym: str) -> dict:
    d = api._req("GET", f"https://data.alpaca.markets/v2/stocks/{sym}/snapshot", params={"feed": "iex"})
    return d["dailyBar"]


def recent_daily(api: Alpaca, sym: str, n: int) -> list[dict]:
    start = (date.today() - timedelta(days=n * 2 + 10)).isoformat()
    end = (date.today() - timedelta(days=1)).isoformat()
    d = api._req("GET", "https://data.alpaca.markets/v2/stocks/bars",
                 params={"symbols": sym, "timeframe": "1Day", "start": start, "end": end, "feed": "sip",
                         "adjustment": "all", "limit": 1000})
    return d["bars"].get(sym, [])


def overnight_vol(bars: list[dict], lookback: int) -> float:
    rets = [bars[i]["o"] / bars[i - 1]["c"] - 1 for i in range(1, len(bars))][-lookback:]
    return statistics.pstdev(rets) if len(rets) >= 5 else float("nan")


def compute_signal(api: Alpaca, st: dict) -> dict:
    sym = CFG["symbol"]
    bar = today_bar(api, sym)
    rng = bar["h"] - bar["l"]
    ibs = (bar["c"] - bar["l"]) / rng if rng > 0 else 0.5
    vol = overnight_vol(recent_daily(api, sym, CFG["vol_lookback"]), CFG["vol_lookback"])
    raw = min(CFG["max_exposure"], CFG["target_daily_std"] / vol) if vol and vol > 0 else CFG["max_exposure"]
    exposure = raw * st.get("brake", 1.0)
    return {"ibs": round(ibs, 4), "price": bar["c"], "high": bar["h"], "low": bar["l"], "vol20": round(vol, 5),
            "exposure": round(exposure, 3), "brake": st.get("brake", 1.0), "signal": ibs < CFG["ibs_threshold"]}


# ---------- fases ----------

def phase_entry(api: Alpaca, st: dict, dry_run: bool) -> None:
    today = str(date.today())
    if st.get("open"):
        log.warning("Ya hay una posición abierta (%s); no se compra de nuevo", st["open"])
        return
    sig = compute_signal(api, st)
    acct = api.account()
    equity = float(acct["equity"])
    row = {"date": today, "mode": "dry-run" if dry_run else "paper", "ibs": sig["ibs"], "vol20": sig["vol20"],
           "exposure": sig["exposure"], "brake": sig["brake"], "equity": round(equity, 2)}
    if not sig["signal"]:
        log.info("Sin señal: IBS %.2f >= %.2f (precio %.2f, rango %.2f-%.2f). No se opera hoy.",
                 sig["ibs"], CFG["ibs_threshold"], sig["price"], sig["low"], sig["high"])
        journal_upsert({**row, "qty": 0, "status": "no-signal"})
        return
    budget = math.floor(min(equity * sig["exposure"], float(acct["cash"])) * 100) / 100
    fractional = CFG["order_mode"] == "fractional"
    qty = round(budget / sig["price"], 4) if fractional else int(budget // sig["price"])
    if (fractional and budget < 1) or (not fractional and qty < 1):
        log.warning("Señal pero sin plata suficiente (presupuesto $%.2f)", budget)
        journal_upsert({**row, "qty": 0, "status": "skipped", "note": "sin plata"})
        return
    log.info("SEÑAL IBS %.2f < %.2f -> comprar %s %s por ~$%.2f (%s, exposición %.0f%%, vol20 %.2f%%)",
             sig["ibs"], CFG["ibs_threshold"], qty, CFG["symbol"], budget if fractional else qty * sig["price"],
             "orden de mercado" if fractional else "subasta de cierre", sig["exposure"] * 100, sig["vol20"] * 100)
    if dry_run:
        journal_upsert({**row, "qty": qty, "status": "dry-run"})
        return
    if fractional:
        o = api.submit_order(symbol=CFG["symbol"], side="buy", type="market", time_in_force="day",
                             notional=f"{budget:.2f}", client_order_id=f"inc-buy-{today}")
    else:
        o = api.submit_order(symbol=CFG["symbol"], side="buy", type="market", time_in_force="cls", qty=str(qty),
                             client_order_id=f"inc-buy-{today}")
    st["open"] = {"date": today, "qty": qty, "buy_order": o["id"]}
    save_state(st)
    journal_upsert({**row, "qty": qty, "buy_order": o["id"], "status": "entered"})


def phase_exit(api: Alpaca, st: dict, dry_run: bool) -> None:
    pos = st.get("open")
    if not pos:
        log.info("Sin posición que cerrar.")
        return
    if pos.get("sell_order"):
        log.info("La venta ya fue enviada (%s).", pos["sell_order"])
        return
    live = api.position(CFG["symbol"])
    qty = float(live["qty"]) if live else 0.0
    if qty <= 0:
        log.warning("La compra de ayer no se llenó; nada que vender.")
        journal_upsert({"date": pos["date"], "status": "not-filled", "pnl_usd": 0})
        st["open"] = None
        save_state(st)
        return
    fractional = CFG["order_mode"] == "fractional"
    log.info("Vender %s %s %s", qty, CFG["symbol"], "a mercado tras la apertura" if fractional else "en la subasta de apertura")
    if dry_run:
        return
    if fractional:
        o = api.close_position(CFG["symbol"]) or {}
    else:
        o = api.submit_order(symbol=CFG["symbol"], side="sell", type="market", time_in_force="opg", qty=str(int(qty)),
                             client_order_id=f"inc-sell-{pos['date']}")
    pos["sell_order"] = o["id"]
    save_state(st)
    journal_upsert({"date": pos["date"], "sell_order": o["id"], "status": "exiting"})


def _order(api: Alpaca, oid: str) -> dict | None:
    return api._req("GET", f"{api.base}/orders/{oid}")


def phase_report(api: Alpaca, st: dict, dry_run: bool) -> None:
    pos = st.get("open")
    if pos and pos.get("sell_order"):
        b, s = _order(api, pos["buy_order"]), _order(api, pos["sell_order"])
        if s and s.get("status") == "filled" and b and b.get("status") == "filled":
            qty = float(s["filled_qty"])
            bp, sp = float(b["filled_avg_price"]), float(s["filled_avg_price"])
            pnl = round((sp - bp) * qty, 2)
            # IBS real al cierre (con la barra SIP de ayer) para medir cuánto se desvía la señal de las 15:48
            bars = recent_daily(api, CFG["symbol"], 3)
            ibs_close = ""
            if bars and bars[-1]["t"][:10] == pos["date"]:
                y = bars[-1]
                ibs_close = round((y["c"] - y["l"]) / (y["h"] - y["l"]), 4) if y["h"] > y["l"] else 0.5
            journal_upsert({"date": pos["date"], "buy_px": bp, "sell_px": sp, "pnl_usd": pnl, "ibs_close": ibs_close,
                            "status": "closed"})
            log.info("RESULTADO %s: compra %.2f -> venta %.2f x %s = %+.2f USD", pos["date"], bp, sp, qty, pnl)
            st["open"] = None
        elif s and s.get("status") in ("canceled", "expired", "rejected"):
            log.error("La venta %s quedó %s; se reintenta en la próxima fase exit", pos["sell_order"], s["status"])
            pos.pop("sell_order", None)
        else:
            log.warning("Órdenes aún no conciliadas (compra %s / venta %s)", b and b.get("status"), s and s.get("status"))
    learn(st)
    if not dry_run:
        save_state(st)


def learn(st: dict) -> None:
    """Actualiza métricas móviles y el freno. Es la parte que 'aprende' día a día."""
    closed = [r for r in journal_rows() if r.get("status") == "closed" and r.get("pnl_usd")]
    if not closed:
        log.info("Aprendizaje: aún no hay operaciones cerradas.")
        return
    pnls = [float(r["pnl_usd"]) for r in closed]
    eq = float(closed[-1]["equity"] or 100000)
    n = CFG["brake_lookback_trades"]
    last = pnls[-n:]
    brake_before = st.get("brake", 1.0)
    st["brake"] = CFG["brake_factor"] if len(last) >= 20 and sum(last) < 0 else 1.0
    exp_per_trade = eq * CFG["expected_usd_per_active_day_pct"]
    for k in (20, 60):
        w = pnls[-k:]
        log.info("Aprendizaje %2d op: %+.0f USD total, %+.1f USD/op (esperado ~%+.0f), %d%% positivas",
                 len(w), sum(w), sum(w) / len(w), exp_per_trade, round(100 * sum(p > 0 for p in w) / len(w)))
    all_days = journal_rows()
    first = all_days[0]["date"]
    cal_days = max(1, (date.today() - date.fromisoformat(first)).days * 252 / 365)
    log.info("Acumulado desde %s: %+.0f USD en %d operaciones = %+.1f USD por día de bolsa (meta: +10)",
             first, sum(pnls), len(pnls), sum(pnls) / cal_days)
    if st["brake"] != brake_before:
        log.warning("FRENO %s: últimos %d resultados suman %+.0f USD -> exposición x%.1f",
                    "ACTIVADO" if st["brake"] < 1 else "LIBERADO", len(last), sum(last), st["brake"])


# ---------- programador ----------

def sessions(api: Alpaca, start: date) -> list[dict]:
    return api.calendar(start.isoformat(), (start + timedelta(days=10)).isoformat())


def events(api: Alpaca, now: datetime) -> list[tuple[datetime, str]]:
    out = []
    for d in sessions(api, now.date()):
        day = date.fromisoformat(d["date"])
        close = datetime.combine(day, datetime.strptime(d["close"], "%H:%M").time(), NY)
        out.append((datetime.combine(day, datetime.strptime(CFG["exit_time"], "%H:%M").time(), NY), "exit"))
        out.append((datetime.combine(day, datetime.strptime(CFG["report_time"], "%H:%M").time(), NY), "report"))
        out.append((close - timedelta(minutes=CFG["entry_minutes_before_close"]), "entry"))
    return sorted(out)


PHASES = {"entry": phase_entry, "exit": phase_exit, "report": phase_report}


def run_phase(api: Alpaca, phase: str, dry_run: bool) -> None:
    st = load_state()
    log.info("=== fase %s (%s) ===", phase, datetime.now(NY).strftime("%Y-%m-%d %H:%M NY"))
    try:
        PHASES[phase](api, st, dry_run)
    except Exception:  # noqa: BLE001
        log.exception("La fase %s falló", phase)
    if not dry_run:
        st = load_state() if phase == "report" else st
        st.setdefault("runs", {})[f"{date.today()}:{phase}"] = datetime.now(NY).isoformat(timespec="seconds")
        save_state(st)


def serve(api: Alpaca) -> None:
    log.info("Bot de ingreso diario iniciado: %s, IBS<%.2f, objetivo std %.2f%%/día", CFG["symbol"],
             CFG["ibs_threshold"], CFG["target_daily_std"] * 100)
    announced = None
    while True:
        try:
            now = datetime.now(NY)
            done = load_state().get("runs", {})
            # una fase se considera perdida si pasaron más de 20 min (las subastas no esperan)
            pending = [(t, p) for t, p in events(api, now) if f"{t.date()}:{p}" not in done
                       and now < t + timedelta(minutes=20)]
            if not pending:
                time.sleep(300)
                continue
            when, phase = pending[0]
            if when <= now:
                run_phase(api, phase, dry_run=False)
                continue
            if announced != (when, phase):
                log.info("Próxima fase: %s el %s NY", phase, when.strftime("%Y-%m-%d %H:%M"))
                announced = (when, phase)
            time.sleep(min(300, (when - now).total_seconds()))
        except Exception:  # noqa: BLE001
            log.exception("Error en el ciclo; reintento en 60 s")
            time.sleep(60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("signal")
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=list(PHASES), required=True)
    r.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    sub.add_parser("serve")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(DATA / "income.log")])
    load_env(ROOT / ".env")
    if not os.environ.get("APCA_API_KEY_ID"):
        while True:
            log.error("Faltan las variables APCA_API_KEY_ID / APCA_API_SECRET_KEY / APCA_API_BASE_URL; esperando...")
            time.sleep(600)
    api = Alpaca()
    if not api.is_paper:
        sys.exit("Este bot solo opera contra la cuenta paper.")
    if args.cmd == "signal":
        print(json.dumps(compute_signal(api, load_state()), indent=2))
    elif args.cmd == "run":
        run_phase(api, args.phase, args.dry_run)
    elif args.cmd == "status":
        a = api.account()
        print(f"PAPER equity=${float(a['equity']):,.2f} cash=${float(a['cash']):,.2f} | estado: {load_state()}")
        learn(load_state())
    elif args.cmd == "serve":
        serve(api)


if __name__ == "__main__":
    main()
