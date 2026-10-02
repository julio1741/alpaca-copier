#!/usr/bin/env python3
"""Copia en Alpaca las operaciones que declaran congresistas (PTR del Clerk de la Cámara).

Uso:
  copier.py init                         marca como vistas las declaraciones existentes (no opera)
  copier.py run --phase open|close|all   busca declaraciones nuevas y ejecuta lo que toca en esa fase
                [--dry-run]              (open = compras, close = ventas)
  copier.py simulate DOCID YEAR          muestra qué haría con una declaración concreta (nunca opera)
  copier.py status                       cuenta, posiciones y acciones pendientes

Reglas (config.json):
  - Compra (P) de acción [ST] u opción [OP]  -> compra la ACCIÓN, monto = % del equity según el rango declarado.
  - Venta (S / S partial)                    -> cierra toda la posición en ese ticker, si la tienes.
  - Ejercicio de calls ("Exercised ...")     -> se ignora: ya se copió al comprar las calls.
  - Donaciones ("Contribution ... / gift")   -> se ignoran: no son una señal de venta.
  - Otros activos (fondos, LLC, bonos) o sin ticker -> se ignoran.
  - Efectivo ocioso -> estacionado en SPY (park_cash_in); se vende SPY para financiar cada compra.
  - Sin plata para una compra -> se vende la posición copiada más antigua (rotate_when_short).

Cada declaración nueva se convierte en acciones pendientes. La fase "open" ejecuta las compras
pendientes y la fase "close" (15 min antes del cierre) las ventas pendientes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from alpaca_client import Alpaca, load_env
from disclosures import (Filing, Transaction, fetch_pdf_text, list_ptr_filings, parse_ptr,
                         search_ptr_filings_live)

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", ROOT))  # en Railway: volumen montado en /data
STATE = DATA / "state" / "state.json"
JOURNAL = DATA / "logs" / "journal.csv"
log = logging.getLogger("copier")


@dataclass
class Decision:
    tx: Transaction
    action: str      # buy | close | skip
    pct: float = 0.0
    reason: str = ""


# ---------- reglas (sin efectos secundarios) ----------

def tier_pct(amount_low: int | None, tiers: list[list[float]]) -> float:
    if amount_low is None:
        return 0.0
    pct = 0.0
    for low, p in tiers:
        if amount_low >= low:
            pct = p
    return pct


def decide(tx: Transaction, cfg: dict) -> Decision:
    desc = f"{tx.description} {tx.asset}".lower()
    if not tx.ticker:
        return Decision(tx, "skip", reason="sin ticker bursátil")
    if tx.asset_type not in cfg["copy_asset_types"]:
        return Decision(tx, "skip", reason=f"tipo de activo {tx.asset_type} no se copia")
    if cfg.get("park_cash_in") and tx.ticker == cfg["park_cash_in"]:
        return Decision(tx, "skip", reason="es el ETF donde se estaciona el efectivo")
    if tx.tx_type == "E":
        return Decision(tx, "skip", reason="canje/spin-off (E)")
    if tx.is_purchase:
        if cfg["skip_option_exercises"] and "exercised" in desc:
            return Decision(tx, "skip", reason="ejercicio de calls (ya copiado al comprarlas)")
        pct = tier_pct(tx.amount_low, cfg["size_tiers_pct"])
        if pct <= 0:
            return Decision(tx, "skip", reason="rango de monto no reconocido")
        why = "compra de calls -> se compra la acción" if tx.asset_type == "OP" else "compra de acciones"
        return Decision(tx, "buy", pct=pct, reason=why)
    if tx.is_sale:
        if cfg["skip_donations"] and any(w in desc for w in ("contribution", "gift", "donat")):
            return Decision(tx, "skip", reason="donación, no es venta")
        return Decision(tx, "close", reason=f"venta declarada ({tx.tx_type})")
    return Decision(tx, "skip", reason=f"tipo de transacción {tx.tx_type} no contemplado")


def plan(txs: list[Transaction], cfg: dict) -> list[Decision]:
    # orden cronológico; el mismo día, ventas antes que compras
    txs = sorted(txs, key=lambda t: (t.tx_date, 0 if t.is_sale else 1))
    return [decide(t, cfg) for t in txs]


# ---------- estado y bitácora ----------

def load_state() -> dict:
    if STATE.exists():
        st = json.loads(STATE.read_text())
        st.setdefault("pending", [])
        return st
    return {"initialized_at": None, "seen_docs": {}, "pending": []}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str))
    tmp.replace(STATE)


JOURNAL_FIELDS = ["ts", "mode", "member", "doc_id", "filing_date", "tx_date", "ticker", "asset_type", "tx_type",
                  "amount_low", "action", "pct", "notional_usd", "qty", "order_id", "status", "reason"]


def journal(row: dict) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    new = not JOURNAL.exists()
    with JOURNAL.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in JOURNAL_FIELDS})


# ---------- descubrimiento ----------

def member_name(m: dict) -> str:
    return f"{m['first']} {m['last']}"


def years_to_check(today: date) -> list[int]:
    # en enero todavía llegan declaraciones del año anterior
    return [today.year - 1, today.year] if today.month == 1 else [today.year]


def filings_for(member: dict, years: list[int]) -> list[Filing]:
    """Buscador en vivo + índice ZIP diario; si uno falla, sirve el otro."""
    found: dict[str, Filing] = {}
    for y in years:
        for source in (list_ptr_filings, search_ptr_filings_live):
            try:
                for f in source(y, member["last"], member["first"]):
                    found.setdefault(f.doc_id, f)  # el ZIP va primero porque trae la fecha real
            except Exception as e:  # noqa: BLE001
                log.warning("Fuente %s falló para %s %s: %s", source.__name__, member_name(member), y, e)
    return sorted(found.values(), key=lambda f: f.doc_id)


def to_pending(d: Decision, filing: Filing, member: str) -> dict:
    t = d.tx
    return {"key": t.key, "member": member, "doc_id": t.doc_id, "filing_date": str(filing.filing_date),
            "tx_date": str(t.tx_date), "ticker": t.ticker, "asset_type": t.asset_type, "tx_type": t.tx_type,
            "amount_low": t.amount_low, "action": d.action, "pct": d.pct, "reason": d.reason,
            "queued_at": datetime.now().isoformat(timespec="seconds")}


def discover(cfg: dict, state: dict, dry_run: bool) -> list[dict]:
    """Lee declaraciones nuevas, anota en la bitácora lo que se ignora y devuelve las acciones a encolar."""
    new_actions = []
    for member in cfg["members"]:
        name = member_name(member)
        for f in filings_for(member, years_to_check(date.today())):
            if f.doc_id in state["seen_docs"]:
                continue
            log.info("Nueva declaración de %s: %s %s", name, f.doc_id, f.pdf_url)
            txs = parse_ptr(fetch_pdf_text(f), f.doc_id)
            if not txs:
                log.error("No se pudo leer ninguna transacción de %s (¿PDF escaneado?). Revisar a mano.", f.doc_id)
                result = "needs_manual_review"
            else:
                for d in plan(txs, cfg):
                    p = to_pending(d, f, name)
                    if d.action == "skip":
                        log.info("SKIP  %-6s %-12s %s", d.tx.ticker, d.tx.tx_type, d.reason)
                        journal({**p, "mode": "dry-run" if dry_run else "", "status": "skipped",
                                 "ts": p["queued_at"]})
                    else:
                        new_actions.append(p)
                result = f"{len(txs)} tx"
            if not dry_run:
                state["seen_docs"][f.doc_id] = {"member": name, "filing_date": str(f.filing_date), "result": result}
    return new_actions


# ---------- ejecución ----------

class Executor:
    """Ejecuta acciones pendientes. Si `park_cash_in` está configurado (SPY), el efectivo ocioso vive en ese
    ETF: se vende lo necesario para financiar cada compra y lo que sobra se vuelve a estacionar.
    Si `rotate_when_short` y no alcanza la plata, se vende la posición copiada comprada hace más tiempo."""

    def __init__(self, api: Alpaca, cfg: dict, state: dict, dry_run: bool):
        self.api, self.cfg, self.state, self.dry_run = api, cfg, state, dry_run
        self.park = cfg.get("park_cash_in")
        self.mode = "dry-run" if dry_run else ("paper" if api.is_paper else "LIVE")
        self.refresh()

    def refresh(self) -> None:
        acct = self.api.account()
        self.equity = float(acct["equity"])
        self.cash = float(acct["cash"])
        self.pos = {p["symbol"]: float(p["market_value"]) for p in self.api.positions()}
        self.spy = self.pos.pop(self.park, 0.0) if self.park else 0.0
        self.buffer = self.equity * self.cfg.get("cash_buffer_pct", 0) / 100

    def _row(self, p: dict) -> dict:
        return {**p, "mode": self.mode, "ts": datetime.now().isoformat(timespec="seconds")}

    def run(self, p: dict) -> bool:
        """Ejecuta una acción pendiente. Devuelve True si queda resuelta (ejecutada o descartada)."""
        row = self._row(p)
        sym = p["ticker"]
        asset = self.api.asset(sym)
        if not asset or not asset.get("tradable"):
            log.warning("SKIP  %-6s no es operable en Alpaca", sym)
            journal({**row, "status": "skipped", "reason": "no operable en Alpaca"})
            return True
        return self._close(p, row) if p["action"] == "close" else self._buy(p, row, asset)

    # --- ventas ---
    def _close(self, p: dict, row: dict, why: str | None = None) -> bool:
        sym = p["ticker"]
        if sym not in self.pos:
            log.info("SKIP  %-6s %s vendió, pero no tienes posición", sym, p["member"])
            journal({**row, "status": "skipped", "reason": f"{p['reason']}; sin posición propia"})
            return True
        value = self.pos.pop(sym)
        log.info("CLOSE %-6s $%.2f (%s)", sym, value, why or f"{p['member']} vendió")
        self.cash += value
        self.state.setdefault("positions", {}).pop(sym, None)
        if self.dry_run:
            journal({**row, "notional_usd": f"{value:.2f}", "status": "dry-run"})
            return True
        o = self.api.close_position(sym) or {}
        journal({**row, "notional_usd": f"{value:.2f}", "order_id": o.get("id", ""), "status": o.get("status", "")})
        return True

    def _rotate_out(self, exclude: str) -> bool:
        """Vende la posición copiada más antigua (por fecha de su última compra copiada)."""
        opened = self.state.setdefault("positions", {})
        cands = sorted((opened.get(t, "0000-00-00"), t) for t in self.pos if t != exclude)
        if not cands:
            return False
        t = cands[0][1]
        p = {"key": f"rotate|{t}|{date.today()}", "member": "-", "doc_id": "", "ticker": t, "action": "close",
             "pct": 0, "reason": f"rotación: la más antigua (comprada {cands[0][0]}) para liberar plata"}
        self._close(p, self._row(p), why=p["reason"])
        return True

    def _order(self, **order) -> dict:
        return {"id": "", "status": "dry-run"} if self.dry_run else self.api.submit_order(**order)

    def _unpark(self, amount: float) -> None:
        amount = math.floor(min(amount, self.spy) * 100) / 100
        if amount < 1:
            return
        log.info("SELL  %-6s $%.2f para financiar compras", self.park, amount)
        o = self._order(symbol=self.park, side="sell", type="market", time_in_force="day", notional=f"{amount:.2f}")
        journal({**self._row({"ticker": self.park, "action": "unpark", "reason": "financiar compra copiada"}),
                 "notional_usd": f"{amount:.2f}", "order_id": o["id"], "status": o["status"]})
        self.spy -= amount
        self.cash += amount

    def park_idle_cash(self) -> None:
        if not self.park:
            return
        amount = math.floor((self.cash - self.buffer) * 100) / 100
        if amount < self.cfg["min_order_usd"]:
            return
        log.info("BUY   %-6s $%.2f estacionando efectivo ocioso", self.park, amount)
        o = self._order(symbol=self.park, side="buy", type="market", time_in_force="day", notional=f"{amount:.2f}")
        journal({**self._row({"ticker": self.park, "action": "park", "reason": "efectivo ocioso"}),
                 "notional_usd": f"{amount:.2f}", "order_id": o["id"], "status": o["status"]})
        self.cash -= amount
        self.spy += amount

    # --- compras ---
    def _buy(self, p: dict, row: dict, asset: dict) -> bool:
        sym, cfg = p["ticker"], self.cfg
        target = self.equity * p["pct"] / 100
        room = self.equity * cfg["max_position_pct"] / 100 - self.pos.get(sym, 0.0)
        want = min(target, max(room, 0.0))
        if want < cfg["min_order_usd"]:
            log.info("SKIP  %-6s tope por posición alcanzado", sym)
            journal({**row, "status": "skipped", "reason": f"{p['reason']}; tope por posición alcanzado"})
            return True

        def available() -> float:
            return self.cash + self.spy - self.buffer if cfg["cash_only"] else float("inf")

        while cfg.get("rotate_when_short") and available() < want and self._rotate_out(exclude=sym):
            pass
        if self.park and self.cash - self.buffer < want:
            self._unpark(want - (self.cash - self.buffer))
        notional = math.floor(min(want, max(available() - self.spy, 0.0)) * 100) / 100
        if notional < cfg["min_order_usd"]:
            log.info("SKIP  %-6s sin plata disponible (objetivo $%.2f)", sym, target)
            journal({**row, "status": "skipped", "reason": f"{p['reason']}; sin plata disponible"})
            return True

        order = {"symbol": sym, "side": "buy", "type": "market", "time_in_force": "day",
                 "client_order_id": f"cp-{p['doc_id']}-{hashlib.sha1(p['key'].encode()).hexdigest()[:12]}"}
        qty = ""
        if asset.get("fractionable"):
            order["notional"] = f"{notional:.2f}"
        else:
            price = self.api.latest_price(sym)
            qty = int(notional // price) if price else 0
            if qty < 1:
                log.info("SKIP  %-6s no fraccionable y el monto no alcanza 1 acción", sym)
                journal({**row, "status": "skipped", "reason": "no fraccionable; monto < 1 acción"})
                return True
            order["qty"] = str(qty)
            notional = qty * price

        log.info("BUY   %-6s $%.2f (%.1f%% equity) %s — %s", sym, notional, p["pct"], p["reason"], p["member"])
        o = self._order(**order)
        journal({**row, "notional_usd": f"{notional:.2f}", "qty": qty, "order_id": o["id"], "status": o["status"]})
        self.pos[sym] = self.pos.get(sym, 0.0) + notional
        self.cash -= notional
        self.state.setdefault("positions", {})[sym] = str(date.today())
        return True


PHASE_ACTIONS = {"open": {"buy"}, "close": {"close"}, "all": {"buy", "close"}}


def cmd_run(cfg: dict, api: Alpaca, phase: str, dry_run: bool) -> None:
    state = load_state()
    if not state["initialized_at"]:
        sys.exit("Primero corre `copier.py init` para fijar la línea base.")
    queue = state["pending"] + discover(cfg, state, dry_run)
    todo = [p for p in queue if p["action"] in PHASE_ACTIONS[phase]]
    later = [p for p in queue if p["action"] not in PHASE_ACTIONS[phase]]
    log.info("Fase %s: %d acciones a ejecutar, %d quedan para la otra fase", phase, len(todo), len(later))

    ex = Executor(api, cfg, state, dry_run)
    failed = []
    for p in todo:
        try:
            if not ex.run(p):
                failed.append(p)
        except Exception as e:  # noqa: BLE001 — una orden fallida no debe frenar las demás
            log.error("ERROR %-6s %s", p["ticker"], e)
            journal({**p, "mode": ex.mode, "ts": datetime.now().isoformat(timespec="seconds"),
                     "status": "error", "reason": str(e)[:300]})
            p["attempts"] = p.get("attempts", 0) + 1
            if p["attempts"] < 3:
                failed.append(p)
    try:
        if any(p["action"] == "close" for p in todo) and not dry_run:
            time.sleep(5)  # dejar que se llenen las ventas antes de reinvertir
            ex.refresh()
        ex.park_idle_cash()
    except Exception as e:  # noqa: BLE001
        log.error("No se pudo estacionar el efectivo: %s", e)
    if not dry_run:
        state["pending"] = later + failed
        state["last_run"] = {"phase": phase, "at": datetime.now().isoformat(timespec="seconds")}
        save_state(state)


def cmd_init(cfg: dict) -> None:
    state = load_state()
    today = date.today()
    for member in cfg["members"]:
        for f in filings_for(member, [today.year - 1, today.year]):
            state["seen_docs"].setdefault(f.doc_id, {"member": member_name(member),
                                                     "filing_date": str(f.filing_date), "result": "baseline"})
    state["initialized_at"] = state["initialized_at"] or datetime.now().isoformat(timespec="seconds")
    save_state(state)
    log.info("Línea base: %d declaraciones marcadas como vistas (no se opera sobre ellas)", len(state["seen_docs"]))


def cmd_simulate(cfg: dict, api: Alpaca, doc_id: str, year: int) -> None:
    for member in cfg["members"]:
        f = next((x for x in filings_for(member, [year]) if x.doc_id == doc_id), None)
        if f:
            break
    else:
        sys.exit(f"No encontré la declaración {doc_id} en {year}")
    ex = Executor(api, cfg, load_state(), dry_run=True)
    for d in plan(parse_ptr(fetch_pdf_text(f), doc_id), cfg):
        if d.action == "skip":
            log.info("SKIP  %-6s %-12s %s", d.tx.ticker, d.tx.tx_type, d.reason)
        else:
            ex.run(to_pending(d, f, member_name(member)))


def cmd_status(api: Alpaca) -> None:
    a = api.account()
    print(f"{'PAPER' if api.is_paper else 'LIVE'}  equity=${float(a['equity']):,.2f}  cash=${float(a['cash']):,.2f}")
    for p in api.positions():
        print(f"  {p['symbol']:6} qty={p['qty']:>10}  valor=${float(p['market_value']):>12,.2f}  "
              f"P/L=${float(p['unrealized_pl']):>10,.2f}")
    st = load_state()
    print(f"Pendientes: {len(st['pending'])}  | última corrida: {st.get('last_run')}")
    for p in st["pending"]:
        print(f"  {p['action']:5} {p['ticker']:6} {p['pct']}%  {p['member']}  {p['reason']}")


def setup_logging() -> None:
    (DATA / "logs").mkdir(parents=True, exist_ok=True)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.StreamHandler(sys.stdout),
                                      logging.FileHandler(DATA / "logs" / "copier.log")])


def load_config() -> dict:
    load_env(ROOT / ".env")
    return json.loads((ROOT / "config.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=list(PHASE_ACTIONS), default="all")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--live", action="store_true", help="permitir cuenta real (por defecto solo paper)")
    s = sub.add_parser("simulate")
    s.add_argument("doc_id")
    s.add_argument("year", type=int)
    sub.add_parser("status")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config()
    if args.cmd == "init":
        return cmd_init(cfg)
    api = Alpaca()
    if args.cmd == "run":
        if not api.is_paper and not args.live and not args.dry_run:
            sys.exit("La URL no es paper. Para operar con dinero real hay que pasar --live explícitamente.")
        cmd_run(cfg, api, args.phase, args.dry_run)
    elif args.cmd == "simulate":
        cmd_simulate(cfg, api, args.doc_id, args.year)
    elif args.cmd == "status":
        cmd_status(api)


if __name__ == "__main__":
    main()
