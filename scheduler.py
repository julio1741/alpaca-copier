#!/usr/bin/env python3
"""Proceso permanente (Railway): corre el copiador dos veces por día hábil de bolsa.

  - apertura + OPEN_DELAY_MIN (1 min)     -> fase "open": revisa declaraciones nuevas y compra
  - cierre  - CLOSE_LEAD_MIN (15 min)     -> fase "close": vuelve a revisar y vende

Los horarios salen del calendario de Alpaca (hora de Nueva York), así que respeta feriados,
cierres tempranos y el cambio de horario de EE.UU. Si el proceso se reinicia en medio de la
sesión y la fase de hoy no corrió, la corre al tiro.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import copier
from alpaca_client import Alpaca

NY = ZoneInfo("America/New_York")
OPEN_DELAY_MIN = 1
CLOSE_LEAD_MIN = 15
MAX_SLEEP_S = 300
log = logging.getLogger("scheduler")


def upcoming_events(api: Alpaca, now: datetime) -> list[tuple[datetime, str, datetime]]:
    """[(cuándo, fase, cierre de esa sesión)] para los próximos días hábiles."""
    start = now.date()
    days = api.calendar(start.isoformat(), (start + timedelta(days=10)).isoformat())
    out = []
    for d in days:
        day = date.fromisoformat(d["date"])
        o = datetime.combine(day, datetime.strptime(d["open"], "%H:%M").time(), NY)
        c = datetime.combine(day, datetime.strptime(d["close"], "%H:%M").time(), NY)
        out.append((o + timedelta(minutes=OPEN_DELAY_MIN), "open", c))
        out.append((c - timedelta(minutes=CLOSE_LEAD_MIN), "close", c))
    return out


def run_phase(cfg: dict, api: Alpaca, phase: str, day: date) -> None:
    log.info("=== Fase %s del %s ===", phase, day)
    try:
        copier.cmd_run(cfg, api, phase, dry_run=False)
    except Exception:  # noqa: BLE001 — el proceso debe seguir vivo para el próximo turno
        log.exception("La fase %s falló", phase)
    st = copier.load_state()
    st.setdefault("runs", {})[f"{day}:{phase}"] = datetime.now(NY).isoformat(timespec="seconds")
    copier.save_state(st)


def main() -> None:
    copier.setup_logging()
    cfg = copier.load_config()
    api = Alpaca()
    if not api.is_paper:
        raise SystemExit("El programador solo corre contra la cuenta paper.")
    if not copier.load_state()["initialized_at"]:
        log.info("Sin estado previo: fijando línea base (no se copian declaraciones ya publicadas).")
        copier.cmd_init(cfg)
    log.info("Programador iniciado. Miembros: %s", ", ".join(copier.member_name(m) for m in cfg["members"]))

    announced = None
    while True:
        try:
            now = datetime.now(NY)
            done = copier.load_state().get("runs", {})
            pending = [(t, ph, c) for t, ph, c in upcoming_events(api, now)
                       if f"{t.date()}:{ph}" not in done and now < c]
            if not pending:
                time.sleep(MAX_SLEEP_S)
                continue
            when, phase, _ = pending[0]
            if when <= now:
                run_phase(cfg, api, phase, when.date())
                continue
            if announced != (when, phase):
                log.info("Próxima fase: %s el %s (hora NY)", phase, when.strftime("%Y-%m-%d %H:%M"))
                announced = (when, phase)
            time.sleep(min(MAX_SLEEP_S, (when - now).total_seconds()))
        except Exception:  # noqa: BLE001 — errores de red: reintentar en un rato
            log.exception("Error en el ciclo del programador; reintento en 60 s")
            time.sleep(60)


if __name__ == "__main__":
    main()
