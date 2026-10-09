#!/usr/bin/env python3
"""¿Hubo momento de alza para vender con ganancia? Análisis de preventas PinkSale (BNB Chain).

Para cada token finalizado con velas horarias (pinksale_outcomes.py):
  precio de preventa = 1 / rate          (en BNB o USDT por token)
  multiplo(t)        = precio_t / precio de preventa
Mide el máximo alcanzado en ventanas (1h, 24h, 7d, 30d) y simula reglas de salida:
  - vender todo en la primera hora (cierre de la 1ª vela)
  - take-profit a +X%: vende si el máximo horario toca X; si no, vende al cierre del día 7 (o de la última vela)
  - take-profit + stop: si antes cae a -Y%, vende ahí (se evalúa con el mínimo de la vela)
Supuestos: se recibe el token al listar (claim inmediato), sin impuesto de venta ni slippage
(en la práctica hay ambos: el resultado real es PEOR que el simulado).
"""
from __future__ import annotations

import json
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "cache" / "pinksale"


def load():
    docs = {d["token"]["address"].lower(): d for d in json.load(open(CACHE / "bsc_ended.json"))}
    rows = []
    for f in (CACHE / "ohlcv").glob("*.json"):
        res = json.loads(f.read_text())
        d = docs.get(f.stem)
        if not d:
            continue
        rows.append((d, res))
    return rows


def presale_price(d: dict) -> float | None:
    rate = int(d["pool"]["rate"] or 0) / 10 ** int(d["token"]["decimals"])
    return 1 / rate if rate > 0 else None


def simulate(bars, p0, tp=None, sl=None, horizon_h=168):
    """Devuelve el múltiplo de salida respecto a p0."""
    if not bars:
        return 0.0
    t0 = bars[0][0]
    for t, o, h, l, c, v in bars:
        if t - t0 >= horizon_h * 3600:
            return c / p0
        if sl is not None and l / p0 <= 1 + sl:
            return 1 + sl
        if tp is not None and h / p0 >= 1 + tp:
            return 1 + tp
    return bars[-1][4] / p0


def main():
    rows = load()
    stat = {}
    for d, res in rows:
        stat[res.get("status")] = stat.get(res.get("status"), 0) + 1
    print("tokens procesados:", len(rows), stat)
    ok = []
    for d, res in rows:
        if res.get("status") != "ok" or not res.get("bars"):
            continue
        p0 = presale_price(d)
        if not p0:
            continue
        bars = res["bars"]
        t0 = bars[0][0]
        win = lambda hrs: [b for b in bars if b[0] - t0 < hrs * 3600]
        mx = lambda hrs: max(b[2] for b in win(hrs)) / p0
        ok.append({
            "sym": d["token"]["symbol"], "year": datetime.fromtimestamp(d["pool"]["endTime"], timezone.utc).year,
            "kyc": bool(d["pool"].get("kycDetails")), "listing_mult": int(d["pool"]["rate"]) / int(d["pool"]["listingRate"]),
            "raised_usd": d["pool"].get("totalRaisedUsd") or 0,
            "open": bars[0][1] / p0, "c1h": bars[0][4] / p0, "max1h": mx(1), "max24h": mx(24), "max7d": mx(168), "max30d": mx(720),
            "c24h": win(24)[-1][4] / p0, "c7d": win(168)[-1][4] / p0, "last": bars[-1][4] / p0, "hours": len(bars), "bars": bars, "p0": p0,
        })
    n = len(ok)
    print(f"con velas válidas: {n}  (las sin pool en GeckoTerminal suelen ser tokens muertos o rug: cuentan como pérdida total)\n")
    pct = lambda k, thr: sum(r[k] >= thr for r in ok) / n * 100
    med = lambda k: st.median(r[k] for r in ok)
    print("Múltiplo sobre el precio de preventa (1,0 = lo que pagaste):")
    for k, lbl in [("open", "apertura"), ("c1h", "cierre 1ª hora"), ("max1h", "máximo 1ª hora"), ("max24h", "máximo 24 h"),
                   ("max7d", "máximo 7 días"), ("max30d", "máximo 30 días"), ("c24h", "cierre 24 h"), ("c7d", "cierre 7 días"),
                   ("last", "última vela (≤30 d)")]:
        print(f"  {lbl:22} mediana {med(k):5.2f}x | ≥1,0x {pct(k,1):4.0f}% | ≥1,5x {pct(k,1.5):4.0f}% | ≥2x {pct(k,2):4.0f}% | ≥5x {pct(k,5):4.0f}%")

    print("\nReglas de salida (resultado medio por cada US$100 invertidos en CADA preventa, sin impuestos ni slippage):")
    rules = [("vender cierre 1ª hora", lambda r: r["c1h"]), ("mantener 7 días", lambda r: r["c7d"])]
    for tp in (0.2, 0.5, 1.0, 2.0):
        rules.append((f"TP +{int(tp*100)}% (si no, día 7)", lambda r, tp=tp: simulate(r["bars"], r["p0"], tp=tp)))
    for tp, sl in ((0.3, -0.3), (0.5, -0.3), (1.0, -0.5)):
        rules.append((f"TP +{int(tp*100)}% / stop {int(sl*100)}%", lambda r, tp=tp, sl=sl: simulate(r["bars"], r["p0"], tp=tp, sl=sl)))
    for name, fn in rules:
        m = [fn(r) for r in ok]
        print(f"  {name:28} medio {st.mean(m)*100-100:+6.1f} | mediana {st.median(m)*100-100:+6.1f} | ganan {sum(x>1 for x in m)/n*100:3.0f}%")
    return ok


if __name__ == "__main__":
    main()
