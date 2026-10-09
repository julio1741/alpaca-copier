#!/usr/bin/env python3
"""Para cada preventa finalizada de PinkSale (BNB Chain) busca su pool en GeckoTerminal y baja las velas
horarias de los primeros 30 días, con el precio expresado en la moneda de la preventa (BNB o USDT).

Entrada:  cache/pinksale/bsc_ended.json   (API interna de PinkSale, pool/list?filterBy=ended)
Salida:   cache/pinksale/ohlcv/{token}.json  (una por token; se reanuda si se corta)

GeckoTerminal (gratis) permite ~30 llamadas/min: 2 llamadas por token -> ~2,2 s por token.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

CACHE = Path(__file__).resolve().parent / "cache" / "pinksale"
OUT = CACHE / "ohlcv"
GT = "https://api.geckoterminal.com/api/v2/networks/bsc"
WBNB = "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def get(url: str, params: dict | None = None) -> dict | None:
    for attempt in range(6):
        r = requests.get(url, headers=H, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    return None


def quote_address(doc: dict) -> str:
    cur = doc["currency"]["address"].lower()
    return WBNB if cur == "0x0000000000000000000000000000000000000000" else cur


def fetch(doc: dict) -> dict:
    token = doc["token"]["address"].lower()
    quote = quote_address(doc)
    pools = get(f"{GT}/tokens/{token}/pools", {"page": 1})
    if not pools or not pools.get("data"):
        return {"status": "sin_pool"}
    # el pool de lanzamiento es el que empareja el token con la moneda de la preventa
    cand = []
    for p in pools["data"]:
        rel = p["relationships"]
        base = rel["base_token"]["data"]["id"].split("_", 1)[1].lower()
        q = rel["quote_token"]["data"]["id"].split("_", 1)[1].lower()
        if {base, q} == {token, quote}:
            cand.append((p["attributes"]["pool_created_at"], p, base == token))
    if not cand:
        return {"status": "sin_pool_moneda"}
    created, pool, token_is_base = sorted(cand, key=lambda x: x[0])[0]
    t0 = int(datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp())
    o = get(f"{GT}/pools/{pool['attributes']['address']}/ohlcv/hour",
            {"limit": 721, "before_timestamp": t0 + 30 * 86400, "currency": "token",
             "token": "base" if token_is_base else "quote"})
    bars = sorted(o["data"]["attributes"]["ohlcv_list"]) if o else []
    return {"status": "ok", "pool": pool["attributes"]["address"], "created": t0, "bars": bars,
            "reserve_usd": pool["attributes"].get("reserve_in_usd")}


def main(since: str, limit: int | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    docs = json.load(open(CACHE / "bsc_ended.json"))
    ts = datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp()
    todo = [d for d in docs if d["pool"]["state"] == 1 and (d["pool"].get("endTime") or 0) >= ts
            and d["pool"].get("rate") and d["pool"].get("listingRate")]
    print(f"{len(todo)} preventas finalizadas desde {since}", flush=True)
    done = 0
    for d in todo[:limit]:
        f = OUT / f"{d['token']['address'].lower()}.json"
        if f.exists():
            continue
        try:
            res = fetch(d)
        except Exception as e:  # noqa: BLE001
            res = {"status": "error", "error": str(e)[:200]}
        f.write_text(json.dumps(res))
        done += 1
        if done % 25 == 0:
            print(f"{done} procesadas", flush=True)
        time.sleep(2.2)
    print("listo", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-01-01", int(sys.argv[2]) if len(sys.argv) > 2 else None)
