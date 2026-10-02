"""Cliente mínimo de la API REST de Alpaca (trading + último precio)."""
from __future__ import annotations

import os
from pathlib import Path

import requests

DATA_URL = "https://data.alpaca.markets/v2"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class Alpaca:
    def __init__(self):
        self.base = os.environ["APCA_API_BASE_URL"].rstrip("/")
        if not self.base.endswith("/v2"):
            self.base += "/v2"
        self.s = requests.Session()
        self.s.headers.update({
            "APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
            "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
        })

    @property
    def is_paper(self) -> bool:
        return "paper-api" in self.base

    def _req(self, method: str, url: str, **kw):
        r = self.s.request(method, url, timeout=30, **kw)
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {url} -> {r.status_code}: {r.text}")
        return r.json() if r.content else None

    def account(self) -> dict:
        return self._req("GET", f"{self.base}/account")

    def clock(self) -> dict:
        return self._req("GET", f"{self.base}/clock")

    def calendar(self, start: str, end: str) -> list[dict]:
        """Días hábiles de NYSE con hora de apertura/cierre en hora de Nueva York (incluye cierres tempranos)."""
        return self._req("GET", f"{self.base}/calendar", params={"start": start, "end": end}) or []

    def asset(self, symbol: str) -> dict | None:
        return self._req("GET", f"{self.base}/assets/{symbol}")

    def position(self, symbol: str) -> dict | None:
        return self._req("GET", f"{self.base}/positions/{symbol}")

    def positions(self) -> list[dict]:
        return self._req("GET", f"{self.base}/positions") or []

    def latest_price(self, symbol: str) -> float | None:
        d = self._req("GET", f"{DATA_URL}/stocks/{symbol}/trades/latest", params={"feed": "iex"})
        return float(d["trade"]["p"]) if d and d.get("trade") else None

    def submit_order(self, **order) -> dict:
        return self._req("POST", f"{self.base}/orders", json=order)

    def close_position(self, symbol: str) -> dict | None:
        return self._req("DELETE", f"{self.base}/positions/{symbol}")
