#!/usr/bin/env python3
"""Compras de ejecutivos (Formulario 4 de la SEC) -> tabla de eventos para backtest.

Fuente: SEC "Insider Transactions Data Sets" trimestrales (research/cache/form4/{q}_form345.zip).
Evento = una presentación (ACCESSION_NUMBER) con compras en mercado abierto (TRANS_CODE == 'P').
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache"
F4 = CACHE / "form4"


def _read(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    with z.open(f"{name}.tsv") as f:
        return pd.read_csv(f, sep="\t", usecols=cols, dtype=str, quoting=3, on_bad_lines="skip")


def load_events() -> pd.DataFrame:
    frames = []
    for zp in sorted(F4.glob("20*q*.zip")):
        with zipfile.ZipFile(zp) as z:
            sub = _read(z, "SUBMISSION", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
                                           "ISSUERNAME", "ISSUERTRADINGSYMBOL", "AFF10B5ONE"])
            own = _read(z, "REPORTINGOWNER", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                                               "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
            tr = _read(z, "NONDERIV_TRANS", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                              "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD",
                                              "SHRS_OWND_FOLWNG_TRANS"])
        tr = tr[(tr.TRANS_CODE == "P") & (tr.TRANS_ACQUIRED_DISP_CD == "A")].copy()
        for c in ("TRANS_SHARES", "TRANS_PRICEPERSHARE", "SHRS_OWND_FOLWNG_TRANS"):
            tr[c] = pd.to_numeric(tr[c], errors="coerce")
        tr["value"] = tr.TRANS_SHARES * tr.TRANS_PRICEPERSHARE
        g = tr.groupby("ACCESSION_NUMBER").agg(
            value=("value", "sum"), shares=("TRANS_SHARES", "sum"),
            owned_after=("SHRS_OWND_FOLWNG_TRANS", "max"), trans_date=("TRANS_DATE", "min"),
            price=("TRANS_PRICEPERSHARE", "mean")).reset_index()
        # un accession puede tener varios dueños (p.ej. fondo + persona); se toma el primero
        own1 = own.drop_duplicates("ACCESSION_NUMBER")
        ev = g.merge(sub[sub.DOCUMENT_TYPE.isin(["4", "4/A"])], on="ACCESSION_NUMBER").merge(
            own1, on="ACCESSION_NUMBER", how="left")
        frames.append(ev)
    ev = pd.concat(frames, ignore_index=True).drop_duplicates("ACCESSION_NUMBER")
    ev["filed"] = pd.to_datetime(ev.FILING_DATE, format="%d-%b-%Y")
    ev["trans_date"] = pd.to_datetime(ev.trans_date, format="%d-%b-%Y", errors="coerce")
    ev["ticker"] = ev.ISSUERTRADINGSYMBOL.fillna("").str.upper().str.strip()
    rel = ev.RPTOWNER_RELATIONSHIP.fillna("")
    title = ev.RPTOWNER_TITLE.fillna("").str.lower()
    ev["is_officer"] = rel.str.contains("Officer")
    ev["is_director"] = rel.str.contains("Director")
    ev["is_10pct"] = rel.str.contains("TenPercent")
    ev["is_ceo_cfo"] = title.str.contains(r"\bceo\b|chief executive|\bcfo\b|chief financial|president")
    prev = ev.owned_after - ev.shares
    ev["pct_increase"] = (ev.shares / prev.where(prev > 0)).fillna(10.0)  # sin tenencia previa = posición nueva
    ev = ev[(ev.ticker != "") & (ev.ticker != "NONE") & ev.value.gt(0) & ev.DOCUMENT_TYPE.eq("4")]
    ev = ev.sort_values("filed").reset_index(drop=True)
    # compra en grupo: ejecutivos distintos de la misma empresa que compraron en los 30 días previos (incluye este)
    ev["cluster"] = 0
    for _, idx in ev.groupby("ISSUERCIK").groups.items():
        sub = ev.loc[idx]
        vals = []
        for i, r in sub.iterrows():
            w = sub[(sub.filed <= r.filed) & (sub.filed > r.filed - pd.Timedelta(days=30))]
            vals.append(w.RPTOWNERCIK.nunique())
        ev.loc[idx, "cluster"] = vals
    return ev


if __name__ == "__main__":
    ev = load_events()
    ev.to_pickle(CACHE / "insider_events.pkl")
    print(len(ev), "eventos de compra;", ev.ticker.nunique(), "tickers;", ev.filed.min().date(), "->", ev.filed.max().date())
    print(ev.groupby(ev.filed.dt.year).size())
    print("valor mediano $", round(ev.value.median()), "| officers", ev.is_officer.mean().round(2),
          "| directors", ev.is_director.mean().round(2), "| 10%", ev.is_10pct.mean().round(2),
          "| CEO/CFO", ev.is_ceo_cfo.mean().round(2), "| cluster>=2", (ev.cluster >= 2).mean().round(2))
    print("días hábiles con al menos un evento:", ev.filed.dt.date.nunique())
