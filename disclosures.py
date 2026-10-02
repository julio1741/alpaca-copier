"""Lectura de las declaraciones oficiales de transacciones (PTR) de la Cámara de Representantes.

Fuente: https://disclosures-clerk.house.gov — el índice anual {año}FD.zip lista todas las
declaraciones; las de tipo "P" (Periodic Transaction Report) tienen su PDF en
ptr-pdfs/{año}/{DocID}.pdf. El texto se extrae con `pdftotext -layout` (poppler).
"""
from __future__ import annotations

import io
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime

import requests

BASE = "https://disclosures-clerk.house.gov/public_disc"
UA = {"User-Agent": "Mozilla/5.0 (personal research; pelosi-copier)"}


@dataclass
class Filing:
    doc_id: str
    filing_date: date
    year: int

    @property
    def pdf_url(self) -> str:
        return f"{BASE}/ptr-pdfs/{self.year}/{self.doc_id}.pdf"


@dataclass
class Transaction:
    doc_id: str
    owner: str            # SP = cónyuge, JT = conjunta, DC = hijo dependiente, "" = ella
    asset: str
    ticker: str | None
    asset_type: str | None  # ST = acción, OP = opción, otros = ignorados
    tx_type: str          # P, S, S (partial), E
    tx_date: date
    notified_date: date
    amount_low: int | None
    description: str = ""
    raw: list[str] = field(default_factory=list)
    seq: int = 0  # posición de la fila en el reporte: distingue transacciones idénticas (misma fecha y rango)

    @property
    def is_purchase(self) -> bool:
        return self.tx_type == "P"

    @property
    def is_sale(self) -> bool:
        return self.tx_type.startswith("S")

    @property
    def key(self) -> str:
        return f"{self.doc_id}|{self.ticker}|{self.asset_type}|{self.tx_type}|{self.tx_date}|{self.amount_low}|{self.seq}"


def list_ptr_filings(year: int, last: str = "Pelosi", first: str = "Nancy") -> list[Filing]:
    r = requests.get(f"{BASE}/financial-pdfs/{year}FD.zip", headers=UA, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        xml = z.read(f"{year}FD.xml").decode("utf-8-sig")
    out = []
    for m in ET.fromstring(xml).iter("Member"):
        if (m.findtext("Last") or "").strip().lower() != last.lower():
            continue
        if not (m.findtext("First") or "").strip().lower().startswith(first.lower()):
            continue
        if (m.findtext("FilingType") or "").strip() != "P":
            continue
        fdate = datetime.strptime(m.findtext("FilingDate").strip(), "%m/%d/%Y").date()
        out.append(Filing(doc_id=m.findtext("DocID").strip(), filing_date=fdate, year=year))
    return sorted(out, key=lambda f: f.filing_date)


SEARCH_URL = "https://disclosures-clerk.house.gov/FinancialDisclosure/ViewMemberSearchResult"
SEARCH_ROW_RE = re.compile(
    r'href="public_disc/ptr-pdfs/(?P<year>\d{4})/(?P<doc>\d+)\.pdf"[^>]*>(?P<name>[^<]+)</a>.*?'
    r'data-label="Filing">(?P<kind>[^<]*)</td>',
    re.DOTALL,
)


def search_ptr_filings_live(year: int, last: str, first: str) -> list[Filing]:
    """Buscador del sitio (sin caché). El ZIP del índice solo se regenera una vez al día."""
    r = requests.post(SEARCH_URL, headers=UA, timeout=60,
                      data={"LastName": last, "FilingYear": str(year), "State": "", "District": ""})
    r.raise_for_status()
    out = []
    for m in SEARCH_ROW_RE.finditer(r.text):
        name = m["name"].lower()
        if last.lower() in name and first.lower() in name and m["kind"].strip().upper().startswith("PTR"):
            out.append(Filing(doc_id=m["doc"], filing_date=date.today(), year=int(m["year"])))
    return out


def fetch_pdf_text(filing: Filing) -> str:
    r = requests.get(filing.pdf_url, headers=UA, timeout=60)
    r.raise_for_status()
    return pdf_bytes_to_text(r.content)


def pdf_bytes_to_text(pdf: bytes) -> str:
    res = subprocess.run(["pdftotext", "-layout", "-", "-"], input=pdf, capture_output=True, check=True)
    return res.stdout.decode("utf-8", errors="replace")


# Línea que abre una transacción: [owner] asset ... TIPO  fecha  fecha  $monto
ENTRY_RE = re.compile(
    r"^[\s\f]*(?:\d{8,}\s+)?(?:(?P<owner>SP|JT|DC)\s+)?(?P<asset>\S.*?)\s*"
    r"(?<!\S)(?P<type>S \(partial\)|P|S|E)\s+"
    r"(?P<tdate>\d{2}/\d{2}/\d{4})\s+(?P<ndate>\d{2}/\d{2}/\d{4})\s+(?P<amount>.*)$",
    re.IGNORECASE,
)
# Los PDF antiguos (2014-2022) traen las letras en mayúsculas/minúsculas mezcladas: "(aaPl) [sT]"
TICKER_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9.\-]{0,5})\)")
TYPE_RE = re.compile(r"\[([A-Za-z]{2})\]")
MONEY_RE = re.compile(r"\$([\d,]+)")
# Líneas "F S : New" / "D : Purchased ..." (los títulos salen con letras sueltas en el PDF)
META_RE = re.compile(
    r"^\s*(?:[A-Z](?:\s+[A-Z])*|filing status|description|comments|location|subholding of)\s*:\s*(?P<val>.*)$",
    re.IGNORECASE,
)
# En el formato antiguo esos campos quedan en la misma línea que el activo
INLINE_META_RE = re.compile(r"\s(filing status|description|comments|location|subholding of)\s*:", re.IGNORECASE)
# Encabezado de tabla que se repite en cada página (una transacción puede quedar partida en dos)
PAGE_HEADER_RE = re.compile(r"^[\s\f]*(ID\s+Owner|Type\s+Date|\$200\?)")
END_OF_TABLE_RE = re.compile(r"^\s*\*\s*For the complete")


def _left_col(line: str) -> str:
    """Texto de la columna de activo en una línea de continuación (corta antes de la columna de monto)."""
    return re.split(r"\s{3,}", line.strip())[0] if line.strip() else ""


def parse_ptr(text: str, doc_id: str) -> list[Transaction]:
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if ENTRY_RE.match(l)]
    txs = []
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        m = ENTRY_RE.match(lines[i])
        asset_parts, desc_parts = [m["asset"].strip()], []
        in_meta = False
        for l in lines[i + 1:end]:
            if END_OF_TABLE_RE.match(l):
                break
            if PAGE_HEADER_RE.match(l):
                continue
            mm = META_RE.match(l)
            if mm:
                in_meta = True
                desc_parts.append(mm["val"].strip())
                continue
            if not in_meta and l.strip():
                asset_parts.append(_left_col(l))
        asset = " ".join(p for p in asset_parts if p)
        cut = INLINE_META_RE.search(asset)
        if cut:
            desc_parts.insert(0, asset[cut.start():].strip())
            asset = asset[:cut.start()].strip()
        tickers = TICKER_RE.findall(asset)
        ticker = tickers[-1].upper() if tickers else None
        tm = TYPE_RE.search(asset)
        atype = tm[1].upper() if tm else None
        if atype is None and ticker:  # formato antiguo sin etiqueta [ST]/[OP]
            atype = "OP" if "option" in " ".join(desc_parts).lower() else "ST"
        money = MONEY_RE.search(m["amount"])
        amount_low = int(money[1].replace(",", "")) if money else None
        txs.append(Transaction(
            doc_id=doc_id,
            owner=(m["owner"] or "").upper(),
            asset=asset,
            ticker=ticker,
            asset_type=atype,
            tx_type=m["type"].upper().replace("(PARTIAL)", "(partial)"),
            tx_date=datetime.strptime(m["tdate"], "%m/%d/%Y").date(),
            notified_date=datetime.strptime(m["ndate"], "%m/%d/%Y").date(),
            amount_low=amount_low,
            description=" | ".join(desc_parts),
            raw=lines[i:end],
            seq=n,
        ))
    return txs
