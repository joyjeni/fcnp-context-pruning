"""Real-world Agri "mandi" (market) price data source, sourced from
data.gov.in, wired in as realistic candidate context for FCNP pruning
demos — replacing/supplementing the synthetic weather/flight-tool
candidates in ``hf_space/app.py``.

Resource: "Current Daily Price of Various Commodities from Various
Markets (Mandi)" — Ministry of Agriculture and Farmers Welfare.
    https://www.data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi
Resource id: ``9ef84268-d588-465a-a308-a864a43d0070``
API docs:    https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070

This exact resource id and query pattern is the one used by the
sibling PhD-pipeline project (Objective 1, "SessionRerank+" /
Karnataka Agri Assistant):
    https://github.com/joyjeni/session-aware-toolbench-rerank
whose README documents the intended data flow for this project
(Objective 4, FCNP): "Receives from Obj4 (FCNP): Receives the pruned
context window back after FCNP compresses mandi data from 50+ records
to top-5." The fetch-and-compress demo in ``hf_space/app.py`` recreates
that exact framing.

Karnataka-coverage gap (confirmed, real, not a workaround to hide)
-------------------------------------------------------------------
This resource is "current day only" — there is no historical query
support, and on any given day it may simply have zero records for a
given state. Karnataka in particular is frequently absent (confirmed
empty across a full paginated scan on 2026-08-18, while 17 other
states had records). Server-side ``filters[...]`` query parameters for
this resource were tested extensively (state.keyword, state, District,
lower/upper case) and none narrowed results — filtering must happen
client-side after fetching full pages. So this module implements a
transparent **Karnataka-first, national-fallback** strategy: try to
find Karnataka records among the fetched pages; if none exist for the
day, fall back to a national top-commodities sample and say so
explicitly in the returned ``AgriSnapshot.used_national_fallback`` flag
and ``notes`` field, rather than silently substituting data.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fcnp.types import ContextElement

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
RESOURCE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
RESOURCE_CITATION = (
    "Current Daily Price of Various Commodities from Various Markets (Mandi), "
    "Ministry of Agriculture and Farmers Welfare, data.gov.in "
    f"(resource {RESOURCE_ID}) — https://www.data.gov.in/resource/"
    "current-daily-price-various-commodities-various-markets-mandi"
)

PREFERRED_STATE = "Karnataka"


@dataclass
class AgriRecord:
    state: str
    district: str
    market: str
    commodity: str
    variety: str
    grade: str
    arrival_date: str
    min_price: str
    max_price: str
    modal_price: str

    @property
    def id(self) -> str:
        parts = [self.state, self.district, self.market, self.commodity]
        slug = "_".join(p.strip().lower().replace(" ", "-") for p in parts if p)
        return slug or "agri_record"

    def describe(self) -> str:
        return (
            f"{self.commodity} ({self.variety}, grade {self.grade}) at {self.market} "
            f"market, {self.district}, {self.state}: modal price \u20b9{self.modal_price}/quintal "
            f"(range \u20b9{self.min_price}\u2013\u20b9{self.max_price}), arrival date {self.arrival_date}."
        )


@dataclass
class AgriSnapshot:
    records: list[AgriRecord]
    fetched_at: str
    total_records_scanned: int
    states_seen: list[str]
    used_national_fallback: bool
    notes: str
    source_url: str = RESOURCE_URL
    source_citation: str = RESOURCE_CITATION

    def to_dict(self) -> dict:
        return {
            "fetched_at": self.fetched_at,
            "total_records_scanned": self.total_records_scanned,
            "states_seen": self.states_seen,
            "used_national_fallback": self.used_national_fallback,
            "notes": self.notes,
            "source_url": self.source_url,
            "source_citation": self.source_citation,
            "records": [r.__dict__ for r in self.records],
        }

    @staticmethod
    def from_dict(d: dict) -> "AgriSnapshot":
        return AgriSnapshot(
            records=[AgriRecord(**r) for r in d["records"]],
            fetched_at=d["fetched_at"],
            total_records_scanned=d["total_records_scanned"],
            states_seen=d["states_seen"],
            used_national_fallback=d["used_national_fallback"],
            notes=d["notes"],
            source_url=d.get("source_url", RESOURCE_URL),
            source_citation=d.get("source_citation", RESOURCE_CITATION),
        )


def _parse_page(raw_json: dict) -> list[AgriRecord]:
    out = []
    for rec in raw_json.get("records", []):
        out.append(
            AgriRecord(
                state=rec.get("state", ""),
                district=rec.get("district", ""),
                market=rec.get("market", ""),
                commodity=rec.get("commodity", ""),
                variety=rec.get("variety", ""),
                grade=rec.get("grade", ""),
                arrival_date=rec.get("arrival_date", ""),
                min_price=rec.get("min_price", ""),
                max_price=rec.get("max_price", ""),
                modal_price=rec.get("modal_price", ""),
            )
        )
    return out


def _select_karnataka_first_with_fallback(
    all_records: list[AgriRecord], n_keep: int = 60,
) -> tuple[list[AgriRecord], bool, str]:
    """Karnataka-first, national-fallback selection (see module docstring)."""
    karnataka = [r for r in all_records if r.state.strip().lower() == PREFERRED_STATE.lower()]
    if karnataka:
        return karnataka[:n_keep], False, (
            f"Found {len(karnataka)} Karnataka records for today; no fallback needed."
        )

    # National fallback: pick a diverse top-commodities sample across
    # whichever states DID report today, so the demo still shows
    # realistic, non-degenerate variety instead of one state's list.
    by_commodity: dict[str, list[AgriRecord]] = {}
    for r in all_records:
        by_commodity.setdefault(r.commodity, []).append(r)
    sample: list[AgriRecord] = []
    for commodity, recs in sorted(by_commodity.items(), key=lambda kv: -len(kv[1])):
        sample.extend(recs[:3])
        if len(sample) >= n_keep:
            break
    return sample[:n_keep], True, (
        "No Karnataka records reported today for this resource (a real, "
        "documented data-coverage gap in this data.gov.in feed — it only "
        "carries the current day's arrivals, and Karnataka does not report "
        "every day). Falling back to a national top-commodities sample "
        f"across {len({r.state for r in all_records})} reporting states."
    )


def fetch_snapshot_via_curl(
    max_pages: int = 14, page_size: int = 500, timeout_s: int = 30,
) -> AgriSnapshot:
    """Fetch a fresh snapshot using ``curl`` (bash-sandbox-friendly path).

    This is the path used to build the static snapshot bundled with the
    Hugging Face Space (``hf_space/data/agri_mandi_snapshot.json``) at
    commit time — it must be run through the ``bash`` tool with
    ``api_credentials=["custom-cred:api.data.gov.in"]`` so the API key
    is injected via the outbound HTTPS proxy. It has no direct use
    inside the deployed Space itself (see ``fetch_snapshot_live`` for
    that path).
    """
    all_records: list[AgriRecord] = []
    for page in range(max_pages):
        offset = page * page_size
        url = f"{RESOURCE_URL}?format=json&limit={page_size}&offset={offset}"
        proc = subprocess.run(
            ["curl", "-s", "--max-time", str(timeout_s), url],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(proc.stdout)
        page_records = _parse_page(data)
        if not page_records:
            break
        all_records.extend(page_records)
        total = int(data.get("total", 0) or 0)
        if offset + page_size >= total:
            break

    selected, used_fallback, notes = _select_karnataka_first_with_fallback(all_records)
    return AgriSnapshot(
        records=selected,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        total_records_scanned=len(all_records),
        states_seen=sorted({r.state for r in all_records if r.state}),
        used_national_fallback=used_fallback,
        notes=notes,
    )


def fetch_snapshot_live(api_key: str, max_pages: int = 14, page_size: int = 500) -> AgriSnapshot:
    """Fetch a fresh snapshot using ``requests`` with an explicit API key.

    Intended for environments with normal outbound internet access (e.g.
    the deployed Hugging Face Space, where ``api_key`` comes from an HF
    Space secret) — NOT for this dev sandbox, whose outbound HTTPS proxy
    has a TLS quirk with the ``requests`` library (curl works fine here;
    see ``fetch_snapshot_via_curl``).
    """
    import requests

    all_records: list[AgriRecord] = []
    for page in range(max_pages):
        offset = page * page_size
        resp = requests.get(
            RESOURCE_URL,
            params={"api-key": api_key, "format": "json", "limit": page_size, "offset": offset},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        page_records = _parse_page(data)
        if not page_records:
            break
        all_records.extend(page_records)
        total = int(data.get("total", 0) or 0)
        if offset + page_size >= total:
            break

    selected, used_fallback, notes = _select_karnataka_first_with_fallback(all_records)
    return AgriSnapshot(
        records=selected,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        total_records_scanned=len(all_records),
        states_seen=sorted({r.state for r in all_records if r.state}),
        used_national_fallback=used_fallback,
        notes=notes,
    )


def load_snapshot(path: str | Path) -> AgriSnapshot:
    """Load a previously-saved snapshot JSON file (the bundled demo asset)."""
    with open(path, "r", encoding="utf-8") as f:
        return AgriSnapshot.from_dict(json.load(f))


def save_snapshot(snapshot: AgriSnapshot, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot.to_dict(), f, indent=2)


def records_to_context_elements(records: list[AgriRecord], embedding_dim: int | None = None) -> list[ContextElement]:
    """Convert AgriRecords into ContextElements (embedding left unset —
    caller embeds ``.text`` with their own encoder, matching how
    ``hf_space/app.py`` already embeds pasted candidate text)."""
    import numpy as np

    elements = []
    for r in records:
        emb = np.zeros(embedding_dim, dtype=np.float32) if embedding_dim else np.zeros(1, dtype=np.float32)
        elements.append(
            ContextElement(
                id=r.id,
                text=r.describe(),
                embedding=emb,
                citations=[RESOURCE_URL],
                metadata={"state": r.state, "commodity": r.commodity, "market": r.market},
            )
        )
    return elements


def snapshot_to_candidate_lines(snapshot: AgriSnapshot, limit: int = 20) -> list[tuple[str, str]]:
    """Return (id, text) pairs matching the 'id | text' textbox format
    ``hf_space/app.py`` already expects, so the Agri example set drops
    into the existing UI with zero structural changes."""
    return [(r.id, r.describe()) for r in snapshot.records[:limit]]
