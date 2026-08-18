"""Unit tests for the data.gov.in Agri mandi-price integration."""

from fcnp.data_gov_agri import (
    AgriRecord,
    AgriSnapshot,
    _select_karnataka_first_with_fallback,
    records_to_context_elements,
    snapshot_to_candidate_lines,
)


def _mk_record(state, commodity="Tomato", market="Test Market"):
    return AgriRecord(
        state=state, district="Test District", market=market, commodity=commodity,
        variety="Local", grade="Medium", arrival_date="18/08/2026",
        min_price="1000", max_price="1500", modal_price="1250",
    )


def test_karnataka_records_preferred_when_present():
    records = [_mk_record("Karnataka"), _mk_record("Tamil Nadu"), _mk_record("Karnataka", commodity="Onion")]
    selected, used_fallback, notes = _select_karnataka_first_with_fallback(records)
    assert used_fallback is False
    assert all(r.state == "Karnataka" for r in selected)
    assert len(selected) == 2


def test_national_fallback_when_no_karnataka():
    records = [
        _mk_record("Tamil Nadu", commodity="Onion"),
        _mk_record("Bihar", commodity="Potato"),
        _mk_record("Bihar", commodity="Potato"),
    ]
    selected, used_fallback, notes = _select_karnataka_first_with_fallback(records)
    assert used_fallback is True
    assert "fallback" in notes.lower() or "no karnataka" in notes.lower()
    assert len(selected) > 0


def test_record_describe_and_id_are_nonempty():
    r = _mk_record("Karnataka")
    assert "Tomato" in r.describe()
    assert "karnataka" in r.id


def test_snapshot_roundtrip_dict():
    r = _mk_record("Karnataka")
    snap = AgriSnapshot(
        records=[r], fetched_at="2026-08-18T00:00:00Z",
        total_records_scanned=1, states_seen=["Karnataka"],
        used_national_fallback=False, notes="ok",
    )
    d = snap.to_dict()
    snap2 = AgriSnapshot.from_dict(d)
    assert snap2.records[0].commodity == "Tomato"
    assert snap2.source_url == snap.source_url


def test_snapshot_to_candidate_lines_format():
    r1, r2 = _mk_record("Karnataka", commodity="Tomato"), _mk_record("Karnataka", commodity="Onion")
    snap = AgriSnapshot(
        records=[r1, r2], fetched_at="x", total_records_scanned=2,
        states_seen=["Karnataka"], used_national_fallback=False, notes="ok",
    )
    lines = snapshot_to_candidate_lines(snap, limit=10)
    assert len(lines) == 2
    for cid, text in lines:
        assert isinstance(cid, str) and isinstance(text, str)
        assert len(cid) > 0 and len(text) > 0


def test_records_to_context_elements():
    records = [_mk_record("Karnataka"), _mk_record("Karnataka", commodity="Onion")]
    elements = records_to_context_elements(records, embedding_dim=8)
    assert len(elements) == 2
    assert all(e.embedding.shape[0] == 8 for e in elements)
    assert all(e.citations for e in elements)
