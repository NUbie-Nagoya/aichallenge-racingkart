from pathlib import Path

from inventory_bags import discover_bags, mark_duplicate_recording_ids


def test_discovery_includes_unfinalized_mcap_and_metadata_only_directories(
    tmp_path: Path,
):
    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "metadata.yaml").write_text("x")
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "recording.json").write_text("{}")
    (partial / "data_0.mcap").write_bytes(b"partial")
    assert discover_bags(tmp_path) == [complete, partial]


def test_duplicate_recording_ids_make_every_duplicate_rejected():
    reports = [
        {"accepted": True, "reasons": [], "recording": {"recording_id": "same"}},
        {"accepted": True, "reasons": [], "recording": {"recording_id": "same"}},
        {"accepted": True, "reasons": [], "recording": {"recording_id": "unique"}},
    ]
    mark_duplicate_recording_ids(reports)
    assert all(not report["accepted"] for report in reports[:2])
    assert all(
        "duplicate recording_id across inventory: same" in report["reasons"]
        for report in reports[:2]
    )
    assert reports[2]["accepted"] is True
