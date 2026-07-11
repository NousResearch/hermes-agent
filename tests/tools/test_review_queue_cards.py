import os
import sys
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def test_store_create_and_resolve_card_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    from tools import review_queue_cards as rq

    card = rq.create_card(
        card_id="831667",
        kind="evidence",
        thesis="Agentic VCs",
        body="FAIR multi-agent AI venture fund claim",
        target="telegram:-1003915682412:3930",
        person="@cfm_sol",
        url="https://x.com/cfm_sol/status/2066780746610831667",
        source="test",
    )

    assert card["status"] == "pending"
    assert card["telegram_chat_id"] == "-1003915682412"
    assert card["telegram_thread_id"] == "3930"

    resolved = rq.resolve_card("831667", "accept", user="Tester")
    assert resolved["status"] == "accepted"
    assert resolved["decided_by"] == "Tester"

    loaded = rq.get_card("831667")
    assert loaded["status"] == "accepted"


def test_render_buttons_transitions_after_accept():
    from tools import review_queue_cards as rq

    pending = {"id": "abc123", "status": "pending"}
    assert rq.button_rows_for_card(pending) == [
        [("✅ Accept", "rq:a:abc123"), ("❌ Deny", "rq:d:abc123")],
        [("🔁 Rescore", "rq:r:abc123"), ("⏭ Skip", "rq:s:abc123")],
    ]

    accepted = {"id": "abc123", "status": "accepted"}
    assert rq.button_rows_for_card(accepted) == [
        [("📝 Add accepted note", "rq:e:abc123"), ("✓ No note", "rq:n:abc123")]
    ]

    denied_pending_note = {"id": "abc123", "status": "denied_pending_note"}
    assert rq.button_rows_for_card(denied_pending_note) == [
        [("📝 Add denial reason", "rq:dn:abc123"), ("✓ No note", "rq:nn:abc123")]
    ]


def test_render_evidence_freeform_body_is_not_duplicated():
    from tools import review_queue_cards as rq

    body = (
        "Thesis Decentralized pre-training\n"
        "ID 745937\n"
        "Person [@mignano](https://x.com/mignano)\n"
        "Post https://x.com/i/web/status/2070190552725745937\n"
        "Rationale USV partner says pressure is building away from frontier labs."
    )
    text = rq.render_card_text(
        {
            "id": "evidence-20260626-745937",
            "kind": "evidence",
            "thesis": "Decentralized pre-training",
            "person": "@mignano",
            "url": "https://x.com/i/web/status/2070190552725745937",
            "source": "weekly expert timeline: @mignano",
            "body": body,
            "status": "pending",
            "decision_note": "",
        }
    )

    assert text.count("Thesis Decentralized pre-training") == 1
    assert text.count("Rationale USV partner") == 1
    assert "Status: Pending" in text


def test_startup_card_accept_creates_startup_note(tmp_path, monkeypatch):
    vault = tmp_path / "Antidote" / "Thesis"
    thesis_dir = vault / "Agentic VCs"
    thesis_dir.mkdir(parents=True)
    (thesis_dir / "Review queue.md").write_text("# Review queue — Agentic VCs\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    monkeypatch.setenv("ANTIDOTE_THESIS_ROOT", str(vault))

    from tools import review_queue_cards as rq

    rq.create_card(
        card_id="startup-test",
        kind="startup",
        thesis="Agentic VCs",
        body=(
            "Startup: Fairmint\n"
            "Website: https://fairmint.co\n"
            "Taxonomy fit: AI-native fundraising rails\n"
            "Market map: agentic VC market map search\n"
            "Founders: founder background TBD\n"
            "Rationale: candidate company in the thesis-adjacent startup landscape."
        ),
        target="telegram:-1003915682412:3930",
        url="https://fairmint.co",
        source="market map search",
    )
    card = rq.resolve_card("startup-test", "accept", user="Tester")

    assert card["status"] == "accepted"
    note = thesis_dir / "Startups" / "Fairmint.md"
    assert note.exists()
    text = note.read_text(encoding="utf-8")
    assert "AI-native fundraising rails" in text
    assert "market map search" in text
    assert (thesis_dir / "Startups" / "Startup map.md").exists()


def test_expert_accept_promotes_pending_display_name_note(tmp_path, monkeypatch):
    vault = tmp_path / "Antidote" / "Thesis"
    people_pending = vault / "Expert Seeds" / "People" / "Pending"
    people_pending.mkdir(parents=True)
    pending = people_pending / "Haseeb Qureshi.md"
    pending.write_text("# Haseeb Qureshi\n\n- Review status: pending review\n", encoding="utf-8")
    for thesis in ("Agentic VCs", "Decentralized pre-training"):
        (vault / thesis).mkdir(parents=True)
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    monkeypatch.setenv("ANTIDOTE_THESIS_ROOT", str(vault))

    from tools import review_queue_cards as rq

    rq.create_card(
        card_id="expert-hosseeb-20260709",
        kind="expert",
        thesis="Agentic VCs, Decentralized pre-training",
        person="Haseeb Qureshi / @hosseeb",
        url="https://x.com/hosseeb/status/1",
        body=(
            "Pending expert/source candidate: Haseeb Qureshi, @hosseeb\n\n"
            "CV note: [[Antidote/Thesis/Expert Seeds/People/hosseeb|hosseeb]]\n"
            "Wikilink: [[Antidote/Thesis/Expert Seeds/People/Pending/Haseeb Qureshi|Haseeb Qureshi]]\n"
            "Credentials / provenance: Dragonfly."
        ),
        target="telegram:-1003915682412:3930",
    )
    card = rq.resolve_card("expert-hosseeb-20260709", "no_note", user="Tester")

    accepted = vault / "Expert Seeds" / "People" / "Haseeb Qureshi.md"
    assert accepted.exists()
    assert not pending.exists()
    assert card["status"] == "accepted_no_note"
    loaded = rq.get_card("expert-hosseeb-20260709")
    assert loaded is not None
    assert loaded["obsidian_path"] == str(accepted)
    assert "Haseeb Qureshi" in loaded["body"]
    assert "from:hosseeb" in (vault / "Agentic VCs" / "Expert seeds.md").read_text(encoding="utf-8")


def test_apply_decision_to_obsidian_appends_review_line(tmp_path, monkeypatch):
    vault = tmp_path / "Antidote" / "Thesis"
    thesis_dir = vault / "Agentic VCs"
    thesis_dir.mkdir(parents=True)
    rq_file = thesis_dir / "Review queue.md"
    rq_file.write_text("# Review queue — Agentic VCs\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    monkeypatch.setenv("ANTIDOTE_THESIS_ROOT", str(vault))

    from tools import review_queue_cards as rq

    rq.create_card(
        card_id="831667",
        kind="evidence",
        thesis="Agentic VCs",
        body="FAIR multi-agent AI venture fund claim",
        target="telegram:-1003915682412:3930",
    )
    card = rq.resolve_card("831667", "deny", user="Tester")
    rq.apply_decision_to_obsidian(card, action="deny")

    text = rq_file.read_text(encoding="utf-8")
    assert "Telegram review decisions" in text
    assert "831667" in text
    assert "deny" in text
    assert "Tester" in text


def test_capture_pending_elaboration_note_updates_card_and_obsidian(tmp_path, monkeypatch):
    vault = tmp_path / "Antidote" / "Thesis"
    thesis_dir = vault / "Agentic VCs"
    thesis_dir.mkdir(parents=True)
    (thesis_dir / "Review queue.md").write_text("# Review queue — Agentic VCs\n", encoding="utf-8")
    thesis_note = thesis_dir / "Agentic VCs.md"
    thesis_note.write_text("# Agentic VCs\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    monkeypatch.setenv("ANTIDOTE_THESIS_ROOT", str(vault))

    from tools import review_queue_cards as rq

    rq.create_card(
        card_id="831667",
        kind="evidence",
        thesis="Agentic VCs",
        body="FAIR multi-agent AI venture fund claim",
        target="telegram:-1003915682412:3930",
    )
    rq.resolve_card("831667", "elaborate", user="Tester")

    card = rq.capture_pending_note(
        telegram_chat_id="-1003915682412",
        telegram_thread_id="3930",
        user="Tester",
        note="this test works",
    )

    assert card is not None
    assert card["status"] == "elaborated"
    assert card["decision_note"] == "this test works"
    assert "this test works" in (thesis_dir / "Review queue.md").read_text(encoding="utf-8")
    assert "this test works" not in thesis_note.read_text(encoding="utf-8")


def test_capture_pending_denial_note_updates_card_and_obsidian(tmp_path, monkeypatch):
    vault = tmp_path / "Antidote" / "Thesis"
    thesis_dir = vault / "Agentic VCs"
    thesis_dir.mkdir(parents=True)
    rq_file = thesis_dir / "Review queue.md"
    rq_file.write_text("# Review queue — Agentic VCs\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    monkeypatch.setenv("ANTIDOTE_THESIS_ROOT", str(vault))

    from tools import review_queue_cards as rq

    rq.create_card(
        card_id="831667",
        kind="evidence",
        thesis="Agentic VCs",
        body="FAIR multi-agent AI venture fund claim",
        target="telegram:-1003915682412:3930",
    )
    card = rq.resolve_card("831667", "deny", user="Tester")
    assert card["status"] == "denied_pending_note"

    note_requested = rq.resolve_card("831667", "deny_note", user="Tester")
    assert note_requested["status"] == "deny_note_requested"

    captured = rq.capture_pending_note(
        telegram_chat_id="-1003915682412",
        telegram_thread_id="3930",
        user="Tester",
        note="too generic; tighten filter to primary sources only",
    )

    assert captured is not None
    assert captured["status"] == "denied_with_note"
    assert captured["decision_note"] == "too generic; tighten filter to primary sources only"
    text = rq_file.read_text(encoding="utf-8")
    assert "denied_with_note" in text
    assert "tighten filter" in text
