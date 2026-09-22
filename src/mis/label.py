"""Post-session labelling: the gate v2 instrument.

Run after an interview. It asks the three session questions, then replays every
card that was shown and asks whether it earned the interruption.

Three deliberate choices, each of which changes what the answers are worth:

* **Session questions come first.** Going card by card first anchors you on the
  individual judgments, and the question that actually decides the gate -- did
  looking at it cost you anything -- is about the session as a whole. Ask it
  before the card replay colours it.

* **Cards are labelled here, not during the interview.** A tap mid-session adds
  attention cost to the exact thing being measured, which makes a failing gate
  uninterpretable: you could not tell whether the tool or the instrument caused
  it. Labelling afterwards costs the session nothing.

* **Each card is shown with the transcript around it.** Recall decays fast and
  "was this useful?" is unanswerable without the moment it fired in. The excerpt
  is what makes a retrospective label honest rather than a guess.

The distinction the labels are built around is usefulness, not correctness. The
2026-09-20 gate failed on a signal that was accurate and already noticed, so
"was it right" measures the wrong thing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .store import SessionStore

#: What a shown card can turn out to be worth. `already_knew` is the interesting
#: one: it is the verdict that failed the last gate, and it is not a bug report.
CARD_LABELS = {
    "n": ("new", "told me something I hadn't noticed"),
    "k": ("already_knew", "true, but I already knew"),
    "w": ("wrong", "it was wrong"),
    "t": ("bad_timing", "right, but the wrong moment"),
}

COST_LEVELS = {"n": "none", "l": "little", "y": "yes"}


def _fmt(ms: int) -> str:
    return f"{ms // 60000}:{(ms // 1000) % 60:02d}"


def _ask(prompt: str, options: dict[str, tuple[str, str] | str]) -> str:
    """Single-keystroke choice. Repeats until one of the keys is given."""
    for key, value in options.items():
        label = value[1] if isinstance(value, tuple) else value
        print(f"    [{key}] {label}")
    while True:
        raw = input(f"  {prompt} ").strip().lower()
        if raw in options:
            chosen = options[raw]
            return chosen[0] if isinstance(chosen, tuple) else chosen
        print("  ? pick one of: " + ", ".join(options))


def _yes_no(prompt: str) -> bool:
    while True:
        raw = input(f"  {prompt} [y/n] ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def _context(conn: sqlite3.Connection, session_id: str, at_ms: int,
             before_ms: int = 45_000) -> list[sqlite3.Row]:
    return list(conn.execute(
        "SELECT offset_ms, speaker, text FROM transcript_chunks "
        "WHERE session_id=? AND offset_ms BETWEEN ? AND ? ORDER BY offset_ms",
        (session_id, max(0, at_ms - before_ms), at_ms),
    ))


def run(db: Path, session_id: str | None) -> int:
    from .report import resolve_session_id

    store = SessionStore(db)
    try:
        sid = resolve_session_id(str(db), session_id)
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row

        cards = [
            r for r in conn.execute(
                "SELECT id, signal_name, value, probability, confidence, window_end_ms "
                "FROM signal_decisions WHERE session_id=? AND visible=1 "
                "AND signal_name!='current_phase' ORDER BY window_end_ms", (sid,))
        ]
        duration = conn.execute(
            "SELECT MAX(offset_ms) AS d FROM transcript_chunks WHERE session_id=?",
            (sid,)).fetchone()["d"] or 0

        print(f"\nSession {sid} — {_fmt(duration)}, {len(cards)} card(s) shown")
        if duration:
            print(f"Card rate: {len(cards) / max(duration / 3_600_000, 1e-9):.0f}/hour")

        # -- the session questions, before any card is replayed --------------
        print("\nFirst, the session as a whole.\n")
        told_new = _yes_no("Did any signal tell you something you hadn't already noticed?")
        misfired = _yes_no("Did any signal fire when it shouldn't have?")
        print("\n  Did looking at it cost you anything in the conversation?")
        cost = _ask("cost?", COST_LEVELS)

        excluded = _yes_no("\n  Was this session confounded (bad audio, wrong setup, "
                           "not a real interview)?")
        note = input("  Anything worth remembering? (enter to skip) ").strip() or None

        store.save_session_label(sid, told_new=told_new, misfired=misfired,
                                 attention_cost=cost, excluded=excluded, note=note)
        print("  saved.")

        if excluded:
            print("\nMarked confounded; skipping card replay. It won't count toward "
                  "the gate.")
            return 0
        if not cards:
            print("\nNo cards were shown, so there is nothing to replay.")
            return 0

        # -- then each card, with the moment it fired in ---------------------
        already = store.card_labels(sid)
        print(f"\nNow the {len(cards)} card(s). Each is shown with the 45 seconds "
              f"before it fired.\n")
        for i, card in enumerate(cards, start=1):
            if card["id"] in already:
                continue
            detail = (f"p={card['probability']:.2f}" if card["probability"] is not None
                      else str(card["value"]))
            print("─" * 72)
            print(f"[{i}/{len(cards)}]  {_fmt(card['window_end_ms'])}  "
                  f"{card['signal_name']}  {detail}")
            for row in _context(conn, sid, card["window_end_ms"]):
                who = "you " if row["speaker"] == "interviewer" else "them"
                print(f"    {who}  {row['text'][:96]}")
            print()
            store.save_card_label(sid, card["id"], _ask("verdict?", CARD_LABELS))

        labels = store.card_labels(sid)
        new = sum(1 for v in labels.values() if v == "new")
        print("─" * 72)
        print(f"\n{new}/{len(labels)} cards told you something new. "
              f"Attention cost: {cost}.")
        print("Run 'mis gate' to see where this leaves the gate.")
        return 0
    finally:
        store.close()
