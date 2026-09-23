"""CLI for offline replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .jev.fake import FakeJevAdapter
from .models import Speaker
from .replay import ReplayResult, replay
from .store import SessionStore


def _fmt_ms(ms: int) -> str:
    return f"{ms // 60000:02d}:{(ms // 1000) % 60:02d}"


def _fmt_value(decision) -> str:
    if decision.probability is not None:
        label = decision.value if isinstance(decision.value, str) else ""
        return f"{label} p={decision.probability:.2f}".strip()
    if decision.confidence is not None:
        return f"{decision.value} (conf {decision.confidence:.2f})"
    return str(decision.value)


def _print_report(result: ReplayResult, *, show_suppressed: bool) -> None:
    print(f"\nsession {result.session.id}  ({result.session.source_ref})")
    print(
        f"sha256  {result.session.source_sha256[:16]}…  "
        f"signals {result.session.signal_set_version}"
    )
    print(f"ticks   {len(result.ticks)}\n")

    print(f"{'time':>6}  {'signal':<20} {'answer':<28} {'status'}")
    print("-" * 78)
    for tick in result.ticks:
        for d in tick.decisions:
            if not d.visible and not show_suppressed:
                continue
            status = "SHOWN" if d.visible else f"hidden: {d.suppressed_reason}"
            print(f"{_fmt_ms(tick.at_ms):>6}  {d.signal_name:<20} {_fmt_value(d):<28} {status}")

    total = len(result.all_decisions)
    shown = len(result.visible_decisions)
    print("-" * 78)
    print(
        f"{shown}/{total} decisions displayed   "
        f"suppression rate {result.policy.suppression_rate:.0%}"
    )

    skipped: dict[str, int] = {}
    for tick in result.ticks:
        for name, reason in tick.skipped.items():
            skipped[f"{name}:{reason}"] = skipped.get(f"{name}:{reason}", 0) + 1
    if skipped:
        print("\nnot asked (precondition failed, no tokens spent):")
        for key, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>4}x  {key}")


def _sessions(args) -> int:
    from .report import resolve_session_id

    if not args.db.exists():
        print(f"{args.db} does not exist. Sessions record automatically when you run "
              f"'cue live'.", file=sys.stderr)
        return 2

    store = SessionStore(args.db)
    try:
        if args.note:
            target, text = args.note
            sid = resolve_session_id(str(args.db), target)
            store.set_notes(sid, text)
            print(f"noted on {sid}")
            return 0

        rows = store.sessions()
        if not rows:
            print("no sessions recorded yet")
            return 0

        print(f"{'id':<14}{'when':<18}{'source':<9}{'length':>8}{'shown':>7}"
              f"{'report':>8}  notes")
        for r in rows:
            when = (r["started_at"] or "")[:16].replace("T", " ")
            ms = r["duration_ms"] or 0
            length = f"{ms // 60000}:{(ms // 1000) % 60:02d}"
            print(f"{r['id']:<14}{when:<18}{r['source_type']:<9}{length:>8}"
                  f"{r['shown']:>7}{'yes' if r['has_report'] else '-':>8}  "
                  f"{r['notes'] or ''}")
        print(f"\n{len(rows)} session(s). 'cue report <id>' for an evidence pack.")
        return 0
    finally:
        store.close()


def _signals(args) -> int:
    """The five questions, as sent. Read-only on purpose.

    Editing them is a source change plus a version bump, not a config tweak,
    because every stored decision is stamped with the version that produced it.
    """
    from .models import SIGNAL_SET_VERSION
    from .signals import SIGNAL_SET, fingerprint

    print(f"signal set {SIGNAL_SET_VERSION}   fingerprint {fingerprint()}")
    print("defined in src/cue/signals.py, pinned by tests/test_signals_pinned.py\n")

    for spec in SIGNAL_SET:
        print("=" * 78)
        print(f"{spec.name}   [{spec.primitive.value}]")
        print("=" * 78)
        print("\nASKED WHEN")
        reason = spec.precondition.__doc__ or spec.precondition.__name__
        print(f"  {reason.strip().splitlines()[0]}")
        print("\nINSTRUCTIONS")
        for line in _wrap(spec.instructions):
            print(f"  {line}")
        if spec.criteria:
            print("\nCRITERIA")
            items = (spec.criteria.items() if isinstance(spec.criteria, dict)
                     else enumerate(spec.criteria))
            for key, text in items:
                body = _wrap(str(text), width=68)
                if not args.full:
                    body = body[:1] + (["..."] if len(body) > 1 else [])
                print(f"  {key}:")
                for line in body:
                    print(f"      {line}")
        print()
    if not args.full:
        print("--full for complete criteria text.")
    return 0


def _wrap(text: str, width: int = 74) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(text.split()), width=width)


def _traces(args) -> int:
    """Traces hold transcript text. Listing them is how you remember to delete."""
    from .trace import listing

    rows = listing(args.dir)
    if not rows:
        print("No traces. Run with --trace to record one.")
        return 0

    total = 0
    print(f"{'file':<44}{'size':>9}{'jev calls':>11}{'lines':>8}")
    for path, size, calls, lines in rows:
        total += size
        print(f"{path.name:<44}{size // 1024:>7}KB{calls:>11}{lines:>8}")
    print(f"\n{len(rows)} trace(s), {total // 1024}KB. These contain transcript text.")

    if args.purge:
        for path, *_ in rows:
            path.unlink()
        print(f"Deleted {len(rows)} trace(s).")
    else:
        print("'cue traces --purge' deletes them all.")
    return 0


def _gate(args) -> int:
    """Gate v2 standing, two-sided on purpose.

    A benefit rate alone would pass a tool that helps occasionally and distracts
    constantly, which is how the 2026-09-20 gate failed. The cost answer is the
    other half and it is not averaged: one session that cost attention is a
    finding, not an outlier to be smoothed away.
    """
    if not args.db.exists():
        print(f"{args.db} does not exist.", file=sys.stderr)
        return 2
    store = SessionStore(args.db)
    try:
        rows = store.gate_rows()
        counted = [r for r in rows if not r["excluded"]]
        if not rows:
            print("No labelled sessions yet. Run 'cue label' after your next one.")
            return 0

        print(f"{'session':<14}{'cards':>7}{'new':>6}{'cost':>9}  notes")
        for r in rows:
            mark = "  (excluded)" if r["excluded"] else ""
            print(f"{r['id'][:12]:<14}{r['labelled']:>7}{r['new_cards']:>6}"
                  f"{r['attention_cost']:>9}  {(r['notes'] or '')[:28]}{mark}")

        cards = sum(r["labelled"] for r in counted)
        new = sum(r["new_cards"] for r in counted)
        costly = [r for r in counted if r["attention_cost"] == "yes"]
        rate = (new / cards * 100) if cards else 0.0

        print()
        print(f"  sessions counted   {len(counted)}/{args.target}")
        print(f"  cards labelled     {cards}")
        print(f"  told me something  {new}  ({rate:.0f}%)")
        print(f"  cost attention     {len(costly)} session(s)")
        print()
        if len(counted) < args.target:
            print(f"  {args.target - len(counted)} more session(s) before the gate is callable.")
        elif costly:
            print("  FAIL on cost. Drop the cards, keep the phase banner.")
        else:
            print(f"  Benefit rate {rate:.0f}%, zero attention cost. "
                  f"Call it against the threshold you fixed in advance.")
        return 0
    finally:
        store.close()


def _live(args) -> int:
    import threading
    import uuid

    import uvicorn

    from .live.session import LiveSession, run_from_capture, run_from_transcript
    from .server import create_app
    from .trace import Trace

    session_id = uuid.uuid4().hex[:12]
    trace = Trace.for_session(session_id, args.trace_dir) if args.trace else None
    if trace is not None:
        print(f"tracing this session to {trace.path}", file=sys.stderr)

    if args.adapter == "typesafe":
        from .jev.typesafe import TypeSafeJevAdapter

        adapter = TypeSafeJevAdapter(trace=trace)
    else:
        adapter = FakeJevAdapter()
        print("FAKE adapter: the strip will move, but the judgments are keyword "
              "heuristics, not Jev.", file=sys.stderr)

    store = None if args.no_db else SessionStore(args.db)
    session = LiveSession(adapter=adapter, tick_ms=args.tick_ms, store=store,
                          session_id=session_id, trace=trace)
    if store is not None:
        print(f"recording to {args.db} as session {session.session_id}", file=sys.stderr)

    if args.simulate is not None:
        worker = threading.Thread(
            target=run_from_transcript, args=(session, args.simulate, args.speed),
            daemon=True, name="simulate")
    else:
        from .live.capture import DeviceError, DualCapture, device_name, resolve_device
        from .live.transcribe import Transcriber

        if args.mic is None:
            print("--mic is required. Run: cue devices", file=sys.stderr)
            return 2
        try:
            mic = resolve_device(args.mic)
            system = resolve_device(args.system) if args.system is not None else None
        except DeviceError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if system is None:
            print("WARNING: no --system device. You will hear yourself but NOT the\n"
                  "         candidate, and three of the signals need their speech.",
                  file=sys.stderr)
        print(f"mic:    [{mic}] {device_name(mic)}", file=sys.stderr)
        if system is not None:
            print(f"system: [{system}] {device_name(system)}", file=sys.stderr)
            if mic == system:
                print("ERROR: mic and system are the same device.", file=sys.stderr)
                return 2
        print(f"loading whisper '{args.model}' (first run downloads it)...", file=sys.stderr)
        transcriber = Transcriber(args.model)
        mic_speaker = (Speaker.CANDIDATE if args.my_role == "candidate"
                       else Speaker.INTERVIEWER)
        if mic_speaker is Speaker.CANDIDATE:
            print("ROLE: you are the CANDIDATE this half; your mic records as "
                  "'candidate'.", file=sys.stderr)
        capture = DualCapture(mic, system, mic_speaker=mic_speaker)
        worker = threading.Thread(
            target=run_from_capture, args=(session, capture, transcriber),
            daemon=True, name="capture")

    app = create_app(session, worker)
    print(f"\n  ->  http://127.0.0.1:{args.port}\n", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cue", description="Mock interview signal replay")
    sub = parser.add_subparsers(dest="command", required=True)

    rp = sub.add_parser("replay", help="Replay a JSONL transcript")
    rp.add_argument("transcript", type=Path)
    rp.add_argument("--db", type=Path, default=None, help="SQLite session log path")
    rp.add_argument("--tick-ms", type=int, default=15_000)
    rp.add_argument("--show-suppressed", action="store_true")
    rp.add_argument("--trace", action="store_true",
                    help="Write a per-session log of every Jev request and response")
    rp.add_argument("--trace-dir", type=Path, default=None,
                    help="Where traces go (default: traces/)")
    rp.add_argument("--json", action="store_true", help="Emit the session log as JSON")
    rp.add_argument(
        "--adapter",
        choices=["fake", "typesafe"],
        default="fake",
        help="fake = offline keyword stand-in, NOT evidence of signal quality",
    )

    dv = sub.add_parser("devices", help="List audio input devices")
    dv.add_argument("--check", action="store_true",
                    help="Listen to each device briefly and report whether audio is "
                         "actually arriving")

    ss = sub.add_parser("sessions", help="List recorded sessions")
    ss.add_argument("--db", type=Path, default=Path("sessions.db"))
    ss.add_argument("--note", nargs=2, metavar=("SESSION", "TEXT"),
                    help="Attach a note to a session")

    rp2 = sub.add_parser("report", help="Evidence pack from a recorded session")
    rp2.add_argument("session", nargs="?", default=None,
                     help="Session id or prefix. Omit for the most recent.")
    rp2.add_argument("--db", type=Path, default=Path("sessions.db"))
    rp2.add_argument("--out", type=Path, default=None, help="Write markdown to a file")
    rp2.add_argument("--no-coverage", action="store_true",
                     help="Skip the topic-coverage pass (no Jev calls)")
    rp2.add_argument("--refresh", action="store_true",
                     help="Regenerate even if a saved report exists")
    rp2.add_argument("--board", type=Path, default=None,
                     help="A whiteboard to analyse alongside the transcript: an "
                          ".excalidraw export, or a screenshot (.png/.jpg) when the "
                          "platform will not let you export. Implies --refresh.")

    lb = sub.add_parser("label", help="Record what was useful, after a session")
    lb.add_argument("session", nargs="?", default=None,
                    help="Session id or prefix. Omit for the most recent.")
    lb.add_argument("--db", type=Path, default=Path("sessions.db"))

    gt = sub.add_parser("gate", help="Where the labelled sessions leave gate v2")
    gt.add_argument("--db", type=Path, default=Path("sessions.db"))
    gt.add_argument("--target", type=int, default=10, help="Sessions the gate needs")

    sg = sub.add_parser("signals", help="Show the exact questions sent to Jev")
    sg.add_argument("--full", action="store_true", help="Include every criterion")

    tr = sub.add_parser("traces", help="List session traces, or delete them")
    tr.add_argument("--dir", type=Path, default=None)
    tr.add_argument("--purge", action="store_true", help="Delete every trace")

    bd = sub.add_parser("board", help="Show the extracted state of a whiteboard")
    bd.add_argument("board", type=Path,
                    help="An .excalidraw export or a screenshot")
    bd.add_argument("--json", action="store_true",
                    help="Emit the state exactly as Jev receives it")

    lv = sub.add_parser("live", help="Listen and show live signals in a browser")
    lv.add_argument("--mic", default=None,
                    help="YOUR microphone (interviewer): device name or index. "
                         "Names are safer -- indexes shift when devices change.")
    lv.add_argument("--system", default=None,
                    help="System audio (candidate): device name or index, e.g. BlackHole")
    lv.add_argument("--simulate", type=Path, default=None,
                    help="Rehearse against a stored transcript instead of a microphone")
    lv.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier")
    lv.add_argument("--model", default="base.en", help="faster-whisper model size")
    lv.add_argument("--tick-ms", type=int, default=20_000)
    lv.add_argument("--port", type=int, default=8765)
    lv.add_argument("--db", type=Path, default=Path("sessions.db"),
                    help="Session log. Recording is on by default; pass --no-db to skip")
    lv.add_argument("--no-db", action="store_true", help="Do not record this session")
    lv.add_argument("--adapter", choices=["fake", "typesafe"], default="typesafe")
    lv.add_argument("--trace", action="store_true",
                    help="Write a per-session log of every Jev request and response")
    lv.add_argument("--trace-dir", type=Path, default=None,
                    help="Where traces go (default: traces/)")
    lv.add_argument("--my-role", choices=["interviewer", "candidate"],
                    default="interviewer",
                    help="Which side of the interview YOU are on. Pass 'candidate' "
                         "for the half of a peer swap where they interview you, or "
                         "every role label in the recording is inverted.")

    args = parser.parse_args(argv)

    if args.command == "devices":
        from .live.capture import list_devices, measure_level

        devices = list_devices()
        if not args.check:
            print("avfoundation audio inputs:")
            for idx, name in devices:
                print(f"  [{idx}]  {name}")
            print("\nYour microphone is the interviewer. System audio (BlackHole) is the")
            print("candidate. Pass devices by NAME -- indexes shift when devices change.")
            print("Run 'cue devices --check' to see which ones are actually receiving audio.")
            return 0

        print("Listening to each device for 3s. Play audio through your call or a")
        print("video first, and speak, so both sides have something to hear.\n")
        silent = []
        for idx, name in devices:
            peak, rms = measure_level(idx)
            verdict = "AUDIO" if peak > 0.01 else "silent"
            if peak <= 0.01:
                silent.append(name)
            print(f"  [{idx}]  {name:<34} peak={peak:0.4f} rms={rms:0.4f}  {verdict}")
        if any("blackhole" in n.lower() for n in silent):
            print("\nBlackHole is silent. Your call audio is not routed to it, so the")
            print("candidate will never be transcribed. Set your OUTPUT device to the")
            print("Multi-Output Device that includes BlackHole (menu bar: option-click")
            print("the volume icon), not to your headset directly.")
        return 0

    if args.command == "signals":
        return _signals(args)

    if args.command == "traces":
        return _traces(args)

    if args.command == "label":
        from .label import run as run_label

        if not args.db.exists():
            print(f"{args.db} does not exist.", file=sys.stderr)
            return 2
        return run_label(args.db, args.session)

    if args.command == "gate":
        return _gate(args)

    if args.command == "board":
        from .board import BoardError, load_board
        from .board_vision import VisionError

        try:
            state = load_board(args.board)
        except (BoardError, VisionError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(state.to_jev_state(), indent=2) if args.json
              else state.render())
        return 0

    if args.command == "sessions":
        return _sessions(args)

    if args.command == "report":
        from .report import build, render, resolve_session_id

        if not args.db.exists():
            print(f"{args.db} does not exist. Sessions record automatically when you "
                  f"run 'cue live'.", file=sys.stderr)
            return 2

        sid = resolve_session_id(str(args.db), args.session)
        store = SessionStore(args.db)
        try:
            text = None if (args.refresh or args.board) else store.load_report(sid)
            if text is None:
                adapter = None
                if not args.no_coverage:
                    from .jev.typesafe import TypeSafeJevAdapter

                    adapter = TypeSafeJevAdapter()
                text = render(build(str(args.db), sid, adapter,
                                    str(args.board) if args.board else None))
                store.save_report(sid, text)
                print(f"generated and saved report for {sid}", file=sys.stderr)
            else:
                print(f"saved report for {sid} (--refresh to regenerate)", file=sys.stderr)
        finally:
            store.close()

        if args.out:
            args.out.write_text(text)
            print(f"wrote {args.out}", file=sys.stderr)
        else:
            print(text)
        return 0

    if args.command == "live":
        return _live(args)


    if args.command == "replay":
        if args.adapter == "typesafe":
            from .jev.typesafe import TypeSafeJevAdapter
            from .trace import Trace

            adapter = TypeSafeJevAdapter(
                trace=Trace.for_session("replay", args.trace_dir) if args.trace else None)
        else:
            adapter = FakeJevAdapter()
            print("using the FAKE adapter: output exercises the pipeline, not Jev", file=sys.stderr)

        store = SessionStore(args.db) if args.db else None
        try:
            result = replay(args.transcript, adapter, store=store, tick_ms=args.tick_ms)
            if args.json:
                if store is None:
                    print("--json requires --db", file=sys.stderr)
                    return 2
                print(json.dumps(store.export_json(result.session.id), indent=2))
            else:
                _print_report(result, show_suppressed=args.show_suppressed)
        finally:
            if store is not None:
                store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
