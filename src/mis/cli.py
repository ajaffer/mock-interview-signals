"""CLI for offline replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .jev.fake import FakeJevAdapter
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


def _live(args) -> int:
    import threading

    import uvicorn

    from .live.session import LiveSession, run_from_capture, run_from_transcript
    from .server import create_app

    if args.adapter == "typesafe":
        from .jev.typesafe import TypeSafeJevAdapter

        adapter = TypeSafeJevAdapter()
    else:
        adapter = FakeJevAdapter()
        print("FAKE adapter: the strip will move, but the judgments are keyword "
              "heuristics, not Jev.", file=sys.stderr)

    session = LiveSession(adapter=adapter, tick_ms=args.tick_ms)

    if args.simulate is not None:
        worker = threading.Thread(
            target=run_from_transcript, args=(session, args.simulate, args.speed),
            daemon=True, name="simulate")
    else:
        from .live.capture import DualCapture, list_devices
        from .live.transcribe import Transcriber

        if args.mic is None:
            print("--mic is required. Run: mis devices", file=sys.stderr)
            return 2
        if args.system is None:
            print("WARNING: no --system device. You will hear yourself but NOT the\n"
                  "         candidate, and three of five signals need their speech.\n"
                  "         Install BlackHole and pass its index.", file=sys.stderr)
        names = dict(list_devices())
        print(f"mic:    [{args.mic}] {names.get(args.mic, '?')}", file=sys.stderr)
        if args.system is not None:
            print(f"system: [{args.system}] {names.get(args.system, '?')}", file=sys.stderr)
        print(f"loading whisper '{args.model}' (first run downloads it)...", file=sys.stderr)
        transcriber = Transcriber(args.model)
        capture = DualCapture(args.mic, args.system)
        worker = threading.Thread(
            target=run_from_capture, args=(session, capture, transcriber),
            daemon=True, name="capture")

    app = create_app(session, worker)
    print(f"\n  ->  http://127.0.0.1:{args.port}\n", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mis", description="Mock interview signal replay")
    sub = parser.add_subparsers(dest="command", required=True)

    rp = sub.add_parser("replay", help="Replay a JSONL transcript")
    rp.add_argument("transcript", type=Path)
    rp.add_argument("--db", type=Path, default=None, help="SQLite session log path")
    rp.add_argument("--tick-ms", type=int, default=15_000)
    rp.add_argument("--show-suppressed", action="store_true")
    rp.add_argument("--json", action="store_true", help="Emit the session log as JSON")
    rp.add_argument(
        "--adapter",
        choices=["fake", "typesafe"],
        default="fake",
        help="fake = offline keyword stand-in, NOT evidence of signal quality",
    )

    sub.add_parser("devices", help="List audio input devices")

    lv = sub.add_parser("live", help="Listen and show live signals in a browser")
    lv.add_argument("--mic", type=int, default=None,
                    help="avfoundation index for YOUR microphone (interviewer)")
    lv.add_argument("--system", type=int, default=None,
                    help="avfoundation index for system audio (candidate), e.g. BlackHole")
    lv.add_argument("--simulate", type=Path, default=None,
                    help="Rehearse against a stored transcript instead of a microphone")
    lv.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier")
    lv.add_argument("--model", default="base.en", help="faster-whisper model size")
    lv.add_argument("--tick-ms", type=int, default=20_000)
    lv.add_argument("--port", type=int, default=8765)
    lv.add_argument("--adapter", choices=["fake", "typesafe"], default="typesafe")

    args = parser.parse_args(argv)

    if args.command == "devices":
        from .live.capture import list_devices

        print("avfoundation audio inputs:")
        for idx, name in list_devices():
            print(f"  [{idx}]  {name}")
        print("\nYour microphone is the interviewer. System audio (BlackHole) is the")
        print("candidate. No BlackHole in the list means only your own voice is heard.")
        return 0

    if args.command == "live":
        return _live(args)


    if args.command == "replay":
        if args.adapter == "typesafe":
            from .jev.typesafe import TypeSafeJevAdapter

            adapter = TypeSafeJevAdapter()
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
