from __future__ import annotations

import json

import pytest

from mis.board import BoardError, load_excalidraw


def _shape(eid, x, y, w, h, kind="rectangle", **kw):
    return {"id": eid, "type": kind, "x": x, "y": y, "width": w, "height": h, **kw}


def _text(eid, x, y, text, w=80, h=25, **kw):
    return {"id": eid, "type": "text", "x": x, "y": y, "width": w, "height": h,
            "text": text, "originalText": text, **kw}


def _arrow(eid, start, end, **kw):
    return {
        "id": eid, "type": "arrow", "x": 0, "y": 0, "width": 10, "height": 10,
        "startBinding": {"elementId": start} if start else None,
        "endBinding": {"elementId": end} if end else None,
        **kw,
    }


def _write(tmp_path, elements, name="b.excalidraw"):
    p = tmp_path / name
    p.write_text(json.dumps({"type": "excalidraw", "version": 2, "elements": elements}))
    return p


def test_deleted_elements_are_not_components(tmp_path):
    board = load_excalidraw(_write(tmp_path, [
        _shape("a", 0, 0, 100, 50),
        _text("at", 10, 10, "Kept", containerId="a"),
        _shape("b", 200, 0, 100, 50, isDeleted=True),
        _text("bt", 210, 10, "Removed", containerId="b", isDeleted=True),
    ]))
    assert [c.label for c in board.components] == ["Kept"]
    assert board.deleted_count == 2


def test_bound_text_labels_its_shape(tmp_path):
    board = load_excalidraw(_write(tmp_path, [
        _shape("a", 0, 0, 100, 50),
        _text("at", 10, 10, "API GW\n* Routing", containerId="a"),
    ]))
    # Only the first meaningful line becomes the name.
    assert [c.label for c in board.components] == ["API GW"]
    assert board.notes == []


def test_free_text_inside_a_shape_labels_it(tmp_path):
    board = load_excalidraw(_write(tmp_path, [
        _shape("a", 0, 0, 200, 100),
        _text("t", 20, 20, "Dashboard service"),
    ]))
    assert [c.label for c in board.components] == ["Dashboard service"]


def test_smallest_containing_shape_wins(tmp_path):
    board = load_excalidraw(_write(tmp_path, [
        _shape("outer", 0, 0, 400, 400),
        _shape("inner", 50, 50, 100, 60),
        _text("t", 60, 60, "Inner"),
    ]))
    labels = {c.id: c.label for c in board.components}
    assert labels["inner"] == "Inner"
    assert labels["outer"].startswith("unlabelled")


def test_label_overflowing_a_narrow_box_still_binds(tmp_path):
    """A long name typed against a thin box lands outside its bounds."""
    board = load_excalidraw(_write(tmp_path, [
        _shape("a", 0, 0, 40, 200),
        _text("t", 10, 90, "AuthenticationSvc", w=180, h=25),
    ]))
    assert [c.label for c in board.components] == ["AuthenticationSvc"]


def test_distant_text_is_a_note_not_a_label(tmp_path):
    board = load_excalidraw(_write(tmp_path, [
        _shape("a", 0, 0, 40, 40),
        _text("t", 900, 900, "Functional Requirements"),
    ]))
    assert board.components[0].label.startswith("unlabelled")
    assert board.notes == ["Functional Requirements"]


def test_arrows_become_relationships_with_labels(tmp_path):
    board = load_excalidraw(_write(tmp_path, [
        _shape("a", 0, 0, 100, 50),
        _text("at", 10, 10, "Client", containerId="a"),
        _shape("b", 300, 0, 100, 50),
        _text("bt", 310, 10, "Service", containerId="b"),
        _arrow("r", "a", "b"),
        _text("rt", 150, 10, "click", containerId="r"),
    ]))
    assert len(board.relationships) == 1
    rel = board.relationships[0]
    assert (rel.source, rel.target, rel.label) == ("Client", "Service", "click")
    assert board.orphans == []


def test_arrow_with_one_loose_end_is_dangling(tmp_path):
    board = load_excalidraw(_write(tmp_path, [
        _shape("a", 0, 0, 100, 50),
        _text("at", 10, 10, "Gateway", containerId="a"),
        _arrow("r", "a", None),
    ]))
    assert board.dangling_arrows == 1
    assert board.relationships[0].target == "?"


def test_fully_unbound_arrow_is_ignored(tmp_path):
    board = load_excalidraw(_write(tmp_path, [_arrow("r", None, None)]))
    assert board.relationships == []


def test_untouched_component_is_an_orphan(tmp_path):
    board = load_excalidraw(_write(tmp_path, [
        _shape("a", 0, 0, 100, 50),
        _text("at", 10, 10, "Connected", containerId="a"),
        _shape("b", 300, 0, 100, 50),
        _text("bt", 310, 10, "Floating", containerId="b"),
        _shape("c", 600, 0, 100, 50),
        _text("ct", 610, 10, "Other", containerId="c"),
        _arrow("r", "a", "c"),
    ]))
    assert [c.label for c in board.orphans] == ["Floating"]


def test_arrow_bound_to_a_bare_text_resolves_to_that_text(tmp_path):
    """Several services on real boards are a label with no box around it."""
    board = load_excalidraw(_write(tmp_path, [
        _shape("a", 0, 0, 100, 50),
        _text("at", 10, 10, "Queue", containerId="a"),
        _text("free", 900, 900, "Flink"),
        _arrow("r", "a", "free"),
    ]))
    assert board.relationships[0].target == "Flink"


def test_jev_state_resolves_counts_in_code(tmp_path):
    board = load_excalidraw(_write(tmp_path, [
        _shape("a", 0, 0, 100, 50),
        _text("at", 10, 10, "A", containerId="a"),
        _shape("b", 300, 0, 100, 50),
        _text("bt", 310, 10, "B", containerId="b"),
        _arrow("r", "a", "b"),
    ]))
    state = board.to_jev_state()
    assert state["component_count"] == 2
    assert state["connection_count"] == 1
    assert state["arrows_with_a_loose_end"] == 0
    assert state["connections"] == ["A -> B"]


def test_rejects_a_file_that_is_not_a_scene(tmp_path):
    p = tmp_path / "x.excalidraw"
    p.write_text(json.dumps({"nope": True}))
    with pytest.raises(BoardError, match="no 'elements' key"):
        load_excalidraw(p)


def test_rejects_invalid_json(tmp_path):
    p = tmp_path / "x.excalidraw"
    p.write_text("{not json")
    with pytest.raises(BoardError, match="not valid JSON"):
        load_excalidraw(p)


def test_peer_swap_inverts_which_stream_is_which():
    """The second half of a peer swap records the operator as the candidate."""
    from mis.live.capture import DualCapture
    from mis.models import Speaker

    default = DualCapture(mic_device=0, system_device=1)
    assert [s.speaker for s in default.streams] == [
        Speaker.INTERVIEWER, Speaker.CANDIDATE,
    ]

    swapped = DualCapture(mic_device=0, system_device=1,
                          mic_speaker=Speaker.CANDIDATE)
    assert [s.speaker for s in swapped.streams] == [
        Speaker.CANDIDATE, Speaker.INTERVIEWER,
    ]


# -- screenshot extraction ------------------------------------------------

class _FakeParsed:
    def __init__(self, parsed):
        self.parsed_output = parsed


class _FakeMessages:
    def __init__(self, parsed):
        self._parsed = parsed
        self.calls = []

    def parse(self, **kw):
        self.calls.append(kw)
        return _FakeParsed(self._parsed)


class _FakeClient:
    def __init__(self, parsed):
        self.messages = _FakeMessages(parsed)


def _scene(components, relationships, notes=()):
    from mis.board_vision import _Component, _Relationship, _Scene

    return _Scene(
        components=[_Component(label=lbl, shape=shape) for lbl, shape in components],
        relationships=[_Relationship(source=s, target=t, label=lab)
                       for s, t, lab in relationships],
        notes=list(notes),
    )


def test_screenshot_becomes_the_same_board_state(tmp_path):
    from mis.board_vision import load_image

    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    client = _FakeClient(_scene(
        components=[("API Gateway", "rectangle"), ("Comment Service", "rectangle")],
        relationships=[("API Gateway", "Comment Service", "post comment")],
        notes=["10M DAU"],
    ))

    board = load_image(png, client=client)
    assert [c.label for c in board.components] == ["API Gateway", "Comment Service"]
    assert board.relationships[0].label == "post comment"
    assert board.notes == ["10M DAU"]
    # A screenshot cannot show what was erased before it was taken.
    assert board.deleted_count == 0
    assert board.extraction == "image"
    assert "screenshot" in board.render()


def test_screenshot_sends_the_image_as_base64(tmp_path):
    import base64

    from mis.board_vision import load_image

    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG bytes")
    client = _FakeClient(_scene([("A", "rectangle")], []))

    load_image(png, client=client)
    content = client.messages.calls[0]["messages"][0]["content"]
    image = next(b for b in content if b["type"] == "image")
    assert image["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(image["source"]["data"]) == b"\x89PNG bytes"


def test_loose_arrow_end_survives_transcription(tmp_path):
    from mis.board_vision import load_image

    png = tmp_path / "shot.png"
    png.write_bytes(b"x")
    client = _FakeClient(_scene([("Gateway", "rectangle")], [("Gateway", "?", None)]))

    board = load_image(png, client=client)
    assert board.dangling_arrows == 1


def test_unsupported_image_type_is_refused(tmp_path):
    from mis.board_vision import VisionError, load_image

    bad = tmp_path / "board.tiff"
    bad.write_bytes(b"x")
    with pytest.raises(VisionError, match="unsupported image type"):
        load_image(bad, client=_FakeClient(_scene([], [])))


def test_load_board_dispatches_on_suffix(tmp_path):
    from mis.board import load_board

    scene = tmp_path / "b.excalidraw"
    scene.write_text(json.dumps({"elements": [
        _shape("a", 0, 0, 100, 50),
        _text("at", 10, 10, "Parsed", containerId="a"),
    ]}))
    board = load_board(scene)
    assert board.extraction == "excalidraw"
    assert [c.label for c in board.components] == ["Parsed"]


# -- transport controls ---------------------------------------------------

def _session():
    from mis.jev.fake import FakeJevAdapter
    from mis.live.session import LiveSession
    return LiveSession(adapter=FakeJevAdapter(), tick_ms=20_000)


def test_standby_discards_audio_until_started():
    """Launching early is in the runbook; that small talk is not the interview."""
    from mis.models import Speaker

    s = _session()
    assert s.mode == "standby"
    s.advance(5_000)
    s.add_utterance(Speaker.INTERVIEWER, "just waiting for them to join", 5_000)
    assert s.state.now_ms == 0
    assert len(s.state.chunks) == 0


def test_start_zeroes_the_clock_at_the_moment_you_press_it():
    from mis.models import Speaker

    s = _session()
    s.advance(90_000)          # 90s of pre-interview idling
    s.start()
    s.add_utterance(Speaker.INTERVIEWER, "shall we begin", 90_000)
    assert s.state.chunks[-1].offset_ms == 0


def test_paused_time_is_not_interview_time():
    from mis.models import Speaker

    s = _session()
    s.advance(0)
    s.start()
    s.advance(10_000)
    assert s.state.now_ms == 10_000

    s.pause()
    s.advance(70_000)          # a minute of break
    assert s.state.now_ms == 10_000, "clock must not move while paused"
    s.add_utterance(Speaker.CANDIDATE, "back in a sec", 70_000)
    assert len(s.state.chunks) == 0, "audio during a pause is discarded"

    s.start()
    s.advance(75_000)
    # 10s before the pause + 5s after it, not 75s of wall clock.
    assert s.state.now_ms == 15_000


def test_stop_is_terminal():
    s = _session()
    s.advance(0)
    s.start()
    s.stop()
    assert s.mode == "stopped"
    s.start()
    assert s.mode == "stopped", "start must not revive a stopped session"


# -- Jev traffic tracing --------------------------------------------------

class _FakeNoul:
    def __init__(self, v): self.noul = v


class _FakeResult:
    model = "jev-1.13.0"
    def __init__(self):
        self.nouls = {"answered_question": _FakeNoul(0.08)}
        self.choices, self.scores = {}, {}
        self.usage = type("U", (), {"input_tokens": 1809, "output_tokens": 123})()


class _FakeTSClient:
    def system_one(self, state, questions):
        return _FakeResult()


def _answered_spec():
    from mis.signals import SIGNAL_SET
    return [s for s in SIGNAL_SET if s.name == "answered_question"]


def test_trace_records_both_directions(tmp_path):
    from mis.jev.typesafe import TypeSafeJevAdapter

    path = tmp_path / "jev.trace.jsonl"
    a = TypeSafeJevAdapter(client=_FakeTSClient(), trace_path=path)
    a.evaluate({"recent_transcript": [{"speaker": "candidate", "text": "hi"}]},
               _answered_spec(), window_start_ms=0, window_end_ms=20_000)

    rec = json.loads(path.read_text().strip())
    assert rec["request"]["state"]["recent_transcript"][0]["text"] == "hi"
    assert rec["request"]["questions"]["answered_question"]["type"] == "noul"
    assert "instructions" in rec["request"]["questions"]["answered_question"]
    assert rec["response"]["answered_question"] == {"type": "noul", "noul": 0.08}
    # A Noul has no confidence; the trace must not invent one.
    assert "confidence" not in rec["response"]["answered_question"]
    assert rec["usage"]["request_input_tokens"] == 1809
    assert rec["cost_cents"] > 0


def test_trace_records_failures_too(tmp_path):
    """The call you most want logged is the one that broke."""
    from mis.jev.typesafe import TypeSafeJevAdapter

    class Boom:
        def system_one(self, *a, **k):
            raise RuntimeError("upstream exploded")

    path = tmp_path / "jev.trace.jsonl"
    a = TypeSafeJevAdapter(client=Boom(), trace_path=path)
    with pytest.raises(RuntimeError):
        a.evaluate({}, _answered_spec(), window_start_ms=0, window_end_ms=1)

    rec = json.loads(path.read_text().strip())
    assert "upstream exploded" in rec["error"]
    assert "request" in rec


def test_no_trace_file_unless_asked(tmp_path):
    from mis.jev.typesafe import TypeSafeJevAdapter

    a = TypeSafeJevAdapter(client=_FakeTSClient())
    a.evaluate({}, _answered_spec(), window_start_ms=0, window_end_ms=1)
    assert list(tmp_path.iterdir()) == []
