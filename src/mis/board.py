"""Whiteboard state extraction (Phase 3, structured source).

Converts an Excalidraw scene into the `components / relationships /
observations` shape the architecture doc specifies, before anything reaches
Jev. Jev is never asked to interpret raw pixels or raw scene JSON -- it judges
this summary.

Two things about Excalidraw scenes that the extraction has to handle, because
getting either wrong silently changes the answer:

* Deleted elements stay in the file with `isDeleted: true`, and a real board
  carries plenty of them: components drawn and then removed. Counting those as
  present would invent structure that is not there.
* A shape's label is bound to it (`containerId`) only when it was typed into
  the shape. Labels typed *next to* a shape are free-floating text that happens
  to sit inside its bounds, which is how most of these boards are drawn.
  Geometry, not just bindings, decides what a box is called.

Counts and adjacency are arithmetic and stay in code; Jev 1.13 cannot count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .jev.adapter import JevAdapter
from .models import Primitive, SignalDecision
from .signals import SignalSpec

#: Element types treated as components rather than annotation.
SHAPE_TYPES = frozenset({"rectangle", "ellipse", "diamond"})


class BoardError(ValueError):
    """Raised when a file cannot be read as an Excalidraw scene."""


@dataclass(frozen=True)
class Component:
    id: str
    label: str
    shape: str

    @property
    def is_labelled(self) -> bool:
        return not self.label.startswith("unlabelled ")


@dataclass(frozen=True)
class Relationship:
    source: str
    target: str
    label: str | None = None

    @property
    def is_dangling(self) -> bool:
        return self.source == "?" or self.target == "?"


@dataclass
class BoardState:
    """The summarized board. This, not the scene, is what Jev sees."""

    source_ref: str
    components: list[Component] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    #: Free text blocks that are not a component's label -- requirements,
    #: entity lists, key/value sketches, stray annotations.
    notes: list[str] = field(default_factory=list)
    deleted_count: int = 0
    #: How this state was produced: "excalidraw" (parsed from a scene file) or
    #: "image" (transcribed from a screenshot by a vision model). A report has
    #: to be able to say which, because a transcription can be wrong in ways a
    #: file parse cannot.
    extraction: str = "excalidraw"

    @property
    def orphans(self) -> list[Component]:
        """Components no arrow touches."""
        touched = set()
        for rel in self.relationships:
            touched.add(rel.source)
            touched.add(rel.target)
        return [c for c in self.components if c.label not in touched]

    @property
    def dangling_arrows(self) -> int:
        return sum(1 for r in self.relationships if r.is_dangling)

    def to_jev_state(self) -> dict:
        """Compact, typed state for a Jev request.

        Counts are resolved here rather than left for the model to derive:
        Jev cannot count, and multi-hop reasoning degrades it, so anything
        arithmetic arrives as its own named field.
        """
        return {
            "components": [c.label for c in self.components],
            "component_count": len(self.components),
            "connections": [
                f"{r.source} -> {r.target}" + (f" ({r.label})" if r.label else "")
                for r in self.relationships
            ],
            "connection_count": len(self.relationships),
            "unconnected_components": [c.label for c in self.orphans],
            "arrows_with_a_loose_end": self.dangling_arrows,
            "written_notes": self.notes,
        }

    def render(self) -> str:
        """Human-readable summary, in the architecture doc's example shape."""
        out = [f"components: {', '.join(c.label for c in self.components) or 'none'}"]
        out.append("relationships:")
        for r in self.relationships:
            arrow = f"  {r.source} -> {r.target}"
            out.append(arrow + (f"  [{r.label}]" if r.label else ""))
        if not self.relationships:
            out.append("  none")
        out.append("observations:")
        out.append(f"  {len(self.components)} components, "
                   f"{len(self.relationships)} connections")
        if self.orphans:
            out.append(f"  unconnected: {', '.join(c.label for c in self.orphans)}")
        if self.dangling_arrows:
            out.append(f"  {self.dangling_arrows} arrow(s) with a loose end")
        if self.deleted_count:
            out.append(f"  {self.deleted_count} element(s) drawn then deleted")
        if self.extraction == "image":
            out.append("  read from a screenshot; anything erased earlier is not visible")
        return "\n".join(out)


#: Board-only judgments, from the Phase 3 list in the planning doc. Each is a
#: bounded decision over the summarized board -- never over the scene JSON.
#: `written_notes` is included in the state because a requirements block that
#: says "out of scope: monitoring" is the difference between a gap and a
#: deliberate exclusion, and Jev reads criteria literally enough to use it.
BOARD_SIGNALS: tuple[SignalSpec, ...] = (
    SignalSpec(
        name="persistence_layer",
        primitive=Primitive.CHOICE,
        instructions=(
            "Looking at `components` and `connections`, how does this design handle "
            "storing data that must outlive a single request? Judge only what the "
            "board shows. A cache is not persistence unless the board treats it as "
            "the system of record."
        ),
        criteria={
            "drawn_as_a_component": (
                "A database, durable store or persistent log appears as its own "
                "component and something connects to it."
            ),
            "named_but_not_connected": (
                "A store appears on the board or in the notes but nothing connects "
                "to it, so its place in the flow is not shown."
            ),
            "absent": "Nothing on the board stores data durably.",
        },
    ),
    SignalSpec(
        name="data_flow_traceable",
        primitive=Primitive.CHOICE,
        instructions=(
            "Using `connections`, can a request be traced from where it enters the "
            "system to where it is stored or answered, without guessing? Treat "
            "`arrows_with_a_loose_end` and `unconnected_components` as breaks in the "
            "path. Judge the drawn path only; do not infer links that are not there."
        ),
        criteria={
            "complete": "Every entry point reaches a store or a response by arrows.",
            "partial": (
                "A main path is traceable but at least one branch stops short or is "
                "left unconnected."
            ),
            "unclear": (
                "The arrows do not establish a path from entry to storage; the "
                "components are largely floating."
            ),
        },
    ),
    SignalSpec(
        name="bottleneck_identified",
        primitive=Primitive.NOUL,
        instructions=(
            "Does the board name a specific place where this system saturates first "
            "-- a hot partition, a write path, a single component under the most "
            "load? Answer yes only for a named bottleneck. Drawing a queue, a cache "
            "or a shard key is a scaling mechanism, not the identification of a "
            "bottleneck, and counts as no on its own."
        ),
        criteria={
            "true": "A specific component or path is called out as the limiting one.",
            "false": (
                "Scaling machinery is drawn, but no specific saturation point is "
                "named."
            ),
        },
    ),
    SignalSpec(
        name="failure_handling_shown",
        primitive=Primitive.NOUL,
        instructions=(
            "Does the board show what happens when a component fails mid-operation "
            "-- retries, replicas, failover, dead-letter handling, an explicitly "
            "drawn fallback path? Answer yes only if it is on the board or in the "
            "written notes as a mechanism. A stated goal such as 'highly available' "
            "or 'fault tolerant' with no mechanism drawn counts as no."
        ),
        criteria={
            "true": "A concrete failure-handling mechanism appears.",
            "false": (
                "Reliability appears only as an aspiration in the requirements, with "
                "no mechanism shown."
            ),
        },
    ),
)


def evaluate_board(
    adapter: JevAdapter, board: BoardState, *, end_ms: int = 0
) -> list[SignalDecision]:
    """Run the board-only signals over one summarized board."""
    return adapter.evaluate(
        board.to_jev_state(), BOARD_SIGNALS, window_start_ms=0, window_end_ms=end_ms
    )


def _slug(label: str) -> str:
    keep = [ch.lower() if ch.isalnum() else "_" for ch in label]
    return "".join(keep).strip("_")[:48] or "component"


def evaluate_component_coverage(
    adapter: JevAdapter,
    board: BoardState,
    transcript: list[dict],
    *,
    end_ms: int = 0,
) -> dict[str, float]:
    """Per component: was it ever explained out loud?

    The planning doc's "component exists on the board but has not been
    explained verbally" signal. One Noul per component, batched over shared
    state. Fuzzy on purpose -- a box labelled `OffersListSvc` is explained by
    someone saying "the offers listing service", which a string match misses
    and Jev does not.

    Returns label -> probability it was explained. Labels that collide after
    slugging are asked once; the answer applies to both.
    """
    named = [c for c in board.components if c.is_labelled]
    if not named:
        return {}

    specs = []
    seen: set[str] = set()
    slugs: dict[str, str] = {}
    for component in named:
        slug = _slug(component.label)
        slugs[component.label] = slug
        if slug in seen:
            continue
        seen.add(slug)
        specs.append(
            SignalSpec(
                name=slug,
                primitive=Primitive.NOUL,
                instructions=(
                    f"The board has a component labelled '{component.label}'. Reading "
                    f"`transcript`, did the candidate ever explain what it does or why "
                    f"it is there -- in any words, not necessarily that label? Answer "
                    f"yes if they described its role, even briefly. Answer no if they "
                    f"only drew it, or merely said its name in passing without saying "
                    f"what it does."
                ),
                criteria={
                    "true": "The candidate said what this component does or why it exists.",
                    "false": (
                        "It was drawn and at most named, with no explanation of its "
                        "role."
                    ),
                },
            )
        )

    decisions = adapter.evaluate(
        {"transcript": transcript, "board_components": [c.label for c in named]},
        specs,
        window_start_ms=0,
        window_end_ms=end_ms,
    )
    by_slug = {d.signal_name: (d.probability or 0.0) for d in decisions}
    return {c.label: by_slug.get(slugs[c.label], 0.0) for c in named}


def _text_of(element: dict) -> str:
    return str(element.get("originalText") or element.get("text") or "").strip()


def _centre(element: dict) -> tuple[float, float]:
    return (
        float(element.get("x", 0)) + float(element.get("width", 0)) / 2,
        float(element.get("y", 0)) + float(element.get("height", 0)) / 2,
    )


def _contains(shape: dict, point: tuple[float, float]) -> bool:
    x, y = point
    sx, sy = float(shape.get("x", 0)), float(shape.get("y", 0))
    return (sx <= x <= sx + float(shape.get("width", 0))
            and sy <= y <= sy + float(shape.get("height", 0)))


#: How far outside a shape a label may sit and still belong to it. A multi-line
#: name typed against a narrow box overflows it, landing tens of pixels clear of
#: a rectangle only a few dozen wide. Dropping it would report a real, named
#: component as unlabelled.
NEAR_PX = 60.0


def _distance_to(shape: dict, point: tuple[float, float]) -> float:
    """Distance from a point to a shape's bounding box; 0 when inside."""
    x, y = point
    sx, sy = float(shape.get("x", 0)), float(shape.get("y", 0))
    ex, ey = sx + float(shape.get("width", 0)), sy + float(shape.get("height", 0))
    dx = max(sx - x, 0.0, x - ex)
    dy = max(sy - y, 0.0, y - ey)
    return (dx * dx + dy * dy) ** 0.5


def _label_from(text: str, limit: int = 40) -> str:
    """First meaningful line of a text block, as a component name."""
    for line in text.splitlines():
        line = line.strip().lstrip("*-• ").strip()
        if line:
            return line[:limit]
    return ""


def load_excalidraw(path: str | Path) -> BoardState:
    """Read an `.excalidraw` file into a summarized board state."""
    path = Path(path)
    try:
        scene = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise BoardError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(scene, dict) or "elements" not in scene:
        raise BoardError(f"{path}: no 'elements' key; is this an Excalidraw scene?")

    raw = scene["elements"]
    live = [e for e in raw if not e.get("isDeleted")]
    deleted = len(raw) - len(live)

    by_id = {e["id"]: e for e in live}
    shapes = [e for e in live if e.get("type") in SHAPE_TYPES]
    texts = [e for e in live if e.get("type") == "text" and _text_of(e)]

    # Pass 1: bound labels (text typed into a shape or onto an arrow).
    bound_label: dict[str, str] = {}
    consumed: set[str] = set()
    for t in texts:
        container = t.get("containerId")
        if container and container in by_id:
            bound_label[container] = _text_of(t)
            consumed.add(t["id"])

    # Pass 2: free text sitting inside a shape is that shape's label. Smallest
    # containing shape wins, so a label inside a box inside a region binds to
    # the box.
    inside_label: dict[str, str] = {}
    for t in texts:
        if t["id"] in consumed:
            continue
        point = _centre(t)
        holders = [s for s in shapes
                   if s["id"] not in bound_label and _contains(s, point)]
        if not holders:
            continue
        smallest = min(holders, key=lambda s: float(s.get("width", 0))
                       * float(s.get("height", 0)))
        # A shape can only take one free-text label; the first wins.
        if smallest["id"] not in inside_label:
            inside_label[smallest["id"]] = _text_of(t)
            consumed.add(t["id"])

    # Pass 3: a shape still without a label takes the nearest text just outside
    # it, for labels that overflow a narrow box.
    for s in shapes:
        if s["id"] in bound_label or s["id"] in inside_label:
            continue
        candidates = [
            (_distance_to(s, _centre(t)), t) for t in texts if t["id"] not in consumed
        ]
        candidates = [(d, t) for d, t in candidates if d <= NEAR_PX]
        if not candidates:
            continue
        _, nearest = min(candidates, key=lambda pair: pair[0])
        inside_label[s["id"]] = _text_of(nearest)
        consumed.add(nearest["id"])

    components: list[Component] = []
    label_by_id: dict[str, str] = {}
    for s in shapes:
        text = bound_label.get(s["id"]) or inside_label.get(s["id"]) or ""
        label = _label_from(text) or f"unlabelled {s['type']}"
        components.append(Component(id=s["id"], label=label, shape=s["type"]))
        label_by_id[s["id"]] = label

    # A free text element can itself act as a component when an arrow binds to
    # it -- several services on these boards are a bare label with no box.
    def resolve(element_id: str | None) -> str | None:
        if element_id is None:
            return None
        if element_id in label_by_id:
            return label_by_id[element_id]
        target = by_id.get(element_id)
        if target is None:
            return None
        if target.get("type") == "text":
            point = _centre(target)
            for s in shapes:
                if _contains(s, point) and s["id"] in label_by_id:
                    return label_by_id[s["id"]]
            return _label_from(_text_of(target))
        return None

    relationships: list[Relationship] = []
    for a in (e for e in live if e.get("type") == "arrow"):
        start = (a.get("startBinding") or {}).get("elementId")
        end = (a.get("endBinding") or {}).get("elementId")
        source = resolve(start) or "?"
        target = resolve(end) or "?"
        if source == "?" and target == "?":
            continue  # a free-floating arrow says nothing about structure
        label = bound_label.get(a["id"])
        relationships.append(
            Relationship(source=source, target=target,
                         label=_label_from(label) if label else None)
        )

    notes = [_text_of(t) for t in texts if t["id"] not in consumed]

    return BoardState(
        source_ref=str(path),
        components=components,
        relationships=relationships,
        notes=notes,
        deleted_count=deleted,
    )


#: Screenshot suffixes routed to the vision extractor.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})


def load_board(path: str | Path) -> BoardState:
    """Load a board from an Excalidraw export or a screenshot.

    The seam the architecture doc asks for: callers downstream of this take a
    `BoardState` and never learn which source produced it.
    """
    path = Path(path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        from .board_vision import load_image

        return load_image(path)
    return load_excalidraw(path)
