"""Whiteboard extraction from a screenshot (ADR 008).

Some interview platforms do not let you export the board. A screenshot is all
there is, so this reads one into the same `BoardState` that
`board.load_excalidraw` produces. Everything downstream is identical from that
point on.

The division of labour matters and is the reason this does not violate the
project's central constraint. This module runs a **vision model as the board
state extractor**: it transcribes shapes, labels and arrows into structured
state and makes no judgement about the design. Jev then judges that structured
state, exactly as it does for an exported board. Jev is still never the
component looking at pixels, and the thing making bounded decisions is still
Jev.

The cost of that is a second model dependency, which the architecture doc
flags. Two consequences worth knowing when reading a report built this way:

* A transcription can be wrong in ways a file parse cannot. `BoardState.extraction`
  records which path produced the state so a report can say so, and the render
  marks it.
* `deleted_count` is always 0 here. A screenshot is a final frame; what was
  drawn and erased earlier leaves no trace, whereas an `.excalidraw` file keeps
  it. Absence of that number is not evidence that nothing was deleted.
"""

from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel, Field

from .board import BoardState, Component, Relationship

#: The extractor. Transcription quality is the whole value here, so this is not
#: a place to economise on model choice -- a mislabelled component silently
#: changes every board signal that follows.
DEFAULT_MODEL = "claude-opus-5"

MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

PROMPT = """This is a whiteboard from a system design interview.

Transcribe it. Do not evaluate it, do not judge whether the design is good, and
do not add components that would make it more complete. Record only what is
actually drawn.

- `components`: every box, circle or labelled node. Use the label exactly as
  written, including misspellings. A node with no readable label gets the label
  "unlabelled".
- `relationships`: every arrow, as source and target component labels, plus the
  arrow's own label if it has one. If one end of an arrow does not land on a
  component, use "?" for that end. Record the direction the arrowhead points.
- `notes`: every block of free text that is not a component label -- the
  problem statement, requirements lists, capacity numbers, schema sketches,
  annotations. Keep them as written, one entry per visually distinct block.

If part of the board is cut off or illegible, leave it out rather than guessing
at it."""


class _Component(BaseModel):
    label: str = Field(description="The node's label, exactly as written")
    shape: str = Field(description="rectangle, ellipse, diamond or other")


class _Relationship(BaseModel):
    source: str = Field(description="Label of the component the arrow starts at, or '?'")
    target: str = Field(description="Label of the component the arrow points to, or '?'")
    label: str | None = Field(default=None, description="The arrow's own label, if any")


class _Scene(BaseModel):
    components: list[_Component]
    relationships: list[_Relationship]
    notes: list[str]


class VisionError(RuntimeError):
    """Raised when a screenshot cannot be read into board state."""


def load_image(
    path: str | Path, *, model: str = DEFAULT_MODEL, client: object | None = None
) -> BoardState:
    """Read a whiteboard screenshot into the same state an export would give."""
    path = Path(path)
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise VisionError(
            f"{path}: unsupported image type {path.suffix!r}. "
            f"Supported: {', '.join(sorted(MEDIA_TYPES))}"
        )

    if client is None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise VisionError(
                "anthropic is not installed. Install the 'vision' extra: "
                "pip install -e '.[vision]'"
            ) from exc
        client = anthropic.Anthropic()

    encoded = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    response = client.messages.parse(
        model=model,
        max_tokens=16_000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        output_format=_Scene,
    )
    scene = response.parsed_output
    if scene is None:
        raise VisionError(f"{path}: the model returned no parsable board state")

    return BoardState(
        source_ref=str(path),
        components=[
            # Ids are synthetic here; a screenshot has no element identity.
            Component(id=f"v{i}", label=c.label.strip() or "unlabelled rectangle",
                      shape=c.shape)
            for i, c in enumerate(scene.components)
        ],
        relationships=[
            Relationship(source=r.source.strip(), target=r.target.strip(),
                         label=(r.label or "").strip() or None)
            for r in scene.relationships
        ],
        notes=[n.strip() for n in scene.notes if n.strip()],
        deleted_count=0,
        extraction="image",
    )
