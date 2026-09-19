"""Typed contracts for the signal pipeline.

The one asymmetry worth knowing before reading anything else: TypeSafe's Noul
primitive returns a probability and *no* confidence value -- the probability is
the confidence. Choice and Score do carry a separate confidence. `SignalDecision`
therefore cannot assume both numbers are present, and `confidence` is None for
every Noul-backed signal. See ADR 005 in docs/jev-architecture.md.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

SIGNAL_SET_VERSION = "v0.1"


class Speaker(StrEnum):
    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"


class Phase(StrEnum):
    REQUIREMENTS = "requirements"
    HIGH_LEVEL_DESIGN = "high_level_design"
    DATA_MODEL = "data_model"
    SCALING = "scaling"
    TRADEOFFS = "tradeoffs"
    WRAP_UP = "wrap_up"
    UNKNOWN = "unknown"


#: Phases in the order a system design interview normally moves through.
#: Used for "has the session reached at least phase X" gates. UNKNOWN is not
#: part of the ordering and never satisfies a gate.
PHASE_ORDER: tuple[Phase, ...] = (
    Phase.REQUIREMENTS,
    Phase.HIGH_LEVEL_DESIGN,
    Phase.DATA_MODEL,
    Phase.SCALING,
    Phase.TRADEOFFS,
    Phase.WRAP_UP,
)


class Primitive(StrEnum):
    """TypeSafe primitive backing a signal. Determines the answer shape."""

    CHOICE = "choice"
    NOUL = "noul"
    SCORE = "score"


class SourceType(StrEnum):
    REPLAY = "replay"
    LIVE = "live"


class Utterance(BaseModel):
    """One line of a source transcript, as imported from JSONL."""

    model_config = ConfigDict(frozen=True)

    t_ms: int = Field(ge=0)
    speaker: Speaker
    text: str


class TranscriptChunk(BaseModel):
    """An utterance as consumed by a specific session run.

    Utterance-level chunking is deliberate: it is the closest offline analogue
    to how a live transcriber emits updates, which keeps replay and live mode on
    the same downstream path (ADR 002).
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    offset_ms: int = Field(ge=0)
    speaker: Speaker
    text: str

    @property
    def estimated_duration_ms(self) -> int:
        """How long this utterance took to say, estimated from word count.

        Transcript timestamps mark when an utterance *started*, so a long
        turn looks instantaneous if you only diff offsets -- which silently
        starved every duration gate. Conversational speech runs about 150
        words per minute. Live mode replaces this with real timing.
        """
        words = len(self.text.split())
        return max(1_000, round(words / 2.5 * 1_000))

    @property
    def is_question(self) -> bool:
        """Whether this reads as a question. Code-side detection on purpose.

        Jev degrades on multi-hop reasoning, so the interviewer's outstanding
        question is resolved here and passed to the model as an explicit named
        state field rather than being something the model must first locate.
        """
        stripped = self.text.strip()
        if stripped.endswith("?"):
            return True
        opener = stripped.lower().split()
        if not opener:
            return False
        return opener[0] in {
            "what", "how", "why", "when", "where", "which", "who",
            "can", "could", "would", "will", "do", "does", "did",
            "is", "are", "was", "were", "should", "tell", "walk", "talk",
        }


class SignalDecision(BaseModel):
    """One Jev answer, normalized across the three primitive shapes."""

    signal_name: str
    signal_version: str = SIGNAL_SET_VERSION
    primitive: Primitive

    #: Choice -> selected option; Score -> weighted level; Noul -> probability of yes.
    value: str | float
    #: Probability of yes. Noul only; None for Choice and Score.
    probability: float | None = None
    #: Distribution concentration. Choice and Score only; ALWAYS None for Noul.
    confidence: float | None = None
    #: Full distribution when the primitive provides one.
    probabilities: dict[str, float] | None = None

    visible: bool = False
    #: Why the policy hid this. None when visible.
    suppressed_reason: str | None = None

    window_start_ms: int
    window_end_ms: int
    latency_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, _context: object) -> None:
        if self.primitive is Primitive.NOUL and self.confidence is not None:
            raise ValueError(
                f"{self.signal_name}: Noul answers carry no confidence value; "
                "its probability is its confidence"
            )
        if self.primitive is Primitive.NOUL and self.probability is None:
            raise ValueError(f"{self.signal_name}: Noul answers must carry a probability")


class Session(BaseModel):
    """A single replay or live run."""

    id: str
    interview_type: str = "system_design"
    source_type: SourceType = SourceType.REPLAY
    #: Path or identifier of the source transcript. None for live sessions.
    source_ref: str | None = None
    #: Content hash of the source file. Without it, editing a fixture silently
    #: invalidates comparisons against earlier runs (ADR 004).
    source_sha256: str | None = None
    signal_set_version: str = SIGNAL_SET_VERSION
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    notes: str | None = None
