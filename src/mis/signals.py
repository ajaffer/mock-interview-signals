"""Signal set v0.1 -- the Phase 0 spec expressed as code.

Each spec carries the question text sent to Jev and a `precondition` that decides
whether the question is worth asking at all. Preconditions run in code because
they are arithmetic and time comparison, which Jev 1.13 does not do reliably; a
signal whose precondition fails is omitted from the request entirely rather than
asked and discarded (ADR 006).

Instructions and criteria state their boundary cases explicitly -- Jev reads
instructions literally, so the exclusions ("thinking out loud is not rambling")
have to live in the text sent to the model, not in a comment here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .models import Phase, Primitive
from .state import QUESTION_RETAIN_MS, WINDOW_MS, RollingState

Precondition = Callable[[RollingState], str | None]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    primitive: Primitive
    instructions: str
    criteria: dict[str, str | None] | list[str] | None = None
    precondition: Precondition = lambda _state: None

    def should_ask(self, state: RollingState) -> str | None:
        """Return a skip reason, or None if the question should be asked."""
        return self.precondition(state)


# -- preconditions --------------------------------------------------------

COLD_START_MS = 60_000
FAIR_CHANCE_MS = 20_000
MIN_CANDIDATE_SPEECH_MS = 90_000
MIN_CONTINUOUS_MS = 45_000


def _phase_precondition(state: RollingState) -> str | None:
    """After the first minute. Before that there is only greeting and audio checks."""
    if state.now_ms < COLD_START_MS:
        return "cold_start"
    return None


def _answered_precondition(state: RollingState) -> str | None:
    """The last interviewer question is between 20 and 120 seconds old.

    Not "still open": nothing tracks whether it was answered. The question is
    simply the most recent one the interviewer asked, and it ages out on a
    timer rather than on being addressed. So for up to two minutes after any
    question, this gate passes regardless of what has happened since.
    """
    elapsed = state.ms_since_interviewer_question
    if elapsed is None:
        return "no_recent_question"
    if elapsed < FAIR_CHANCE_MS:
        return "fair_chance_window"
    return None


def _clarity_precondition(state: RollingState) -> str | None:
    """At least 90 seconds of candidate speech in the window to judge."""
    if state.candidate_speech_ms_in_window < MIN_CANDIDATE_SPEECH_MS:
        return "insufficient_candidate_speech"
    return None


def _rambling_precondition(state: RollingState) -> str | None:
    """The candidate is 45 seconds into an uninterrupted stretch."""
    if state.continuous_candidate_ms < MIN_CONTINUOUS_MS:
        return "short_turn"
    return None


def _tradeoff_precondition(state: RollingState) -> str | None:
    """The session has reached high level design."""
    if not state.has_reached(Phase.HIGH_LEVEL_DESIGN):
        return "phase_not_reached"
    return None


# -- the five signals -----------------------------------------------------

CURRENT_PHASE = SignalSpec(
    name="current_phase",
    primitive=Primitive.CHOICE,
    instructions=(
        "Which part of a system design interview is happening in the most recent "
        "part of `recent_transcript`? Judge only what is being discussed now, not "
        "what was discussed earlier in the window."
    ),
    criteria={
        "requirements": (
            "Clarifying scope, users, scale, constraints, or what the system must do."
        ),
        "high_level_design": (
            "Naming components and how they connect; the overall shape of the system."
        ),
        "data_model": "Schemas, entities, keys, or how data is stored and structured.",
        "scaling": "Load, bottlenecks, sharding, caching, replication, or failure under growth.",
        "tradeoffs": "Weighing named alternatives against each other and justifying a choice.",
        "wrap_up": "Summarizing, naming risks or next steps, or closing the session.",
        "unknown": (
            "Greetings, audio or tool checks, scheduling, or anything not yet part of "
            "the interview itself. Choose this rather than forcing a phase."
        ),
    },
    precondition=_phase_precondition,
)

ANSWERED_QUESTION = SignalSpec(
    name="answered_question",
    primitive=Primitive.NOUL,
    instructions=(
        "The interviewer asked `latest_interviewer_question`. Considering what the "
        "candidate has said since, has the candidate addressed that specific question? "
        "Answer yes if they addressed it, even partially or imperfectly. Answer no if "
        "they spoke about a related but different topic, or did not engage with it."
    ),
    criteria={
        "true": "The candidate engaged with the question that was actually asked.",
        "false": (
            "The candidate spoke fluently about something adjacent but never addressed "
            "the specific question asked."
        ),
    },
    precondition=_answered_precondition,
)

CLARITY = SignalSpec(
    name="clarity",
    primitive=Primitive.SCORE,
    instructions=(
        "How easy is the candidate to follow in `recent_transcript`? Judge only how "
        "followable the explanation is. Do not judge whether the design is correct or "
        "good -- a clearly explained wrong design is clear and belongs at the top."
    ),
    criteria=[
        "Jumping between topics with no stated connection; a listener cannot tell what "
        "problem is being solved.",
        "Individual statements make sense but their order does not; the listener has to "
        "reconstruct the thread themselves.",
        "A recognizable structure is being followed, but with unexplained jumps or terms "
        "introduced without grounding.",
        "A clear sequence with stated transitions; each claim connects to the ones before it.",
        "States where it is going, goes there, and closes the loop; a listener could "
        "summarize it accurately afterwards.",
    ],
    precondition=_clarity_precondition,
)

RAMBLING_RISK = SignalSpec(
    name="rambling_risk",
    primitive=Primitive.NOUL,
    instructions=(
        "Has the candidate drifted away from the thread they started in "
        "`recent_transcript`? Answer yes only for drift: circling the same point, or "
        "moving onto tangents that do not serve the question at hand. Thinking out loud "
        "and working through a problem step by step is expected in a system design "
        "interview and is not drift. A long answer that stays on its thread is not drift."
    ),
    criteria={
        "true": (
            "The candidate has left the thread they started -- unprompted tangents, or "
            "restating the same point without adding to it."
        ),
        "false": (
            "The candidate is still working the thread, including thinking out loud, "
            "estimating, or reasoning step by step at length."
        ),
    },
    precondition=_rambling_precondition,
)

#: CUT in v0.2. Inverted against human judgment across five real runs, and all
#: three rewrites inverted too. Kept defined for the record and for anyone
#: re-testing it; deliberately absent from SIGNAL_SET. See the evidence section
#: in docs/phase-0-signal-spec.md.
TRADEOFF_COVERAGE = SignalSpec(
    name="tradeoff_coverage",
    primitive=Primitive.NOUL,
    instructions=(
        "In `recent_transcript`, is the candidate weighing alternatives against each "
        "other -- naming more than one option, comparing them, and justifying which they "
        "chose? Answer no if they are describing a single design without considering "
        "what else they could have done, even if the description is detailed."
    ),
    criteria={
        "true": "Named alternatives are compared and a choice among them is justified.",
        "false": "A single approach is narrated with no alternatives considered.",
    },
    precondition=_tradeoff_precondition,
)



#: v0.3. The only signal in the set that came from a user describing a real
#: problem rather than from reasoning about what might help. After running a
#: mock, the interviewer's complaint was that the candidate "was reluctant to go
#: deep and kept answering in just voice and was avoiding drawing to go in
#: deeper". `answered_question` scored all 23 of those exchanges as answered --
#: correctly, by its own wording, which only asks whether a question was
#: addressed.
#:
#: The artifact half of that complaint (not drawing) is invisible to audio and
#: needs whiteboard state. The verbal half is not.
ANSWER_DEPTH = SignalSpec(
    name="answer_depth",
    primitive=Primitive.NOUL,
    instructions=(
        "The interviewer asked `latest_interviewer_question`. Is the candidate's "
        "response so far STAYING SHALLOW -- describing in general terms what they "
        "would do, without saying what specifically, how, or why? Answer yes if the "
        "response is mostly intent and naming ('we'd need some kind of queue', 'we "
        "would scale that horizontally', 'we'd use some smart merge') with no "
        "mechanism, no named component, no numbers, and no reasoning. Answer no if "
        "they give specifics: a named technology with a reason, a concrete data "
        "shape, an actual number, or a described mechanism. Judge depth only. A "
        "short answer that is specific is NOT shallow. A long answer that names "
        "nothing concrete IS shallow."
    ),
    criteria={
        "true": (
            "Generalities and intent only -- what they would do, with no mechanism, "
            "named component, number, or reason given."
        ),
        "false": (
            "Specifics present: a named choice with a reason, a concrete mechanism, "
            "a data shape, or a number."
        ),
    },
    precondition=_answered_precondition,
)

#: Everything that changes what Jev is shown or when it is asked. Named
#: explicitly rather than hashing source, so a comment or a docstring edit does
#: not trip the pin while a threshold change does.
_BEHAVIOURAL_CONSTANTS = {
    "COLD_START_MS": COLD_START_MS,
    "FAIR_CHANCE_MS": FAIR_CHANCE_MS,
    "MIN_CANDIDATE_SPEECH_MS": MIN_CANDIDATE_SPEECH_MS,
    "MIN_CONTINUOUS_MS": MIN_CONTINUOUS_MS,
    "QUESTION_RETAIN_MS": QUESTION_RETAIN_MS,
    "WINDOW_MS": WINDOW_MS,
}


def fingerprint(specs: Sequence[SignalSpec] = ()) -> str:
    """A hash of every word sent to Jev.

    The questions are versioned artifacts, not configuration. `signal_version`
    is stamped on every stored decision so a replay from weeks ago stays
    comparable to one from today, and that guarantee is worthless if the
    wording can drift while the version stays put.

    A test pins this value. Reword a question, add a criterion, change a
    primitive, and the test fails and tells you to bump SIGNAL_SET_VERSION.
    That is the intended workflow, not an obstacle: changing what Jev is asked
    should be a deliberate act with a version attached.
    """
    h = hashlib.sha256()
    # The gates decide which ticks produce a judgment at all, and the window
    # decides how much transcript each one sees. Moving either changes the data
    # collected just as surely as rewording a question does, so both are pinned.
    for name, value in sorted(_BEHAVIOURAL_CONSTANTS.items()):
        h.update(f"{name}={value}".encode())
    for spec in (specs or SIGNAL_SET):
        h.update(spec.name.encode())
        h.update(spec.primitive.value.encode())
        h.update(spec.instructions.encode())
        h.update(json.dumps(spec.criteria, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


SIGNAL_SET: tuple[SignalSpec, ...] = (
    CURRENT_PHASE,
    ANSWERED_QUESTION,
    CLARITY,
    RAMBLING_RISK,
    ANSWER_DEPTH,
)

BY_NAME: dict[str, SignalSpec] = {s.name: s for s in SIGNAL_SET}
