# mock-interview-signals

An interviewer-side signal layer for human-led mock interviews. Transcript state goes in;
a small set of bounded, typed judgments comes out, enough for an interviewer to decide
whether to probe, redirect, or move on, while the interview is still running.

Live signals are for the interviewer only. The candidate never sees them.

**[Watch it run](https://ajaffer.github.io/interview-signals.html)**, a replay of a
recorded session: real Jev decisions over a synthetic transcript, no API calls.

Offline replay, live audio capture, whiteboard extraction and the post-session
evidence pack all work today.

## What is actually proven

Stated up front because it is the part most likely to be overclaimed.

| Signal | Evidence |
|---|---|
| `current_phase` | Holds up across six interviews. The one the interviewer reports using. |
| `clarity` | Reasonable, evidence across several sessions. |
| `answer_depth` | New and unvalidated. Treat as a hypothesis. |
| `answered_question` | **Known to miss.** Scored every exchange in one session as answered when at least two were not. |
| `rambling_risk` | Fires rarely. A written prediction that it never would was falsified. |

A usefulness gate on 2026-09-20 **failed**, not because a signal was wrong, but because
it was right about something the interviewer had already noticed. Accuracy is not
usefulness, and the gate is written to measure the second. `cue label` and `cue gate`
exist to settle it with data rather than recollection.

Measured on a real 10-minute session: 119 judgments, 8 shown (**93% suppressed**), 81 of
200 questions never sent because a precondition blocked them, **$0.0025** total (a quarter of a cent), **119ms**
median per batched call.

## Try it

Three levels, by how much setup you are willing to do.

**1. Watch a recorded session.** Nothing to install:
[ajaffer.github.io/interview-signals.html](https://ajaffer.github.io/interview-signals.html)

**2. Run the real thing against a recorded transcript.** Five minutes, no API key, no
microphone, any operating system:

```bash
git clone https://github.com/ajaffer/mock-interview-signals.git
cd mock-interview-signals
python3 -m venv .venv && ./.venv/bin/pip install -e ".[live]"

./.venv/bin/cue live --simulate transcripts/fixtures/sample-001-url-shortener.jsonl \
  --speed 10 --adapter fake
```

Open <http://127.0.0.1:8765>, press **Start**, and watch it run at ten times speed.
`--adapter fake` substitutes keyword heuristics for the model, so this needs no
credentials. The judgments are not real, but the pipeline, the suppression and the
end-of-session summary all are.

Swap in `--adapter typesafe` with a `TYPESAFE_API_KEY` for real judgments over the same
transcript, for about a cent.

**3. Use it in an actual interview.** macOS only, and the audio setup is the fiddly part:

```bash
./.venv/bin/pip install -e ".[all]"
brew install blackhole-2ch          # then build a Multi-Output Device
export TYPESAFE_API_KEY=...
./.venv/bin/cue live --mic <name> --system BlackHole
```

First run downloads a Whisper model, so do that before an interview rather than during
one, and run `cue devices --check` beforehand: if the loopback device reads silent you
capture only your own voice, which is the failure that ruins a whole session and gives
no sign until afterwards.

## Quick start

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/cue replay transcripts/fixtures/sample-001-url-shortener.jsonl
```

That runs against a **fake adapter**, keyword heuristics that exercise the pipeline
without network or credentials. It is not a proxy for model judgment, and no threshold
should be tuned against it.

For real judgments:

```bash
./.venv/bin/pip install -e ".[jev]"
export TYPESAFE_API_KEY=...
./.venv/bin/cue replay <transcript.jsonl> --adapter typesafe --db sessions.db
```

Useful flags: `--show-suppressed` (see what was hidden and why), `--db` (persist a session
log), `--json` (export it), `--tick-ms` (evaluation cadence).

Live, against a real interview:

```bash
./.venv/bin/cue live --mic <name> --system <name>   # add --my-role candidate in a peer swap
./.venv/bin/cue label                               # afterwards: what was actually useful
./.venv/bin/cue gate                                # where that leaves the usefulness gate
```

`cue report --board board.excalidraw` adds whiteboard analysis to the evidence pack.

To see exactly what is sent to Jev and what comes back, add `--trace`. It writes one
file per session under `traces/`, recording start, pause, resume and stop alongside
every request and response with latency, tokens and cost. Traces contain transcript
text, so they are gitignored; `cue traces` lists them and `cue traces --purge` deletes
them.

## How it fits together

**[ARCHITECTURE.md](ARCHITECTURE.md)** has the pipeline in full, with diagrams:
how it decides what to ask, how the request is built, and how 93% of correct
answers get discarded before anyone sees them.

```
JSONL transcript -> chunker -> rolling state -> preconditions -> Jev adapter
  -> signal policy -> interviewer output / session log
```

| Module | Responsibility |
|---|---|
| `transcript.py` | JSONL import, speaker normalization, content hashing |
| `state.py` | Bounded window, code-resolved facts (outstanding question, durations) |
| `signals.py` | The five signals: question text, criteria, and preconditions |
| `trace.py` | Per-session log of everything sent to and received from Jev |
| `jev/` | The only place that talks to TypeSafe. Protocol + fake + real |
| `policy.py` | Thresholds, dwell, rate limits, display budget |
| `store.py` | SQLite session log |
| `replay.py` | The tick loop |
| `live/` | Dual-stream capture, local transcription, the live session |
| `board.py` | Whiteboard state from an Excalidraw export or a screenshot |
| `report.py` | Post-session evidence pack |
| `label.py` | Post-session labelling, the usefulness-gate instrument |

Three things are load-bearing and easy to break:

**Noul answers carry no confidence.** Its probability *is* its confidence. `SignalDecision`
rejects a Noul that arrives with one, because policy code gating on a field that does not
exist would silently never fire.

**Preconditions gate the request, not the response.** All timing and counting happens in
code, the model is unreliable at both, and a signal whose precondition fails is left out
of the batch entirely rather than asked and discarded.

**The evaluator and the policy are separate layers.** One answers "what does the model say";
the other answers "should a person see it". A correct signal shown at the wrong moment is
still bad behavior. Do not merge them.

## The questions

The five questions are Python constants in `signals.py`. Nothing generates them and
nothing varies them; every tick sends the same wording and only the transcript window
underneath changes.

```bash
./.venv/bin/cue signals          # all five, as sent
./.venv/bin/cue signals --full   # with complete criteria
```

They are deliberately **not** configuration. Every stored decision is stamped with
`signal_version`, which is what lets a replay from weeks ago be compared to one from
today, and that guarantee only holds while the version and the wording move together.
So a fingerprint over the question text is pinned in `tests/test_signals_pinned.py`:
reword a question without bumping `SIGNAL_SET_VERSION` and the suite fails and tells
you what to do.

Changing what the model is asked should be a deliberate act with a version attached.

## Operating notes

The signals are the easy part. Running this during a real interview, where someone's
time and money are on the line, needs a few rules that are not in the code.

**Nothing here is load-bearing.** If it breaks mid-interview, close the tab and keep
interviewing. The tool is not in the loop and nothing depends on it.

**Glance at it in pauses only.** While they are drawing, while they are thinking. Never
mid-answer. Attention spent on the strip is attention taken from the person being
interviewed.

**When a signal fires, do not act on it immediately.** Ask first whether you had already
noticed. That question is the whole experiment, and so far the answer is usually yes.

**If it distracts you even once, close it.** A signal that is accurate and distracting is
still a failure. The interview is real and the candidate's time is worth more than the
data.

**Tell them it is running**, before it is running, in one sentence.

That fourth rule is why this project has a usefulness gate rather than an accuracy
benchmark. Being right is not the bar.

## Transcripts

`transcripts/fixtures/` holds synthetic transcripts and is tracked. `transcripts/private/`
is for real ones and is gitignored, the directory is default-deny, so anything dropped
there is ignored unless explicitly allow-listed.

## Tests

```bash
./.venv/bin/python -m pytest
./.venv/bin/ruff check .
```

Tests build their own fixtures inline, so they do not depend on anything gitignored.
