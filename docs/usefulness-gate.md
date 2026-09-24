# The usefulness gate

Written before session 1. Not changed after that.

This is the second attempt. The first failed on a signal that was accurate and
that I had already noticed, which is why the criteria below are about novelty
and attention cost rather than correctness.

## What I am testing

The four cards: `answered_question`, `answer_depth`, `clarity`, `rambling_risk`.

Not `current_phase`. That is a banner, it does not interrupt, and it already works. It
stays either way.

## Pass needs both

- At least **25%** of shown cards labelled "told me something I hadn't noticed"
- **Zero** sessions where I answer "yes" to "did looking at it cost you anything"

"A little" is fine. "Yes" is not, and one is enough to fail.

Sample: **10 sessions**, not counting excluded ones.

Why 25%: below one useful card in four I would stop looking at the strip anyway.

## Excluding a session

Only for these reasons:

- Audio failed. One side missing, or transcription fell behind.
- Wrong `--my-role` setting.
- Not a real interview.

A bad result is not a reason to exclude.

## If it fails

Drop the cards. Keep the phase banner.

Write up what I found either way.

## Protocol

These change what the data says, so they are fixed too:

- Press Start when the interview starts, not when the app launches
- Pause for breaks and while they draw
- `cue label` straight after, while I still remember

The questions are fixed in `src/cue/label.py`. Do not reword them mid-run.

Consent and which platforms I may run on are in the runbook. They apply whether or
not a gate is running.

## Record

Pre-registered:

First session:
