"""Rule-based answer extraction and scoring for OpenTune financial QA.

Every downstream number in the repo depends on this module. It implements the
frozen contract in docs/task-spec.md: pure rules, no ML, never raises on weird
model output. Gold and predictions run through the same parse path (spec rule:
multi-answer disambiguation uses one shared resolver).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

# Scoring tolerance (piecewise, per task-spec.md):
#   correct iff |pred - gold| <= tol where
#   tol = 0.001 * |gold|  if |gold| >= REL_TOL_FLOOR (big-magnitude slack)
#         ABS_TOL         otherwise (cent-level strictness)
ABS_TOL = Decimal("0.01")
REL_TOL = Decimal("0.001")
REL_TOL_FLOOR = Decimal("1000")


class Status(StrEnum):
    OK = "ok"
    NO_ANSWER = "no_answer_line"
    NON_NUMERIC = "non_numeric_answer"


@dataclass(frozen=True)
class ExtractionResult:
    status: Status
    raw: str | None = None  # raw text captured from the ANSWER line
    value: Decimal | None = None  # canonical value (percent divided by 100)
    is_percent: bool = False
    n_answer_lines: int = 0  # answer-line count; >1 signals self-correction/degeneracy


@dataclass(frozen=True)
class ScoreResult:
    correct: bool
    extraction: ExtractionResult
    gold_value: Decimal | None = None
    diff: Decimal | None = None


# \b after "answer" so "Answers: 42" does not match.
_ANSWER_RE = re.compile(r"^\s*answer\b\s*[:\-–]?\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
_UNICODE_MINUS = str.maketrans({"−": "-", "–": "-", "—": "-"})
_CURRENCY_RE = re.compile(r"^[$€£]\s*")
_NUMBER_RE = re.compile(r"^-?\d*\.?\d+$")


def _parse_number(raw: str) -> tuple[Decimal, bool] | None:
    """Parse to (canonical_value, is_percent); None if not a clean number."""
    s = raw.translate(_UNICODE_MINUS)
    s = s.strip().strip("`").strip()
    s = s.strip("\"'").strip()
    if not s:
        return None

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()

    is_percent = False
    if s.endswith("%"):
        is_percent = True
        s = s[:-1].strip()
    elif s.lower().endswith("percent"):
        is_percent = True
        s = s[:-7].strip()

    s = _CURRENCY_RE.sub("", s)
    s = s.replace(",", "")

    # Reject anything that isn't a plain decimal (units, junk, etc.)
    if not _NUMBER_RE.match(s):
        return None
    try:
        value = Decimal(s)
    except InvalidOperation:
        return None

    if negative:
        value = -value
    if is_percent:
        value = value / 100
    return value, is_percent


def extract_answer(text: str) -> ExtractionResult:
    """Extract the final ANSWER value. Last non-empty answer line wins."""
    if not text or not text.strip():
        return ExtractionResult(status=Status.NO_ANSWER)

    matches = _ANSWER_RE.findall(text)
    n = len(matches)
    if n == 0:
        return ExtractionResult(status=Status.NO_ANSWER)

    raw = ""
    for m in reversed(matches):
        if m.strip():
            raw = m.strip()
            break
    if not raw:
        return ExtractionResult(status=Status.NON_NUMERIC, raw=None, n_answer_lines=n)

    parsed = _parse_number(raw)
    if parsed is None:
        return ExtractionResult(status=Status.NON_NUMERIC, raw=raw, n_answer_lines=n)

    value, is_percent = parsed
    return ExtractionResult(
        status=Status.OK, raw=raw, value=value, is_percent=is_percent, n_answer_lines=n
    )


def _tolerance(gold_value: Decimal) -> Decimal:
    magnitude = abs(gold_value)
    if magnitude >= REL_TOL_FLOOR:
        return magnitude * REL_TOL
    return ABS_TOL


def score(prediction: str, gold: str) -> ScoreResult:
    """Score a prediction against a gold string per docs/task-spec.md."""
    extraction = extract_answer(prediction)

    gold_parsed = _parse_number(gold)
    gold_value = gold_parsed[0] if gold_parsed else None

    if extraction.status is not Status.OK or extraction.value is None or gold_value is None:
        return ScoreResult(correct=False, extraction=extraction, gold_value=gold_value)

    diff = abs(extraction.value - gold_value)
    return ScoreResult(
        correct=diff <= _tolerance(gold_value),
        extraction=extraction,
        gold_value=gold_value,
        diff=diff,
    )
