from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A closed-open interval of UTC datetimes derived from a question.

    ``start`` is inclusive; ``end`` is exclusive. Use ``label`` when surfacing
    the window back to the user or to the LLM.
    """

    start: datetime
    end: datetime
    label: str


@dataclass(frozen=True, slots=True)
class RecencyIntent:
    """Marker that the user asked about the most-recently modified files.

    Carries the spoken label (e.g. ``"latest"``) so it can be surfaced back.
    """

    label: str


_MONTH_NAMES = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}


# Words/phrases that mean "sort by modified_at desc, give me the top few".
# Order matters for label selection — first match wins.
_RECENCY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bmost\s+recent(?:ly)?\b", "most recent"),
    (r"\brecently\s+(?:modified|updated|edited|worked|changed)\b", "most recent"),
    (r"\blatest\b", "latest"),
    (r"\bnewest\b", "newest"),
    (r"\blast\s+(?:file|files|thing|things|one|ones)\b", "latest"),
)


def parse_recency_intent(question: str) -> RecencyIntent | None:
    """Return a ``RecencyIntent`` when the user is asking about the freshest
    files (e.g. *what is the latest file I worked on?*), else ``None``.

    Recency is distinct from a :class:`TimeWindow`: it doesn't bound a
    time range, it asks for "sort by modified_at desc" instead. The chat
    layer branches on this so retrieval skips embedding similarity entirely.
    """
    if not question:
        return None
    q = question.lower()
    for pattern, label in _RECENCY_PATTERNS:
        if re.search(pattern, q):
            return RecencyIntent(label=label)
    return None


def parse_time_intent(question: str, *, now: datetime | None = None) -> TimeWindow | None:
    """Extract a time window from a free-form question, or return ``None``.

    Conservative regex-only parser. Recognises:

    * ``today``, ``yesterday``
    * ``this week``, ``last week``
    * ``this month``, ``last month``
    * ``last N day(s)/week(s)/month(s)`` and ``past N ...``
    * ``in <Month>`` and ``in <Month> <year>``

    Times are anchored in UTC because that's how chunks are stored. A user
    near the date-line will see at most a few-hour drift on day-boundary
    queries; acceptable for v1.
    """
    if not question:
        return None
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    q = question.lower()

    # Order matters: more specific first.
    today_start = _start_of_day(now)
    today_end = today_start + timedelta(days=1)

    if re.search(r"\byesterday\b", q):
        start = today_start - timedelta(days=1)
        return TimeWindow(start=start, end=today_start, label="yesterday")

    if re.search(r"\btoday\b", q):
        return TimeWindow(start=today_start, end=today_end, label="today")

    if re.search(r"\b(this week)\b", q):
        start = _start_of_week(today_start)
        return TimeWindow(start=start, end=start + timedelta(days=7), label="this week")

    if re.search(r"\b(last week|previous week)\b", q):
        start_this = _start_of_week(today_start)
        start_last = start_this - timedelta(days=7)
        return TimeWindow(start=start_last, end=start_this, label="last week")

    if re.search(r"\b(this month)\b", q):
        start, end = _bounds_of_month(today_start.year, today_start.month)
        return TimeWindow(start=start, end=end, label="this month")

    if re.search(r"\b(last month|previous month)\b", q):
        prev_year = today_start.year if today_start.month > 1 else today_start.year - 1
        prev_month = today_start.month - 1 if today_start.month > 1 else 12
        start, end = _bounds_of_month(prev_year, prev_month)
        return TimeWindow(start=start, end=end, label="last month")

    m = re.search(r"\b(?:last|past)\s+(\d{1,3})\s+day(?:s)?\b", q)
    if m:
        n = max(int(m.group(1)), 1)
        return TimeWindow(
            start=today_start - timedelta(days=n - 1),
            end=today_end,
            label=f"the last {n} days",
        )

    m = re.search(r"\b(?:last|past)\s+(\d{1,3})\s+week(?:s)?\b", q)
    if m:
        n = max(int(m.group(1)), 1)
        return TimeWindow(
            start=today_start - timedelta(weeks=n),
            end=today_end,
            label=f"the last {n} weeks",
        )

    m = re.search(r"\b(?:last|past)\s+(\d{1,3})\s+month(?:s)?\b", q)
    if m:
        n = max(int(m.group(1)), 1)
        return TimeWindow(
            start=_subtract_months(today_start, n),
            end=today_end,
            label=f"the last {n} months",
        )

    # "in <Month>" or "in <Month> <year>"
    m = re.search(
        r"\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"(?:\s+(\d{4}))?\b",
        q,
    )
    if m:
        month = _MONTH_NAMES[m.group(1)]
        year = int(m.group(2)) if m.group(2) else _infer_year_for_month(now, month)
        start, end = _bounds_of_month(year, month)
        label = f"{calendar.month_name[month]} {year}"
        return TimeWindow(start=start, end=end, label=label)

    return None


def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _start_of_week(day_start: datetime) -> datetime:
    # ISO week: Monday = 0
    return day_start - timedelta(days=day_start.weekday())


def _bounds_of_month(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=UTC)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(year, month + 1, 1, tzinfo=UTC)
    return start, end


def _subtract_months(dt: datetime, n: int) -> datetime:
    month = dt.month - n
    year = dt.year
    while month <= 0:
        month += 12
        year -= 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _infer_year_for_month(now: datetime, month: int) -> int:
    """If a bare month name is in the future this calendar year, assume the user
    means last year. Otherwise this year."""
    if month > now.month:
        return now.year - 1
    return now.year
