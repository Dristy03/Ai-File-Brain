from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_file_brain.core.time_intent import parse_recency_intent, parse_time_intent

# Anchor every test at a known reference. 2026-05-07 is a Thursday.
NOW = datetime(2026, 5, 7, 14, 30, tzinfo=UTC)


def test_no_time_intent_returns_none():
    assert parse_time_intent("explain how ranking works", now=NOW) is None
    assert parse_time_intent("", now=NOW) is None
    assert parse_time_intent("   ", now=NOW) is None


def test_yesterday():
    w = parse_time_intent("what was I working on yesterday?", now=NOW)
    assert w is not None
    assert w.start == datetime(2026, 5, 6, tzinfo=UTC)
    assert w.end == datetime(2026, 5, 7, tzinfo=UTC)
    assert w.label == "yesterday"


def test_today():
    w = parse_time_intent("what files did I touch today?", now=NOW)
    assert w is not None
    assert w.start == datetime(2026, 5, 7, tzinfo=UTC)
    assert w.end == datetime(2026, 5, 8, tzinfo=UTC)


def test_this_week_starts_on_monday():
    # NOW is Thursday → this week starts Monday May 4.
    w = parse_time_intent("summarise what I edited this week", now=NOW)
    assert w is not None
    assert w.start == datetime(2026, 5, 4, tzinfo=UTC)
    assert w.end == datetime(2026, 5, 11, tzinfo=UTC)


def test_last_week():
    w = parse_time_intent("what was last week's progress", now=NOW)
    assert w is not None
    assert w.start == datetime(2026, 4, 27, tzinfo=UTC)
    assert w.end == datetime(2026, 5, 4, tzinfo=UTC)


def test_this_month():
    w = parse_time_intent("notes from this month", now=NOW)
    assert w is not None
    assert w.start == datetime(2026, 5, 1, tzinfo=UTC)
    assert w.end == datetime(2026, 6, 1, tzinfo=UTC)


def test_last_month():
    w = parse_time_intent("anything from last month?", now=NOW)
    assert w is not None
    assert w.start == datetime(2026, 4, 1, tzinfo=UTC)
    assert w.end == datetime(2026, 5, 1, tzinfo=UTC)


def test_last_n_days():
    w = parse_time_intent("what changed in the last 3 days", now=NOW)
    assert w is not None
    # Inclusive of today, so window covers 3 calendar days ending tomorrow 00:00.
    assert w.end - w.start == timedelta(days=3)
    assert w.label == "the last 3 days"


def test_past_n_weeks():
    w = parse_time_intent("summarize the past 2 weeks", now=NOW)
    assert w is not None
    assert (w.end - w.start) == timedelta(days=15)  # 14 days back + today
    assert w.label == "the last 2 weeks"


def test_in_named_month_assumes_recent_year():
    # "in March" — March is in the past relative to NOW (May), so same year.
    w = parse_time_intent("what was I working on in March?", now=NOW)
    assert w is not None
    assert w.start == datetime(2026, 3, 1, tzinfo=UTC)
    assert w.end == datetime(2026, 4, 1, tzinfo=UTC)
    assert w.label == "March 2026"


def test_in_named_month_in_future_assumes_last_year():
    # "in November" — November is later in the calendar than May, so user must
    # mean last November.
    w = parse_time_intent("did I write anything in November?", now=NOW)
    assert w is not None
    assert w.start == datetime(2025, 11, 1, tzinfo=UTC)
    assert w.end == datetime(2025, 12, 1, tzinfo=UTC)


def test_in_named_month_with_explicit_year():
    w = parse_time_intent("anything from in January 2024?", now=NOW)
    assert w is not None
    assert w.start == datetime(2024, 1, 1, tzinfo=UTC)
    assert w.end == datetime(2024, 2, 1, tzinfo=UTC)


def test_more_specific_phrase_wins():
    # "yesterday" beats a stray "this week" elsewhere in the sentence.
    w = parse_time_intent(
        "yesterday I started something, but I'll finish this week", now=NOW
    )
    assert w is not None
    assert w.label == "yesterday"


def test_naive_now_is_treated_as_utc():
    naive = datetime(2026, 5, 7, 14, 30)  # no tzinfo
    w = parse_time_intent("yesterday", now=naive)
    assert w is not None
    assert w.start.tzinfo == UTC


@pytest.mark.parametrize(
    "phrase",
    [
        "yesterday's notes",
        "what about today's status",
        "anything new this week?",
        "show me last month's diff",
    ],
)
def test_punctuation_does_not_break_word_boundaries(phrase):
    assert parse_time_intent(phrase, now=NOW) is not None


# ---- recency intent -------------------------------------------------


@pytest.mark.parametrize(
    "phrase,expected_label",
    [
        ("what is the latest file I worked on?", "latest"),
        ("show me the newest file", "newest"),
        ("what is the most recent thing I edited", "most recent"),
        ("most recently modified file", "most recent"),
        ("what was the last thing I worked on", "latest"),
        ("show me the last file in the index", "latest"),
    ],
)
def test_recency_intent_recognised(phrase, expected_label):
    intent = parse_recency_intent(phrase)
    assert intent is not None
    assert intent.label == expected_label


@pytest.mark.parametrize(
    "phrase",
    [
        "what did I work on yesterday",
        "summarise this week",
        "explain how chunking works",
        "",
        "   ",
    ],
)
def test_no_recency_intent(phrase):
    assert parse_recency_intent(phrase) is None
