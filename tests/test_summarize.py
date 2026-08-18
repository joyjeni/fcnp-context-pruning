"""Unit tests for the extractive summarizer (improvement #2 default)."""

from fcnp import extractive_summarize


def test_short_text_passthrough():
    text = "Just a few words here."
    assert extractive_summarize(text, max_tokens=40) == text


def test_long_text_gets_shortened():
    sentences = [f"This is filler sentence number {i} about nothing important." for i in range(20)]
    text = " ".join(sentences)
    summary = extractive_summarize(text, query=None, max_tokens=20)
    assert len(summary.split()) <= 25  # small slack for boundary sentence
    assert len(summary) < len(text)


def test_query_relevant_sentence_is_preferred():
    text = (
        "The weather today is sunny with a light breeze. "
        "Flight AI202 to Bengaluru departs at 6pm and mandi prices for tomato rose sharply this week. "
        "Traffic on the highway was unusually light this morning."
    )
    summary = extractive_summarize(text, query="mandi tomato prices", max_tokens=15)
    assert "tomato" in summary.lower() or "mandi" in summary.lower()


def test_empty_text_returns_empty():
    assert extractive_summarize("", max_tokens=10) == ""
