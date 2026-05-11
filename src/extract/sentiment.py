"""Sentiment scoring. VADER by default; FinBERT available as opt-in."""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = None


def vader_score(text: str) -> float:
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader.polarity_scores(text)["compound"]


def score_batch(texts: list[str], backend: str = "vader") -> list[float]:
    if backend == "vader":
        return [vader_score(t) for t in texts]
    if backend == "finbert":
        return _finbert_batch(texts)
    raise ValueError(f"Unknown backend: {backend}")


def _finbert_batch(texts: list[str]) -> list[float]:
    from transformers import pipeline
    pipe = pipeline("text-classification", model="ProsusAI/finbert", truncation=True)
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    results = pipe(texts, batch_size=32)
    return [label_map[r["label"]] * r["score"] for r in results]
