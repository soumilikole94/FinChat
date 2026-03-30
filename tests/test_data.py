from pathlib import Path

from finchat.data import article_dedup_key, load_articles
from finchat.retrieval import build_index


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_news.json"


def test_load_articles_flattens_dataset():
    articles = load_articles(DATA_PATH)

    assert len(articles) == 138
    assert {article.ticker for article in articles} == {
        "AAPL",
        "AMZN",
        "IBM",
        "INTC",
        "MSFT",
        "NFLX",
        "NVDA",
    }
    assert all(article.title for article in articles)
    assert all(article.link.startswith("https://") for article in articles)


def test_cleaning_and_deduplication():
    articles = load_articles(DATA_PATH)
    assert any("Story Continues" not in article.full_text for article in articles)
    assert all("View Comments" not in article.full_text for article in articles)

    dedup_keys = {article_dedup_key(article) for article in articles}
    index = build_index(articles)

    assert len(index.articles) == len(dedup_keys)
    assert len(index.articles) < len(articles)
    duplicate_link = "https://finance.yahoo.com/video/wall-street-bullish-cash-levels-145356726.html"
    assert sum(1 for article in index.articles if article.link == duplicate_link) == 1


def test_deduplication_preserves_related_tickers_and_picks_sensible_primary_ticker():
    index = build_index(load_articles(DATA_PATH))
    article = next(
        article
        for article in index.articles
        if article.title == "Nvidia slashes stake in emerging rival as AI arms race heats up"
    )

    assert article.ticker == "NVDA"
    assert set(article.related_tickers) == {"AAPL", "NVDA"}
