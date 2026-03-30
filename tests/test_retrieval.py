from pathlib import Path
from finchat.data import load_articles
from finchat.retrieval import build_index, retrieve


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_news.json"


def test_retrieval_finds_expected_ticker():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("What did Apple tease about its new product launch?", index=index)

    assert contexts
    assert contexts[0].article.ticker == "AAPL"
    assert "Apple" in contexts[0].article.title


def test_index_builds_dense_chunk_embeddings():
    index = build_index(load_articles(DATA_PATH))

    assert index.chunks
    assert index.vector_store.chunk_embeddings.shape[0] == len(index.chunks)
    assert index.vector_store.chunk_embeddings.shape[1] > 0


def test_retrieval_returns_full_text_excerpts_not_just_headlines():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("What is the latest Apple news in the dataset?", index=index)

    assert contexts
    combined_excerpt = " ".join((contexts[0].excerpt,) + contexts[0].supporting_excerpts).lower()
    assert combined_excerpt != contexts[0].article.title.lower()
    assert (
        "planning to launch a product" in combined_excerpt
        or "apple intelligence" in combined_excerpt
        or "in-house cellular modem chip" in combined_excerpt
        or "february 19" in combined_excerpt
        or "feb. 19" in combined_excerpt
    )
