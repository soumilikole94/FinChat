# FinChat

FinChat is a small retrieval-grounded chat app for exploring the provided stock news dataset. It answers questions strictly from `data/stock_news.json`, chunks article `full_text`, builds hash-projected TF-IDF vectors for those chunks in a lightweight in-memory vector store, and uses cosine similarity to retrieve supporting context before generating an answer.

This repository runs in two modes:

- Out of the box, it works locally with the built-in retrieval and fallback summarizer. No API key is required.
- Optional LLM mode can use OpenAI, Anthropic, Gemini, or an OpenAI-compatible endpoint if the user pastes a compatible API key into the sidebar.

The main flow lives in `service.py`, with data cleanup in `data.py`, retrieval in `retrieval.py`, a thin orchestration layer in `answering.py`, provider integrations in `llm.py`, local summary logic in `local_summary.py`, and the Streamlit UI in `ui.py` plus `app.py`.

## Why this design

This assignment is small enough that we do not need a separate hosted database, but large enough that plain keyword matching feels weak. The implementation aims for a middle ground:

- Keep the app self-contained and runnable locally.
- Separate the chat UI from the data, retrieval, and answer-generation logic.
- Ground every answer in retrieved dataset articles.
- Use a local vector-store-style retrieval layer over article body text instead of headline matching.
- Support both an optional LLM flow and a deterministic local fallback.
- Make the important parts easy to test.

## Project structure

```text
.
├── app.py
├── data/stock_news.json
├── requirements.txt
├── src/finchat
│   ├── answering.py
│   ├── constants.py
│   ├── data.py
│   ├── env.py
│   ├── llm.py
│   ├── local_summary.py
│   ├── models.py
│   ├── retrieval.py
│   ├── service.py
│   └── ui.py
└── tests
```

## Setup

1. Create a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the app.

```bash
streamlit run app.py
```

Optional: paste an LLM API key into the app sidebar to enable provider-backed synthesis. The key stays only in browser session state and is not stored in the repo.

4. Run the tests.

```bash
pytest
```

## How it works

### 1. Data normalization

`load_articles()` flattens the ticker-keyed JSON into a list of `Article` records. During loading it:

- normalizes whitespace
- removes obvious boilerplate such as `Story Continues`, `View Comments`, and `READ NEXT`
- keeps the source link so answers can cite the supporting articles

Why it matters: the raw news text contains noise, and retrieval quality drops quickly if we index that noise directly.

### 2. Deduplication

Some articles appear under multiple tickers with the same title and link. Before indexing, the app deduplicates them using a normalized `title + link` key.

Why it matters: duplicate wire items can crowd out the ranking and make the answer look less thoughtful.

### 3. Chunking

Long articles are split into overlapping chunks of about 120 words with a 30-word overlap.

Why it matters: many articles include a lot of unrelated finance-site filler. Chunking helps the retriever focus on the part of the article that actually matches the user’s question.

### 4. Retrieval

The retriever builds a lightweight in-memory vector store from `full_text` chunks:

- each chunk is projected into a dense vector using a deterministic hashed TF-IDF representation
- the user query is embedded with the same projection
- chunks are ranked by cosine similarity
- top chunk matches are merged back into article-level contexts for synthesis

Titles are not used to create embeddings. They are only used later for source display and light quality penalties that demote obvious roundup or clickbait articles.

It also expands queries with ticker aliases like `AAPL/Apple` and `NVDA/NVIDIA`, then boosts chunks from matched tickers.

Why it matters: this keeps retrieval grounded in article body text while staying local, fast, and dependency-light.

### 5. Answer generation

The app always retrieves local context first.

- If a supported LLM key is set and the matching SDK is installed, the retrieved context is sent to that provider with strict instructions to answer only from those sources.
- Otherwise, the app falls back to a deterministic local summarizer that selects the most relevant sentences from the retrieved contexts.

Why it matters: the assignment allows either approach, so this supports both a strong local default and a more fluent optional LLM mode without assuming the user has a specific vendor account.

### 6. Streamlit UI

The UI is intentionally thin:

- `st.chat_input` collects questions
- `st.session_state` stores chat history
- a sidebar shows supported tickers and sample prompts
- each answer includes a source list with links and excerpts

Why it matters: Style is not the focus, so the UI stays functional and leaves most of the effort in the code quality and testability.

## Example prompts

- `What is the latest Apple news in the dataset?`
- `Summarize the NVIDIA coverage.`
- `What is happening with Intel?`
- `Compare NVIDIA and Apple based on the dataset.`
- `What does the dataset say about Apple's AI plans in China?`
- `How is IBM using AI in the dataset?`
- `Is Apple the company related to the fruit apple?`

## Testing strategy

The tests cover the behaviors most likely to matter:

- loading and flattening the dataset correctly
- removing boilerplate and deduplicating repeated wire items
- building chunk embeddings and returning body-text excerpts from the vector store
- ranking the expected ticker for a known query
- synthesizing from multiple retrieved articles in local mode
- refusing to invent unsupported answers
- swapping in a mock LLM client for deterministic testing
- initializing the service without requiring an API key

## Reasonable assumptions

- The app answers strictly from the provided JSON dataset.
- The dataset is bundled in the repo so reviewers can run it locally.
- Chat history is in-memory only.
- Retrieval quality matters more than styling for this assignment.
