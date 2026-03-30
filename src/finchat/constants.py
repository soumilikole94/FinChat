from pathlib import Path
import os

from .env import load_env_file


REPO_ROOT = Path(__file__).resolve().parents[2]
load_env_file(REPO_ROOT / ".env")
NEWS_DATA_PATH = REPO_ROOT / "data" / "stock_news.json"
INSUFFICIENT_SUPPORT_MESSAGE = "I couldn't find enough support in the provided dataset to answer that confidently."

TICKER_ALIASES = {
    "AAPL": ("apple", "aapl"),
    "AMZN": ("amazon", "amzn"),
    "IBM": ("ibm",),
    "INTC": ("intel", "intc"),
    "MSFT": ("microsoft", "msft", "openai"),
    "NFLX": ("netflix", "nflx"),
    "NVDA": ("nvidia", "nvda"),
}

COMPANY_NAMES = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "IBM": "IBM",
    "INTC": "Intel",
    "MSFT": "Microsoft",
    "NFLX": "Netflix",
    "NVDA": "NVIDIA",
}

SUPPORTED_TICKERS = tuple(sorted(TICKER_ALIASES))
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"))
