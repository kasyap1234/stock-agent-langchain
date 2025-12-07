# Project Documentation

## Tech Stack
- Python with LangChain + LangGraph for multi-agent orchestration.
- LLMs: ChatGroq `qwen/qwen3-32b` for analysts/ensemble; Critic prefers Gemini 2.5 Flash with Groq `openai/gpt-oss-120b` fallback (keys via `.env`).
- Data/analysis libs: `yfinance` (prices, options, holders), `pandas`, `ta` for indicators and regime checks.
- Search/news: `duckduckgo-search` via LangChain community tool for open-web headlines and sentiment.
- Infra/utilities: `python-dotenv` for config loading, `structlog` logging, `argparse` CLI, retry middleware for yfinance/web requests.
- Tests: `pytest` suite in `tests/` (simulation, prompts, trajectories).

## Data & News Sources
- Market data: `yfinance` powers historical prices, multi-timeframe indicators, options sentiment, insider transactions, and institutional holdings (`src/tools/market_data.py`, `src/tools/enhanced_data.py`, `src/tools/fundamentals.py`).
- Fundamentals: `yfinance.Ticker.info` for valuation, profitability, and growth metrics (`get_fundamental_metrics`, `get_growth_metrics`).
- News & sentiment: DuckDuckGo search via LangChain tool; used by Fundamental and Sentiment analysts to pull recent articles, forum chatter, and catalysts.
- Optional provider keys (see `.env`): `ALPHA_VANTAGE_KEY`, `FINNHUB_API_KEY` for richer data if wired in later; `GOOGLE_API_KEY` for Gemini critic.

## Onboarding Guide
1) Prereqs: Python 3.10+ with `uv` or `pip`, `GROQ_API_KEY` (required), optional `GOOGLE_API_KEY`, `ALPHA_VANTAGE_KEY`, `FINNHUB_API_KEY`.
2) Setup:
   - `python -m venv .venv && source .venv/bin/activate`
   - `uv pip install -r requirements.txt` (or `pip install -r requirements.txt`)
   - Create `.env` with keys (see `README.md`).
3) Run:
   - `./run.sh TICKER.NS` (loads `.env`, activates venv) or `python main.py TICKER.NS --verbose`.
4) Mental model:
   - `main.py` loads env → LangGraph `workflow` (`src/graph/workflow.py`).
   - Flow: ContextSetup → Planner → sequential Technical/Fundamental/Sentiment analysts (self-correcting) → Ensemble (regime-aware strategy voting) → Critic (backtests + confidence) → Final synthesis.
   - Key agents in `src/agents/*`; tools in `src/tools/*`; validation/retry in `src/validation` and `src/middleware`.
5) Contributing tips:
   - Add new data tools under `src/tools/` and register with the relevant agent factory.
   - Maintain structured logging (`ToolLogger`/`AgentLogger`) and validators when fetching data.
   - Keep prompts concise; use `--verbose` during debugging and run `pytest` before PRs.
6) Common issues:
   - Missing `GROQ_API_KEY` → run exits early.
   - Rate limits on Groq → analysts are sequential to reduce 429s; rerun or lower concurrency if modified.
   - Empty yfinance responses → check ticker suffix (e.g., `.NS`) and internet connectivity.

