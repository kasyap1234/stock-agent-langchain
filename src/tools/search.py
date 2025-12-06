from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
import time
from src.utils.logging_config import ToolLogger
from src.middleware.retry_handler import retry_web_search

# Initialize logger
logger = ToolLogger("web_search")

@tool
def web_search(query: str) -> str:
    """
    Performs a web search to find news, sentiment, or fundamental data with retry on failures.
    Args:
        query: The search query.
    Returns:
        Search results.
    """
    start_time = time.time()

    try:
        result = _perform_search_with_retry(query)
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(
            ticker="N/A",
            data_type="web_search",
            success=True,
            latency_ms=latency_ms,
            records_fetched=len(result) if result else 0
        )
        return result

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(
            ticker="N/A",
            data_type="web_search",
            success=False,
            latency_ms=latency_ms,
            error=str(e)
        )
        return f"Error performing web search: {str(e)}"


@retry_web_search
def _perform_search_with_retry(query: str) -> str:
    """Helper function to perform web search with automatic retry."""
    search = DuckDuckGoSearchRun()
    return search.run(query)
