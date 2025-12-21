"""
News fetching tools for Indian stock analysis.

Provides news from multiple sources:
- MoneyControl (Indian financial news)
- Economic Times RSS feeds
- NSE corporate announcements
- NewsAPI (global, rate-limited)
- Google News RSS (fallback)
"""

import os
import re
import time
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from urllib.parse import quote, urljoin
from langchain.tools import tool
from src.utils.logging_config import ToolLogger
from src.tools.cache import (
    news_cache,
    newsapi_limiter,
    google_rss_limiter,
    cached,
)

# Initialize logger
logger = ToolLogger("news")


# ============================================================
# Data Models
# ============================================================

@dataclass
class NewsArticle:
    """Structured news article data."""
    title: str
    source: str
    published_date: datetime
    url: str
    summary: Optional[str] = None
    ticker: Optional[str] = None
    relevance_score: float = 0.0

    def to_string(self) -> str:
        """Format for LLM consumption."""
        date_str = self.published_date.strftime('%Y-%m-%d %H:%M')
        summary_text = f"\n   Summary: {self.summary}" if self.summary else ""
        return f"[{self.source}] {date_str}: {self.title}{summary_text}\n   URL: {self.url}"


# ============================================================
# Relevance Keywords for Filtering
# ============================================================

RELEVANCE_KEYWORDS = {
    'high': [
        'earnings', 'results', 'quarterly', 'revenue', 'profit', 'loss',
        'acquisition', 'merger', 'buyback', 'dividend', 'split', 'bonus',
        'target', 'upgrade', 'downgrade', 'rating', 'guidance', 'outlook'
    ],
    'medium': [
        'management', 'CEO', 'CFO', 'expansion', 'contract', 'order',
        'partnership', 'investment', 'launch', 'product', 'deal'
    ],
    'low': [
        'sector', 'industry', 'market', 'economy', 'stocks'
    ]
}


def calculate_relevance(text: str, ticker: str = "") -> float:
    """
    Calculate relevance score for article text.

    Args:
        text: Article title or content
        ticker: Stock ticker for matching

    Returns:
        Relevance score 0.0 to 1.0
    """
    text_lower = text.lower()
    score = 0.0

    # Check for ticker mention
    if ticker:
        ticker_base = ticker.replace('.NS', '').replace('.BO', '').lower()
        if ticker_base in text_lower:
            score += 0.4

    # Check relevance keywords
    for keyword in RELEVANCE_KEYWORDS['high']:
        if keyword in text_lower:
            score += 0.15
    for keyword in RELEVANCE_KEYWORDS['medium']:
        if keyword in text_lower:
            score += 0.08
    for keyword in RELEVANCE_KEYWORDS['low']:
        if keyword in text_lower:
            score += 0.03

    return min(1.0, score)


def filter_articles(
    articles: List[NewsArticle],
    ticker: str = "",
    min_relevance: float = 0.1,
    max_age_days: int = 7
) -> List[NewsArticle]:
    """Filter and sort articles by relevance and freshness."""
    now = datetime.now()
    cutoff = now - timedelta(days=max_age_days)

    filtered = []
    for article in articles:
        # Check age
        if article.published_date < cutoff:
            continue

        # Calculate relevance if not set
        if article.relevance_score == 0.0:
            article.relevance_score = calculate_relevance(
                f"{article.title} {article.summary or ''}",
                ticker
            )

        if article.relevance_score >= min_relevance:
            filtered.append(article)

    # Sort by relevance (desc) then date (desc)
    filtered.sort(key=lambda a: (a.relevance_score, a.published_date), reverse=True)
    return filtered


# ============================================================
# Source Fetchers
# ============================================================

def _fetch_google_news_rss(query: str, ticker: str = "") -> List[NewsArticle]:
    """
    Fetch news from Google News RSS (no API key required).

    Args:
        query: Search query
        ticker: Optional ticker for relevance scoring

    Returns:
        List of NewsArticle objects
    """
    import feedparser

    articles = []

    try:
        if not google_rss_limiter.acquire(blocking=True, timeout=30):
            logger.log_fetch(ticker or "N/A", "google_rss", False, 0, 0, "rate_limited")
            return articles

        # Google News RSS search URL
        encoded_query = quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        start_time = time.time()
        feed = feedparser.parse(url)
        latency_ms = (time.time() - start_time) * 1000

        for entry in feed.entries[:15]:
            try:
                # Parse publication date
                pub_date = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])

                # Extract source from title (Google News format: "Title - Source")
                title = entry.get('title', '')
                source = 'Google News'
                if ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    if len(parts) == 2:
                        title = parts[0]
                        source = parts[1]

                article = NewsArticle(
                    title=title,
                    source=source,
                    published_date=pub_date,
                    url=entry.get('link', ''),
                    summary=entry.get('summary', '')[:200] if entry.get('summary') else None,
                    ticker=ticker
                )
                article.relevance_score = calculate_relevance(
                    f"{article.title} {article.summary or ''}",
                    ticker
                )
                articles.append(article)

            except Exception:
                continue

        logger.log_fetch(ticker or "N/A", "google_rss", True, latency_ms, len(articles))

    except Exception as e:
        logger.log_fetch(ticker or "N/A", "google_rss", False, 0, 0, str(e))

    return articles


def _fetch_economic_times_rss(ticker: str = "", categories: List[str] = None) -> List[NewsArticle]:
    """
    Fetch news from Economic Times RSS feeds.

    Args:
        ticker: Stock ticker for filtering
        categories: List of categories to fetch (default: markets, stocks)

    Returns:
        List of NewsArticle objects
    """
    import feedparser

    ET_RSS_FEEDS = {
        'markets': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
        'stocks': 'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
        'ipo': 'https://economictimes.indiatimes.com/markets/ipo/rssfeeds/2886573.cms',
        'tech': 'https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms',
        'banking': 'https://economictimes.indiatimes.com/industry/banking/finance/rssfeeds/13358359.cms',
    }

    if categories is None:
        categories = ['markets', 'stocks']

    articles = []

    for category in categories:
        if category not in ET_RSS_FEEDS:
            continue

        try:
            start_time = time.time()
            feed = feedparser.parse(ET_RSS_FEEDS[category])
            latency_ms = (time.time() - start_time) * 1000

            for entry in feed.entries[:10]:
                try:
                    pub_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])

                    article = NewsArticle(
                        title=entry.get('title', ''),
                        source=f'Economic Times ({category})',
                        published_date=pub_date,
                        url=entry.get('link', ''),
                        summary=entry.get('summary', '')[:200] if entry.get('summary') else None,
                        ticker=ticker
                    )
                    article.relevance_score = calculate_relevance(
                        f"{article.title} {article.summary or ''}",
                        ticker
                    )
                    articles.append(article)

                except Exception:
                    continue

            logger.log_fetch(ticker or "N/A", f"et_rss_{category}", True, latency_ms, len(feed.entries))

        except Exception as e:
            logger.log_fetch(ticker or "N/A", f"et_rss_{category}", False, 0, 0, str(e))

    return articles


def _fetch_newsapi(query: str, ticker: str = "", days_back: int = 7) -> List[NewsArticle]:
    """
    Fetch news from NewsAPI.org (free tier: 100 requests/day).

    Args:
        query: Search query
        ticker: Optional ticker for relevance scoring
        days_back: How far back to search (max 30 for free tier)

    Returns:
        List of NewsArticle objects
    """
    import requests

    articles = []
    api_key = os.getenv('NEWSAPI_KEY')

    if not api_key:
        return articles

    try:
        if not newsapi_limiter.acquire(blocking=True, timeout=30):
            remaining = newsapi_limiter.get_remaining_day()
            logger.log_fetch(ticker or "N/A", "newsapi", False, 0, 0, f"rate_limited_daily_{remaining}_remaining")
            return articles

        from_date = (datetime.now() - timedelta(days=min(days_back, 30))).strftime('%Y-%m-%d')

        start_time = time.time()
        response = requests.get(
            'https://newsapi.org/v2/everything',
            params={
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': 20,
                'apiKey': api_key
            },
            timeout=15
        )
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            for item in data.get('articles', []):
                try:
                    pub_date = datetime.now()
                    if item.get('publishedAt'):
                        pub_date = datetime.fromisoformat(
                            item['publishedAt'].replace('Z', '+00:00')
                        ).replace(tzinfo=None)

                    article = NewsArticle(
                        title=item.get('title', ''),
                        source=item.get('source', {}).get('name', 'NewsAPI'),
                        published_date=pub_date,
                        url=item.get('url', ''),
                        summary=item.get('description', '')[:200] if item.get('description') else None,
                        ticker=ticker
                    )
                    article.relevance_score = calculate_relevance(
                        f"{article.title} {article.summary or ''}",
                        ticker
                    )
                    articles.append(article)

                except Exception:
                    continue

            logger.log_fetch(ticker or "N/A", "newsapi", True, latency_ms, len(articles))
        else:
            logger.log_fetch(ticker or "N/A", "newsapi", False, latency_ms, 0, f"status_{response.status_code}")

    except Exception as e:
        logger.log_fetch(ticker or "N/A", "newsapi", False, 0, 0, str(e))

    return articles


def _fetch_nse_announcements(symbol: str, days_back: int = 30) -> List[NewsArticle]:
    """
    Fetch corporate announcements from NSE India.

    Args:
        symbol: NSE symbol (e.g., RELIANCE without .NS)
        days_back: How far back to fetch

    Returns:
        List of NewsArticle objects
    """
    import requests

    articles = []

    # Clean symbol
    clean_symbol = symbol.replace('.NS', '').replace('.BO', '').upper()

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        # NSE requires session with cookies
        session = requests.Session()
        session.headers.update(headers)

        # Get initial cookies
        session.get('https://www.nseindia.com', timeout=10)

        start_time = time.time()
        response = session.get(
            f'https://www.nseindia.com/api/corporate-announcements',
            params={
                'index': 'equities',
                'symbol': clean_symbol
            },
            timeout=15
        )
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            cutoff = datetime.now() - timedelta(days=days_back)

            for item in data[:20]:
                try:
                    # Parse date
                    date_str = item.get('an_dt', '')
                    pub_date = datetime.now()
                    if date_str:
                        try:
                            pub_date = datetime.strptime(date_str, '%d-%b-%Y')
                        except ValueError:
                            pass

                    if pub_date < cutoff:
                        continue

                    subject = item.get('desc', '') or item.get('subject', '')

                    article = NewsArticle(
                        title=subject[:200],
                        source='NSE Announcement',
                        published_date=pub_date,
                        url=f"https://www.nseindia.com/companies-listing/corporate-filings-announcements",
                        summary=item.get('attchmntText', '')[:200] if item.get('attchmntText') else None,
                        ticker=symbol,
                        relevance_score=0.9  # Official announcements are highly relevant
                    )
                    articles.append(article)

                except Exception:
                    continue

            logger.log_fetch(symbol, "nse_announcements", True, latency_ms, len(articles))
        else:
            logger.log_fetch(symbol, "nse_announcements", False, latency_ms, 0, f"status_{response.status_code}")

    except Exception as e:
        logger.log_fetch(symbol, "nse_announcements", False, 0, 0, str(e))

    return articles


def _fetch_moneycontrol_news(ticker: str, company_name: str = "") -> List[NewsArticle]:
    """
    Fetch news from MoneyControl search.

    Args:
        ticker: Stock ticker
        company_name: Company name for search

    Returns:
        List of NewsArticle objects
    """
    import requests
    from bs4 import BeautifulSoup

    articles = []
    clean_symbol = ticker.replace('.NS', '').replace('.BO', '')

    try:
        search_query = company_name if company_name else clean_symbol
        url = f"https://www.moneycontrol.com/news/tags/{search_query.lower().replace(' ', '-')}.html"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        start_time = time.time()
        response = requests.get(url, headers=headers, timeout=15)
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find news list items
            news_items = soup.select('.news_list li, .newsList li, .article_box')[:10]

            for item in news_items:
                try:
                    # Get title and link
                    title_elem = item.select_one('h2 a, h3 a, .title a, a.title')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')

                    # Get date if available
                    date_elem = item.select_one('.article_date, .date, time, .timmment')
                    pub_date = datetime.now()
                    if date_elem:
                        date_text = date_elem.get_text(strip=True)
                        # Try common date formats
                        for fmt in ['%B %d, %Y', '%d %b %Y', '%Y-%m-%d']:
                            try:
                                pub_date = datetime.strptime(date_text, fmt)
                                break
                            except ValueError:
                                continue

                    # Get summary
                    summary_elem = item.select_one('p, .desc, .summary')
                    summary = summary_elem.get_text(strip=True)[:200] if summary_elem else None

                    article = NewsArticle(
                        title=title,
                        source='MoneyControl',
                        published_date=pub_date,
                        url=link if link.startswith('http') else f"https://www.moneycontrol.com{link}",
                        summary=summary,
                        ticker=ticker
                    )
                    article.relevance_score = calculate_relevance(
                        f"{article.title} {article.summary or ''}",
                        ticker
                    )
                    articles.append(article)

                except Exception:
                    continue

            logger.log_fetch(ticker, "moneycontrol", True, latency_ms, len(articles))
        else:
            logger.log_fetch(ticker, "moneycontrol", False, latency_ms, 0, f"status_{response.status_code}")

    except Exception as e:
        logger.log_fetch(ticker, "moneycontrol", False, 0, 0, str(e))

    return articles


# ============================================================
# Utility Functions
# ============================================================

def get_company_name(ticker: str) -> str:
    """Get company name from ticker using yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('shortName', '') or info.get('longName', '')
    except Exception:
        return ""


# ============================================================
# LangChain Tools
# ============================================================

@tool
def get_stock_news(ticker: str, days_back: int = 7) -> str:
    """
    Fetches recent news for a stock from multiple Indian and global sources.

    Sources (in priority order):
    1. NSE Corporate Announcements (official filings)
    2. MoneyControl (Indian financial news)
    3. Economic Times RSS (Indian market news)
    4. NewsAPI (global, rate-limited to 100/day)
    5. Google News RSS (fallback)

    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS", "TCS.NS")
        days_back: Number of days to look back (default: 7)

    Returns:
        Formatted news summary with source attribution
    """
    start_time = time.time()
    cache_key = f"stock_news_{ticker}_{days_back}_{datetime.now().strftime('%Y%m%d_%H')}"

    # Check cache
    cached_result = news_cache.get(cache_key)
    if cached_result:
        logger.log_fetch(ticker, "stock_news", True, 0, 0, "cache_hit")
        return cached_result

    all_articles: List[NewsArticle] = []
    sources_used = []
    company_name = get_company_name(ticker)

    # 1. NSE Announcements (highest priority for official news)
    try:
        nse_articles = _fetch_nse_announcements(ticker, days_back)
        all_articles.extend(nse_articles)
        if nse_articles:
            sources_used.append("NSE")
    except Exception:
        pass

    # 2. MoneyControl
    try:
        mc_articles = _fetch_moneycontrol_news(ticker, company_name)
        all_articles.extend(mc_articles)
        if mc_articles:
            sources_used.append("MoneyControl")
    except Exception:
        pass

    # 3. Economic Times RSS
    try:
        et_articles = _fetch_economic_times_rss(ticker, ['markets', 'stocks'])
        all_articles.extend(et_articles)
        if et_articles:
            sources_used.append("Economic Times")
    except Exception:
        pass

    # 4. NewsAPI (if key available and not rate limited)
    if os.getenv('NEWSAPI_KEY'):
        try:
            search_query = company_name or ticker.replace('.NS', '').replace('.BO', '')
            newsapi_articles = _fetch_newsapi(search_query, ticker, days_back)
            all_articles.extend(newsapi_articles)
            if newsapi_articles:
                sources_used.append("NewsAPI")
        except Exception:
            pass

    # 5. Google News RSS (fallback)
    try:
        search_query = f"{company_name or ticker.replace('.NS', '')} stock India"
        google_articles = _fetch_google_news_rss(search_query, ticker)
        all_articles.extend(google_articles)
        if google_articles:
            sources_used.append("Google News")
    except Exception:
        pass

    # Filter and sort articles
    filtered_articles = filter_articles(all_articles, ticker, min_relevance=0.1, max_age_days=days_back)

    # Deduplicate by title similarity
    seen_titles = set()
    unique_articles = []
    for article in filtered_articles:
        title_key = article.title.lower()[:50]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(article)

    # Format report
    latency_ms = (time.time() - start_time) * 1000

    report = f"""
NEWS SUMMARY: {ticker} (Last {days_back} days)
{'='*60}

"""

    # Separate official announcements
    official = [a for a in unique_articles if a.source == 'NSE Announcement']
    news = [a for a in unique_articles if a.source != 'NSE Announcement']

    if official:
        report += "OFFICIAL ANNOUNCEMENTS (NSE):\n"
        for i, article in enumerate(official[:5], 1):
            report += f"{i}. [{article.published_date.strftime('%Y-%m-%d')}] {article.title}\n"
            if article.summary:
                report += f"   Details: {article.summary}\n"
            report += "\n"

    if news:
        report += "RECENT NEWS:\n"
        for i, article in enumerate(news[:8], 1):
            relevance = "HIGH" if article.relevance_score > 0.5 else "MEDIUM" if article.relevance_score > 0.2 else "LOW"
            report += f"{i}. [{article.source}] {article.published_date.strftime('%Y-%m-%d')}: {article.title}\n"
            if article.summary:
                report += f"   Summary: {article.summary[:150]}...\n"
            report += f"   Relevance: {relevance}\n\n"

    if not official and not news:
        report += "No significant news found for this period.\n"

    # Summary stats
    positive_keywords = ['growth', 'profit', 'beat', 'upgrade', 'acquisition', 'expansion', 'strong']
    negative_keywords = ['loss', 'decline', 'downgrade', 'concern', 'weak', 'risk', 'lawsuit']

    positive_count = sum(1 for a in unique_articles
                        if any(k in a.title.lower() for k in positive_keywords))
    negative_count = sum(1 for a in unique_articles
                        if any(k in a.title.lower() for k in negative_keywords))

    report += f"""
NEWS SENTIMENT INDICATORS:
- Positive headlines: {positive_count}
- Negative headlines: {negative_count}
- Neutral headlines: {len(unique_articles) - positive_count - negative_count}

SOURCES USED: {', '.join(sources_used) if sources_used else 'None available'}
{'='*60}
"""

    # Cache result
    news_cache.set(cache_key, report, ttl_seconds=1800)

    logger.log_fetch(ticker, "stock_news", True, latency_ms, len(unique_articles))

    return report


@tool
def get_indian_market_news(sectors: str = "all") -> str:
    """
    Fetches broad Indian market news from Economic Times.

    Useful for understanding market-wide sentiment and sector trends.

    Args:
        sectors: Comma-separated sectors or "all" (e.g., "banking,tech,markets")
                Available: markets, stocks, ipo, tech, banking

    Returns:
        Market-wide news summary
    """
    start_time = time.time()
    cache_key = f"market_news_{sectors}_{datetime.now().strftime('%Y%m%d_%H')}"

    cached_result = news_cache.get(cache_key)
    if cached_result:
        return cached_result

    # Parse sectors
    if sectors.lower() == "all":
        categories = ['markets', 'stocks', 'tech', 'banking']
    else:
        categories = [s.strip().lower() for s in sectors.split(',')]

    articles = _fetch_economic_times_rss("", categories)
    articles = filter_articles(articles, "", min_relevance=0.0, max_age_days=3)

    report = f"""
INDIAN MARKET NEWS
{'='*60}
Sectors: {', '.join(categories)}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} IST

"""

    for category in categories:
        cat_articles = [a for a in articles if category in a.source.lower()][:5]
        if cat_articles:
            report += f"\n{category.upper()}:\n"
            for article in cat_articles:
                report += f"- {article.title}\n"

    report += f"""
{'='*60}
"""

    news_cache.set(cache_key, report, ttl_seconds=1800)
    latency_ms = (time.time() - start_time) * 1000
    logger.log_fetch("market", "indian_market_news", True, latency_ms, len(articles))

    return report


@tool
def get_nse_announcements(ticker: str, days_back: int = 30) -> str:
    """
    Fetches official corporate announcements from NSE India.

    These are official filings including:
    - Board meeting outcomes
    - Financial results
    - Dividend announcements
    - AGM/EGM notices
    - Other material disclosures

    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS")
        days_back: How far back to search (default: 30)

    Returns:
        Official NSE announcements
    """
    start_time = time.time()
    cache_key = f"nse_ann_{ticker}_{days_back}_{datetime.now().strftime('%Y%m%d')}"

    cached_result = news_cache.get(cache_key)
    if cached_result:
        return cached_result

    articles = _fetch_nse_announcements(ticker, days_back)

    report = f"""
NSE CORPORATE ANNOUNCEMENTS: {ticker}
{'='*60}
Period: Last {days_back} days

"""

    if articles:
        for i, article in enumerate(articles, 1):
            report += f"{i}. [{article.published_date.strftime('%Y-%m-%d')}] {article.title}\n"
            if article.summary:
                report += f"   {article.summary}\n"
            report += "\n"
    else:
        report += "No announcements found for this period.\n"
        report += "Note: NSE API may require browser session. Check ticker format.\n"

    report += f"""
{'='*60}
Source: NSE India (www.nseindia.com)
"""

    news_cache.set(cache_key, report, ttl_seconds=3600)
    latency_ms = (time.time() - start_time) * 1000
    logger.log_fetch(ticker, "nse_announcements_tool", True, latency_ms, len(articles))

    return report


@tool
def search_financial_news(query: str, source: str = "auto") -> str:
    """
    Search for specific financial news across sources.

    Args:
        query: Search query (e.g., "RBI policy rate", "IT sector outlook India")
        source: "newsapi", "google", "et", or "auto" (tries sources in order)

    Returns:
        Search results from specified or best available source
    """
    start_time = time.time()
    cache_key = f"search_{hashlib.md5(query.encode()).hexdigest()[:10]}_{source}"

    cached_result = news_cache.get(cache_key)
    if cached_result:
        return cached_result

    articles = []
    source_used = ""

    if source == "auto" or source == "newsapi":
        if os.getenv('NEWSAPI_KEY'):
            articles = _fetch_newsapi(query, "", 7)
            if articles:
                source_used = "NewsAPI"

    if not articles and (source == "auto" or source == "google"):
        articles = _fetch_google_news_rss(query, "")
        if articles:
            source_used = "Google News"

    if not articles and (source == "auto" or source == "et"):
        # For ET, we search across all categories
        articles = _fetch_economic_times_rss("", ['markets', 'stocks', 'tech', 'banking'])
        # Filter by query
        query_lower = query.lower()
        articles = [a for a in articles if query_lower in a.title.lower() or
                   (a.summary and query_lower in a.summary.lower())]
        if articles:
            source_used = "Economic Times"

    report = f"""
FINANCIAL NEWS SEARCH: "{query}"
{'='*60}
Source: {source_used or 'No results found'}

"""

    if articles:
        for i, article in enumerate(articles[:10], 1):
            report += f"{i}. [{article.source}] {article.published_date.strftime('%Y-%m-%d')}\n"
            report += f"   {article.title}\n"
            if article.summary:
                report += f"   {article.summary[:150]}...\n"
            report += f"   URL: {article.url}\n\n"
    else:
        report += "No matching articles found.\n"
        report += "Try different keywords or check if NewsAPI key is configured.\n"

    report += f"""
{'='*60}
"""

    news_cache.set(cache_key, report, ttl_seconds=1800)
    latency_ms = (time.time() - start_time) * 1000
    logger.log_fetch("search", "financial_news_search", True, latency_ms, len(articles))

    return report


# Export all tools
NEWS_TOOLS = [
    get_stock_news,
    get_indian_market_news,
    get_nse_announcements,
    search_financial_news,
]
