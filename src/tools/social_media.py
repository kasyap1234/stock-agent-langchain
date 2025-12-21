"""
Social Media Sentiment Tools for Indian Stock Analysis.

Provides sentiment data from:
- Twitter/X (via free Nitter scraping)
- Reddit (r/IndianStreetBets, r/IndiaInvestments via PRAW)
"""

import os
import time
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from langchain.tools import tool
from src.utils.logging_config import ToolLogger
from src.tools.cache import (
    social_cache,
    twitter_limiter,
    reddit_limiter,
)

# Initialize logger
logger = ToolLogger("social_media")


# ============================================================
# Data Structures
# ============================================================

@dataclass
class SocialMention:
    """Represents a single social media mention."""
    source: str  # 'twitter' or 'reddit'
    content: str
    timestamp: datetime
    engagement: int  # likes + retweets or upvotes
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # 'Bullish', 'Bearish', 'Neutral'
    author: str
    url: Optional[str] = None


@dataclass
class SocialSentimentResult:
    """Aggregated sentiment result from social media."""
    ticker: str
    source: str
    total_mentions: int
    sentiment_score: float  # -1 (bearish) to 1 (bullish)
    sentiment_label: str  # 'Bullish', 'Bearish', 'Neutral'
    volume_trend: str  # 'Increasing', 'Stable', 'Decreasing'
    top_mentions: List[SocialMention]
    analysis_time: datetime
    confidence: float  # 0-1 based on data quality


# ============================================================
# Indian Stock Cashtag Mapping
# ============================================================

INDIAN_CASHTAGS = {
    'RELIANCE.NS': ['$RELIANCE', '$RIL', 'Reliance Industries', 'Reliance'],
    'TCS.NS': ['$TCS', 'TCS', 'Tata Consultancy'],
    'HDFCBANK.NS': ['$HDFCBANK', 'HDFC Bank', 'HDFC'],
    'INFY.NS': ['$INFY', 'Infosys', 'INFY'],
    'ICICIBANK.NS': ['$ICICIBANK', 'ICICI Bank', 'ICICI'],
    'HINDUNILVR.NS': ['$HUL', 'Hindustan Unilever', 'HUL'],
    'SBIN.NS': ['$SBIN', 'SBI', 'State Bank'],
    'BHARTIARTL.NS': ['$BHARTIARTL', 'Airtel', 'Bharti Airtel'],
    'ITC.NS': ['$ITC', 'ITC Limited', 'ITC'],
    'KOTAKBANK.NS': ['$KOTAKBANK', 'Kotak Bank', 'Kotak'],
    'LT.NS': ['$LT', 'L&T', 'Larsen Toubro'],
    'AXISBANK.NS': ['$AXISBANK', 'Axis Bank'],
    'WIPRO.NS': ['$WIPRO', 'Wipro'],
    'HCLTECH.NS': ['$HCLTECH', 'HCL Tech', 'HCL Technologies'],
    'ASIANPAINT.NS': ['$ASIANPAINT', 'Asian Paints'],
    'MARUTI.NS': ['$MARUTI', 'Maruti Suzuki', 'Maruti'],
    'TITAN.NS': ['$TITAN', 'Titan Company'],
    'SUNPHARMA.NS': ['$SUNPHARMA', 'Sun Pharma', 'Sun Pharmaceutical'],
    'BAJFINANCE.NS': ['$BAJFINANCE', 'Bajaj Finance'],
    'TATAMOTORS.NS': ['$TATAMOTORS', 'Tata Motors'],
    'TATASTEEL.NS': ['$TATASTEEL', 'Tata Steel'],
    'ADANIENT.NS': ['$ADANIENT', 'Adani Enterprises', 'Adani'],
    'ULTRACEMCO.NS': ['$ULTRACEMCO', 'UltraTech Cement'],
    'NTPC.NS': ['$NTPC', 'NTPC Limited'],
    'POWERGRID.NS': ['$POWERGRID', 'Power Grid'],
}


def get_search_terms(ticker: str) -> List[str]:
    """Get search terms for a ticker."""
    base_symbol = ticker.replace('.NS', '').replace('.BO', '')

    if ticker in INDIAN_CASHTAGS:
        return INDIAN_CASHTAGS[ticker]

    return [f'${base_symbol}', base_symbol]


# ============================================================
# Sentiment Analysis (Rule-based)
# ============================================================

BULLISH_KEYWORDS = [
    'buy', 'bullish', 'long', 'breakout', 'strong', 'accumulate',
    'undervalued', 'target', 'upside', 'growth', 'positive',
    'rocket', 'moon', 'diamond hands', 'hodl', 'add', 'calls',
    'excellent', 'beat', 'outperform', 'upgrade', 'jackpot',
    'multibagger', 'wealth creator', 'holding', 'dip buying'
]

BEARISH_KEYWORDS = [
    'sell', 'bearish', 'short', 'breakdown', 'weak', 'avoid',
    'overvalued', 'downside', 'negative', 'crash', 'dump', 'puts',
    'exit', 'cut loss', 'downgrade', 'miss', 'disappoint',
    'paper hands', 'red flag', 'warning', 'risk', 'scam',
    'fraud', 'manipulation', 'trap', 'book profit'
]


def analyze_text_sentiment(text: str) -> tuple:
    """
    Simple rule-based sentiment analysis for social media.

    Returns:
        (score, label) where score is -1 to 1
    """
    text_lower = text.lower()

    bullish_count = sum(1 for word in BULLISH_KEYWORDS if word in text_lower)
    bearish_count = sum(1 for word in BEARISH_KEYWORDS if word in text_lower)

    total = bullish_count + bearish_count
    if total == 0:
        return 0.0, "Neutral"

    score = (bullish_count - bearish_count) / total

    if score > 0.2:
        return score, "Bullish"
    elif score < -0.2:
        return score, "Bearish"
    else:
        return score, "Neutral"


# ============================================================
# Twitter/X Scraping (via Nitter)
# ============================================================

# Nitter instances - these are volunteer-maintained and may go down
# The scraper will try each instance in order and fall back to the next
NITTER_INSTANCES = [
    'https://nitter.poast.org',
    'https://nitter.privacydev.net',
    'https://nitter.net',
]

# Cache for healthy Nitter instances
_healthy_instances: List[str] = []
_last_health_check: Optional[datetime] = None
HEALTH_CHECK_INTERVAL = timedelta(minutes=30)


def _check_nitter_health(instance: str, timeout: int = 5) -> bool:
    """
    Check if a Nitter instance is healthy with a lightweight request.

    Args:
        instance: Nitter instance URL
        timeout: Request timeout in seconds

    Returns:
        True if instance responds successfully
    """
    import requests
    try:
        response = requests.head(instance, timeout=timeout, allow_redirects=True)
        return response.status_code < 400
    except Exception:
        return False


def _get_healthy_nitter_instances() -> List[str]:
    """
    Get list of healthy Nitter instances, with caching.

    Performs health checks periodically and caches results.
    Falls back to all instances if none are healthy.
    """
    global _healthy_instances, _last_health_check

    now = datetime.now()

    # Return cached healthy instances if still valid
    if _last_health_check and (now - _last_health_check) < HEALTH_CHECK_INTERVAL:
        if _healthy_instances:
            return _healthy_instances

    # Perform health checks
    healthy = []
    for instance in NITTER_INSTANCES:
        if _check_nitter_health(instance):
            healthy.append(instance)

    _last_health_check = now
    _healthy_instances = healthy

    # Fall back to all instances if none are healthy
    if not healthy:
        logger.logger.warning("no_healthy_nitter_instances",
                             message="All Nitter instances appear down, trying all")
        return NITTER_INSTANCES

    return healthy


def _scrape_twitter_nitter(query: str, limit: int = 20) -> List[Dict]:
    """
    Scrape Twitter via Nitter instances (free, no API needed).

    Uses health-checked instances with automatic fallback.

    Args:
        query: Search query
        limit: Max tweets to fetch

    Returns:
        List of tweet dicts
    """
    import requests
    from bs4 import BeautifulSoup

    tweets = []
    instances = _get_healthy_nitter_instances()

    for instance in instances:
        try:
            if not twitter_limiter.acquire(blocking=True, timeout=30):
                continue

            # URL encode query
            from urllib.parse import quote
            search_url = f"{instance}/search?f=tweets&q={quote(query)}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(search_url, headers=headers, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find tweet items
                tweet_items = soup.select('.timeline-item, .tweet-body')[:limit]

                for item in tweet_items:
                    try:
                        # Get tweet content
                        content_elem = item.select_one('.tweet-content, .tweet-text')
                        if not content_elem:
                            continue

                        tweet_text = content_elem.get_text(strip=True)

                        # Get engagement (likes, retweets)
                        engagement = 0
                        stats = item.select('.tweet-stat, .icon-container')
                        for stat in stats:
                            num_text = stat.get_text(strip=True)
                            nums = re.findall(r'\d+', num_text)
                            if nums:
                                engagement += int(nums[0])

                        # Get author
                        author_elem = item.select_one('.username, .tweet-name-row a')
                        author = author_elem.get_text(strip=True) if author_elem else 'unknown'
                        author = author.replace('@', '')

                        tweets.append({
                            'content': tweet_text,
                            'engagement': engagement,
                            'timestamp': datetime.now(),
                            'author': author,
                            'source': 'twitter'
                        })

                    except Exception:
                        continue

                if tweets:
                    logger.log_fetch("N/A", f"twitter_nitter", True, 0, len(tweets))
                    break

        except Exception as e:
            logger.log_fetch("N/A", f"twitter_nitter_{instance}", False, 0, 0, str(e))
            continue

    return tweets


def fetch_twitter_sentiment(ticker: str) -> List[SocialMention]:
    """
    Fetch Twitter mentions for a stock.

    Args:
        ticker: Stock ticker

    Returns:
        List of SocialMention objects

    Note:
        Tweet timestamps are approximate (scraping limitation).
        Date filtering is not available with Nitter scraping.
    """
    search_terms = get_search_terms(ticker)
    all_mentions = []

    for term in search_terms[:2]:  # Limit to avoid rate limits
        try:
            tweets = _scrape_twitter_nitter(term, limit=15)

            for tweet in tweets:
                score, label = analyze_text_sentiment(tweet['content'])

                mention = SocialMention(
                    source='twitter',
                    content=tweet['content'][:500],
                    timestamp=tweet.get('timestamp', datetime.now()),
                    engagement=tweet.get('engagement', 0),
                    sentiment_score=score,
                    sentiment_label=label,
                    author=tweet.get('author', 'unknown'),
                    url=None
                )
                all_mentions.append(mention)

        except Exception as e:
            logger.log_fetch(ticker, "twitter_fetch", False, 0, 0, str(e))

    return all_mentions


# ============================================================
# Reddit Fetching (PRAW)
# ============================================================

def _get_reddit_client():
    """
    Get PRAW Reddit client.

    Requires environment variables:
    - REDDIT_CLIENT_ID
    - REDDIT_CLIENT_SECRET
    """
    try:
        import praw
    except ImportError:
        logger.logger.warning("praw_not_installed")
        return None

    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT', 'StockAnalyzer/1.0 by YourUsername')

    if not client_id or not client_secret:
        logger.logger.warning("reddit_credentials_missing")
        return None

    try:
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    except Exception as e:
        logger.logger.error("reddit_client_error", error=str(e))
        return None


def fetch_reddit_sentiment(
    ticker: str,
    subreddits: List[str] = None,
    days_back: int = 7,
    limit: int = 30
) -> List[SocialMention]:
    """
    Fetch Reddit mentions from Indian investing subreddits.

    Args:
        ticker: Stock ticker
        subreddits: List of subreddits (default: IndianStreetBets, IndiaInvestments)
        days_back: How many days to look back
        limit: Max posts per subreddit

    Returns:
        List of SocialMention objects
    """
    if subreddits is None:
        subreddits = ['IndianStreetBets', 'IndiaInvestments']

    search_terms = get_search_terms(ticker)
    all_mentions = []

    reddit = _get_reddit_client()
    if reddit is None:
        return all_mentions

    for subreddit_name in subreddits:
        try:
            if not reddit_limiter.acquire(blocking=True, timeout=30):
                continue

            subreddit = reddit.subreddit(subreddit_name)

            for term in search_terms[:2]:
                try:
                    query = f"{term}"

                    for post in subreddit.search(query, time_filter='week', limit=limit):
                        # Combine title and body
                        text = f"{post.title} {post.selftext or ''}"
                        score, label = analyze_text_sentiment(text)

                        post_time = datetime.fromtimestamp(post.created_utc)

                        mention = SocialMention(
                            source='reddit',
                            content=text[:500],
                            timestamp=post_time,
                            engagement=post.score + post.num_comments,
                            sentiment_score=score,
                            sentiment_label=label,
                            author=str(post.author) if post.author else 'deleted',
                            url=f"https://reddit.com{post.permalink}"
                        )
                        all_mentions.append(mention)

                except Exception as e:
                    logger.log_fetch(ticker, f"reddit_{subreddit_name}", False, 0, 0, str(e))

            logger.log_fetch(ticker, f"reddit_{subreddit_name}", True, 0, len(all_mentions))

        except Exception as e:
            logger.log_fetch(ticker, f"reddit_{subreddit_name}", False, 0, 0, str(e))

    return all_mentions


# ============================================================
# Aggregation Utilities
# ============================================================

def aggregate_social_sentiment(
    mentions: List[SocialMention],
    source: str = 'all'
) -> SocialSentimentResult:
    """
    Aggregate sentiment from multiple mentions with volume weighting.

    Higher engagement posts get more weight.

    Args:
        mentions: List of SocialMention objects
        source: Filter by source ('twitter', 'reddit', or 'all')

    Returns:
        SocialSentimentResult
    """
    if source != 'all':
        mentions = [m for m in mentions if m.source == source]

    if not mentions:
        return SocialSentimentResult(
            ticker='N/A',
            source=source,
            total_mentions=0,
            sentiment_score=0.0,
            sentiment_label='Neutral',
            volume_trend='Unknown',
            top_mentions=[],
            analysis_time=datetime.now(),
            confidence=0.0
        )

    # Volume-weighted sentiment
    total_weight = sum(max(m.engagement, 1) for m in mentions)
    weighted_score = sum(
        m.sentiment_score * max(m.engagement, 1) / total_weight
        for m in mentions
    )

    # Determine label
    if weighted_score > 0.15:
        label = "Bullish"
    elif weighted_score < -0.15:
        label = "Bearish"
    else:
        label = "Neutral"

    # Volume trend
    now = datetime.now()
    recent = [m for m in mentions if (now - m.timestamp).days <= 1]
    older = [m for m in mentions if (now - m.timestamp).days > 1]

    if len(recent) > len(older) * 1.5:
        volume_trend = "Increasing"
    elif len(recent) < len(older) * 0.5:
        volume_trend = "Decreasing"
    else:
        volume_trend = "Stable"

    # Confidence based on sample size
    confidence = min(len(mentions) / 15, 1.0)

    # Top mentions by engagement
    top_mentions = sorted(mentions, key=lambda m: m.engagement, reverse=True)[:5]

    return SocialSentimentResult(
        ticker='N/A',
        source=source,
        total_mentions=len(mentions),
        sentiment_score=weighted_score,
        sentiment_label=label,
        volume_trend=volume_trend,
        top_mentions=top_mentions,
        analysis_time=datetime.now(),
        confidence=confidence
    )


# ============================================================
# LangChain Tools
# ============================================================

@tool
def get_twitter_sentiment(ticker: str) -> str:
    """
    Fetch and analyze Twitter/X sentiment for an Indian stock.

    Uses free scraping methods (no paid API required).
    Analyzes cashtags like $RELIANCE, $TCS and stock discussions.

    Args:
        ticker: Stock ticker symbol (e.g., "RELIANCE.NS", "TCS.NS")

    Returns:
        Twitter sentiment analysis report with:
        - Overall sentiment (Bullish/Bearish/Neutral)
        - Mention count and volume trend
        - Top influential tweets
        - Confidence score
    """
    start_time = time.time()
    cache_key = f"twitter_{ticker}_{datetime.now().strftime('%Y%m%d_%H')}"

    # Check cache
    cached_result = social_cache.get(cache_key)
    if cached_result:
        logger.log_fetch(ticker, "twitter_sentiment", True, 0, 0, "cache_hit")
        return cached_result

    try:
        mentions = fetch_twitter_sentiment(ticker)
        result = aggregate_social_sentiment(mentions, source='twitter')
        result.ticker = ticker

        report = f"""
TWITTER SENTIMENT ANALYSIS: {ticker}
{'='*60}

OVERALL SENTIMENT: {result.sentiment_label}
Sentiment Score: {result.sentiment_score:+.2f} (-1 bearish to +1 bullish)
Confidence: {result.confidence*100:.0f}%

VOLUME ANALYSIS:
Total Mentions (3 days): {result.total_mentions}
Volume Trend: {result.volume_trend}

"""

        if result.top_mentions:
            report += "TOP DISCUSSIONS:\n"
            for i, mention in enumerate(result.top_mentions[:3], 1):
                snippet = mention.content[:100] + "..." if len(mention.content) > 100 else mention.content
                report += f"{i}. @{mention.author} (Engagement: {mention.engagement})\n"
                report += f"   \"{snippet}\"\n"
                report += f"   Sentiment: {mention.sentiment_label}\n\n"
        else:
            report += "No significant discussions found.\n"
            report += "Note: Twitter scraping may be limited. Try web search for more results.\n"

        report += f"""
INTERPRETATION:
"""
        if result.sentiment_label == "Bullish":
            report += "- Positive social buzz, retail interest is high\n"
            report += "- Watch for momentum continuation\n"
        elif result.sentiment_label == "Bearish":
            report += "- Negative sentiment prevailing\n"
            report += "- May indicate selling pressure ahead\n"
        else:
            report += "- Mixed/neutral social sentiment\n"
            report += "- Focus on technical/fundamental factors\n"

        if result.volume_trend == "Increasing":
            report += "- Rising discussion volume - increasing interest\n"
        elif result.volume_trend == "Decreasing":
            report += "- Declining discussion - interest waning\n"

        report += f"""
{'='*60}
Note: Based on public social media data. Use as one input among many.
"""

        social_cache.set(cache_key, report, ttl_seconds=1800)

        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "twitter_sentiment", True, latency_ms, result.total_mentions)

        return report

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "twitter_sentiment", False, latency_ms, 0, str(e))
        return f"Error fetching Twitter sentiment for {ticker}: {str(e)}"


@tool
def get_reddit_sentiment(ticker: str) -> str:
    """
    Fetch and analyze Reddit sentiment from Indian investing communities.

    Analyzes posts from:
    - r/IndianStreetBets (retail traders, momentum plays)
    - r/IndiaInvestments (long-term investors, value focus)

    Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.

    Args:
        ticker: Stock ticker symbol (e.g., "RELIANCE.NS", "TCS.NS")

    Returns:
        Reddit sentiment analysis report with:
        - Overall sentiment (Bullish/Bearish/Neutral)
        - Subreddit breakdown
        - Top posts and discussions
        - Confidence score
    """
    start_time = time.time()
    cache_key = f"reddit_{ticker}_{datetime.now().strftime('%Y%m%d_%H')}"

    cached_result = social_cache.get(cache_key)
    if cached_result:
        logger.log_fetch(ticker, "reddit_sentiment", True, 0, 0, "cache_hit")
        return cached_result

    # Check if Reddit credentials are available
    if not os.getenv('REDDIT_CLIENT_ID') or not os.getenv('REDDIT_CLIENT_SECRET'):
        return f"""
REDDIT SENTIMENT ANALYSIS: {ticker}
{'='*60}

ERROR: Reddit API credentials not configured.

To enable Reddit sentiment analysis:
1. Go to https://www.reddit.com/prefs/apps
2. Create a new "script" type application
3. Set environment variables:
   - REDDIT_CLIENT_ID=your_client_id
   - REDDIT_CLIENT_SECRET=your_client_secret

{'='*60}
"""

    try:
        mentions = fetch_reddit_sentiment(ticker, days_back=7, limit=30)
        result = aggregate_social_sentiment(mentions, source='reddit')
        result.ticker = ticker

        # Separate by subreddit
        isb_count = sum(1 for m in mentions if 'indianstreetbets' in (m.url or '').lower())
        ii_count = sum(1 for m in mentions if 'indiainvestments' in (m.url or '').lower())

        report = f"""
REDDIT SENTIMENT ANALYSIS: {ticker}
{'='*60}

OVERALL SENTIMENT: {result.sentiment_label}
Sentiment Score: {result.sentiment_score:+.2f} (-1 bearish to +1 bullish)
Confidence: {result.confidence*100:.0f}%

SUBREDDIT BREAKDOWN:
- r/IndianStreetBets: ~{isb_count} mentions (Retail traders, speculative)
- r/IndiaInvestments: ~{ii_count} mentions (Long-term, value-focused)

VOLUME ANALYSIS:
Total Mentions (7 days): {result.total_mentions}
Volume Trend: {result.volume_trend}

"""

        if result.top_mentions:
            report += "TOP DISCUSSIONS:\n"
            for i, mention in enumerate(result.top_mentions[:3], 1):
                snippet = mention.content[:100] + "..." if len(mention.content) > 100 else mention.content
                report += f"{i}. u/{mention.author} (Score: {mention.engagement})\n"
                report += f"   \"{snippet}\"\n"
                if mention.url:
                    report += f"   Link: {mention.url}\n"
                report += f"   Sentiment: {mention.sentiment_label}\n\n"
        else:
            report += "No significant Reddit discussions found.\n"

        report += f"""
INTERPRETATION:
"""
        if isb_count > ii_count * 2:
            report += "- Heavy r/IndianStreetBets activity - FOMO/momentum play\n"
            report += "- Potential for high volatility, watch for reversal\n"
        elif ii_count > isb_count:
            report += "- r/IndiaInvestments discussion - Fundamental interest\n"
            report += "- Likely longer-term investment thesis\n"

        if result.sentiment_label == "Bullish":
            report += "- Positive retail sentiment on Reddit\n"
        elif result.sentiment_label == "Bearish":
            report += "- Negative retail sentiment on Reddit\n"

        report += f"""
{'='*60}
Note: Reddit sentiment often reflects retail trader views.
"""

        social_cache.set(cache_key, report, ttl_seconds=3600)

        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "reddit_sentiment", True, latency_ms, result.total_mentions)

        return report

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "reddit_sentiment", False, latency_ms, 0, str(e))
        return f"Error fetching Reddit sentiment for {ticker}: {str(e)}"


@tool
def get_social_sentiment_aggregate(ticker: str) -> str:
    """
    Get aggregated social media sentiment from all sources (Twitter + Reddit).

    Provides a combined view with volume-weighted sentiment across platforms.
    Best used for overall social sentiment assessment.

    Args:
        ticker: Stock ticker symbol (e.g., "RELIANCE.NS")

    Returns:
        Combined social sentiment report with:
        - Aggregate sentiment score
        - Platform comparison
        - Volume and trending analysis
        - Trading implications
    """
    start_time = time.time()
    cache_key = f"social_agg_{ticker}_{datetime.now().strftime('%Y%m%d_%H')}"

    cached_result = social_cache.get(cache_key)
    if cached_result:
        return cached_result

    try:
        # Fetch from both sources
        twitter_mentions = []
        reddit_mentions = []

        try:
            twitter_mentions = fetch_twitter_sentiment(ticker)
        except Exception as e:
            logger.log_fetch(ticker, "social_agg_twitter", False, 0, 0, str(e))

        try:
            if os.getenv('REDDIT_CLIENT_ID'):
                reddit_mentions = fetch_reddit_sentiment(ticker, days_back=7)
        except Exception as e:
            logger.log_fetch(ticker, "social_agg_reddit", False, 0, 0, str(e))

        all_mentions = twitter_mentions + reddit_mentions

        twitter_result = aggregate_social_sentiment(twitter_mentions, 'twitter')
        reddit_result = aggregate_social_sentiment(reddit_mentions, 'reddit')
        combined_result = aggregate_social_sentiment(all_mentions, 'all')
        combined_result.ticker = ticker

        # Determine if trending
        is_trending = (
            combined_result.volume_trend == "Increasing" and
            combined_result.total_mentions > 10
        )

        report = f"""
SOCIAL MEDIA SENTIMENT AGGREGATE: {ticker}
{'='*60}

COMBINED SENTIMENT: {combined_result.sentiment_label}
Aggregate Score: {combined_result.sentiment_score:+.2f} (-1 to +1)
Confidence: {combined_result.confidence*100:.0f}%
Trending: {'YES - High activity detected' if is_trending else 'No'}

PLATFORM BREAKDOWN:
{'='*60}
| Platform | Mentions | Sentiment | Score   | Trend      |
|----------|----------|-----------|---------|------------|
| Twitter  | {twitter_result.total_mentions:>8} | {twitter_result.sentiment_label:>9} | {twitter_result.sentiment_score:>+.2f}   | {twitter_result.volume_trend:>10} |
| Reddit   | {reddit_result.total_mentions:>8} | {reddit_result.sentiment_label:>9} | {reddit_result.sentiment_score:>+.2f}   | {reddit_result.volume_trend:>10} |
| COMBINED | {combined_result.total_mentions:>8} | {combined_result.sentiment_label:>9} | {combined_result.sentiment_score:>+.2f}   | {combined_result.volume_trend:>10} |
{'='*60}

TRADING IMPLICATIONS:
"""

        # Consensus check
        if twitter_result.sentiment_label == reddit_result.sentiment_label and twitter_result.total_mentions > 0 and reddit_result.total_mentions > 0:
            if twitter_result.sentiment_label == "Bullish":
                report += "- CONSENSUS BULLISH: Both platforms positive\n"
                report += "- Strong retail support, momentum play possible\n"
            elif twitter_result.sentiment_label == "Bearish":
                report += "- CONSENSUS BEARISH: Both platforms negative\n"
                report += "- Caution advised, selling pressure likely\n"
            else:
                report += "- CONSENSUS NEUTRAL: No strong directional bias\n"
        elif twitter_result.total_mentions > 0 and reddit_result.total_mentions > 0:
            report += "- DIVERGENCE: Platforms showing different signals\n"
            report += f"  Twitter: {twitter_result.sentiment_label}, Reddit: {reddit_result.sentiment_label}\n"
            report += "- Mixed signals, wait for clarity\n"
        else:
            report += "- LIMITED DATA: Not all platforms have sufficient mentions\n"

        if is_trending:
            report += "\nTRENDING ALERT:\n"
            report += "- High social media activity detected\n"
            report += "- Expect increased volatility\n"
            report += "- Monitor for breakout/breakdown\n"

        report += f"""
CONFIDENCE FACTORS:
- Sample size: {combined_result.total_mentions} mentions ({'>15 Good' if combined_result.total_mentions > 15 else '<15 Limited'})
- Platform coverage: {'Both' if twitter_mentions and reddit_mentions else 'Partial'}

CAVEATS:
- Social sentiment is ONE factor - combine with technicals/fundamentals
- Retail sentiment can be contrarian indicator at extremes
- Large spikes may indicate manipulation

{'='*60}
"""

        social_cache.set(cache_key, report, ttl_seconds=1800)

        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "social_sentiment_aggregate", True, latency_ms, len(all_mentions))

        return report

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "social_sentiment_aggregate", False, latency_ms, 0, str(e))
        return f"Error fetching social sentiment for {ticker}: {str(e)}"


# Export all tools
SOCIAL_MEDIA_TOOLS = [
    get_twitter_sentiment,
    get_reddit_sentiment,
    get_social_sentiment_aggregate,
]
