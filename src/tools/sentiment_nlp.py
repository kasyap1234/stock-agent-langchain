"""
Hybrid NLP Sentiment Analysis for Indian Stock Analysis.

Provides:
- FinBERT (ProsusAI/finbert) for financial text sentiment
- VADER as fallback for general text
- Aggregation and sector-specific adjustments
"""

import time
import threading
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from langchain.tools import tool
from src.utils.logging_config import ToolLogger
from src.tools.cache import sentiment_cache
from src.tools.search import web_search

# Initialize logger
logger = ToolLogger("sentiment_nlp")


# ============================================================
# Data Models
# ============================================================

@dataclass
class SentimentScore:
    """Individual sentiment score for a piece of text."""
    text: str  # Truncated text for reference
    score: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0 to 1
    source: str  # "finbert" or "vader"
    label: str  # "positive", "negative", "neutral"


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment from multiple sources."""
    overall_score: float  # Weighted average: -1 to +1
    overall_confidence: float  # Combined confidence: 0 to 1
    overall_label: str  # "Bullish", "Bearish", "Neutral"
    num_articles: int
    positive_count: int
    negative_count: int
    neutral_count: int
    individual_scores: List[SentimentScore]
    summary: str


# ============================================================
# Sector-Specific Sentiment Keywords
# ============================================================

SECTOR_SENTIMENT_KEYWORDS = {
    'Technology': {
        'positive': ['deal win', 'large deal', 'TCV', 'order book', 'guidance raise',
                    'digital', 'cloud', 'AI adoption', 'margin expansion', 'hiring'],
        'negative': ['attrition', 'visa denial', 'guidance cut', 'furlough',
                    'client concentration', 'currency hedge loss', 'layoffs']
    },
    'Financial Services': {
        'positive': ['NIM expansion', 'CASA growth', 'credit growth', 'PCR improve',
                    'asset quality', 'ROA improve', 'book value growth', 'deposits'],
        'negative': ['NPA rise', 'slippage', 'RBI action', 'provisioning spike',
                    'GNPA', 'NNPA', 'restructured book', 'fraud']
    },
    'Healthcare': {
        'positive': ['USFDA approval', 'ANDA approval', 'no observations',
                    'market exclusivity', 'para IV win', 'biosimilar launch'],
        'negative': ['FDA warning', 'form 483', 'import alert', 'price erosion',
                    'clinical trial fail', 'generic competition']
    },
    'Consumer Goods': {
        'positive': ['volume growth', 'market share', 'rural recovery', 'premium mix',
                    'distribution expansion', 'price hike'],
        'negative': ['volume decline', 'raw material cost', 'competition', 'GST impact',
                    'weak rural', 'market share loss']
    },
    'Energy': {
        'positive': ['GRM expansion', 'refining margin', 'capacity addition',
                    'price hike', 'inventory gain', 'subsidy receipt'],
        'negative': ['GRM decline', 'crude spike', 'demand weakness', 'subsidy burden',
                    'inventory loss', 'shutdown']
    },
    'Industrials': {
        'positive': ['order inflow', 'order book', 'capacity expansion', 'infrastructure',
                    'PLI benefit', 'export order'],
        'negative': ['order cancellation', 'execution delay', 'cost overrun',
                    'working capital', 'receivables']
    }
}


# ============================================================
# Model Manager (Singleton)
# ============================================================

class SentimentModelManager:
    """
    Singleton manager for NLP model loading with caching.
    Ensures models are loaded once and reused across calls.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._finbert = None
                    cls._instance._vader = None
                    cls._instance._finbert_failed = False
                    cls._instance._finbert_loading = False
        return cls._instance

    def is_finbert_available(self) -> bool:
        """Check if FinBERT can be loaded."""
        if self._finbert_failed:
            return False
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False

    def get_finbert(self) -> Optional[Tuple[Any, Any]]:
        """
        Lazy-load FinBERT model with caching.

        Returns:
            Tuple of (model, tokenizer) or None if unavailable
        """
        if self._finbert is not None:
            return self._finbert

        if self._finbert_failed or self._finbert_loading:
            return None

        with self._lock:
            if self._finbert is not None:
                return self._finbert

            self._finbert_loading = True

            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch

                model_name = "ProsusAI/finbert"
                logger.logger.info("loading_finbert", model=model_name)

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)

                # Move to GPU if available
                if torch.cuda.is_available():
                    model = model.cuda()
                    logger.logger.info("finbert_gpu_enabled")

                model.eval()  # Set to evaluation mode
                self._finbert = (model, tokenizer)
                logger.logger.info("finbert_loaded_successfully")

            except Exception as e:
                logger.logger.warning("finbert_load_failed", error=str(e))
                self._finbert_failed = True
                self._finbert = None

            finally:
                self._finbert_loading = False

        return self._finbert

    def get_vader(self) -> Any:
        """Lazy-load VADER analyzer."""
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
            except ImportError:
                logger.logger.warning("vader_import_failed")
                return None
        return self._vader


# Global model manager instance
_model_manager = SentimentModelManager()


# ============================================================
# Sentiment Scoring Functions
# ============================================================

def score_text_finbert(text: str) -> Optional[SentimentScore]:
    """
    Score text sentiment using FinBERT (ProsusAI/finbert).

    FinBERT is specifically trained on financial text:
    - Reuters financial news
    - Financial PhraseBank

    Args:
        text: Text to analyze

    Returns:
        SentimentScore or None if FinBERT unavailable
    """
    finbert = _model_manager.get_finbert()
    if finbert is None:
        return None

    model, tokenizer = finbert

    try:
        import torch

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Move to same device as model
        if next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT outputs: [positive, negative, neutral]
        probs = predictions[0].cpu().tolist()
        labels = ["positive", "negative", "neutral"]

        # Get top prediction
        max_idx = probs.index(max(probs))
        label = labels[max_idx]
        confidence = probs[max_idx]

        # Convert to score: positive=+1, negative=-1
        # Score = positive_prob - negative_prob (ranges from -1 to +1)
        score = probs[0] - probs[1]

        return SentimentScore(
            text=text[:100],
            score=score,
            confidence=confidence,
            source="finbert",
            label=label
        )

    except Exception as e:
        logger.logger.warning("finbert_inference_error", error=str(e))
        return None


def score_text_vader(text: str) -> Optional[SentimentScore]:
    """
    Score text sentiment using VADER.

    VADER is rule-based, good for social media and general text.
    Used as fallback when FinBERT unavailable.

    Args:
        text: Text to analyze

    Returns:
        SentimentScore or None if VADER unavailable
    """
    vader = _model_manager.get_vader()
    if vader is None:
        return None

    try:
        scores = vader.polarity_scores(text)
        compound = scores['compound']  # Already -1 to +1

        # Determine label
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        # Confidence from how decisive the score is
        confidence = min(abs(compound) * 1.2, 1.0)

        return SentimentScore(
            text=text[:100],
            score=compound,
            confidence=confidence,
            source="vader",
            label=label
        )

    except Exception as e:
        logger.logger.warning("vader_error", error=str(e))
        return None


def score_text_hybrid(text: str, prefer_finbert: bool = True) -> Optional[SentimentScore]:
    """
    Score text using hybrid approach.

    Priority:
    1. FinBERT (if available and preferred)
    2. VADER (always available, rule-based)

    Args:
        text: Text to analyze
        prefer_finbert: Whether to prefer FinBERT over VADER

    Returns:
        SentimentScore from best available model
    """
    if not text or len(text.strip()) < 10:
        return None

    # Clean text
    text = re.sub(r'\s+', ' ', text.strip())

    # Try FinBERT first
    if prefer_finbert:
        score = score_text_finbert(text)
        if score is not None:
            return score

    # Fallback to VADER
    score = score_text_vader(text)
    if score is not None:
        # Slightly lower confidence for VADER (less accurate for financial text)
        score.confidence *= 0.85
        return score

    return None


def batch_score_texts(texts: List[str], prefer_finbert: bool = True) -> List[SentimentScore]:
    """
    Score multiple texts efficiently.

    Args:
        texts: List of texts to analyze
        prefer_finbert: Whether to prefer FinBERT

    Returns:
        List of SentimentScore objects
    """
    scores = []

    for text in texts:
        score = score_text_hybrid(text, prefer_finbert)
        if score:
            scores.append(score)

    return scores


# ============================================================
# Aggregation Functions
# ============================================================

def aggregate_sentiments(scores: List[SentimentScore]) -> AggregatedSentiment:
    """
    Aggregate multiple sentiment scores into overall assessment.

    Uses confidence-weighted averaging:
    - Higher confidence scores contribute more
    - Accounts for article count (more articles = higher confidence)

    Args:
        scores: List of individual SentimentScore objects

    Returns:
        AggregatedSentiment with overall metrics
    """
    if not scores:
        return AggregatedSentiment(
            overall_score=0.0,
            overall_confidence=0.0,
            overall_label="Neutral",
            num_articles=0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            individual_scores=[],
            summary="No articles analyzed."
        )

    # Count by label
    positive_count = sum(1 for s in scores if s.label == "positive")
    negative_count = sum(1 for s in scores if s.label == "negative")
    neutral_count = sum(1 for s in scores if s.label == "neutral")

    # Confidence-weighted average score
    total_weight = sum(s.confidence for s in scores)
    if total_weight > 0:
        weighted_score = sum(s.score * s.confidence for s in scores) / total_weight
    else:
        weighted_score = 0.0

    # Overall confidence based on:
    # 1. Average model confidence
    # 2. Agreement between articles
    # 3. Sample size
    avg_confidence = sum(s.confidence for s in scores) / len(scores)

    # Agreement bonus: if most articles agree, boost confidence
    max_count = max(positive_count, negative_count, neutral_count)
    agreement_ratio = max_count / len(scores)
    agreement_bonus = (agreement_ratio - 0.33) * 0.3  # 0 to 0.2 bonus

    # Sample size bonus (diminishing returns)
    sample_bonus = min(len(scores) / 10, 0.2)  # Max 0.2 bonus at 10+ articles

    overall_confidence = min(1.0, avg_confidence + agreement_bonus + sample_bonus)

    # Determine overall label
    if weighted_score > 0.15:
        overall_label = "Bullish"
    elif weighted_score < -0.15:
        overall_label = "Bearish"
    else:
        overall_label = "Neutral"

    # Generate summary
    summary = f"Analyzed {len(scores)} articles: {positive_count} positive, {neutral_count} neutral, {negative_count} negative. "
    if overall_label == "Bullish":
        summary += f"Overall sentiment is BULLISH with score {weighted_score:.2f}."
    elif overall_label == "Bearish":
        summary += f"Overall sentiment is BEARISH with score {weighted_score:.2f}."
    else:
        summary += f"Overall sentiment is NEUTRAL with mixed signals."

    return AggregatedSentiment(
        overall_score=weighted_score,
        overall_confidence=overall_confidence,
        overall_label=overall_label,
        num_articles=len(scores),
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=neutral_count,
        individual_scores=scores,
        summary=summary
    )


def apply_sector_boost(
    score: SentimentScore,
    sector: str,
    text: str
) -> SentimentScore:
    """
    Apply sector-specific keyword boosting to sentiment score.

    Financial news often uses sector-specific terminology that
    generic models may miss.

    Args:
        score: Original SentimentScore
        sector: Stock sector (e.g., "Technology", "Financial Services")
        text: Full text for keyword matching

    Returns:
        Adjusted SentimentScore
    """
    if sector not in SECTOR_SENTIMENT_KEYWORDS:
        return score

    keywords = SECTOR_SENTIMENT_KEYWORDS[sector]
    text_lower = text.lower()
    adjustment = 0.0

    for keyword in keywords.get('positive', []):
        if keyword.lower() in text_lower:
            adjustment += 0.08

    for keyword in keywords.get('negative', []):
        if keyword.lower() in text_lower:
            adjustment -= 0.08

    # Apply adjustment (capped)
    adjusted_score = max(-1.0, min(1.0, score.score + adjustment))

    return SentimentScore(
        text=score.text,
        score=adjusted_score,
        confidence=score.confidence,
        source=score.source,
        label=score.label
    )


# ============================================================
# Text Extraction
# ============================================================

def extract_articles_from_search(search_results: str) -> List[str]:
    """
    Parse web search results to extract individual article snippets.

    DuckDuckGo returns concatenated snippets; this splits them.

    Args:
        search_results: Raw search results string

    Returns:
        List of article text snippets
    """
    if not search_results:
        return []

    articles = []

    # Split by common separators
    # DuckDuckGo often separates results with "..." or newlines
    parts = re.split(r'\.\.\.|(?:\n\n+)', search_results)

    for part in parts:
        cleaned = part.strip()
        # Filter out very short or empty segments
        if len(cleaned) > 50:
            articles.append(cleaned)

    # If no splits worked, treat entire text as one article
    if not articles and len(search_results) > 50:
        articles = [search_results]

    return articles[:15]  # Limit to 15 articles


# ============================================================
# LangChain Tool
# ============================================================

@tool
def analyze_sentiment_nlp(query: str, ticker: str = "", sector: str = "") -> str:
    """
    Perform NLP-based sentiment analysis on search results.

    Uses FinBERT (financial NLP) for accurate scoring with VADER fallback.
    Provides quantitative sentiment scores from -1 (bearish) to +1 (bullish).

    Workflow:
    1. Execute web search with query
    2. Extract article snippets
    3. Score each snippet with FinBERT/VADER
    4. Apply sector-specific adjustments
    5. Aggregate scores
    6. Return structured report

    Args:
        query: Search query (e.g., "RELIANCE.NS stock news sentiment")
        ticker: Optional ticker for context
        sector: Optional sector for keyword boosting (e.g., "Technology", "Financial Services")

    Returns:
        Structured sentiment report with quantitative scores
    """
    start_time = time.time()
    cache_key = f"nlp_sentiment_{ticker}_{hash(query) % 10000}_{sector}"

    # Check cache
    cached_result = sentiment_cache.get(cache_key)
    if cached_result:
        logger.log_fetch(ticker or "N/A", "sentiment_nlp", True, 0, 0, "cache_hit")
        return cached_result

    try:
        # Step 1: Perform web search
        search_query = query
        if ticker and ticker not in query:
            search_query = f"{ticker} {query}"

        search_results = web_search.invoke(search_query)

        # Step 2: Extract article snippets
        articles = extract_articles_from_search(search_results)

        if not articles:
            return f"""
NLP SENTIMENT ANALYSIS: {ticker or query}
{'='*50}

No articles found for analysis.
Try a different search query or check if the ticker is correct.

{'='*50}
"""

        # Step 3: Score each article
        scores = batch_score_texts(articles, prefer_finbert=True)

        # Step 4: Apply sector adjustments
        if sector:
            scores = [apply_sector_boost(s, sector, s.text) for s in scores]

        # Step 5: Aggregate
        aggregated = aggregate_sentiments(scores)

        # Step 6: Format report
        model_used = scores[0].source if scores else "none"
        latency_ms = (time.time() - start_time) * 1000

        report = f"""
NLP SENTIMENT ANALYSIS: {ticker or query}
{'='*60}

QUANTITATIVE SCORES:
- Overall Sentiment Score: {aggregated.overall_score:+.2f} (-1 bearish to +1 bullish)
- Confidence: {aggregated.overall_confidence*100:.0f}%
- Classification: {aggregated.overall_label}
- Model Used: {model_used.upper()}

ARTICLE BREAKDOWN:
- Articles Analyzed: {aggregated.num_articles}
- Positive Articles: {aggregated.positive_count}
- Neutral Articles: {aggregated.neutral_count}
- Negative Articles: {aggregated.negative_count}

"""

        # Add top individual scores
        if aggregated.individual_scores:
            report += "SAMPLE ARTICLES:\n"
            for i, score in enumerate(aggregated.individual_scores[:5], 1):
                sentiment_icon = "+" if score.score > 0.1 else "-" if score.score < -0.1 else "~"
                report += f"{i}. [{sentiment_icon}] {score.text}...\n"
                report += f"   Score: {score.score:+.2f} | Confidence: {score.confidence:.0%}\n\n"

        report += f"""
INTERPRETATION:
"""

        if aggregated.overall_label == "Bullish":
            report += "- Positive sentiment detected in news coverage\n"
            report += "- Market participants appear optimistic\n"
        elif aggregated.overall_label == "Bearish":
            report += "- Negative sentiment detected in news coverage\n"
            report += "- Caution advised - market concerns evident\n"
        else:
            report += "- Mixed or neutral sentiment in coverage\n"
            report += "- No strong directional bias from news\n"

        if aggregated.overall_confidence < 0.5:
            report += "- LOW CONFIDENCE: Limited data or conflicting signals\n"
        elif aggregated.overall_confidence > 0.75:
            report += "- HIGH CONFIDENCE: Strong agreement across sources\n"

        if sector:
            report += f"- Sector-specific keywords ({sector}) applied\n"

        report += f"""
{'='*60}
Analysis completed in {latency_ms:.0f}ms
"""

        # Cache result
        sentiment_cache.set(cache_key, report, ttl_seconds=900)

        logger.log_fetch(
            ticker or "N/A",
            "sentiment_nlp",
            True,
            latency_ms,
            aggregated.num_articles
        )

        return report

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker or "N/A", "sentiment_nlp", False, latency_ms, 0, str(e))
        return f"Error analyzing sentiment: {str(e)}"


def get_sentiment_score(text: str, sector: str = "") -> Dict[str, Any]:
    """
    Get numeric sentiment score for a piece of text.

    Utility function for use by other modules (e.g., confidence scoring).

    Args:
        text: Text to analyze
        sector: Optional sector for keyword boosting

    Returns:
        Dict with score, confidence, label, source
    """
    score = score_text_hybrid(text, prefer_finbert=True)

    if score is None:
        return {
            'score': 0.0,
            'confidence': 0.0,
            'label': 'neutral',
            'source': 'none'
        }

    if sector:
        score = apply_sector_boost(score, sector, text)

    return {
        'score': score.score,
        'confidence': score.confidence,
        'label': score.label,
        'source': score.source
    }


# Export
SENTIMENT_NLP_TOOLS = [
    analyze_sentiment_nlp,
]
