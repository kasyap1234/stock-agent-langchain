from typing import Dict, Any, List

class ConfidenceScorer:
    """
    Calculates a multi-dimensional confidence score for trade predictions.
    """
    
    def __init__(self):
        self.weights = {
            "signal_alignment": 0.25,
            "data_quality": 0.10,
            "regime_match": 0.15,
            "backtest_robustness": 0.20,
            "monte_carlo": 0.10,
            "volatility_penalty": 0.10,
            "sentiment_confidence": 0.10
        }

    def calculate_score(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the final confidence score and provides a breakdown.
        
        Args:
            analysis_data: Dictionary containing results from all agents and tools.
            
        Returns:
            Dict with 'score' (0-100) and 'breakdown' (details).
        """
        scores = {}
        
        # 1. Signal Alignment (Technical, Fundamental, Sentiment)
        scores["signal_alignment"] = self._score_alignment(analysis_data)
        
        # 2. Data Quality (Data freshness, completeness)
        scores["data_quality"] = self._score_data_quality(analysis_data)
        
        # 3. Regime Match (Strategy vs Market Condition)
        scores["regime_match"] = self._score_regime_match(analysis_data)
        
        # 4. Backtest Robustness (Walk-Forward results)
        scores["backtest_robustness"] = self._score_backtest(analysis_data)
        
        # 5. Monte Carlo (Expected Value & Tail Risk)
        scores["monte_carlo"] = self._score_monte_carlo(analysis_data)
        
        # 6. Volatility Penalty
        scores["volatility_penalty"] = self._score_volatility(analysis_data)

        # 7. Sentiment Confidence (NLP model confidence, article count, cross-source agreement)
        scores["sentiment_confidence"] = self._score_sentiment_confidence(analysis_data)

        # Calculate Weighted Average
        final_score = 0.0
        breakdown = []
        
        for factor, weight in self.weights.items():
            raw_score = scores.get(factor, 0)
            contribution = raw_score * weight
            final_score += contribution
            breakdown.append(f"{factor.replace('_', ' ').title()}: {raw_score:.0f}/100 (Weight: {weight*100:.0f}%) -> +{contribution:.1f}")
            
        return {
            "score": min(100, max(0, final_score)),
            "breakdown": "\n".join(breakdown)
        }

    def _score_alignment(self, data: Dict[str, Any]) -> float:
        """
        Score alignment using quantitative sentiment when available.

        Enhanced to use numeric sentiment_score from NLP analysis.
        """
        signals = []

        # Technical signal
        if "technical_analysis" in data:
            tech = data["technical_analysis"].lower()
            signals.append(1 if "bullish" in tech else -1 if "bearish" in tech else 0)

        # Fundamental signal
        if "fundamental_analysis" in data:
            fund = data["fundamental_analysis"].lower()
            signals.append(1 if "positive" in fund else -1 if "negative" in fund else 0)

        # NEW: Use quantitative sentiment score if available (-1 to +1)
        if "sentiment_score" in data:
            # Direct numeric score from NLP analysis
            nlp_score = data["sentiment_score"]
            signals.append(nlp_score)  # Already in -1 to +1 range
        elif "sentiment_analysis" in data:
            # Fallback to text parsing
            sent = data["sentiment_analysis"].lower()
            signals.append(1 if "positive" in sent or "bullish" in sent
                          else -1 if "negative" in sent or "bearish" in sent
                          else 0)

        if not signals:
            return 50.0

        # Calculate agreement based on average signal direction and magnitude
        avg_signal = sum(signals) / len(signals)

        # Check if all signals agree on direction
        all_positive = all(s > 0 for s in signals)
        all_negative = all(s < 0 for s in signals)

        if all_positive or all_negative:
            # Full agreement - high confidence scaled by magnitude
            agreement_score = abs(avg_signal) * 100
        else:
            # Mixed signals - lower confidence
            agreement_score = 30 + abs(avg_signal) * 20

        return min(100, max(0, agreement_score))

    def _score_sentiment_confidence(self, data: Dict[str, Any]) -> float:
        """
        Score confidence in sentiment analysis specifically.

        Factors:
        - NLP model confidence
        - Number of articles analyzed
        - Agreement between NLP and options sentiment
        """
        base_score = 50.0

        # NLP model confidence (0 to 1)
        if "sentiment_confidence" in data:
            base_score = data["sentiment_confidence"] * 100

        # Article count bonus (more articles = higher confidence)
        if "articles_analyzed" in data:
            article_boost = min(data["articles_analyzed"] * 3, 20)
            base_score += article_boost

        # Agreement between NLP and options sentiment
        if "options_sentiment" in data and "sentiment_score" in data:
            options = 1 if "bullish" in str(data["options_sentiment"]).lower() else \
                     -1 if "bearish" in str(data["options_sentiment"]).lower() else 0
            news = data["sentiment_score"]

            if (options > 0 and news > 0) or (options < 0 and news < 0):
                base_score += 10  # Agreement bonus
            elif options != 0 and news != 0 and options * news < 0:
                base_score -= 10  # Disagreement penalty

        return min(100, max(0, base_score))

    def _score_data_quality(self, data: Dict[str, Any]) -> float:
        # Placeholder: Assume 100 unless flagged
        # In real impl, check timestamps in data
        return 100.0

    def _score_regime_match(self, data: Dict[str, Any]) -> float:
        regime = data.get("regime", "Unknown")
        strategy = data.get("strategy", "Unknown") # e.g. "Trend Following"
        
        if regime == "STRONG_TRENDING" and "Trend" in strategy:
            return 100.0
        elif regime == "RANGING" and "Mean Reversion" in strategy:
            return 100.0
        elif regime == "Unknown":
            return 50.0
        else:
            return 50.0 # Neutral match

    def _score_backtest(self, data: Dict[str, Any]) -> float:
        # Parse backtest report for "Robustness Score"
        report = data.get("backtest_report", "")
        if "Robustness Score: 100%" in report: return 100.0
        if "Robustness Score: 75%" in report: return 85.0
        if "Robustness Score: 50%" in report: return 60.0
        if "Robustness Score: 25%" in report: return 40.0
        if "Robustness Score: 0%" in report: return 20.0
        return 50.0 # Default if no report

    def _score_monte_carlo(self, data: Dict[str, Any]) -> float:
        report = data.get("monte_carlo_report", "")
        if "Positive expected value" in report: return 90.0
        if "Slightly positive" in report: return 60.0
        if "Negative expected value" in report: return 20.0
        return 50.0

    def _score_volatility(self, data: Dict[str, Any]) -> float:
        # Higher volatility -> Lower score (Penalty)
        # We return a score where 100 is BEST (Low Volatility)
        regime = data.get("regime", "")
        if "VOLATILE" in regime:
            return 40.0
        return 90.0
