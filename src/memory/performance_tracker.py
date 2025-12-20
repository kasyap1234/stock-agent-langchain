"""
Persistent memory system for tracking prediction performance and learning.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from langchain.tools import tool

# Memory file location (default to project root .memory, overridable via env)
DEFAULT_MEMORY_DIR = Path(__file__).resolve().parents[2] / ".memory"
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", DEFAULT_MEMORY_DIR))
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_FILE = MEMORY_DIR / "predictions.json"


def load_predictions() -> List[Dict]:
    """Load historical predictions from memory."""
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_prediction(ticker: str, recommendation: str, entry: float, target: float,
                   stop_loss: float, confidence: int, sector: str = None, regime: str = None,
                   date: str = None, reasoning: str = None):
    """Save a new prediction to memory with optional reasoning."""
    predictions = load_predictions()

    prediction = {
        "ticker": ticker,
        "date": date or datetime.now().strftime("%Y-%m-%d"),
        "sector": sector,
        "regime": regime,
        "recommendation": recommendation,
        "entry": entry,
        "target": target,
        "stop_loss": stop_loss,
        "confidence": confidence,
        "reasoning": reasoning,
        "outcome": "PENDING",
        "actual_return": None
    }

    predictions.append(prediction)

    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions, f, indent=2)

def find_similar_examples(sector: str, regime: str, outcome: str = "WIN", limit: int = 3) -> List[Dict]:
    """
    Finds similar historical examples based on sector and market regime.
    """
    predictions = load_predictions()
    
    # Filter by outcome first
    candidates = [p for p in predictions if p.get("outcome") == outcome]
    
    # Score similarity
    scored = []
    for p in candidates:
        score = 0
        if p.get("sector") == sector:
            score += 2
        if p.get("regime") == regime:
            score += 1
        
        if score > 0:
            scored.append((score, p))
            
    # Sort by score (descending) and date (newest first)
    scored.sort(key=lambda x: (x[0], x[1]["date"]), reverse=True)
    
    return [item[1] for item in scored[:limit]]


@tool
def check_past_performance(ticker: str = None) -> str:
    """
    Checks historical prediction performance for a specific ticker or overall.
    
    Args:
        ticker: Optional ticker to filter by
    
    Returns:
        Win rate, average return, and performance history
    """
    try:
        predictions = load_predictions()
        
        if not predictions:
            return "No historical predictions found. First time analyzing stocks."
        
        # Filter by ticker if specified
        if ticker:
            predictions = [p for p in predictions if p['ticker'] == ticker]
            if not predictions:
                return f"No historical predictions for {ticker}"
        
        # Calculate statistics
        total = len(predictions)
        wins = len([p for p in predictions if p.get('outcome') == 'WIN'])
        losses = len([p for p in predictions if p.get('outcome') == 'LOSS'])
        pending = len([p for p in predictions if p.get('outcome') == 'PENDING'])
        
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        
        # Average confidence of wins vs losses
        win_confs = [p['confidence'] for p in predictions if p.get('outcome') == 'WIN']
        loss_confs = [p['confidence'] for p in predictions if p.get('outcome') == 'LOSS']
        
        avg_win_conf = sum(win_confs) / len(win_confs) if win_confs else 0
        avg_loss_conf = sum(loss_confs) / len(loss_confs) if loss_confs else 0
        
        report = f"""
HISTORICAL PERFORMANCE {'for ' + ticker if ticker else '(All Stocks)'}:
{'='*50}
Total Predictions: {total}
Wins: {wins}
Losses: {losses}
Pending: {pending}

Win Rate: {win_rate:.1f}%
Avg Confidence (Wins): {avg_win_conf:.0f}%
Avg Confidence (Losses): {avg_loss_conf:.0f}%

LEARNING:
"""
        if win_rate > 60:
            report += "Good track record - trust the analysis\n"
        elif win_rate > 50:
            report += "Moderate track record - be selective\n"
        else:
            report += "Poor track record - recommend avoiding for now\n"
        
        if avg_win_conf > avg_loss_conf + 10:
            report += "High-confidence calls perform better\n"
        elif avg_loss_conf > avg_win_conf:
            report += "Overconfident on losses - calibrate confidence down\n"
        
        # Show recent predictions
        recent = sorted(predictions, key=lambda x: x['date'], reverse=True)[:5]
        report += f"\nRecent {len(recent)} Predictions:\n"
        for p in recent:
            outcome_icon = {"WIN": "WIN", "LOSS": "LOSS", "PENDING": "PENDING"}.get(p['outcome'], "?")
            report += f"  {outcome_icon} {p['ticker']} ({p['date']}): {p['recommendation']} @Rs{p['entry']} - {p['outcome']}\n"
        
        report += "=" * 50
        return report
        
    except Exception as e:
        return f"Error checking performance: {str(e)}"


@tool
def get_sector_performance_history(sector: str) -> str:
    """
    Analyzes historical performance by sector.

    Args:
        sector: Sector name (e.g., "Technology", "Financial Services", "Healthcare")

    Returns:
        Sector-specific win rate, insights, and example reasoning from past wins
    """
    try:
        predictions = load_predictions()

        if not predictions:
            return "No historical data available"

        # Filter by sector
        sector_preds = [p for p in predictions if p.get("sector") == sector]

        if not sector_preds:
            return f"No historical predictions for sector: {sector}"

        # Calculate sector-specific stats
        total = len(sector_preds)
        wins = [p for p in sector_preds if p.get("outcome") == "WIN"]
        losses = [p for p in sector_preds if p.get("outcome") == "LOSS"]

        win_rate = (len(wins) / (len(wins) + len(losses)) * 100) if (len(wins) + len(losses)) > 0 else 0

        # Average return for wins
        win_returns = [p.get("actual_return", 0) for p in wins if p.get("actual_return")]
        avg_win_return = sum(win_returns) / len(win_returns) if win_returns else 0

        # Regime breakdown
        regime_stats = {}
        for p in sector_preds:
            regime = p.get("regime", "unknown")
            if regime not in regime_stats:
                regime_stats[regime] = {"total": 0, "wins": 0}
            regime_stats[regime]["total"] += 1
            if p.get("outcome") == "WIN":
                regime_stats[regime]["wins"] += 1

        report = f"""
SECTOR PERFORMANCE: {sector}
{'='*50}
Total Predictions: {total}
Wins: {len(wins)} | Losses: {len(losses)}
Win Rate: {win_rate:.1f}%
Avg Win Return: {avg_win_return:.1f}%

PERFORMANCE BY REGIME:
"""
        for regime, stats in regime_stats.items():
            regime_wr = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0
            report += f"  {regime}: {stats['wins']}/{stats['total']} ({regime_wr:.0f}% win rate)\n"

        # Show reasoning from recent wins (few-shot learning context)
        recent_wins = sorted(wins, key=lambda x: x.get("date", ""), reverse=True)[:3]
        if recent_wins:
            report += f"\nSUCCESSFUL REASONING EXAMPLES:\n"
            for p in recent_wins:
                reasoning = p.get("reasoning", "No reasoning recorded")
                report += f"\n  {p['ticker']} ({p['date']}) - {p['recommendation']}:\n"
                report += f"  \"{reasoning[:200]}{'...' if len(reasoning) > 200 else ''}\"\n"
                report += f"  Result: +{p.get('actual_return', 0):.1f}%\n"

        report += "\n" + "=" * 50
        return report

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_similar_winning_trades(sector: str, regime: str) -> str:
    """
    Retrieves similar winning trades for few-shot learning context.

    Args:
        sector: Stock sector (e.g., "Technology", "Financial Services")
        regime: Market regime (e.g., "trending_up", "ranging", "volatile")

    Returns:
        Detailed examples of similar successful trades with reasoning
    """
    try:
        examples = find_similar_examples(sector, regime, outcome="WIN", limit=3)

        if not examples:
            return f"No similar winning examples found for {sector} in {regime} regime."

        report = f"""
SIMILAR SUCCESSFUL TRADES
Sector: {sector} | Regime: {regime}
{'='*50}
"""
        for i, ex in enumerate(examples, 1):
            report += f"""
EXAMPLE {i}: {ex['ticker']} ({ex['date']})
Recommendation: {ex['recommendation']}
Entry: Rs{ex['entry']} | Target: Rs{ex['target']} | Stop: Rs{ex['stop_loss']}
Confidence: {ex['confidence']}%
Result: +{ex.get('actual_return', 0):.1f}%

REASONING:
{ex.get('reasoning', 'No reasoning recorded')}

KEY TAKEAWAYS:
"""
            # Extract key patterns from reasoning
            reasoning = ex.get('reasoning', '').lower()
            if 'usd-inr' in reasoning or 'rupee' in reasoning:
                report += "- Currency movement was a factor\n"
            if 'vix' in reasoning:
                report += "- VIX levels influenced position sizing\n"
            if 'breakout' in reasoning or 'breakdown' in reasoning:
                report += "- Technical breakout/breakdown pattern\n"
            if 'rbi' in reasoning or 'policy' in reasoning:
                report += "- Monetary policy was considered\n"
            if 'fii' in reasoning or 'dii' in reasoning:
                report += "- Institutional flows were tracked\n"
            if 'results' in reasoning or 'earnings' in reasoning:
                report += "- Earnings timing was factored in\n"

        report += "\n" + "=" * 50
        return report

    except Exception as e:
        return f"Error: {str(e)}"
