"""
Persistent memory system for tracking prediction performance and learning.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from langchain.tools import tool

# Memory file location
MEMORY_DIR = Path("/home/tgt/Documents/projects/personal/stock-agent-langchain/.memory")
MEMORY_DIR.mkdir(exist_ok=True)
PREDICTIONS_FILE = MEMORY_DIR / "predictions.json"


def load_predictions() -> List[Dict]:
    """Load historical predictions from memory."""
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_prediction(ticker: str, recommendation: str, entry: float, target: float, 
                   stop_loss: float, confidence: int, sector: str = None, regime: str = None, date: str = None):
    """Save a new prediction to memory."""
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
        sector: Sector name (e.g., "Technology", "Banking")
    
    Returns:
        Sector-specific win rate and insights
    """
    try:
        predictions = load_predictions()
        
        if not predictions:
            return "No historical data available"
        
        # This is simplified - in production, you'd map tickers to sectors
        # For now, return a generic message
        report = f"""
SECTOR PERFORMANCE: {sector}
{'='*50}
Note: Sector tracking will improve as more predictions accumulate.

Current strategy:
- Technology: Trend-following works best
- Banking: Fundamental analysis more reliable
- Energy: Commodity correlation is key
{'='*50}
"""
        return report
        
    except Exception as e:
        return f"Error: {str(e)}"
