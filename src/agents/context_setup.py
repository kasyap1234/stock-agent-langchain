import yfinance as yf
from src.tools.regime_detection import _fetch_regime_data, _calculate_regime_indicators, _classify_regime
from src.tools.market_data import fetch_realtime_quote_structured
from typing import Dict, Any

def setup_analysis_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup node that runs before planning to populate context (Sector, Regime).
    """
    ticker = state["ticker"]
    context_updates = {}
    
    try:
        # 1. Get Sector
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        context_updates["sector"] = sector

        # 2. Live quote (validated and cached)
        context_updates["quote"] = fetch_realtime_quote_structured(ticker)
        
        # 3. Get Regime
        # We use the internal functions from regime_detection to get the raw classification
        df = _fetch_regime_data(ticker, period="6mo")
        if not df.empty and len(df) >= 50:
            regime_data = _calculate_regime_indicators(df)
            classification = _classify_regime(regime_data)
            context_updates["regime"] = classification["regime"]
        else:
            context_updates["regime"] = "Unknown"
            
    except Exception as e:
        print(f"Error setting up context: {e}")
        context_updates["sector"] = "Unknown"
        context_updates["regime"] = "Unknown"
        context_updates["quote"] = {
            "ticker": ticker,
            "price": None,
            "currency": "N/A",
            "as_of": None,
            "stale": True,
            "stale_reason": str(e),
            "source": "yfinance"
        }
        
    return context_updates
