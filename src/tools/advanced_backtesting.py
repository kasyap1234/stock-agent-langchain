"""
Advanced backtesting framework with detailed metrics and visualization.
Adds basic data hygiene and structured payloads for downstream agents.
"""
import time
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from langchain.tools import tool
from sklearn.model_selection import TimeSeriesSplit

from src.validation.data_validators import MarketDataValidator
from src.tools.forecast_model import _make_dataset, train_forecast_model
from src.utils.logging_config import ToolLogger

_adv_validator = MarketDataValidator()
_adv_logger = ToolLogger("advanced_backtesting")

@tool
def advanced_backtest(
    ticker: str,
    strategy: str = "swing",
    days_back: int = 90,
    slippage_bps: int = 10,
    fee_bps: int = 5,
) -> str:
    """
    Runs comprehensive backtest with detailed performance metrics.
    
    Args:
        ticker: Stock ticker
        strategy: "swing", "momentum", "reversion"
        days_back: Historical period to test
        slippage_bps: Per-leg slippage in basis points (default 10bps = 0.10%)
        fee_bps: Per-leg commission/fees in basis points (default 5bps = 0.05%)
    
    Returns:
        Detailed backtest report with Sharpe ratio, max drawdown, win rate, etc.
    """
    try:
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back * 2)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            return f"No data for {ticker}"

        df = df.dropna(subset=["Close"])
        if df.empty:
            return f"No clean price data for {ticker}"

        per_leg_cost = (slippage_bps + fee_bps) / 10000

        # Calculate indicators
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)
        
        # Simulate trades based on strategy
        trades = []
        position = None
        
        for i in range(50, len(df)):
            if strategy == "momentum":
                # Golden cross strategy
                if df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i] and df['SMA_20'].iloc[i-1] <= df['SMA_50'].iloc[i-1]:
                    if position is None:
                        position = {
                            'entry_price': df['Close'].iloc[i],
                            'entry_date': df.index[i],
                            'type': 'LONG'
                        }
                elif position and df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i]:
                    # Exit
                    exit_price = df['Close'].iloc[i]
                    effective_entry = position['entry_price'] * (1 + per_leg_cost)
                    effective_exit = exit_price * (1 - per_leg_cost)
                    pnl = ((effective_exit - effective_entry) / effective_entry) * 100
                    trades.append({
                        'entry': position['entry_price'],
                        'exit': exit_price,
                        'pnl': pnl,
                        'days': (df.index[i] - position['entry_date']).days
                    })
                    position = None
        
        # Close open position
        if position:
            exit_price = df['Close'].iloc[-1]
            effective_entry = position['entry_price'] * (1 + per_leg_cost)
            effective_exit = exit_price * (1 - per_leg_cost)
            pnl = ((effective_exit - effective_entry) / effective_entry) * 100
            trades.append({
                'entry': position['entry_price'],
                'exit': exit_price,
                'pnl': pnl,
                'days': (df.index[-1] - position['entry_date']).days
            })
        
        if not trades:
            return "No trades generated in backtest period"
        
        # Calculate metrics
        total_trades = len(trades)
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = (len(wins) / total_trades) * 100
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        total_return = sum([t['pnl'] for t in trades])
        avg_holding_days = np.mean([t['days'] for t in trades])
        trade_density = total_trades / max(days_back, 1)
        round_trip_cost_pct = per_leg_cost * 2 * 100
        
        # Sharpe ratio (simplified)
        returns = [t['pnl'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 / avg_holding_days) if np.std(returns) > 0 else 0
        
        # Max drawdown
        cumulative = np.cumsum([t['pnl'] for t in trades])
        max_dd = np.min(cumulative - np.maximum.accumulate(cumulative))
        
        report = f"""
ADVANCED BACKTEST: {ticker} ({strategy.upper()} Strategy)
{'='*60}
Period: {days_back} days | Total Trades: {total_trades}

PERFORMANCE METRICS:
Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)
Avg Win: +{avg_win:.2f}%
Avg Loss: {avg_loss:.2f}%
Profit Factor (risk/reward): {risk_reward:.2f}

Total Return: {total_return:.2f}%
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_dd:.2f}%
Avg Holding Period: {avg_holding_days:.1f} days
Costs (round trip): {round_trip_cost_pct:.2f}% | Slippage {slippage_bps}bps, Fees {fee_bps}bps
Trade Density: {trade_density:.2f} trades/day

RISK ASSESSMENT:
"""
        if sharpe > 1.5:
            report += "Excellent risk-adjusted returns\n"
        elif sharpe > 1.0:
            report += "Good risk-adjusted returns\n"
        elif sharpe > 0.5:
            report += "Moderate risk-adjusted returns\n"
        else:
            report += "Poor risk-adjusted returns\n"
        
        if max_dd < -15:
            report += "High drawdown risk - use small position size\n"
        elif max_dd < -10:
            report += "Moderate drawdown - manage risk carefully\n"
        else:
            report += "Acceptable drawdown levels\n"

        if trade_density > 0.7:
            report += "High trade density - risk of overtrading; review filters\n"
        elif trade_density < 0.05:
            report += "Sparse trades - signals may be too restrictive\n"

        report += f"\n{'='*60}"

        payload = {
            "ticker": ticker,
            "strategy": strategy,
            "days_back": days_back,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "risk_reward": risk_reward,
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "avg_holding_days": avg_holding_days,
            "trades": trades,
            "slippage_bps": slippage_bps,
            "fee_bps": fee_bps,
            "round_trip_cost_pct": round_trip_cost_pct,
            "trade_density_per_day": trade_density,
        }

        return report + f"\nJSON:{json.dumps(payload, default=str)}"

    except Exception as e:  # noqa: PERF203 - user-facing error path
        return f"Backtest error: {str(e)}"


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _dynamic_round_trip_cost(df: pd.DataFrame, slippage_bps: int, fee_bps: int) -> float:
    """Volume/volatility-aware round-trip cost as a decimal fraction."""
    base_per_leg = (slippage_bps + fee_bps) / 10000
    if df is None or df.empty:
        return base_per_leg * 2

    range_pct = ((df["High"] - df["Low"]) / df["Close"]).tail(20).mean()
    vol_factor = 1 + min(range_pct * 5, 1.0) if pd.notna(range_pct) else 1.0

    volume_mean = df["Volume"].tail(20).mean()
    liquidity_factor = 1.0
    if volume_mean and not pd.isna(volume_mean):
        if volume_mean < 500_000:
            liquidity_factor = 1.25
        elif volume_mean > 2_000_000:
            liquidity_factor = 0.85

    return base_per_leg * vol_factor * liquidity_factor * 2


@tool
def model_signal_backtest(
    ticker: str,
    period: str = "1y",
    horizon_days: int = 10,
    threshold_pct: float = 0.0,
    n_splits: int = 4,
    purge: int = 5,
    slippage_bps: int = 10,
    fee_bps: int = 5,
    simulations: int = 300,
) -> str:
    """
    Backtest model-driven swing trades using the GBDT forecaster with purged CV signals.

    Trades are taken when the forecasted horizon return exceeds `threshold_pct`.
    Costs are adjusted for liquidity/volatility and Monte Carlo bootstrapping
    provides confidence bands on aggregated returns.
    """
    start_time = time.time()
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df = _adv_validator.sanitize_dataframe(ticker, df)

        if df.empty:
            return f"No data for {ticker}"

        validations = _adv_validator.validate_all(ticker, df)
        for check, (passed, reason) in validations.items():
            if not passed and check != "corporate_actions":
                return f"Data quality issue for {ticker}: {reason}"

        X, y = _make_dataset(df, horizon=horizon_days)
        if X.empty or len(y) < 120:
            return f"Insufficient data for model backtest on {ticker}"

        result = train_forecast_model(X, y, n_splits=n_splits, purge=purge)

        round_trip_cost = _dynamic_round_trip_cost(df, slippage_bps, fee_bps)
        trades = []
        for pred, actual in result.cv_predictions:
            if pred <= threshold_pct:
                continue
            net_return = actual - round_trip_cost
            trades.append({"pred": pred, "actual": actual, "net": net_return})

        if not trades:
            return f"No qualifying trades generated for {ticker} with threshold {threshold_pct:.2%}"

        net_returns = [t["net"] for t in trades]
        wins = [r for r in net_returns if r > 0]
        losses = [r for r in net_returns if r <= 0]

        total_return = float(sum(net_returns))
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        win_rate = (len(wins) / len(net_returns)) * 100
        profit_factor = abs(sum(wins) / sum(losses)) if losses else float("inf")
        trade_density = len(trades) / max(len(y), 1)

        mc_p90 = mc_p50 = mc_p10 = None
        if len(net_returns) >= 2:
            samples = np.random.choice(
                net_returns, size=(simulations, len(net_returns)), replace=True
            )
            dist = samples.sum(axis=1)
            mc_p90 = float(np.percentile(dist, 90))
            mc_p50 = float(np.percentile(dist, 50))
            mc_p10 = float(np.percentile(dist, 10))

        payload = {
            "ticker": ticker,
            "horizon_days": horizon_days,
            "threshold_pct": threshold_pct,
            "cv_metrics": result.metrics,
            "trades": trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "trade_density": trade_density,
            "round_trip_cost": round_trip_cost,
            "mc": {"p90": mc_p90, "p50": mc_p50, "p10": mc_p10, "simulations": simulations},
        }

        mc_p90_str = f"{mc_p90:+.2%}" if mc_p90 is not None else "n/a"
        mc_p50_str = f"{mc_p50:+.2%}" if mc_p50 is not None else "n/a"
        mc_p10_str = f"{mc_p10:+.2%}" if mc_p10 is not None else "n/a"
        latest_cal_str = (
            f"{result.latest_prediction_calibrated:+.2%}"
            if result.latest_prediction_calibrated is not None
            else "n/a"
        )

        summary = f"""
MODEL SIGNAL BACKTEST: {ticker} | Horizon {horizon_days}d | Trades: {len(trades)}
CV RMSE/MAPE (mean): {result.metrics['rmse_cv_mean']} / {result.metrics['mape_cv_mean']}
Win rate: {win_rate:.1f}% | Profit factor: {profit_factor:.2f} | Total return: {total_return:+.2%}
Costs (round trip, dynamic): {round_trip_cost:.2%} | Trade density: {trade_density:.3f} per bar
MC (total return): p90 {mc_p90_str}, p50 {mc_p50_str}, p10 {mc_p10_str}
Latest forecast (raw/calibrated): {result.latest_prediction:+.2%} / {latest_cal_str}
"""
        _adv_logger.log_fetch(
            ticker=ticker,
            data_type="model_signal_backtest",
            success=True,
            latency_ms=(time.time() - start_time) * 1000,
            records_fetched=len(df),
        )
        return summary + f"\nJSON:{json.dumps(payload, default=str)}"

    except Exception as e:
        _adv_logger.log_fetch(
            ticker=ticker,
            data_type="model_signal_backtest",
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            records_fetched=0,
            error=str(e),
        )
        return f"Model signal backtest error for {ticker}: {str(e)}"


@tool
def monte_carlo_simulation(ticker: str, days_ahead: int = 30, simulations: int = 1000, scenario: str = "normal") -> str:
    """
    Runs Monte Carlo simulation to estimate potential outcomes.
    
    Args:
        ticker: Stock ticker
        days_ahead: Days to simulate
        simulations: Number of simulations to run
        scenario: "normal", "bull", "bear", or "fat_tail"
    
    Returns:
        Probability distribution of returns
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")

        if df.empty:
            return f"No data for {ticker}"

        df = df.dropna(subset=["Close"])
        if df.empty:
            return f"No clean price data for {ticker}"

        # Calculate daily returns statistics
        returns = df['Close'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Adjust parameters based on scenario
        if scenario == "bull":
            mean_return += (std_return * 0.1) # Slight drift up
        elif scenario == "bear":
            mean_return -= (std_return * 0.1) # Slight drift down
        
        current_price = df['Close'].iloc[-1]
        
        # Run simulations
        final_prices = []
        for _ in range(simulations):
            price = current_price
            for _ in range(days_ahead):
                if scenario == "fat_tail":
                    # Use Student's t-distribution (df=3) for fatter tails
                    daily_return = mean_return + np.random.standard_t(3) * std_return
                else:
                    daily_return = np.random.normal(mean_return, std_return)
                price *= (1 + daily_return)
            final_prices.append(price)
        
        final_prices = np.array(final_prices)
        
        # Calculate percentiles
        p10 = np.percentile(final_prices, 10)
        p50 = np.percentile(final_prices, 50)
        p90 = np.percentile(final_prices, 90)
        
        # Expected returns
        ret_10 = ((p10 - current_price) / current_price) * 100
        ret_50 = ((p50 - current_price) / current_price) * 100
        ret_90 = ((p90 - current_price) / current_price) * 100
        
        report = f"""
MONTE CARLO SIMULATION: {ticker} ({scenario.upper()} Scenario)
{'='*60}
Simulations: {simulations} | Forecast: {days_ahead} days
Current Price: Rs{current_price:.2f}

EXPECTED OUTCOMES:
90th Percentile (Optimistic): Rs{p90:.2f} ({ret_90:+.1f}%)
50th Percentile (Median): Rs{p50:.2f} ({ret_50:+.1f}%)
10th Percentile (Pessimistic): Rs{p10:.2f} ({ret_10:+.1f}%)

PROBABILITY ANALYSIS:
Chance of >5% gain: {(final_prices > current_price * 1.05).mean() * 100:.1f}%
Chance of >10% gain: {(final_prices > current_price * 1.10).mean() * 100:.1f}%
Chance of <5% loss: {(final_prices < current_price * 0.95).mean() * 100:.1f}%

TRADING IMPLICATIONS:
"""
        if ret_50 > 5:
            report += "Positive expected value - favorable setup\n"
        elif ret_50 > 0:
            report += "Slightly positive - marginal trade\n"
        else:
            report += "Negative expected value - avoid\n"
        
        risk_reward = abs(ret_90 - ret_50) / abs(ret_50 - ret_10) if ret_10 != ret_50 else 0
        report += f"Risk-Reward Asymmetry: {risk_reward:.2f}\n"

        payload = {
            "ticker": ticker,
            "scenario": scenario,
            "days_ahead": days_ahead,
            "simulations": simulations,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "ret_10_pct": ret_10,
            "ret_50_pct": ret_50,
            "ret_90_pct": ret_90,
            "prob_gt_5": (final_prices > current_price * 1.05).mean() * 100,
            "prob_gt_10": (final_prices > current_price * 1.10).mean() * 100,
            "prob_lt_5_loss": (final_prices < current_price * 0.95).mean() * 100,
            "risk_reward_asymmetry": risk_reward,
        }
        
        report += f"{'='*60}"
        return report + f"\nJSON:{json.dumps(payload, default=str)}"
        
    except Exception as e:  # noqa: PERF203 - user-facing error path
        return f"Monte Carlo error: {str(e)}"
