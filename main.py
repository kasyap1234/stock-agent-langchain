"""
Optimized main entry point with parallel execution and streaming.
"""
from dotenv import load_dotenv, find_dotenv, dotenv_values
from pathlib import Path
from langchain_core.messages import HumanMessage
import argparse
import os
import warnings
import time

warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

def load_env():
    """
    Load environment variables from common locations.
    Tries project root, current working directory, then nearest .env via find_dotenv.
    """
    candidates = [
        Path(__file__).resolve().parent / ".env",  # project root (where main.py lives)
        Path.cwd() / ".env",                       # current working directory
    ]
    for path in candidates:
        load_dotenv(path, override=False)
        # Fallback: directly parse and inject if still missing
        if not os.getenv("GROQ_API_KEY") and path.exists():
            vals = dotenv_values(path)
            if vals.get("GROQ_API_KEY"):
                os.environ.setdefault("GROQ_API_KEY", vals["GROQ_API_KEY"])

    # Fallback: nearest .env upwards from cwd
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
        if not os.getenv("GROQ_API_KEY"):
            vals = dotenv_values(found)
            if vals.get("GROQ_API_KEY"):
                os.environ.setdefault("GROQ_API_KEY", vals["GROQ_API_KEY"])


def main():
    load_env()

    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found!")
        print("\nSet it with: export GROQ_API_KEY=your_key_here")
        return

    # Import after key check so ChatGroq isn't instantiated without credentials
    from src.graph.workflow import app

    parser = argparse.ArgumentParser(description="Optimized Swing Trade Bot")
    parser.add_argument("ticker", type=str, help="Stock Ticker (e.g., RELIANCE.NS)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    args = parser.parse_args()

    ticker = args.ticker
    print(f"\nAnalyzing {ticker} (Optimized - 3x Faster)...\n")

    initial_state = {
        "messages": [HumanMessage(content=f"Analyze {ticker} for swing trade.")],
        "ticker": ticker,
        "next": "Planner"
    }

    start_time = time.time()
    
    try:
        # Progress indicators
        print("Planner: Creating custom strategy...")
        
        for s in app.stream(initial_state):
            if "__end__" not in s:
                if "ParallelAnalysts" in s:
                    print("Running Technical, Fundamental & Sentiment in parallel...")
                elif "Ensemble" in s:
                    print("Ensemble: Multi-strategy voting...")
                elif "Critic" in s:
                    print("Critic: Validation & backtesting...")
                elif "FinalSynthesis" in s:
                    elapsed = time.time() - start_time
                    print(f"\nAnalysis complete in {elapsed:.1f}s\n")
                    print("=" * 70)
                    print(s["FinalSynthesis"]["messages"][0].content)
                    print("=" * 70)
        
        print(f"\nTotal time: {time.time() - start_time:.2f}s")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
