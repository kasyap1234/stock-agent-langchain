"""
Test script for Phase 2 improvements.

Tests:
1. Regime detection
2. Multi-timeframe analysis
3. Enhanced ensemble voting

Run: python test_phase2.py
"""

from src.tools.regime_detection import detect_market_regime
from src.tools.market_data import multi_timeframe_analysis, calculate_indicators
from src.agents.ensemble import create_ensemble_agent

def test_regime_detection(ticker="RELIANCE.NS"):
    """Test regime detection tool."""
    print("=" * 80)
    print(f"TEST 1: REGIME DETECTION for {ticker}")
    print("=" * 80)

    try:
        result = detect_market_regime(ticker)
        print(result)
        print("\nRegime detection working!\n")
        return True
    except Exception as e:
        print(f"\nRegime detection failed: {e}\n")
        return False


def test_multi_timeframe(ticker="RELIANCE.NS"):
    """Test multi-timeframe analysis."""
    print("=" * 80)
    print(f"TEST 2: MULTI-TIMEFRAME ANALYSIS for {ticker}")
    print("=" * 80)

    try:
        result = multi_timeframe_analysis(ticker)
        print(result)
        print("\nMulti-timeframe analysis working!\n")
        return True
    except Exception as e:
        print(f"\nMulti-timeframe analysis failed: {e}\n")
        return False


def test_enhanced_ensemble(ticker="RELIANCE.NS"):
    """Test enhanced ensemble agent with regime awareness."""
    print("=" * 80)
    print(f"TEST 3: REGIME-AWARE ENSEMBLE for {ticker}")
    print("=" * 80)

    try:
        # Create ensemble agent
        agent = create_ensemble_agent()

        # Test invoke with a simple query
        messages = [{"role": "user", "content": f"Analyze {ticker} using regime-aware ensemble voting. First detect the regime, then run all strategies and apply weighted voting."}]

        print("Running ensemble agent (this may take 30-60 seconds)...")
        result = agent.invoke({"messages": messages})

        print("\nAgent Response:")
        print(result['messages'][-1].content)
        print("\nRegime-aware ensemble working!\n")
        return True

    except Exception as e:
        print(f"\nEnsemble test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Phase 2 tests."""
    print("\n" + "=" * 80)
    print("PHASE 2 IMPLEMENTATION TESTS")
    print("=" * 80 + "\n")

    ticker = "RELIANCE.NS"  # You can change this to any Indian stock ticker

    results = []

    # Test 1: Regime Detection
    results.append(("Regime Detection", test_regime_detection(ticker)))

    # Test 2: Multi-Timeframe Analysis
    results.append(("Multi-Timeframe Analysis", test_multi_timeframe(ticker)))

    # Test 3: Enhanced Ensemble (requires LLM, may take time)
    print("\nNOTE: Test 3 requires API calls and may take 30-60 seconds...")
    response = input("Run enhanced ensemble test? (y/n): ")

    if response.lower() == 'y':
        results.append(("Regime-Aware Ensemble", test_enhanced_ensemble(ticker)))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! Phase 2 improvements are working correctly.")
    else:
        print(f"\n{total - passed} test(s) failed. Check error messages above.")

    return passed == total


if __name__ == "__main__":
    # Set up environment
    import os
    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not found in environment.")
        print("Some tests may fail without API key.")
        print("Add GROQ_API_KEY to your .env file.\n")

    # Run tests
    success = run_all_tests()

    # Exit code
    exit(0 if success else 1)
