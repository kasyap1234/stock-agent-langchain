# Quick Start Guide

## Environment Setup

### 1. Create .env File
Create a `.env` file in the project root with your API keys:
```bash
# Required: Groq API Key (primary LLM service)
GROQ_API_KEY=sk-your-groq-key-here

# Optional: Enhanced market data APIs
ALPHA_VANTAGE_KEY=your-alpha-vantage-key-here
FINNHUB_API_KEY=your-finnhub-key-here

# Optional: LangChain/Gemini fallback integration
GOOGLE_API_KEY=your-google-gemini-key-here
```

### 2. Install Dependencies
```bash
# Install Python dependencies with uv
uv pip install -r requirements.txt

# Activate virtual environment
source .venv/bin/activate
```

### 3. Verify Setup
Test your configuration:
```bash
./run.sh TCS.NS
```

If you see "GROQ_API_KEY not found", ensure your `.env` file exists and contains the correct API key.

## Easiest Way to Run (Recommended)

Use the provided shell script:

```bash
./run.sh TCS.NS
```

That's it! The script automatically:
- Loads the API key from `.env`
- Activates the virtual environment
- Runs the analysis

## Alternative Methods

### Method 1: Direct command with API key
```bash
GROQ_API_KEY=sk-your-groq-key-here uv run main.py TCS.NS
```

### Method 2: Manual venv activation
```bash
source venv/bin/activate
export GROQ_API_KEY=sk-your-groq-key-here
python main.py TCS.NS
```

## Examples

```bash
# Analyze Reliance
./run.sh RELIANCE.NS

# Analyze Infosys
./run.sh INFY.NS

# Analyze HDFC Bank
./run.sh HDFCBANK.NS
```

## Troubleshooting

If you get "GROQ_API_KEY not found":
1. Make sure `.env` file exists in the project root
2. Check that it contains: `GROQ_API_KEY=your_key_here`
3. Try Method 1 with the explicit API key

## Output

The bot will:
1. Fetch stock data and calculate technical indicators
2. Search for recent news and events
3. Analyze market sentiment
4. Generate a comprehensive trade call with:
   - Bias (Bullish/Bearish/Neutral)
   - Entry price
   - Target levels
   - Stop loss
   - Risk factors
