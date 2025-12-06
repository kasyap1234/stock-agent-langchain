#!/bin/bash
# Optimized runner with parallel execution

if [ -z "$1" ]; then
    echo "Usage: ./run_optimized.sh TICKER"
    echo "Example: ./run_optimized.sh TCS.NS"
    exit 1
fi

# Load API key from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if API key is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "Error: GROQ_API_KEY not found in .env file"
    exit 1
fi

# Run with .venv
source .venv/bin/activate
python main.py "$1"
