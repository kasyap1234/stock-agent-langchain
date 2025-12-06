from typing import List, Dict
from src.memory.performance_tracker import find_similar_examples

def format_example(example: Dict) -> str:
    """Formats a single prediction example."""
    return (
        f"Example (Outcome: {example.get('outcome', 'WIN')}):\n"
        f"Ticker: {example['ticker']}\n"
        f"Context: {example.get('sector', 'N/A')} | {example.get('regime', 'N/A')}\n"
        f"Analysis: {example['recommendation']}\n"
        f"Result: Entry {example['entry']} -> Target {example['target']}\n"
    )

def get_dynamic_prompt(agent_type: str, sector: str = None, regime: str = None) -> str:
    """
    Generates a dynamic prompt section with few-shot examples.
    """
    if not sector or not regime:
        return ""
        
    examples = find_similar_examples(sector, regime, outcome="WIN", limit=3)
    
    if not examples:
        return ""
        
    prompt_section = "\n\n### SUCCESSFUL EXAMPLES FROM SIMILAR CONDITIONS\n"
    prompt_section += "Here are past successful analyses in similar market conditions:\n\n"
    
    for ex in examples:
        prompt_section += format_example(ex) + "\n"
        
    prompt_section += "Use these examples to guide your reasoning style and depth.\n"
    
    return prompt_section
