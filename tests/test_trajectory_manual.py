import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.trajectory_evaluator import TrajectoryEvaluator
from src.agents.self_correcting import SelfCorrectingAgent
from langchain_core.messages import HumanMessage, AIMessage

class MockAgent:
    def __init__(self):
        self.call_count = 0
        
    def invoke(self, state):
        self.call_count += 1
        messages = state.get("messages", [])
        
        # First call: Return bad output
        if self.call_count == 1:
            return {"messages": messages + [AIMessage(content="I think the stock is good. Buy it.")]}
        
        # Second call (retry): Return good output
        return {"messages": messages + [AIMessage(content="I have checked the multi-timeframe analysis. Weekly is up, Daily is up. Buy.")]}

def test_evaluator():
    print("Testing Trajectory Evaluator...")
    evaluator = TrajectoryEvaluator()
    
    # Test case: Bad trajectory (Technical Analyst missing multi-timeframe)
    messages = [
        HumanMessage(content="Analyze RELIANCE.NS"),
        AIMessage(content="I think the stock is good. Buy it.")
    ]
    
    score = evaluator.evaluate_trajectory(messages, "Technical_Analyst")
    print(f"Score: {score.score}")
    print(f"Feedback: {score.feedback}")
    print(f"Needs Correction: {score.needs_correction}")
    
    if score.needs_correction:
        print("Evaluator correctly flagged bad trajectory.")
    else:
        print("Evaluator failed to flag bad trajectory.")

def test_self_correction():
    print("\nTesting Self-Correction Wrapper...")
    mock_agent = MockAgent()
    # We need to mock the evaluator too to avoid real LLM calls in this unit test logic check, 
    # but for this manual test we can use the real one or a mock.
    # Let's use a mock evaluator for deterministic logic testing.
    
    class MockEvaluator:
        def evaluate_trajectory(self, messages, name):
            from src.evaluation.trajectory_evaluator import TrajectoryScore
            # If the last message is short/bad
            last_msg = messages[-1].content
            if "multi-timeframe" not in last_msg and "Buy it" in last_msg:
                return TrajectoryScore(0.5, "Bad reasoning", "Use multi-timeframe analysis", True)
            return TrajectoryScore(1.0, "Good job", "Keep it up", False)
            
    agent = SelfCorrectingAgent(mock_agent, "Technical_Analyst", evaluator=MockEvaluator())
    
    initial_state = {"messages": [HumanMessage(content="Analyze RELIANCE.NS")]}
    result = agent.invoke(initial_state)
    
    print(f"Final Result Messages: {len(result['messages'])}")
    print(f"Final Content: {result['messages'][-1].content}")
    print(f"Agent Call Count: {mock_agent.call_count}")
    
    if mock_agent.call_count == 2:
        print("Self-correction triggered retry.")
    else:
        print("Self-correction failed to trigger retry.")

if __name__ == "__main__":
    # test_evaluator() # Uncomment to test with real LLM (requires API key)
    test_self_correction()
