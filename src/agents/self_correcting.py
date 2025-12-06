from typing import Dict, Any
from langchain_core.messages import HumanMessage
from src.evaluation.trajectory_evaluator import TrajectoryEvaluator

class SelfCorrectingAgent:
    def __init__(self, agent: Any, name: str, evaluator: TrajectoryEvaluator = None):
        self.agent = agent
        self.name = name
        self.evaluator = evaluator or TrajectoryEvaluator()
        
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invokes the agent with self-correction logic.
        """
        # 1. Initial execution
        result = self.agent.invoke(state)
        
        # 2. Evaluate trajectory
        # The result state contains the full history including new messages
        messages = result.get("messages", [])
        if not messages:
            return result
            
        evaluation = self.evaluator.evaluate_trajectory(messages, self.name)
        
        # 3. Self-Correction Loop (Max 1 retry to avoid infinite loops)
        if evaluation.needs_correction:
            print(f"[{self.name}] Self-Correction Triggered: {evaluation.reasoning}")
            print(f"[{self.name}] Feedback: {evaluation.feedback}")
            
            # Create feedback message
            feedback_msg = HumanMessage(
                content=f"REFLECTION: Your previous execution had issues. \nFeedback: {evaluation.feedback}\n\nPlease fix these issues and provide the updated analysis."
            )
            
            # Update state with feedback
            # We need to be careful with how the agent handles state. 
            # For LangGraph agents, we can usually pass the updated list of messages.
            # The 'result' is the state after the first run.
            retry_state = result.copy()
            retry_state["messages"] = list(messages) + [feedback_msg]
            
            # Retry execution
            final_result = self.agent.invoke(retry_state)
            
            # Optional: Log the correction
            print(f"[{self.name}] Correction complete.")
            return final_result
            
        return result
