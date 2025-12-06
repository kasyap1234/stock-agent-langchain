from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import re
import json

@dataclass
class TrajectoryScore:
    score: float
    reasoning: str
    feedback: str
    needs_correction: bool

class EvaluationResult(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation of the score")
    feedback: str = Field(description="Constructive feedback for the agent")
    needs_correction: bool = Field(description="Whether the agent needs to correct its course")


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags from model output before JSON parsing."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

class TrajectoryEvaluator:
    def __init__(self, model_name: str = "qwen/qwen3-32b"):
        self.llm = ChatGroq(model=model_name, temperature=0, max_retries=5)
        self.parser = JsonOutputParser(pydantic_object=EvaluationResult)
        
    def evaluate_trajectory(self, messages: List[BaseMessage], agent_name: str) -> TrajectoryScore:
        """
        Evaluates the execution trajectory of an agent.
        """
        # Extract the conversation history
        conversation_text = ""
        for msg in messages:
            role = msg.type
            content = msg.content
            conversation_text += f"{role.upper()}: {content}\n\n"
            
        # Select prompt based on agent
        if "Technical" in agent_name:
            criteria = (
                "- Did the agent use 'multi_timeframe_analysis'? (CRITICAL)\n"
                "- Did it check Weekly, Daily, and 4H timeframes?\n"
                "- Is the entry/exit logic consistent with the analysis?\n"
                "- Did it provide specific price levels?"
            )
        elif "Fundamental" in agent_name:
            criteria = (
                "- Did the agent check for upcoming earnings?\n"
                "- Did it identify major recent news?\n"
                "- Did it avoid relying solely on old data?"
            )
        elif "Sentiment" in agent_name:
            criteria = (
                "- Did the agent cite specific sources or recent articles?\n"
                "- Is the sentiment classification supported by evidence?"
            )
        else:
            criteria = (
                "- Is the reasoning logical and step-by-step?\n"
                "- Did the agent answer the user's request directly?"
            )
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Quality Assurance AI for a Stock Trading Agent system. "
                      "Your job is to evaluate the execution trajectory of a sub-agent and ensure it followed protocol.\n\n"
                      "CRITICAL CRITERIA for {agent_name}:\n{criteria}\n\n"
                      "Evaluate the following conversation history. "
                      "If the agent missed critical steps (especially multi-timeframe analysis for Technical Analyst), "
                      "give a low score (<0.7) and provide specific feedback to fix it.\n"
                      "If the agent did well, give a high score (>0.9) and 'Keep up the good work' as feedback.\n\n"
                      "{format_instructions}"),
            ("human", "{conversation}")
        ])
        
        try:
            # Get raw response from LLM
            response = (prompt | self.llm).invoke({
                "agent_name": agent_name,
                "criteria": criteria,
                "conversation": conversation_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Extract content and strip thinking tags
            raw_output = response.content if hasattr(response, 'content') else str(response)
            cleaned_output = strip_thinking_tags(raw_output)
            
            # Parse the cleaned JSON
            result = json.loads(cleaned_output)
            
            return TrajectoryScore(
                score=result["score"],
                reasoning=result["reasoning"],
                feedback=result["feedback"],
                needs_correction=result["needs_correction"] or result["score"] < 0.7
            )
            
        except Exception as e:
            # Fallback in case of parsing error
            print(f"Error evaluating trajectory: {e}")
            return TrajectoryScore(
                score=1.0, # Assume good if we can't evaluate
                reasoning="Evaluation failed",
                feedback="",
                needs_correction=False
            )
