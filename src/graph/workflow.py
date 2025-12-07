"""
Optimized workflow with parallel execution and error handling.
"""
from langgraph.graph import StateGraph, END
from src.agents.supervisor import AgentState, synthesize_final_call
from src.agents.analysts import create_technical_analyst, create_fundamental_analyst, create_sentiment_analyst
from src.agents.critic import create_critic_agent
from src.agents.planner import create_planner_agent
from src.agents.ensemble import create_ensemble_agent
from src.agents.self_correcting import SelfCorrectingAgent  # NEW
from src.agents.dynamic_prompts import get_dynamic_prompt
from src.agents.context_setup import setup_analysis_context # NEW
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import functools
import warnings
import asyncio
from typing import Dict

warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

# Create all agents
planner_agent = create_planner_agent()
# Wrap analysts with Self-Correction
tech_agent = SelfCorrectingAgent(create_technical_analyst(), "Technical_Analyst")
fund_agent = SelfCorrectingAgent(create_fundamental_analyst(), "Fundamental_Analyst")
sent_agent = SelfCorrectingAgent(create_sentiment_analyst(), "Sentiment_Analyst")

ensemble_agent = create_ensemble_agent()
critic_agent = create_critic_agent()

def agent_node_with_error_handling(state, agent, name):
    """Agent node with enhanced error handling."""
    try:
        # Inject dynamic prompt if context is available
        sector = state.get("sector")
        regime = state.get("regime")
        
        if sector and regime:
            prompt = get_dynamic_prompt(name, sector, regime)
            if prompt:
                # Create a local state copy with injected prompt
                # We prepend it to messages so it acts as system context
                local_state = state.copy()
                local_state["messages"] = list(state["messages"]) + [SystemMessage(content=prompt)]
                result = agent.invoke(local_state)
            else:
                result = agent.invoke(state)
        else:
            result = agent.invoke(state)

        last_message = result["messages"][-1]
        return {"messages": [HumanMessage(content=last_message.content, name=name)]}
    except Exception as e:
        error_msg = f"Error in {name}: {str(e)}\nReturning partial analysis."
        return {"messages": [HumanMessage(content=error_msg, name=name)]}

def run_analysts_sequential(state: AgentState) -> Dict:
    """
    Run Technical, Fundamental, and Sentiment analysts sequentially to reduce
    concurrent LLM calls (helps avoid Groq 429 rate limits).
    """
    try:
        agents = [
            (tech_agent, "Technical_Analyst"),
            (fund_agent, "Fundamental_Analyst"),
            (sent_agent, "Sentiment_Analyst"),
        ]
        all_messages = []
        for agent, name in agents:
            result = agent_node_with_error_handling(state, agent, name)
            all_messages.extend(result["messages"])
        return {"messages": all_messages}
    except Exception as e:
        error_msg = f"Error in analysts: {str(e)}"
        return {"messages": [HumanMessage(content=error_msg, name="Analysts")]}

# Create Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("ContextSetup", setup_analysis_context) # NEW
workflow.add_node("Planner", functools.partial(agent_node_with_error_handling, agent=planner_agent, name="Planner"))
workflow.add_node("ParallelAnalysts", run_analysts_sequential)  # lowered concurrency
workflow.add_node("Ensemble", functools.partial(agent_node_with_error_handling, agent=ensemble_agent, name="Ensemble"))
workflow.add_node("Critic", functools.partial(agent_node_with_error_handling, agent=critic_agent, name="Critic"))
workflow.add_node("FinalSynthesis", synthesize_final_call)

# Optimized flow: ContextSetup -> Planner -> All 3 Analysts in Parallel -> Ensemble -> Critic -> Final
workflow.add_edge("ContextSetup", "Planner")
workflow.add_edge("Planner", "ParallelAnalysts")
workflow.add_edge("ParallelAnalysts", "Ensemble")
workflow.add_edge("Ensemble", "Critic")
workflow.add_edge("Critic", "FinalSynthesis")
workflow.add_edge("FinalSynthesis", END)

workflow.set_entry_point("ContextSetup")

app = workflow.compile()
