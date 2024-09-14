#from typing import TypedDict, Literal

#import os
#from dotenv import load_dotenv
#load_dotenv()
#ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KE')
#TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from langgraph.graph import StateGraph, END
from base_agent.utils.nodes import call_agent_model, agent_route, get_help, extract_task, agent_tool_node
from base_agent.utils.expert_nodes import call_database, create_plan, task_handler, database_handler, user_handler, human_feedback, calculation_handler, llm_handler, task_router, output_handler, feedback_handler
from base_agent.utils.state import AgentState, ExpertState

# Define expert graph

expert_flow = StateGraph(ExpertState)

expert_flow.add_node("InitialRetrieval", call_database)
expert_flow.add_node("CreatePlan", create_plan)
expert_flow.add_node("TaskRouter", task_router)
expert_flow.add_node("DataBaseHandler", database_handler)
expert_flow.add_node("UserHandler", user_handler)
expert_flow.add_node("HumanFeedback", human_feedback)
expert_flow.add_node("FeedbackHandler", feedback_handler)
expert_flow.add_node("CalculationHandler", calculation_handler)
expert_flow.add_node("LLMHandler", llm_handler)
expert_flow.add_node("OutputHandler", output_handler)

expert_flow.set_entry_point("InitialRetrieval")

expert_flow.add_edge("InitialRetrieval", "CreatePlan")
expert_flow.add_edge("CreatePlan", "TaskRouter")

expert_flow.add_conditional_edges(
    "TaskRouter",
    task_handler,
    {
        "database_query": "DataBaseHandler",
        "user_query": "UserHandler",
        "calculation": "CalculationHandler",
        "LLM": "LLMHandler",
        "end": "OutputHandler"
    },
)

expert_flow.add_edge("DataBaseHandler", "TaskRouter")
expert_flow.add_edge("UserHandler", "HumanFeedback")
expert_flow.add_edge("HumanFeedback", "FeedbackHandler")
expert_flow.add_edge("FeedbackHandler", "TaskRouter")
expert_flow.add_edge("CalculationHandler", "TaskRouter")
expert_flow.add_edge("LLMHandler", "TaskRouter")
expert_flow.add_edge("OutputHandler", END)



# Define parent graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_agent_model)
workflow.add_node("DocumentSearch", agent_tool_node)
workflow.add_node("GetHelp", get_help)
workflow.add_node("InvokeExpertModel", extract_task)
workflow.add_node("expert", expert_flow.compile(interrupt_before=["HumanFeedback"]))


# Set the entrypoint as `agent`
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    agent_route,
    
    {
        "doc": "DocumentSearch",
        "help": "GetHelp",
        "expert": "InvokeExpertModel",
        "end": END
    },
)


workflow.add_edge("DocumentSearch", "agent")
workflow.add_edge("GetHelp", END)
workflow.add_edge("InvokeExpertModel", "expert")
workflow.add_edge("expert", END)

graph = workflow.compile()