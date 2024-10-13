from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from base_agent.utils.nodes import call_agent_model, agent_route, get_help, extract_task, agent_tool_node
from base_agent.utils.expert_nodes import call_database, create_plan, task_handler, database_handler, user_handler, human_feedback, calculation_handler, llm_handler, task_router, output_handler, feedback_handler
from base_agent.utils.state import AgentState

# Define parent graph
workflow = StateGraph(AgentState)

# Agent nodes
workflow.add_node("agent", call_agent_model)
workflow.add_node("DocumentSearch", agent_tool_node)
workflow.add_node("GetHelp", get_help)
workflow.add_node("InvokeExpertModel", extract_task)

# Expert nodes
workflow.add_node("InitialRetrieval", call_database)
workflow.add_node("CreatePlan", create_plan)
workflow.add_node("TaskRouter", task_router)
workflow.add_node("DataBaseHandler", database_handler)
workflow.add_node("UserHandler", user_handler)
workflow.add_node("HumanFeedback", human_feedback)
workflow.add_node("FeedbackHandler", feedback_handler)
workflow.add_node("CalculationHandler", calculation_handler)
workflow.add_node("LLMHandler", llm_handler)
workflow.add_node("OutputHandler", output_handler)

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
workflow.add_edge("InvokeExpertModel", "InitialRetrieval")


workflow.add_edge("InitialRetrieval", "CreatePlan")
workflow.add_edge("CreatePlan", "TaskRouter")

workflow.add_conditional_edges(
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

workflow.add_edge("DataBaseHandler", "TaskRouter")
workflow.add_edge("UserHandler", "HumanFeedback")
workflow.add_edge("HumanFeedback", "FeedbackHandler")
workflow.add_edge("FeedbackHandler", "TaskRouter")
workflow.add_edge("CalculationHandler", "TaskRouter")
workflow.add_edge("LLMHandler", "TaskRouter")
workflow.add_edge("OutputHandler", END)


# Compile the graph
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_before=['HumanFeedback'])