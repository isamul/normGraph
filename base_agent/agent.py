"""
This module defines the workflow for an agent using a state graph. The workflow includes various nodes representing different tasks and handlers, and edges that define the transitions between these nodes. The workflow is compiled with a memory saver for checkpointing.

Modules and Classes:
- StateGraph: Class to define the state graph.
- END: Constant to define the end state.
- MemorySaver: Class for saving checkpoints.
- AgentState: Class representing the state of the agent.

Functions:
- call_agent_model: Function to call the agent model.
- agent_route: Function to route the agent.
- get_help: Function to get help.
- extract_task: Function to extract a task.
- agent_tool_node: Function representing an agent tool node.
- call_database: Function to call the database.
- create_plan: Function to create a plan.
- task_handler: Function to handle tasks.
- database_handler: Function to handle database queries.
- user_handler: Function to handle user queries.
- human_feedback: Function to handle human feedback.
- calculation_handler: Function to handle calculations.
- llm_handler: Function to handle LLM tasks.
- task_router: Function to route tasks.
- output_handler: Function to handle output.
- feedback_handler: Function to handle feedback.
"""

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

# Define conditional edges from the `agent` node
workflow.add_conditional_edges(
    "agent",
    agent_route,
    
    {
        "doc": "DocumentSearch", # Route to document search
        "help": "GetHelp", # Route to get help
        "expert": "InvokeExpertModel", # Route to invoke expert model
        "end": END # End the workflow
    },
)

# Define edges between nodes (agent)
workflow.add_edge("DocumentSearch", "agent")
workflow.add_edge("GetHelp", END)
workflow.add_edge("InvokeExpertModel", "InitialRetrieval")


workflow.add_edge("InitialRetrieval", "CreatePlan")
workflow.add_edge("CreatePlan", "TaskRouter")


# Define conditional edges from the `TaskRouter` node
workflow.add_conditional_edges(
    "TaskRouter",
    task_handler,
    {
        "database_query": "DataBaseHandler", # Route to database handler
        "user_query": "UserHandler", # Route to user handler
        "calculation": "CalculationHandler", # Route to calculation handler
        "LLM": "LLMHandler", # Route to LLM handler
        "end": "OutputHandler" # Route to output handler
    },
)

# Define edges between nodes (expert)
workflow.add_edge("DataBaseHandler", "TaskRouter") # Loop back to task router after database handler
workflow.add_edge("UserHandler", "HumanFeedback") # Move to human feedback after user handler
workflow.add_edge("HumanFeedback", "FeedbackHandler") # Move to feedback handler after human feedback
workflow.add_edge("FeedbackHandler", "TaskRouter") # Loop back to task router after feedback handler
workflow.add_edge("CalculationHandler", "TaskRouter") # Loop back to task router after calculation handler
workflow.add_edge("LLMHandler", "TaskRouter") # Loop back to task router after LLM handler
workflow.add_edge("OutputHandler", END) # End workflow after output handler


# Compile the graph with a memory saver for checkpointing
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_before=['HumanFeedback'])