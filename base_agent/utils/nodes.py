"""
This module defines various nodes and utility functions used in the agent's workflow. It includes functions to get language models, route tasks, call models, extract tasks, and provide help. The module also sets up a callback handler for logging and monitoring.

Modules and Classes:
- ChatOpenAI: Class to interact with OpenAI's chat models.
- SearchDataBase: Tool for searching a database.
- ToolNode: Class to define a tool node.
- AIMessage, ToolMessage: Classes for handling messages.
- CallbackHandler: Class for handling callbacks.

Functions:
- _get_model: Function to get a language model based on the model name.
- agent_route: Function to determine the next step based on the agent's state.
- call_agent_model: Function to call the agent model.
- extract_task: Function to extract a task from the agent's state.
- get_help: Function to provide help information.
- call_expert_model: Function to call an expert model.
"""

from functools import lru_cache
from langchain_openai import ChatOpenAI
from base_agent.utils.tools import SearchDataBase, agent_tools
from base_agent.utils.prompts import agent_system_prompt_de
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage
from langfuse.callback import CallbackHandler

# Initialize the callback handler for logging and monitoring
# can be disabled by removing the callback handler from the node "call_agent_model"
langfuse_handler = CallbackHandler(
    secret_key="",
    public_key="",
    host="https://cloud.langfuse.com", # 🇪🇺 EU region
)

# Cache the model instances to avoid redundant initializations
@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "base":
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o", streaming=True)
        return llm
    elif model_name == "mini-t":
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", streaming=True)
        model = llm.bind_tools([SearchDataBase], tool_choice="required")
        return model
    elif model_name == "mini":
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", streaming=False)
        return llm
    elif model_name == "agent":
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", streaming=True)
        model = llm.bind_tools(agent_tools)
        return model

# Define the function that determines whether to continue or not
def agent_route(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish

    print(last_message)

    # Case 1: No tool call
    if not last_message.tool_calls:
        return "end"
    # Case 2: GetHelp tool is called
    elif last_message.tool_calls[0]['name'] == "GetHelp":
        return "help"
    # Case 3: AdvancedDocumentRetriever tool is called
    elif last_message.tool_calls[0]['name'] == "InvokeExpertModel":
        return "expert"
    # Case 3: DocumentRetriever tool is called
    else:
        return "doc"
    

# Define the function that calls the model
def call_agent_model(state):
    messages = state["messages"]
    messages = [{"role": "system", "content": agent_system_prompt_de}] + messages
    #model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model_name = 'agent'
    model = _get_model(model_name)
    response = model.invoke(messages, config={"callbacks": [langfuse_handler]})
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define node that extracts the task from the tool call
def extract_task(state):
    messages = state["messages"]
    last_message = messages[-1]
    task = last_message.tool_calls[-1]['args']['task']
    print(task)
    tool_id = last_message.tool_calls[0]['id']

    return {"task": task, "messages": [ToolMessage(content="Invoking the Expert Model with task: " + str(task), tool_call_id=tool_id, role="ai")]}

# Define function that returns help-section
def get_help(state):
    messages = state["messages"]
    tool_id = messages[-1].tool_calls[0]['id']
    return {"messages": [ToolMessage(content="This is the help section", tool_call_id=tool_id, role="ai")]}

def call_expert_model(state):
    task = state["task"]
    return {"messages": [AIMessage(content="This is the expert model with the task: " + task, role="ai")]}
    

# Define the function to execute tools
agent_tool_node = ToolNode(agent_tools)

#--------------------------------------------


