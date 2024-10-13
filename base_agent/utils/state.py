# This file defines the data structures and types used to represent the state of an agent in the system.
# It includes classes for individual steps, plans, and the overall agent state, which are used to manage and execute tasks sequentially.

import operator
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import List, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from base_agent.utils.tools import StepResult

class Step(BaseModel):
    """Step to contribute to solving a task sequentially. Includes the task desctiption as well as the optional data to use."""

    step_number: str = Field(
        description="The step number is a unique identifier for each step in the plan. The step number is in the format '#EX', where X is the step number. For example, the first step is #E1, the second step is #E2, and so on.",
    )

    step_type: str = Field(
        description="There are four types of possible steps: DataBase, Human, WolframAlpha, and LLM. For DataBase, specify database_query as step_type. For Human, specify user_query as step_type. For WolframAlpha, specify calculation as step_type. For LLM, specify LLM as step_type.",
        enum=['database_query', 'user_query', 'calculation', 'LLM']
    )
    step_input: str = Field(
        description="The step_input must contain the input for the current step. The input is specified in the brackets '[...]' right after the step_type."
    )
    dependencies: List[str] = Field(
        description="Dependencies can be recognized by the #EX format contained in the step inputs. If a step depends on the output of another step, the step number of the dependent step is specified here as a list. If there are no dependencies, this field is empty.",
    )

class Plan(BaseModel):
    """Plan to follow to solve the given task. Includes the steps to be executed in sequential order. Steps can be a Database query, a Calculation, or a User query."""

    steps: List[Step] = Field(
        description="Different steps to follow, which should be in sorted sequential order."
    )


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: str
    plan: Plan
    plan_index: int
    step_results: Annotated[StepResult, operator.add]
    context: str
    response: str
    log: str
    