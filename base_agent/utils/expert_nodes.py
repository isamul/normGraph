"""
This module defines various expert nodes and utility functions used in the agent's workflow. It includes functions to get language models, handle dependencies, call databases, create plans, route tasks, handle user queries, and perform calculations. The module also sets up a callback handler for logging and monitoring.

Modules and Classes:
- ChatOpenAI: Class to interact with OpenAI's chat models.
- SearchDataBase: Tool for searching a database.
- ToolNode: Class to define a tool node.
- AIMessage: Class for handling AI messages.
- CallbackHandler: Class for handling callbacks.
- Plan, StepResult, Calculation, Conclusion: Classes for handling different types of steps and results.

Functions:
- _get_model: Function to get a language model based on the model name.
- add_dependencies: Function to add dependencies to a step.
- add_dependencies_to_string: Function to add dependencies to a string.
- call_database: Function to call the database.
- create_plan: Function to create a plan.
- task_router: Function to route tasks.
- task_handler: Function to handle tasks.
- database_handler: Function to handle database queries.
- user_handler: Function to handle user queries.
- human_feedback: Function to handle human feedback.
- feedback_handler: Function to handle feedback.
- calculation_handler: Function to handle calculations.
- llm_handler: Function to handle LLM tasks.
- output_handler: Function to handle output.
"""

from functools import lru_cache
from langchain_openai import ChatOpenAI
from base_agent.utils.tools import Plan, StepResult, Calculation, Conclusion, SearchDataBase, parse_steps_fixed, DataBase, sort_steps, wa
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from openai import OpenAI
from base_agent.utils.prompts import planner_prompt, extractor_prompt, reasoning_prompt, calculator_prompt, output_prompt

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
    elif model_name == "calculator":
        client = OpenAI()
        assistant = client.beta.assistants.retrieve(
            assistant_id='asst_7VQGXcbxkAgXMYNLns3e5tTU'
            )
        return client, assistant

# Function to add dependencies to a step
def add_dependencies(step, dependencies, dependency_results):
    for dependency in dependencies:
            for result in dependency_results:
                # check if result is a StepResult object or a dictionary
                if isinstance(result, StepResult):
                    if result.step_number == dependency:
                        step.step_input = step.step_input.replace(dependency, result.result)
                else:
                    if result['step_number'] == dependency:
                        step.step_input = step.step_input.replace(dependency, result['result'])

    return step

# Function to add dependencies to a string
def add_dependencies_to_string(step, dependencies, dependency_results):
    dependency_string = ""
    for dependency in dependencies:
            for result in dependency_results:
                # check if result is a StepResult object or a dictionary
                if isinstance(result, StepResult):
                    if result.step_number == dependency:
                        dependency_string += f"{dependency} = {result.result}\n"

                        #step.step_input = step.step_input.replace(dependency, result.result)
                else:
                    if result['step_number'] == dependency:
                        dependency_string += f"{dependency} = {result['result']}\n"
                        #step.step_input = step.step_input.replace(dependency, result['result'])

    return dependency_string

# Function for the initial database query
def call_database(state):
    task = state["task"]
    context = ""
    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Create a search query for each question that is asked for in the user query."),
            ("user", "{task}"),
        ]
    )
    retriever_model = _get_model("mini-t")

    retriever = retriever_prompt | retriever_model
    retriever_output = retriever.invoke({"task": task})

    for calls in retriever_output.tool_calls:
        print(calls['args']['query'])
        res = SearchDataBase(query=calls['args']['query'], data_type=calls['args']['data_type'], category=calls['args']['category'])
        context += res['retrieved information']

    return {"context": context}

# Function to create a plan
def create_plan(state):
    task = state["task"]
    context = state["context"]
    model = _get_model("base")

    result = model.invoke(planner_prompt.format(task=task, context=context))
    
    steps = parse_steps_fixed(result.content)
    print(steps)
    sorted_step_order = sort_steps(steps)
    sorted_steps = sorted(steps, key=lambda step: sorted_step_order.index(step.step_number))
    plan = Plan(steps=sorted_steps)
   

    return {"plan": plan, "plan_index": 0, "step_results": []}

# Dummy-Function to route tasks
def task_router(state):
    return {"log": 'routing to next task...'}

# Function to route tasks
def task_handler(state):
    index = state["plan_index"]
    plan = state["plan"]
    step_count = len(plan.steps)

    print("Index: " + str(index) + ", Step count: " + str(step_count))
    if index < (step_count):
        current_step = plan.steps[index]

        if current_step.step_type == "database_query":
            return "database_query"
        elif current_step.step_type == "user_query":
            return "user_query"
        elif current_step.step_type == "calculation":
            return "calculation"
        elif current_step.step_type == "LLM":
            return "LLM"
            
    elif index == (step_count):
        return "end"
    
# Function to handle database queries
def database_handler(state):
    index = state["plan_index"]
    plan = state["plan"]
    current_step = plan.steps[index]

    if current_step.dependencies != []:
        dependencies = current_step.dependencies
        dependency_results = state["step_results"]

        current_step = add_dependencies(current_step, dependencies, dependency_results)

    res = SearchDataBase(query=current_step.step_input, data_type="", category="")
    res_str = res['retrieved information']

    sr = StepResult(step_number=current_step.step_number, result=res_str)

    return {"step_results": [sr], "plan_index": index + 1}

# Function to initiate user feedback process
def user_handler(state):
    index = state["plan_index"]
    plan = state["plan"]
    current_step = plan.steps[index]

    if current_step.dependencies != []:
        dependencies = current_step.dependencies
        dependency_results = state["step_results"]

        current_step = add_dependencies(current_step, dependencies, dependency_results)

    user_query = current_step.step_input


    return {"messages": [AIMessage(user_query)]}

# Dummy-Function to wait for human feedback
def human_feedback(state):
    return {"log": 'waiting for user feedback...'}

# Function to parse feedback
def feedback_handler(state):
    messages = state["messages"]
    last_message = messages[-1].content
    index = state["plan_index"]
    plan = state["plan"]
    current_step = plan.steps[index]

    question = current_step.step_input
    model = _get_model("mini")

    result = model.invoke(extractor_prompt.format(question=question, answer=last_message))
    

    sr = StepResult(step_number=current_step.step_number, result=result.content)
    return {"step_results": [sr], "plan_index": index + 1}

# Function to handle calculations
def calculation_handler(state):
    index = state["plan_index"]
    plan = state["plan"]
    current_step = plan.steps[index]
    dependency_string = ""

    if current_step.dependencies != []:
        dependencies = current_step.dependencies
        dependency_results = state["step_results"]

        dependency_string = add_dependencies_to_string(current_step, dependencies, dependency_results)

    #print(dependency_string)
    model = _get_model("mini")
    structured_model = model.with_structured_output(Calculation, method="json_schema") 
    result = structured_model.invoke(calculator_prompt.format(task=current_step.step_input, variables=dependency_string))

    #print("Calculator Input: " + str(result.problem_plain_text))

    print("Augmented Step Input: " + str(current_step.step_input))

    calc_client, calc_model = _get_model("calculator")
    thread = calc_client.beta.threads.create()
    
    message = calc_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=result.problem_plain_text
        )
    
    run = calc_client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=calc_model.id
        )
    
    if run.status == 'completed': 
        messages = calc_client.beta.threads.messages.list(
            thread_id=thread.id
        )

    response = messages.data[0].content[0].text.value
    sr = StepResult(step_number=current_step.step_number, result=response)
    return {"step_results": [sr], "plan_index": index + 1}


# Function to call LLM as a tool of the expert model
def llm_handler(state):
    index = state["plan_index"]
    plan = state["plan"]
    current_step = plan.steps[index]
    context = state["context"]

    if current_step.dependencies != []:
        dependencies = current_step.dependencies
        dependency_results = state["step_results"]

        current_step = add_dependencies(current_step, dependencies, dependency_results)



    model = _get_model("base")
    result = model.invoke(reasoning_prompt.format(context=context, task=current_step.step_input))

    print("Augmented Step Input: " + str(current_step.step_input))  

    sr = StepResult(step_number=current_step.step_number, result=result.content)

    return {"step_results": [sr], "plan_index": index + 1}

# Function to generate the final output
def output_handler(state):
    step_results = state["step_results"]
    context = state["context"]

    task = state["task"]
    plan = state["plan"]

    plan_string = ""
    result_string = ""
    for step in plan.steps:
        plan_string += f"Step {step.step_number} [{step.step_type}]: {step.step_input}, depending on steps: [{step.dependencies}]\n"

    for result in step_results:
        if isinstance(result, StepResult):
            result_string += f"Step {result.step_number} result: {result.result}\n"
        else:
            result_string += f"Step {result['step_number']} result: {result['result']}\n"

    model = _get_model("mini")
    structured_model = model.with_structured_output(Conclusion, method="json_schema")
    

    result = structured_model.invoke(output_prompt.format(context=context, task=task, plan=plan_string, step_results=result_string))

    messages = []
    messages.append(AIMessage(result.conclusion))
    messages.append(AIMessage("References: " + str(result.citations)))
   

    return {"messages": messages}