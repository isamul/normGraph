from functools import lru_cache
from langchain_openai import ChatOpenAI
from base_agent.utils.tools import Plan, StepResult, Calculation, Conclusion, SearchDataBase, parse_steps_fixed, DataBase, sort_steps, wa
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
#from pydantic import BaseModel, Field
#from typing import List
from base_agent.utils.prompts import planner_prompt, extractor_prompt, reasoning_prompt, calculator_prompt, output_prompt


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
# bind tools?
    
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
        res = DataBase(query=calls['args']['query'], data_type=calls['args']['data_type'], category=calls['args']['category'])
        context += res['retrieved information']

    return {"context": context}

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
    #sr = StepResult(step_number="E0", result="Plan created")

    return {"plan": plan, "plan_index": 0, "step_results": []}

def task_router(state):
    return {"log": 'routing to next task...'}

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
    

def database_handler(state):
    index = state["plan_index"]
    plan = state["plan"]
    current_step = plan.steps[index]

    if current_step.dependencies != []:
        dependencies = current_step.dependencies
        dependency_results = state["step_results"]

        current_step = add_dependencies(current_step, dependencies, dependency_results)

    #res = DataBase(query=current_step.step_input, data_type="placeholder", category="placeholder")

    sr = StepResult(step_number=current_step.step_number, result="Area 2")

    #sr = {"step_number": current_step.step_number, "result": "Database response"}
    #step_results.append(sr)


    return {"step_results": [sr], "plan_index": index + 1}

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


def human_feedback(state):
    return {"log": 'waiting for user feedback...'}


def feedback_handler(state):
    messages = state["messages"]
    last_message = messages[-1].content
    index = state["plan_index"]
    plan = state["plan"]
    current_step = plan.steps[index]

    # add gpt-4o to extract relevant information from the user feedback according to the user_query
    question = current_step.step_input
    model = _get_model("mini")

    result = model.invoke(extractor_prompt.format(question=question, answer=last_message))
    
    #step_results.append(StepResult(step_number=current_step.step_number, result=last_message))

    sr = StepResult(step_number=current_step.step_number, result=result.content)
    #sr = {"step_number": current_step.step_number, "result": result.content}
    return {"step_results": [sr], "plan_index": index + 1}

def calculation_handler(state):
    index = state["plan_index"]
    plan = state["plan"]
    current_step = plan.steps[index]
    dependency_string = ""

    if current_step.dependencies != []:
        dependencies = current_step.dependencies
        dependency_results = state["step_results"]

        #current_step = add_dependencies(current_step, dependencies, dependency_results)
        dependency_string = add_dependencies_to_string(current_step, dependencies, dependency_results)

    # wolfram alpha should be prompted to solve the given equation
    # llm model should read the input and provide the raw math problem
    print(dependency_string)
    model = _get_model("mini")
    structured_model = model.with_structured_output(Calculation, method="json_schema") 
    result = structured_model.invoke(calculator_prompt.format(task=current_step.step_input, variables=dependency_string))

    print("Calculator Input: " + str(result.problem_plain_text))

    print("Augmented Step Input: " + str(current_step.step_input))

    res = wa.query(result.problem_plain_text)
    sr = StepResult(step_number=current_step.step_number, result=next(res.results).text)

    return {"step_results": [sr], "plan_index": index + 1}

def llm_handler(state):
    index = state["plan_index"]
    plan = state["plan"]
    current_step = plan.steps[index]
    context = state["context"]

    if current_step.dependencies != []:
        dependencies = current_step.dependencies
        dependency_results = state["step_results"]
        #print(dependency_results)
        #print("Hello from llm")

        current_step = add_dependencies(current_step, dependencies, dependency_results)



    model = _get_model("base")
    result = model.invoke(reasoning_prompt.format(context=context, task=current_step.step_input))

    print("Augmented Step Input: " + str(current_step.step_input))  

    sr = StepResult(step_number=current_step.step_number, result=result.content)

    return {"step_results": [sr], "plan_index": index + 1}

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
    
    #result = model.invoke(output_prompt.format(task=task, plan=plan_string, step_results=result_string)) old output method

    result = structured_model.invoke(output_prompt.format(context=context, task=task, plan=plan_string, step_results=result_string))

    messages = []
    messages.append(AIMessage(result.conclusion))
    messages.append(AIMessage("References: " + str(result.citations)))
   

    return {"messages": messages}