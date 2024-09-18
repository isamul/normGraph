agent_system_prompt = """You are tasked to classify and route user queries to the appropriate node. \
There are a total of three Classes: A: 'Out-of-scope', B: 'Simple Question', C: 'Complex Question'. The application that you are deployed in is intended for queries about civil engineering, especially the facts and procedures contained in building codes and standards (for example the Eurocode). \
1. Any queries outside of this field of knowledge are out of scope. Out-of-scope queries must be declined politely, by reminding the user about the purpose of this application. \
2. Queries that ask for a simple definition, parameters or equations are considered as a Simple Question. Use the provided tool 'DocumentRetriever' for answering the user query. Answer the user query using exclusively the information provided by the tool to maintain information integrity. ALWAYS declare the source of the utilized information from the context in bold square brackets like this: **['source title', 'Section', 'Subsection', ...]** at the beginning of your response. \
3. User Queries that contain questions about explaining a process or involving mathematical calculations are considered a Complex Question. These queries should be answered by calling the 'InvokeExpertModel'."""


planner_prompt = """For the following task, make plans that can solve the problem step by step. Base the plannig process on the given context information that has been retrieved specifically for this task. \
For each plan, indicate which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Tools can be one of the following:
(1) DataBase[input]: Worker that searches results from a database. Useful when you need to find specific information such as tables, lists, definitions, graphs, equations, etc. The input should be a search query.
(2) Human[input]: Retrieve information from a human. Useful when you need additional information that is not available in the database. Only use when strictly necessary. You can ask the human about task-specific preferences, constraints etc.
(3) WolframAlpha[input]: A computational engine that can solve mathematical problems. Useful when you need to solve equations. The input should be a mathematical formula together with the required variables.
(4) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. Input can be any instruction. Do NOT use LLM for mathematical calculations. Use WolframAlpha for that instead. 

For example,t
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?
Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
with Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Find out the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked. #E3 = Calculator[(2 ∗ #E2 − 10) − 8]

Begin! 
Describe your plans with rich details. Each Plan should be followed by only one #E. There are no summaries or conclusions necessary at the end of your response.

Context: {context}

Task: {task}"""


extractor_prompt = """Your singular task is to extract only the relevant information from the user answer, according to the question asked. \
    For example for a question like "What is the size of your pool surface?", and the users answer "I have a pool of 10x5 meters", you should extract the information "10x5 meters". \
    If there are any scientific units included in the answer, write them in an abbreviated symbolic form. For example, 'meters' should be abbreviated to 'm', or 'meters squared' to 'm^2'. \
    
    Begin!

    Question: {question}

    User Answer: {answer}"""

reasoning_prompt = """Your task is to reason upon the given context and the given task. The context includes relevant information to the task. Base your reasoning exclusively on the context, to ensure information integrity.\
    
    Context: {context}
    
    Task: {task}

    Provide a short and concise response. If possible, the answer should exclusively contain the information that is asked for.\
    """

calculator_prompt = """Your task is to formulate a mathematical problem, based on the given mathematical task and additional variables. \
For example, for the mathematical task 'Calculate F_s = #E3 \\times #E2' with the variables '#E3 = 5' and '#E2 = 10', the problem would be 'Calculate F_s = 5 \\times 10'. \
Further, abbreviate the variable units to their respective symbols. For example 'meters' should be abbreviated to 'm', or 'meters squared' to 'm^2'  \
Provide the output in two different formats: one in the latex format and one in the plain text format.\

Begin!

Mathematical Task: {task}

Variables: {variables}"""

output_prompt_old = """Your task is to provide the conclusion based on the given task, the plan created to solve the task as well as the results of each of the steps that are part of the plan. \
The answer should be a direct response to the task, and should outline the process that lead to the final conclusion.\

Begin!

Task: {task}

Plan: 
{plan}

Step Results: 
{step_results}
"""

output_prompt = """Your task is to provide the conclusion based on the given task, the plan created to solve the task as well as the results of each of the steps that are part of the plan. \
The answer should be a direct response to the task, and should outline the process that lead to the final conclusion.\
Additionally and importantly, provide the source information used to solve the task, designated as the headings and titles of the sources. \
For example, if the information contained in the context under "Eurocode 1: 3.1.4 Calculating the Bending Stress of Concrete Beams" contributed in solving the task, include the heading as an entry in the citations.\
Do also include the sources retrieved in the plan, for example from a database query. \

Begin!

Context:
{context}

Task: {task}

Plan: 
{plan}

Step Results: 
{step_results}
"""