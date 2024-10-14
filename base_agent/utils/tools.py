"""
This module provides a set of tools and utilities for interacting with various APIs and databases It includes:

1. Import Dependencies:
   - Imports necessary libraries and modules, including Pydantic for data validation, Neo4j for database interaction.

2. Environment Variables:
   - Defines environment variables for connecting to Neo4j and VoyageAI.

3. Data Classes:
   - Defines several Pydantic data models (`Step`, `Plan`, `StepResult`, `Calculation`, `Conclusion`) to structure and validate data used in the application.

4. Utility Functions:
   - functions for embedding, ranking, and retrieving data from databases.

5. Tool Definitions:
   - Defines tools using the `@tool` decorator for document retrieval, expert model invocation, and database searching, each with specific input schemas and descriptions.

6. Agent Tools:
   - Aggregates the defined tools into a list for use by an agent.
"""

#--------------Import Dependencies-------------------#
from typing import List
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1, Field as FieldV1
import os
import re
#from openai import OpenAI
import voyageai
from neo4j import GraphDatabase
from collections import defaultdict, deque

#----------------- Define envs -----------------#
neo4j_uri = os.environ["NEO4J_URI"]
neo4j_user = os.environ["NEO4J_USER"]
neo4j_password = os.environ["NEO4J_PASSWORD"]
wolfram_alpha_appid = os.environ["WOLFRAM_ALPHA_APPID"]
#EMBEDDING_MODEL  = "text-embedding-3-small" # can be shortened
EMBEDDING_MODEL = 'voyage-multilingual-2'
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
#openai_client = OpenAI ()
vo = voyageai.Client()

print("Driver and OpenAI client initialized...")

#----------------- Define Data Classes -----------------#
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

class StepResult(BaseModel):
    step_number: str
    result: str

class Calculation(BaseModel):
    """Calculation to be performed by the Calculator tool. Includes the mathematical calculation in latex and plain text format."""

    problem_latex: str = Field(description="Mathematical calculation to be performed. Input the equation in latex format.")
    problem_plain_text: str = Field(description="Mathematical calculation to be performed. Input the equation in plain text format.")

class Conclusion(BaseModel):
    """The conclusion to the task that was solved by the system, given the plan created to solve the task, as well as the results of each of the steps that are part of the plan. The conclusion also contains the source information used to solve the task as citations."""
    
    conclusion: str = Field(description="The conclusion based on the given task, the plan created to solve the task as well as the results of each of the steps that are part of the plan. The answer should be a direct response to the task, and should outline the process that lead to the final conclusion. Use markdown to format the text and make it more easily readable.")
    citations: List[str] = Field(description="Source information used to solve the task, designated as the headings and titles of the sources. The citations consists of strings in a list format. Only include the headings and titles of the sources, not the full content of the sources. A citation entry consists of the source document title printed in brackets (e.g. [Eurocode 1]), as well as the specific section number and title of the source used (e.g. 4.5.6 Calculating Wind Load Configurations) -> Exemplary citation entry: 'Eurocode 1: 4.5.6 Calculating Wind Load Configurations'. Also include tables, equations graphs or similar information, that was retrieved during the plan execution. Only extract the relevant information used for the task, and exclude any irrelevant information from the context.")

class Section:
    def __init__(self, id, parent_id, title='', num='', elements=[], isReference=False):
        self.id = id
        self.parent_id = parent_id
        self.title = title
        self.num = num
        self.elements = elements  # Subsections or chunks
        self.isReference = isReference

    def __str__(self):
        section_str = ""
        if self.isReference:
            section_str += f"{self.num} {self.title}\n"  # Double line break after the title

            for element in self.elements:
                section_str += f"{element.__str__()}"

            return section_str
        else:
            section_str += f"{self.num} {self.title}\n"  # Double line break after the title

            for element in self.elements:
                section_str += f"{element.__str__()}"

            return section_str

class Chunk:
    def __init__(self, id, content, rank, type, references=None):
        self.id = id
        self.content = content
        self.rank = rank
        self.type = type
        self.references = references if references else []  # List of referenced sections

    def __str__(self):
        chunk_str = f"\n\n{self.content}"  # Double line break after chunk content
        for ref in self.references:
            chunk_str += f"\n\nReferenziert: {ref.__str__()}\n\n"  # Double line break after each reference

        return chunk_str

#----------------- Define Retrieval Utils -----------------#
def get_embedding(client, text, model):
    response = client.embed(
                    texts=text,
                    model=model,
                    input_type="query",
                )
    return response.embeddings[0]

def reciprocal_rank_fusion(queries, d, k, searchResults, rank_func):
    # based on code from https://safjan.com/implementing-rank-fusion-in-python/ by Krystian Safjan
    return sum([1.0 / (k + rank_func(searchResults[q], d)) if d in searchResults[q] else 0 for q in queries])

def rank_func(results, d):
    return results.index(d) + 1 # adding 1 because ranks start from 1

def gather_unique_values(dictionary):
    unique_values = set()
    for key in dictionary:
        for value in dictionary[key]:
            unique_values.add(value)
    return list(unique_values)

def apply_reciprocal_rank_fusion(dist_results, queries, searchResults):
    # Create the dictionary with the function outputs
    k = 60
    result_dict = {doc: reciprocal_rank_fusion(queries, doc, k, searchResults, rank_func) for doc in dist_results}
    # Sort the dictionary by values in descending order
    sorted_result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_result_dict

def RRFGraphQuery(query: str, k: int, driver: GraphDatabase.driver, client: voyageai.Client):
    """
    Takes a query and returns the top k results from the graph database
    """
    # Perform TextSearch
    indexName = "titles" # Index containing section and chapter titles
    textCypher= f"CALL db.index.fulltext.queryNodes('{indexName}', '{query}') YIELD node, score RETURN DISTINCT node.title AS title, node.id AS id, score"

    textResults, summary, _ = driver.execute_query(
    textCypher, indexName=indexName)

    print("Text search results retrieved...")

    # Perform VectorSearch
    vecIndex = 'content-embeddings-vo'
    resultCount = k
    queryEmbedding = get_embedding(client, query, EMBEDDING_MODEL)
    print("Embedding generated...")
    vectorCypher = '''
WITH $queryEmbedding AS queryVector
CALL db.index.vector.queryNodes($vecIndex, $resultCount, queryVector)
YIELD node, score WHERE score > 0.8
MATCH (node)<-[:HAS_EMBEDDING]-(chunk)-[:PART_OF]->(root)
RETURN DISTINCT root.title as title, root.id AS id, MAX(score) AS maxScore
'''
    vectorResults, summary, _ = driver.execute_query(
    vectorCypher, queryEmbedding=queryEmbedding, vecIndex=vecIndex, resultCount=resultCount)

    print("Vector search results retrieved...")

    searchResults = {'textSearch': [result['id'] for result in textResults], 'vecSearch': [result['id'] for result in vectorResults]}
    unique_values = gather_unique_values(searchResults)
    print("Unique values gathered...")

    # Perform Reciprocal Rank Fusion
    queries = ['textSearch', 'vecSearch']
    ranked_results = apply_reciprocal_rank_fusion(unique_values, queries, searchResults)
    print("RRF performed...")
    return ranked_results

def parse_records_to_dict(elements):
    new_list = [dict(element) for element in elements]
    return new_list

def RetrieveSections(results, driver):
    cypher = """
CALL apoc.cypher.runMany(
  'MATCH (chunk)-[:PART_OF]->(section)-[:PART_OF]->(parent)-[:PART_OF]->(superparent)
   WHERE section.id IN $ids
   RETURN "chunk" AS type, superparent.title AS super_title, superparent.num AS super_num, superparent.id AS super_id, parent.num AS parent_num, parent.title AS parent_title, parent.id AS parent_id, section.num AS num, section.title AS title, section.id AS section_id, chunk.id, chunk.content AS content, chunk.`sequence-num` AS rank
   ORDER BY rank;

   MATCH (ref)<-[:REFERENCES]-(chunk)-[:PART_OF]->(section)
   WHERE section.id IN $ids
   RETURN "reference" AS type, ref.id AS ref_id, labels(ref) AS element_type, chunk.id AS chunk_id
',
  {ids: $ids},
  {statistics: false}
);
"""
    node_information, summary, _ = driver.execute_query(cypher, ids=results)
    node_information = parse_records_to_dict(node_information)

    return node_information

def RetrieveReferences(ref_id):
    refCypher = """MATCH (chunk)-[:PART_OF]->(section)
WHERE section.id IN $ids
RETURN section.id AS parent_id, section.title AS title, section.num AS num, chunk.id AS chunk_id, chunk.content AS content, chunk.`sequence-num` AS rank
ORDER BY rank
"""
    ref_section, summary, _ = driver.execute_query(refCypher, ids=ref_id)
    ref_section = parse_records_to_dict(ref_section)
    return ref_section

def parse_query_response(query_response):
    # Create dictionaries to hold sections and chunks by their IDs
    sections = {}
    chunks = {}

    # First pass: Create Sections and Chunks from the query response
    for row in query_response:
        result = row['result']
        if result['type'] == 'chunk':
            # Create super section
            super_id = result['super_id']
            super_title = result['super_title']
            super_num = result['super_num']
            
            if super_id not in sections:
                super_section = Section(id=super_id, parent_id=None, title=super_title, num=super_num, elements=[])
                sections[super_id] = super_section

            # Create a parent section
            parent_id = result['parent_id']
            parent_title = result['parent_title']
            parent_num = result['parent_num']

            if parent_id not in sections:
                parent_section = Section(id=parent_id, parent_id=super_id, title=parent_title, num=parent_num, elements=[])
                sections[parent_id] = parent_section

            # Create a Section
            section_id = result['section_id']
            title = result['title']
            num = result['num']
            parent_id = result['parent_id']


            if section_id not in sections:
                section = Section(id=section_id, parent_id=parent_id, title=title, num=num, elements=[])
                sections[section_id] = section


            chunk_id = result['chunk.id']
            content = result['content']
            rank = result['rank']
            chunk_type = result['type']
            chunk = Chunk(id=chunk_id, content=content, rank=rank, type=chunk_type)

            sections[section_id].elements.append(chunk)
        
            # Create a chunk
            #chunk_id = result['chunk.id']
            #content = result['content']
            #rank = result['rank']
            #chunk_type = result['type']
            #chunk = Chunk(id=chunk_id, content=content, rank=rank, type=chunk_type)

            # Add chunk to the chunks dictionary
            #chunks[chunk_id] = chunk


        elif result['type'] == 'reference':
            # This is a reference, so update the chunk it refers to
            chunk_id = result['chunk_id']
            ref_id = result['ref_id']
            found = False

            for section in sections.values():
                if found:
                    break
                print(section)
                for element in section.elements:
                    if element.id == chunk_id:

                        # Query the referenced section
                        # Instantiate a new section object
                        ref_data = RetrieveReferences([ref_id])
                        ref_chunks = []
                        for e in ref_data:
                            chunk = Chunk(e['chunk_id'], e['content'], e['rank'], type="")
                            ref_chunks.append(chunk)

                        ref_section = Section(ref_data[0]['parent_id'], parent_id=None, title=ref_data[0]['title'], num=ref_data[0]['num'], elements=ref_chunks, isReference=True)
                        element.references.append(ref_section)
                        found = True
                        break
                    


            #if chunk_id in chunks:
            #    chunks[chunk_id].references.append(ref_id)
            #    print(ref_id)

        else:
            # Create a section
            section_id = result['section_id']
            title = result.get('title')
            num = result.get('num')
            parent_id = result['parent_id']

            # Check if section already exists, otherwise create it
            if section_id not in sections:
                section = Section(id=section_id, title=title, num=num)
                sections[section_id] = section

    # Second pass: Build the hierarchy of sections and attach chunks
    root_section = None
    
    for section in sections.values():
        if section.parent_id is None:
            root_section = section
            break

    assigned_sections = []
    for section in sections.values():
        if section.parent_id == root_section.id:
            root_section.elements.append(section)
            assigned_sections.append(section.id)

    for section in sections.values():
        if section.id not in assigned_sections:
            for parent_section in root_section.elements:
                if section.parent_id == parent_section.id:
                    parent_section.elements.append(section)
                    assigned_sections.append(section.id)
                    break




    return root_section

def reduce_linebreaks(text):
    return re.sub(r'\n{3,}', '\n\n', text)

def parse_steps_fixed(input_string):
    # Split the input by "Plan:" to isolate each plan
    plans = input_string.split("Plan:")
    steps = []
    
    # Define the regex patterns for step_number, step_type, and dependencies
    step_number_pattern = r"(#E\d+)"
    step_input_pattern = r"\[(.*?)\]"  # To extract the content within square brackets
    step_type_patterns = {
        'database_query': r"DataBase",
        'LLM': r"LLM",
        'user_query': r"Human",
        'calculation': r"WolframAlpha"
    }

    # Iterate over each plan (skip the first empty split result)
    for plan in plans[1:]:
        # Extract step_number
        step_number_match = re.search(step_number_pattern, plan)
        step_number = step_number_match.group(1) if step_number_match else None
        
        # Determine step_type
        step_type = None
        for key, pattern in step_type_patterns.items():
            if re.search(pattern, plan):
                step_type = key
                break

        # Extract step_input from within square brackets after the step type
        step_input_match = re.search(step_input_pattern, plan)
        step_input = step_input_match.group(1) if step_input_match else None

        # Extract dependencies (other step numbers referenced)
        dependencies = re.findall(r"(#E\d+)", plan)
        dependencies.remove(step_number)  # Exclude the current step_number from dependencies

        # Create a Step object
        step = Step(step_number=step_number, step_type=step_type, step_input=step_input.strip('"'), dependencies=dependencies)
        steps.append(step)

    return steps

def sort_steps(steps):
    # Step 1: Build a graph and in-degree map
    graph = defaultdict(list)
    in_degree = {step.step_number: 0 for step in steps}
    
    # Populate the graph and in-degree based on dependencies
    for step in steps:
        for dep in step.dependencies:
            graph[dep].append(step.step_number)
            in_degree[step.step_number] += 1

    # Step 2: Use a queue to process steps with zero in-degree (i.e., no dependencies)
    zero_in_degree_queue = deque([step.step_number for step in steps if in_degree[step.step_number] == 0])

    sorted_steps = []

    # Step 3: Process the queue
    while zero_in_degree_queue:
        current_step_number = zero_in_degree_queue.popleft()
        sorted_steps.append(current_step_number)

        # Decrease the in-degree of the neighbors
        for neighbor in graph[current_step_number]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_queue.append(neighbor)

    # Check if there was a cycle (i.e., not all steps are sorted)
    if len(sorted_steps) != len(steps):
        raise ValueError("The steps contain a cycle, so no valid execution order exists.")
    
    return sorted_steps
#----------------- Define the LLM tools -----------------#

@tool
async def DocumentRetriever(query: str, data_type: str):
    """Call to retrieve relevant documents from a specialized database."""

    results = RRFGraphQuery(query, 5, driver, vo)
    
    #print("DocumentRetriever: Results retrieved...")
    keys = [key for key in results.keys()]
    results = RetrieveSections(keys, driver)
    root_section = parse_query_response(results)

    context = root_section.__str__()
    context = reduce_linebreaks(context)
    
    # ToDo: Support type filters
    #context = "Document Placeholder"
    return {'retrieved information': context}
    

class DocumentRetrieverInput(BaseModelV1):
    query: str = FieldV1(description="Searchquery to retrieve relevant context information from a database containing specialized knowledge regarding civil engineering topics. Only search for one subject at a time. Make multiple calls for different subjects.")
    data_type: str = FieldV1(description="Category of the subject to be searched. Choose from the predifined categories depending on the type of information the user query is about. A coefficient is considered a parameter.", 
                          enum=["Definition", "Parameter", "Equation", "Table", "Process", "Proof", "Other"])
    

DocumentRetriever.args_schema = DocumentRetrieverInput
DocumentRetriever.name = "DocumentRetriever"
DocumentRetriever.description = "Returns a list of relevant document snippets for a simple textual user query retrieved from a specialized database for civil engineering and building code topics. Call this tool for answering simple questions regarding definitions, parameters, equations or similar questions that can be answered by retrieving a small amount of information. Only search for one subject at a time. Make multiple calls for different subjects."


@tool 
async def InvokeExpertModel(task: str):
    """Call to invoke the expert model for answering complex user queries."""

    return ["Expert Placeholder"]

class InvokeExpertModelInput(BaseModelV1):
    task: str = FieldV1(description="Task to be performed by the expert model. Input the unmodified user given task as, while including all relevant information from the conversation history if applicable, for the best possible results. Don't change nuances or details of the user query, like omitting 'my' or 'mine'. The expert models behavior depends on these little nuances.")

InvokeExpertModel.args_schema = InvokeExpertModelInput
InvokeExpertModel.name = "InvokeExpertModel"
InvokeExpertModel.description = "Returns the result of a complex user query, retrieved from the expert model. Call this tool to answer complex user queries, that require a detailed explanation or a highly specialized answer regarding civil engineering based proofs, processes, or mathematical calculations. Include all relevant information from the user as well as the conversation history to ensure the best possible results."

@tool
async def SearchDataBase(query: str, data_type: str, category: str):
    """Call to retrieve relevant documents required for answering the user query from a database, containing information about civil engineering processes and terminology."""
    
    # modified version of the DocumentRetriever tool containing additional category information for the PLM to use
    results = RRFGraphQuery(query, 3, driver, vo)
    #print("SearchDataBase: Results retrieved...")
    keys = [key for key in results.keys()]
    results = RetrieveSections(keys, driver)
    root_section = parse_query_response(results)

    context = root_section.__str__()
    context = reduce_linebreaks(context)
    
    #context = "Context Placeholder" + f"Query: {query}, Data Type: {data_type}, Category: {category}"
    return {'retrieved information': context}

class SearchDataBaseInput(BaseModelV1):
    query: str = FieldV1(description="Searchquery to retrieve relevant context information for highly complex user queries from a specialized database.")
    data_type: str = FieldV1(description="Category of the subject to be searched. Choose from the predifined categories depending on the type of information the user query is about.", 
                          enum=["Definition", "Parameter", "Equation", "Table", "Process", "Proof", "Other"])
    category: str = FieldV1(description="Thematic category of civil engineering to be searched. Choose depending on the topic of the user query: 'DIN 1993-1-3' for Snowloads",
                       enum=["DIN 1993-1-3", "Other"])
    
SearchDataBase.args_schema = SearchDataBaseInput
SearchDataBase.name = "SearchDataBase"
SearchDataBase.description = "Returns a list of relevant document snippets for a complex user query, retrieved from a specialized database for civil engineering and building code topics. Call this tool, in case of questions regarding civil engineering based proofs, processes, or mathematical calculations."

@tool
async def GetHelp(query: str):
    """Call to display information to the user, on how to use this application."""
    return ["Help Placeholder"]

class GetHelpInput(BaseModelV1):
    is_needed: bool = FieldV1(description="Boolean value to determine if the user needs help or not.", 
                            enum=[True, False])

GetHelp.args_schema = GetHelpInput
GetHelp.name = "GetHelp"
GetHelp.description = "Returns information on how to use this application. Call this tool to display information to the user, on how to use this application. If the user seems to be lost or needs help, call this tool to provide them with the necessary information."

agent_tools = [DocumentRetriever, InvokeExpertModel, GetHelp]
