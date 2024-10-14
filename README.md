![NormGraph](img/banner.png)

<a href="https://doi.org/10.5281/zenodo.13930597"><img src="https://zenodo.org/badge/857379406.svg" alt="DOI"></a>

# Companion Repository to the Masters Thesis "Natural Language Processing for Standards"

This repository contains the source code of the base application logic without the dedicated UI, as well as a few helper notebooks for the data ingetion process and benchmark. 

The web-app implementation *normAI* with a dedicated UI is available in a separate repository [here](https://github.com/isamul/normAI).


## NormGraph base application
The base application directory is structured as follows:
```
LANGGRAPH-BASE
├── base_agent                  # folder containing the main application
│   └── utils                   # contains the components the application consists of
│       ├── expert_nodes.py     # graph nodes used in the expert-module
│       ├── nodes.py            # grpah nodes used in the agent-module
│       ├── prompts.py          # contains all prompts used with the PLM in the application
│       ├── state.py            # contains class structure of the application (state machine states)
│       └── tools.py            # supporting functions used throughout the nodes
│   ├── agent.py                # constructs the applications' graph architecture
│   └── requirements.txt        # contains dependencies necessary for executing the application
├── img
├── .env
├── .gitignore
├── application_output.ipynb    # used for debugging purposes
├── assistant.ipynb             # used for debugging purposes
├── langgraph.json              # required for deployment in LangGraph Studio
├── helper_notebooks_benchmark  # contains notebooks used for the benchmark
└── helper_notebooks_ingestion  # contains notebooks required for the initial data ingestion
```

Run the following steps to run the NormGraph in the development environment LangGraph Studio (currently MacOS only):
1. Clone this repository
```shell
git clone https://github.com/isamul/normGraph.git
```
   
2. Download [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) and follow the install instructions
3. A Docker runtime needs to be present. Docker desktop can be downloaded from [here](https://www.docker.com/products/docker-desktop/).
4. Add environment variables (API-Keys) to the `.env` file. For that copy the `.env.example` using the command below:
```shell
cp .env.example .env
```
5. Open LangGraph Studio and select the cloned directory to run the application

⚠️ Before running NormGraph
- running NormGraph requires a [neo4j](https://neo4j.com) Graph-Database containing data of standard documents (follow steps in `helper_notebooks_ingestion` below to setup a neo4j instance)
- the application NormAI (NormGraph + UI) (add repo) can be run on windows as well


### NormGraph Application Graph-Structure
![Application architecture](img/application_architecture_.png)


### NormGraph in LangGraph UI development environment

![NormGraph in LangGraph Studio](img/LangGraphUI.png)

## Helper Notebooks

The helper notebooks contain smaller utilities required for running the main application. The notebooks in `helper_notebooks_ingestion` need to be run in advance for the application to be functional.


```
helper_notebooks_benchmark
└── results
   ├── Benchmark_results.xlsx   # excel sheet aggregating all results, comments and statistics of the benchmark
   ├── token_counts_4o.csv      # csv-file with imput- and output-token count of the 4o reference model
   └── token_counts_4o+RAG.csv  # csv-file with imput- and output-token count of the 4o+RAG reference model
├── benchmark_4o.ipynb          # Notebook used for benchmarking the 4o reference model
├── benchmark_4o+RAG.ipynb      # Notebook used for benchmarking the 4o+RAG reference model
└── questions.csv               # csv list containing all benchmark questions
```

```
helper_notebooks_ingestion
├── img
├── markdown_ingestion.ipynb    # Notebook required for ingesting markdown into the neo4j database
└── voyage_embed.ipynb          # Notebook required for creating text embeddings for the contents of the database
```



