# Companion Repository to the Masters Thesis "Natural Language Processing for Standards"

This repository contains the source code of the base application logic without the dedicated UI, as well as a few helper notebooks for the data ingetion process and benchmark.


## NormGraph base application
The base application directory is structured as follows:
```
LANGGRAPH-BASE
├── base_agent  # folder containing the main application
│   └── utils   # contains the components the application consists of
│       ├── expert_nodes.py  # graph nodes used in the expert-module
│       ├── nodes.py         # grpah nodes used in the agent-module
│       ├── prompts.py       # contains all prompts used with the PLM in the application
│       ├── state.py         # contains class structure of the application (state machine states)
│       └── tools.py         # supporting functions used throughout the nodes
│   ├── agent.py             # constructs the applications' graph architecture
│   └── requirements.txt     # contains dependencies necessary for executing the application
├── img
├── .env
├── .gitignore
├── application_output.ipynb    # used for debugging purposes
├── assistant.ipynb             # used for debugging purposes
└── langgraph.json              # required for deployment in LangGraph Studio
```

Run the following steps to run the NormGraph in the development environment LangGraph Studio (currently MacOS only):
1. Clone this repository
```shell
git clone https://github.com/langchain-ai/langgraph-example.git
```
   
2. Download ([LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) and follow the install instructions
3. A Docker runtime needs to be present. Docker desktop can be downloaded from [here](https://www.docker.com/products/docker-desktop/).
4. Add environment variables (API-Keys) to the `.env` file. For that copy the `.env.example` using the command below:
```shell
cp .env.example .env
```
5. Open LangGraph Studio and select the cloned directory to run  

- the application NormAI (NormGraph + UI) (add repo) can be run on windows as well


### NormGraph Application Graph-Structure
![Application architecture](img/application_architecture_.png)


### NormGraph in LangGraph UI development environment

![NormGraph in LangGraph Studio](img/LangGraphUI.png)

## Helper Notebooks

...
