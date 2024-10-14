# Benchmark

The benchmark has been conducted with three configuraions:
- NormGraph
- 4o
- 4o+RAG

This folder contains the benchmarking notebooks for automatically testing the *4o* as well as the *4o+RAG* reference models. *NormGraph* itself has been benchmarked using LangGraph Studio manually, while using [Langfuse](https://langfuse.com) for token-count data.

The folder is structured as follows:

```shell
helper_notebooks_benchmark
└── results
   ├── Benchmark_results.xlsx   # excel sheet aggregating all results, comments and statistics of the benchmark
   ├── token_counts_4o.csv      # csv-file with imput- and output-token count of the 4o reference model
   └── token_counts_4o+RAG.csv  # csv-file with imput- and output-token count of the 4o+RAG reference model
├── benchmark_4o.ipynb          # Notebook used for benchmarking the 4o reference model
├── benchmark_4o+RAG.ipynb      # Notebook used for benchmarking the 4o+RAG reference model
└── questions.csv               # csv list containing all benchmark questions
```