# Data ingestion pipeline

The notebooks in this folder need to be run in advance before the main application can be deployed. 

Steps:
- Use LLamaParse to parse a standards document from PDF to markdown
- follow instructions in `markdown_ingestion.ipynb` to create a hierarchical database conaining the standards data
- follow instructions in `voyage_embed.ipynb` to create the corresponding vector embeddings


## Visualization of the Data ingestion process:
![Data ingestion](img/data_ingestion.png)

## Database schema of the neo4j graph-database to be created:
![DB schema](img/db_schema.png)
