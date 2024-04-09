# This file is for creating embeddings for Neo4j unstructured data course

from langchain.chains import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

from llm import gpt35 as llm


from langchain_community.graphs import Neo4jGraph


# Variables for the Neo4j database
NEO4J_URI="bolt://34.205.75.218:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="sod-thin-governor"

# connect to the Neo4j database sandbox
# graph = Neo4jGraph(
#     url="bolt://3.231.19.137:7687",
#     username="neo4j",
#     password="dollar-reliabilities-pyramid",
# )

# Define the test connection function
def test_connection():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        # NEO4J_URI,
        "bolt://localhost:7687",
        auth=(NEO4J_USERNAME, "12345678")
    )
    try:
        driver.verify_connectivity()
        connected = True
    except Exception as e:
        connected = False

    driver.close()

    print(f"Connected to Neo4j: {connected}")
    return connected

# Test the connection
test_connection()