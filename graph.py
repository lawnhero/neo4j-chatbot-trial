import streamlit as st
from langchain_community.graphs import Neo4jGraph


graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password='12345678',
)

# graph2 = Neo4jGraph(
#     url="bolt://3.86.36.30:7687",
#     username="neo4j",
#     password="technician-usages-fuse",
# )