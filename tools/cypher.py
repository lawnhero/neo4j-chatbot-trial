from langchain.chains import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

from llm import gpt35 as llm
from graph import graph


# Define the qa prompt
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Your task is to convert the user's query based on the schema <SCHEMA>{schema}</SCHEMA>.

When genrating Cypher queries, follow these guidelines:
- When required, use COUNT() function in your Cypher query, and DO NOT use SIZE() function.
- Use only the provided relationship types and properties in the schema.
- Do not use any other relationship types or properties that are not provided.
- When creating alias, use camelCase. For example, "AS movie title" should be "AS movieTitle".

Fine Tuning:

For movie titles that begin with "The", move "the" to the end. 
<examples>
- "The 39 Steps" becomes "39 Steps, The"
- "the Matrix" becomes "Matrix, The"
- "Toy Story" remains "Toy Story"
</examples>

Here is an example of a Cypher query that finds the shortest path between two Persons in the graph:
<example>
``` cypher
MATCH path = shortestPath(
  (p1:Person {{name: "Viola Davis"}})-[:ACTED_IN|DIRECTED*]-(p2:Person {{name: "Kevin Bacon"}})
)
WITH path, p1, p2, relationships(path) AS rels
RETURN
  p1 {{ .name, .born, link:'https://www.themoviedb.org/person/'+ p1.tmdbId }} AS start,
  p2 {{ .name, .born, link:'https://www.themoviedb.org/person/'+ p2.tmdbId }} AS end,
  reduce(output = '', i in range(0, length(path)-1) |
    output + CASE
      WHEN i = 0 THEN
       startNode(rels[i]).name + CASE WHEN type(rels[i]) = 'ACTED_IN' THEN ' played '+ rels[i].role +' in 'ELSE ' directed ' END + endNode(rels[i]).title
       ELSE
         ' with '+ startNode(rels[i]).name + ', who '+ CASE WHEN type(rels[i]) = 'ACTED_IN' THEN 'played '+ rels[i].role +' in '
    ELSE 'directed '
      END + endNode(rels[i]).title
      END
  ) AS pathBetweenPeople
```
</example>

Based on the user query: <QUERY>{question}</QUERY>,
Generate the corresponding Cypher Query:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

# Define the chain for cypher query generation
cypher_qa = GraphCypherQAChain.from_llm(
    llm,          # (1)
    graph=graph,  # (2)
    verbose=True,  # (3)
    cypher_prompt=cypher_prompt
)

