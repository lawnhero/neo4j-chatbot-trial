import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA


# This file is in the solutions folder to separate the solution
# from the starter project code.
# from llm import llm, embeddings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()

# tag::vector[]
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # <1>
    url=st.secrets["NEO4J_URI"],             # <2>
    username=st.secrets["NEO4J_USERNAME"],   # <3>
    password=st.secrets["NEO4J_PASSWORD"],   # <4>
    index_name="moviePlots",                 # <5>
    node_label="Movie",                      # <6>
    text_node_property="plot",               # <7>
    embedding_node_property="plotEmbedding", # <8>
    retrieval_query="""
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        directors: [ (person)-[:DIRECTED]->(node) | person.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
"""
)
# end::vector[]

retriever = neo4jvector.as_retriever(search_kwargs={"k": 3})

kg_qa = RetrievalQA.from_chain_type(
    llm,                  # <1>
    chain_type="stuff",   # <2>
    retriever=retriever,  # <3>
    verbose=True          # <4>
)

# kg_qa = RetrievalQA.from_llm(
#     llm,                  # <1>
#     # chain_type="stuff",   # <2>
#     retriever=retriever,  # <3>
#     verbose=True          # <4>
# )
# end::qa[]


# tag::generate-response[]
def generate_vector_search_response(prompt):
    """
    Use the Neo4j Vector Search Index
    to augment the response from the LLM
    """

    # Handle the response
    response = kg_qa.invoke({"query": prompt})

    return response['result']
# end::generate-response[]

# USE lcel to create a new chain for retrival 
template = """
    You are a movie expert providing information about movies.
    Be as helpful as possible and return as much information as possible.
    Answer following query by summarizing the relevant context. 
    Query: {query}
    Context: {context}
    """
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template(template)
setup_retrieval = RunnableParallel(
    {"context": retriever,
        "query": RunnablePassthrough()}
)

chain = setup_retrieval | prompt | llm | StrOutputParser()

# try to debug with streamlit interface
# st.set_page_config("Neo4j and LLM", page_icon=":movie_camera:")
# st.title("Movie Suggestion: Neo4j and LLM")

# # Handle any user input
# if prompt := st.chat_input("What is up?"):
#     # Display user message in chat message container
#     with st.chat_message('human'):
#         st.write(prompt)
    
#     # Generate a response
#     response = chain.invoke(prompt)
#     with st.chat_message('assistant'):
#         st.write(response.content)
#         st.write(retriever.get_relevant_documents(prompt))
