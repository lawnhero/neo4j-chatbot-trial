# tag::llm[]
import streamlit as st
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=st.secrets["OPENAI_MODEL"],
)
# end::llm[]

# tag::embedding[]
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
# end::embedding[]
