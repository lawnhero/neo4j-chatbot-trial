
from langchain_openai import ChatOpenAI

# Define the llm from OpenAI
gpt35 = ChatOpenAI(temperature=0
    # model=st.secrets["OPENAI_MODEL"],
)
# Define the embeddings from OpenAI
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


# Define the llm from Claude
import os
from langchain_anthropic import ChatAnthropic
# load the api key from the environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# initialize the chat client
llm_claude = ChatAnthropic(
        # model='claude-3-sonnet-20240229',
        model='claude-3-haiku-20240307',
        temperature=0,
        max_tokens=512,)
