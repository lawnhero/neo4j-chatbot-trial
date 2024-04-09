from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Include the LLM from a previous lesson
from llm import gpt35 as llm, llm_claude

from langchain.tools import Tool
# from tools.vector import kg_qa, generate_vector_search_response , chain as vector_chain
from tools.cypher import cypher_qa
# Define the tools
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
    ),
    # Tool.from_function(
    #     name="Vector Search Index",  # (1)
    #     description="Provides information about movie plots using Vector Search", # (2)
    #     func = kg_qa.invoke, # (3)
    #     return_direct=False
    # ), 
    Tool.from_function(
        name="Graph Cypher QA Chain",  # (1)
        description="Provides information about Movies, their Actors, Directors and User reviews", # (2)
        func = cypher_qa.invoke, # (3)
        # return_direct=True
    ),
]
# Define the memory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)
# Define agent prompt

from langchain_core.prompts import PromptTemplate
agent_prompt = PromptTemplate.from_template("""
You are a movie expert dedicated to providing detailed information about movies, including data on actors and directors. Your goal is to assist users by offering comprehensive insights based on the queries received.

Guidelines for Interactions:
- Focus solely on inquiries related to movies, actors, or directors. Avoid addressing questions outside these topics.
- Rely exclusively on the information derived from the context or generated through the use of specified tools. Do not utilize pre-existing knowledge not contained within the current interaction.

Tools at Your Disposal:
<TOOLS>
{tools}
</TOOLS>

When deciding whether a tool is necessary, strictly follow this structured approach:
```
Thought: Do I need to use a tool? Yes
Action:  action to take, must be one of [{tool_names}]
Action Input: all necessary inputs to the action 
Observation: the result of the action
```
                                            
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [the final response]
```
Additional format for the final response:
- Finalize your response directly addressing the user's query. 
- If the query is directly answerable with tool output or previous context, structure the response to immediately address the query.
                                            
Begin!
Previous conversation history:<history>{chat_history}</history>
New input: <input>{input}</input>
{agent_scratchpad}
""")

# Pull the agent prmopt from the hub
# agent_prompt = hub.pull("hwchase17/react-chat")

agent_prompt2 = PromptTemplate.from_template(
    """
You are a movie expert dedicated to providing detailed information about movies, including data on actors and directors. Your goal is to assist users by offering comprehensive insights based on the queries received.

Guidelines for Interactions:
- Focus solely on inquiries related to movies, actors, or directors. Avoid addressing questions outside these topics.
- Rely exclusively on the information derived from the context or generated through the use of specified tools. Do not utilize pre-existing knowledge not contained within the current interaction.

Tools at Your Disposal:
<TOOLS>{tools}</TOOLS>

Your toolkit enables you to fetch and process data relevant to user queries. When deciding whether a tool is necessary, follow this structured approach:

<TOOL_Logic>
```
Thought: Do I need to use a tool? Yes/No
If Yes:
  Action: action to take, must be within [{tool_names}]
  Action Input: input for the action
  Observation: the output or response from the action

If No:
  Final Answer: [your response based on the information available]
```
</TOOL_Logic>
Please adhere to this protocol for all interactions, ensuring a systematic and transparent process.

Begin each new interaction with a recap of previous conversation history, if any, and the new user input:
Previous conversation history:<history>{chat_history}</history>
New input: <input>{input}</input>
<scratchpad>{agent_scratchpad}</scratchpad>
"""
)



# Create the agent
agent = create_react_agent(llm, tools, agent_prompt)
# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    # return_intermediate_steps=True,
    )

# First define a functiont to convert the message
def memory2str(memory: ConversationBufferWindowMemory):
    messages = memory.chat_memory.messages
    memory_list = [
        f"Human: {mem.content}" if isinstance(mem, HumanMessage) \
        else f"AI: {mem.content}" for mem in messages
        ]
    memory_str = "\n".join(memory_list)
    return memory_str

def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent_executor.invoke({"input": prompt,
                                      "chat_history": memory2str(memory)})

    return response['output']


"""
Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation
"""