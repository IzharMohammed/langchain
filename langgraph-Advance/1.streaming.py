from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## Reducers
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:Annotated[list,add_messages]


llm_groq=ChatGroq(model="Llama3-8b-8192")
print(llm_groq)
rs = llm_groq.invoke("Hyy i am izhar, i like to solve real world problems using programming")
print(rs)

## creating nodes
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

def superbot(state:State):
    return {"messages":[llm_groq.invoke(state["messages"])]}

graph=StateGraph(State)

## node
graph.add_node("superBot",superbot)

## Edges
graph.add_edge(START,"superBot")
graph.add_edge("superBot",END) 

graph_builder = graph.compile(checkpointer=memory)

## Invocation

config={"configurable":{"thread_id":"1"}}

graph_builder.invoke({"messages":"Hyy, i am izhar and i like to solve real world problems"},config)

## Streaming the responses with stream method
# Create a thread
config={"configurable":{"thread_id":"2"}}

for chunk in graph_builder.stream({"messages":"Hyy, i am izhar and i like to solve real world problems"},config,stream_mode="updates"):
    print(chunk)

for chunk in graph_builder.stream({"messages":"Hyy, i am izhar and i like to solve real world problems"},config,stream_mode="values"):
    print(chunk)

