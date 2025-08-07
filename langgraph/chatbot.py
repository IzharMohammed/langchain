from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END

## Reducers
from typing import Annotated
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="Llama3-8b-8192")

response = llm.invoke("Can u tell me about yourself ??")
print("content",response.content)
print("usage",response.response_metadata)

class State(TypedDict):
    messages:Annotated[list,add_messages]

def superbot(state:State):
    return {"messages":[llm.invoke(state["messages"])]}
graph=StateGraph(State)

## node
graph.add_node("superBot",superbot)

## Edges
graph.add_edge(START,"superBot")
graph.add_edge("superBot",END)

graph_builder = graph.compile()

# Display graph (only works in Jupyter/IPython environments)
try:
    from IPython.display import Image, display
    display(Image(graph_builder.get_graph().draw_mermaid_png()))
except ImportError:
    print("IPython not available - graph visualization requires Jupyter notebook")
    # Alternative: save the graph image to a file
    graph_builder.get_graph().draw_mermaid_png().save("graph.png")
    print("Graph saved as graph.png")

graph_builder.invoke({"messgaes":"Hy my name is izhar"})

## Streaming the responses
for event in  graph_builder.stream({"messages":"Hello, My name is izhar"},stream_mode="values"):
    print(event)