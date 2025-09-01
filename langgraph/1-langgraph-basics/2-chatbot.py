# Import required modules
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Initialize Groq API key and LLM model
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="Llama3-8b-8192")

# Test the LLM with a sample prompt
response = llm.invoke("Can you tell me about yourself?")
print("LLM Response Content:", response.content)
print("Token Usage Metadata:", response.response_metadata)

"""
State Definition:
The State is a TypedDict that defines the structure of data flowing through the graph.
It contains a list of messages annotated with add_messages, which helps manage conversation history.
"""
class State(TypedDict):
    messages: Annotated[list, add_messages]

"""
Node Function:
The superbot function is our main processing node that:
1. Takes the current state containing messages
2. Invokes the LLM with those messages
3. Returns a new state with the LLM's response added
"""
def superbot(state: State):
    # The LLM processes the messages and returns a response
    llm_response = llm.invoke(state["messages"])
    # We return a new state containing the response
    return {"messages": [llm_response]}

"""
Graph Construction:
We create a StateGraph with our defined State structure.
The graph will manage the flow of messages through our system.
"""
graph = StateGraph(State)

# Add our superbot function as a node in the graph
graph.add_node("superBot", superbot)

"""
Define Graph Edges:
- START -> superBot: Initial input goes to our processing node
- superBot -> END: After processing, the flow terminates
"""
graph.add_edge(START, "superBot")
graph.add_edge("superBot", END)

# Compile the graph to finalize and validate the structure
graph_builder = graph.compile()

"""
Graph Visualization:
Attempt to display the graph using Mermaid diagram in Jupyter.
If IPython isn't available, save the visualization to a file.
"""
try:
    from IPython.display import Image, display
    display(Image(graph_builder.get_graph().draw_mermaid_png()))
except ImportError:
    print("IPython not available - graph visualization requires Jupyter notebook")
    # Save the graph visualization as a PNG file
    graph_builder.get_graph().draw_mermaid_png().save("graph.png")
    print("Graph visualization saved as graph.png")

"""
Graph Invocation:
Run the graph with sample input.
Note: There's a typo in "messgaes" - should be "messages"
"""
result = graph_builder.invoke({"messages": "Hi my name is izhar"})
print("Graph Invocation Result:", result)

"""
Streaming Responses:
Process input in streaming mode to handle large responses efficiently.
The stream_mode="values" returns the state values at each step.
"""
print("\nStreaming Responses:")
for event in graph_builder.stream({"messages": "Hello, My name is izhar"}, stream_mode="values"):
    # Each event represents the state at that point in the graph execution
    print("Stream Event:", event)