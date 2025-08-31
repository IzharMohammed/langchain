"""
Building a Chain Using LangGraph

This module demonstrates how to build a chain using LangGraph with:
1. Chat messages as graph state
2. Chat models in graph nodes
3. Binding tools to LLM
4. Executing tool calls in graph nodes
"""

from dotenv import load_dotenv
load_dotenv()

import os
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display


# =============================================================================
# 1. USING CHAT MESSAGES AS GRAPH STATE
# =============================================================================

class State(TypedDict):
    """
    State schema using chat messages with add_messages reducer.
    
    The add_messages reducer ensures messages are appended to the existing list
    rather than overriding the previous state.
    """
    messages: Annotated[list[AnyMessage], add_messages]


# =============================================================================
# 2. USING CHAT MODELS
# =============================================================================

# Initialize the chat model
llm = ChatGroq(model="qwen/qwen3-32b")
# llm=ChatGroq(model="Llama3-8b-8192")


# =============================================================================
# 3. BINDING TOOLS TO LLM
# =============================================================================

def add(a: int, b: int) -> int:
    """
    Add two integers.
    
    Args:
        a (int): First integer
        b (int): Second integer
    """
    return a + b

# Bind the tool to the LLM
llm_with_tools = llm.bind_tools([add])


# =============================================================================
# 4. GRAPH NODES
# =============================================================================

def llm_tool(state: State):
    """
    Node function that processes the state using the LLM with tools.
    
    Args:
        state (State): The current state containing messages
        
    Returns:
        dict: Updated state with LLM response
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# =============================================================================
# BUILDING THE GRAPH
# =============================================================================

def build_graph():
    """
    Build and compile the LangGraph with LLM and tool nodes.
    
    Returns:
        Compiled graph
    """
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("llm_tool", llm_tool)
    builder.add_node("tools", ToolNode([add]))
    
    # Add edges
    builder.add_edge(START, "llm_tool")
    builder.add_conditional_edges(
        "llm_tool",
        tools_condition  # Routes to tools if tool call, else to END
    )
    builder.add_edge("tools", END)
    
    return builder.compile()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Build the graph
    graph = build_graph()
    
    # Display the graph structure
    display(Image(graph.get_graph().draw_mermaid_png()))
    
    # Test with tool call
    print("=== Testing with tool call ===")
    messages = graph.invoke({"messages": "What is 2 plus 2"})
    for message in messages["messages"]:
        message.pretty_print()
    
    # Test without tool call
    print("\n=== Testing without tool call ===")
    messages = graph.invoke({"messages": "What is Machine Learning"})
    for message in messages["messages"]:
        message.pretty_print()


# messages=[AIMessage(content=f"Please tell me how can i help",name="LLM-Model")]
# messages.append(HumanMessage(content=f"I want to learn coding",name="Izhar"))
# messages.append(AIMessage(content=f"Which programming language you want to learn",name="LLM-Model"))
# messages.append(HumanMessage(content=f"I want to learn rust",name="Izhar"))

# for message in messages:
#     message.pretty_print()

# llm=ChatGroq(model="Llama3-8b-8192")
# # llm=ChatGroq(model="qwen/qwen3-32b")
# # llm=ChatGroq(model="openai/gpt-oss-120b")

# response = llm.invoke(messages)
# print("Response from LLM",response.content)
# print("Metadata",response.response_metadata)

