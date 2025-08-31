"""
State Schema With DataClasses in LangGraph

This module demonstrates different ways to define state schemas for LangGraph StateGraph:
1. Using TypedDict from typing_extensions
2. Using Python dataclasses

The state schema represents the structure and types of data that the graph will use.
All nodes are expected to communicate with that schema.

LangGraph offers flexibility in how you define your state schema, accommodating 
various Python types and validation approaches.
"""

from typing_extensions import TypedDict
from typing import Literal
from dataclasses import dataclass
import random
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


# =============================================================================
# TYPEDDICT APPROACH
# =============================================================================

class TypedDictState(TypedDict):
    """
    State schema defined using TypedDict.
    
    TypedDict allows specifying keys and their corresponding value types.
    These are type hints that can be used by static type checkers or IDEs
    to catch potential type-related errors, but they are not enforced at runtime.
    
    Attributes:
        name (str): The name of the person
        game (Literal["cricket", "badminton"]): The game they want to play
    """
    name: str
    game: Literal["cricket", "badminton"]


def play_game(state: TypedDictState):
    """Node function that processes the initial state"""
    print("---Play Game node has been called--")
    return {"name": state['name'] + " want to play "}


def cricket(state: TypedDictState):
    """Node function for cricket path"""
    print("-- Cricket node has been called--")
    return {"name": state["name"] + " cricket", "game": "cricket"}


def badminton(state: TypedDictState):
    """Node function for badminton path"""
    print("-- badminton node has been called--")
    return {"name": state["name"] + " badminton", "game": "badminton"}


def decide_play(state: TypedDictState) -> Literal["cricket", "badminton"]:
    """
    Conditional function that randomly decides which path to take
    
    Returns:
        Literal["cricket", "badminton"]: The chosen path
    """
    if random.random() < 0.5:
        return "cricket"
    else:
        return "badminton"


# Build the graph with TypedDict state
def build_typed_dict_graph():
    """Build and compile the graph using TypedDict state schema"""
    builder = StateGraph(TypedDictState)
    
    # Add nodes
    builder.add_node("playgame", play_game)
    builder.add_node("cricket", cricket)
    builder.add_node("badminton", badminton)
    
    # Define graph flow
    builder.add_edge(START, "playgame")
    builder.add_conditional_edges("playgame", decide_play)
    builder.add_edge("cricket", END)
    builder.add_edge("badminton", END)
    
    # Compile the graph
    graph = builder.compile()
    return graph


# =============================================================================
# DATACLASS APPROACH  
# =============================================================================

@dataclass
class DataClassState:
    """
    State schema defined using Python dataclasses.
    
    Dataclasses offer a concise syntax for creating classes that are primarily
    used to store data. They provide runtime validation and other utilities.
    
    Attributes:
        name (str): The name of the person
        game (Literal["badminton", "cricket"]): The game they want to play
    """
    name: str
    game: Literal["badminton", "cricket"]


def play_game_dc(state: DataClassState):
    """Node function that processes the initial state (dataclass version)"""
    print("---Play Game node has been called--")
    return {"name": state.name + " want to play "}


def cricket_dc(state: DataClassState):
    """Node function for cricket path (dataclass version)"""
    print("-- Cricket node has been called--")
    return {"name": state.name + " cricket", "game": "cricket"}


def badminton_dc(state: DataClassState):
    """Node function for badminton path (dataclass version)"""
    print("-- badminton node has been called--")
    return {"name": state.name + " badminton", "game": "badminton"}


def decide_play_dc(state: DataClassState) -> Literal["cricket", "badminton"]:
    """
    Conditional function that randomly decides which path to take (dataclass version)
    
    Returns:
        Literal["cricket", "badminton"]: The chosen path
    """
    if random.random() < 0.5:
        return "cricket"
    else:
        return "badminton"


# Build the graph with dataclass state
def build_dataclass_graph():
    """Build and compile the graph using dataclass state schema"""
    builder = StateGraph(DataClassState)
    
    # Add nodes
    builder.add_node("playgame", play_game_dc)
    builder.add_node("cricket", cricket_dc)
    builder.add_node("badminton", badminton_dc)
    
    # Define graph flow
    builder.add_edge(START, "playgame")
    builder.add_conditional_edges("playgame", decide_play_dc)
    builder.add_edge("cricket", END)
    builder.add_edge("badminton", END)
    
    # Compile the graph
    graph = builder.compile()
    return graph


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example usage with TypedDict
    print("=== TypedDict Graph ===")
    typed_dict_graph = build_typed_dict_graph()
    
    # Display the graph visualization
    display(Image(typed_dict_graph.get_graph().draw_mermaid_png()))
    
    # Invoke the graph with sample input
    result = typed_dict_graph.invoke({"name": "Krish", "game": "cricket"})
    print(f"Result: {result}")
    
    # Example usage with dataclass
    print("\n=== Dataclass Graph ===")
    dataclass_graph = build_dataclass_graph()
    
    # Display the graph visualization
    display(Image(dataclass_graph.get_graph().draw_mermaid_png()))
    
    # Invoke the graph with sample input
    result = dataclass_graph.invoke(DataClassState(name="KRish", game="cricket"))
    print(f"Result: {result}")