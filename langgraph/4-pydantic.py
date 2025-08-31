"""
Pydantic Data Validation in LangGraph

This module demonstrates how to use Pydantic BaseModel for state schema validation
in LangGraph StateGraph. Pydantic provides runtime validation with proper error messages.
"""

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, ValidationError


class State(BaseModel):
    """
    State schema defined using Pydantic BaseModel.
    
    Pydantic provides runtime validation with detailed error messages
    when the input data doesn't match the expected schema.
    
    Attributes:
        name (str): The name of the person (must be a string)
    """
    name: str


def example_node(state: State):
    """
    Node function that processes the state.
    
    Args:
        state (State): The validated state object
        
    Returns:
        dict: Updated state values
    """
    return {"name": "Hello " + state.name}


def build_and_test_graph():
    """
    Build the graph and test it with valid and invalid inputs.
    """
    # Build the graph
    builder = StateGraph(State)
    builder.add_node("example_node", example_node)
    builder.add_edge(START, "example_node")
    builder.add_edge("example_node", END)
    
    graph = builder.compile()
    
    # Test with valid input
    print("=== Testing with valid input ===")
    try:
        result = graph.invoke({"name": "Krish"})
        print(f"Success: {result}")
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Test with invalid input (integer instead of string)
    print("\n=== Testing with invalid input ===")
    try:
        result = graph.invoke({"name": 123})  # This should fail
        print(f"Unexpected success: {result}")
    except ValidationError as e:
        print(f"Expected validation error: {e}")
    
    return graph


if __name__ == "__main__":
    graph = build_and_test_graph()