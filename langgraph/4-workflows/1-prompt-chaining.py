"""
Prompt Chaining with LangGraph Tutorial
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from typing import List
import time

# Load environment variables
load_dotenv()

# Set up API keys
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Uncomment if using OpenAI
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the language model
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=1.1,
    timeout=30,  # Add timeout
    max_retries=2  # Add retry logic
)

# llm = ChatOpenAI(model="gpt-4o")  # Uncomment if using OpenAI

# Test the LLM connection
result = llm.invoke("Hello")
print("LLM connection test:", result.content)

# Define the state structure for the graph
class State(TypedDict):
    """State definition for the prompt chaining workflow"""
    topic: str
    story: str
    improved_story: str
    final_story: str

# Define node functions for the graph

def generate_story(state: State):
    """Generate a one-sentence story premise based on the topic"""
    msg = llm.invoke(f"Write a one sentence story premise about {state['topic']}")
    return {"story": msg.content}

def check_conflict(state: State):
    """Check if the generated story meets quality criteria"""
    if "?" in state["story"] or "!" in state["story"]:
        return "Fail"
    return "Pass"

def improved_story(state: State):
    """Enhance the story premise with vivid details"""
    msg = llm.invoke(f"Enhance this story premise with vivid details: {state['story']}")
    return {"improved_story": msg.content}

def polish_story(state: State):
    """Add an unexpected twist to the improved story"""
    msg = llm.invoke(f"Add an unexpected twist to this story premise: {state['improved_story']}")
    return {"final_story": msg.content}

# Build the graph
graph = StateGraph(State)

# Add nodes to the graph
graph.add_node("generate", generate_story)
graph.add_node("improve", improved_story)
graph.add_node("polish", polish_story)

# Define the edges between nodes
graph.add_edge(START, "generate")
graph.add_conditional_edges("generate", check_conflict, {"Pass": "improve", "Fail": "generate"})
graph.add_edge("improve", "polish")
graph.add_edge("polish", END)

# Compile the graph
compiled_graph = graph.compile()
print("compiled_graph",compiled_graph)


# Run the graph with a sample topic
state = {"topic": "Agentic AI Systems"}
result = compiled_graph.invoke(state)

# Print the results
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

print("\nImproved Story")
print("-" * 30)
print(result["improved_story"])

print("\nPolished Story")
print("-" * 30)
print(result["final_story"])

# Additional functions that were mentioned but not implemented in the notebook

# def enhanced_check_conflict(state: State):
#     """Enhanced validation with multiple checks"""
#     story = state["story"]
#     # Multiple validation checks
#     if "?" in story or "!" in story:
#         return "Fail"
#     if len(story.split()) < 5:  # Minimum word count
#         return "Fail"
#     if len(story) > 200:  # Maximum length
#         return "Fail"
#     return "Pass"

# class EnhancedState(TypedDict):
#     """Enhanced state with additional tracking fields"""
#     topic: str
#     story: str
#     improved_story: str
#     final_story: str
#     iteration_count: int
#     validation_errors: List[str]
#     generation_history: List[dict]

# def track_metrics(state: State):
#     """Track performance metrics for the workflow"""
#     # Track token usage, time, success rates
#     metrics = {
#         "tokens_used": len(state.get("story", "")) + len(state.get("improved_story", "")) + len(state.get("final_story", "")),
#         "processing_time": time.time() - state.get("start_time", time.time()),
#         "iterations": state.get("iteration_count", 0)
#     }
#     return {"metrics": metrics}

# def request_human_review(state: State):
#     """Node for human intervention when needed"""
#     # Send for human approval
#     if state.get("needs_human_review"):
#         return "awaiting_review"
#     return "auto_approve"

# # Example of how you might set up parallel processing nodes
# def add_characters(state: State):
#     """Add character development to the story"""
#     msg = llm.invoke(f"Add detailed characters to this story: {state['improved_story']}")
#     return {"improved_story": msg.content}

# def add_setting(state: State):
#     """Add detailed setting to the story"""
#     msg = llm.invoke(f"Add detailed setting to this story: {state['improved_story']}")
#     return {"improved_story": msg.content}

# def add_conflict(state: State):
#     """Add conflict to the story"""
#     msg = llm.invoke(f"Add conflict to this story: {state['improved_story']}")
#     return {"improved_story": msg.content}

# # Example workflow patterns mentioned in the notebook

# def customer_support_workflow():
#     """Example: Customer support automation workflow"""
#     # This would be implemented similarly to the story generation workflow
#     pass

# def code_generation_workflow():
#     """Example: Code generation workflow"""
#     # This would be implemented similarly to the story generation workflow
#     pass

# def research_assistant_workflow():
#     """Example: Research assistant workflow"""
#     # This would be implemented similarly to the story generation workflow
#     pass

# if __name__ == "__main__":
#     print("Prompt Chaining with LangGraph")
#     print("Script converted from Jupyter notebook")
    
#     # You can add additional test cases here
#     test_topics = ["Space Exploration", "Time Travel", "Artificial Intelligence"]
    
#     for topic in test_topics:
#         print(f"\nTesting with topic: {topic}")
#         state = {"topic": topic}
#         result = compiled_graph.invoke(state)
#         print(f"Generated story length: {len(result.get('final_story', ''))} characters")