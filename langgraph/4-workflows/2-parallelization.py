"""
Parallelization in LangGraph Tutorial
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# Load environment variables
load_dotenv()

# Set up API keys
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Uncomment if using OpenAI
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the language model
llm = ChatGroq(
    model="qwen/qwen3-32b",  # Changed from qwen/qwen3-32b
    temperature=0.1,
    timeout=30,  # Add timeout
    max_retries=2  # Add retry logic
)


# llm = ChatOpenAI(model="gpt-4o")  # Uncomment if using OpenAI

# Test the LLM connection
result = llm.invoke("Hello")
print("LLM connection test:", result.content)

# Define the state structure for parallel processing
class State(TypedDict):
    """State definition for parallel processing workflow"""
    topic: str
    characters: str
    settings: str
    premises: str
    story_intro: str

# Define node functions for parallel processing

def generate_characters(state: State):
    """Generate character descriptions - runs in parallel"""
    msg = llm.invoke(f"Create two character names and brief traits for a story about {state['topic']}")
    return {"characters": msg.content}

def generate_setting(state: State):
    """Generate a story setting - runs in parallel"""
    msg = llm.invoke(f"Describe a vivid setting for a story about {state['topic']}")
    return {"settings": msg.content}

def generate_premise(state: State):
    """Generate a story premise - runs in parallel"""
    msg = llm.invoke(f"Write a one-sentence plot premise for a story about {state['topic']}")
    return {"premises": msg.content}

def combine_elements(state: State):
    """Combine characters, setting, and premise into an intro - runs after parallel tasks complete"""
    msg = llm.invoke(
        f"Write a short story introduction using these elements:\n"
        f"Characters: {state['characters']}\n"
        f"Setting: {state['settings']}\n"
        f"Premise: {state['premises']}"
    )
    return {"story_intro": msg.content}

# Build the graph for parallel processing
graph = StateGraph(State)

# Add nodes to the graph
graph.add_node("character", generate_characters)
graph.add_node("setting", generate_setting)
graph.add_node("premise", generate_premise)
graph.add_node("combine", combine_elements)

# Define edges for parallel execution
# All three generation nodes start from START (parallel execution)
graph.add_edge(START, "character")
graph.add_edge(START, "setting")
graph.add_edge(START, "premise")

# All three generation nodes feed into the combine node
graph.add_edge("character", "combine")
graph.add_edge("setting", "combine")
graph.add_edge("premise", "combine")

# The combine node is the final step
graph.add_edge("combine", END)

# Compile the graph
compiled_graph = graph.compile()

# Visualize the graph (for Jupyter notebook)
# Note: This requires IPython and may not work in all Python environments
try:
    graph_image = compiled_graph.get_graph().draw_mermaid_png()
    display(Image(graph_image))
except Exception as e:
    print(f"Graph visualization not available: {e}")
    print("Graph structure:")
    print("START → character, setting, premise (parallel)")
    print("character, setting, premise → combine (merge)")
    print("combine → END")

# Run the graph with a sample topic
state = {"topic": "time travel"}
result = compiled_graph.invoke(state)

# Print the results
print("\n" + "="*50)
print("PARALLEL PROCESSING RESULTS")
print("="*50)

print("\nCharacters:")
print("-" * 30)
print(result.get("characters", "Not generated"))

print("\nSetting:")
print("-" * 30)
print(result.get("settings", "Not generated"))

print("\nPremise:")
print("-" * 30)
print(result.get("premises", "Not generated"))

print("\nStory Introduction:")
print("-" * 30)
print(result.get("story_intro", "Not generated"))

# Key Benefits Section
print("\n" + "="*50)
print("KEY BENEFITS OF PARALLELIZATION")
print("="*50)
