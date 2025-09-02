"""
Routing in LangGraph 
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Set up API keys
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Uncomment if using OpenAI
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the language model
llm = ChatGroq(
    model="qwen/qwen3-32b", 
    temperature=0.1,
    timeout=30,  # Add timeout
    max_retries=2  # Add retry logic
)

# llm = ChatOpenAI(model="gpt-4o")  # Uncomment if using OpenAI

# Test the LLM connection
result = llm.invoke("Hello")
print("LLM connection test:", result.content)

# Schema for structured output to use as routing logic
class Route(BaseModel):
    """Schema for routing decisions"""
    step: Literal["poem", "story", "joke"] = Field(
        description="The next step in the routing process"
    )

# Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)

# Define the state structure for routing
class State(TypedDict):
    """State definition for routing workflow"""
    input: str
    decision: str
    output: str

# Node functions for routing workflow

def llm_call_1(state: State):
    """Write a story based on input"""
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_2(state: State):
    """Write a joke based on input"""
    print("LLM call 2 is called - Writing a joke")
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_3(state: State):
    """Write a poem based on input"""
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_router(state: State):
    """Route the input to the appropriate node using structured output"""
    decision = router.invoke(
        [
            SystemMessage(
                content="Route the input to story, joke or poem based on the user's request"
            ),
            HumanMessage(content=state["input"])
        ]
    )
    return {"decision": decision.step}

# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    """Determine which node to visit next based on routing decision"""
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"

# Build the routing workflow
router_builder = StateGraph(State)

# Add nodes to the graph
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

# Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

# Compile the workflow
router_workflow = router_builder.compile()

# Visualize the graph (for Jupyter notebook)
# Note: This requires IPython and may not work in all Python environments
try:
    graph_image = router_workflow.get_graph().draw_mermaid_png()
    display(Image(graph_image))
except Exception as e:
    print(f"Graph visualization not available: {e}")
    print("Graph structure:")
    print("START → llm_call_router (decision node)")
    print("llm_call_router → llm_call_1, llm_call_2, or llm_call_3 (conditional routing)")
    print("llm_call_1, llm_call_2, llm_call_3 → END")

# Test the routing workflow
test_inputs = [
    "Write me a joke about sanji from one piece",
    "Tell me a story about consistency. make it short",
    "Compose a poem about luffy from one piece"
]

for i, test_input in enumerate(test_inputs, 1):
    print(f"\nTest {i}: {test_input}")
    state = router_workflow.invoke({"input": test_input})
    print(f"Decision: {state.get('decision', 'No decision')}")
    print(f"Output: {state.get('output', 'No output')}")

# Key Concepts Section
print("\n" + "="*50)
print("KEY CONCEPTS OF ROUTING IN LANGGRAPH")
print("="*50)

print("""
- Dynamic Flow: Unlike a linear sequence, routing lets the graph adapt to intermediate results.
- Condition Logic: You define rules (e.g., \"if this, go here; if that, go there\").
- Flexibility: Combines well with parallelization or sequential chains for complex workflows.
""")

# Benefits of Routing
print("\n" + "="*50)
print("BENEFITS OF ROUTING")
print("="*50)

print("""
- Adaptability: Workflows can respond to different input types or conditions
- Modularity: Each specialized node can be optimized for specific tasks
- Maintainability: Routing logic is centralized and easy to modify
- Scalability: Easy to add new routes and specialized processing nodes
""")

# Advanced Routing Techniques
def demonstrate_advanced_routing():
    """Show advanced routing patterns and techniques"""
    
    print("\n" + "="*50)
    print("ADVANCED ROUTING TECHNIQUES")
    print("="*50)
    
    # Multi-level routing
    print("\n1. Multi-level Routing:")
    print("   - Route based on multiple criteria or conditions")
    print("   - Create hierarchical decision trees")
    
    # Fallback routing
    print("\n2. Fallback Routing:")
    print("   - Default routes for unexpected inputs")
    print("   - Error handling and recovery paths")
    
    # Dynamic node creation
    print("\n3. Dynamic Node Routing:")
    print("   - Create nodes dynamically based on routing decisions")
    print("   - Adaptive workflow structures")

# Real-world use cases for routing
def routing_use_cases():
    """Show practical applications of routing in LangGraph"""
    
    print("\n" + "="*50)
    print("PRACTICAL ROUTING USE CASES")
    print("="*50)
    
    # Customer service routing
    print("\n1. Customer Service:")
    print("   - Route inquiries to appropriate departments")
    print("   - Escalate complex issues to human agents")
    
    # Content moderation
    print("\n2. Content Moderation:")
    print("   - Route content based on type (text, image, video)")
    print("   - Different moderation rules for different content types")
    
    # Data processing pipelines
    print("\n3. Data Processing:")
    print("   - Route data to different processing nodes based on format")
    print("   - Handle different data schemas and validation rules")

if __name__ == "__main__":
    print("Routing in LangGraph")
    print("Script converted from Jupyter notebook")
    
    # Run demonstrations
    demonstrate_advanced_routing()
    routing_use_cases()
    
    # Additional test cases
    additional_tests = [
        "Create a funny story about robots",
        "Make me laugh with a programming joke",
        "Write a romantic poem about technology"
    ]
    
    print("\n" + "="*50)
    print("ADDITIONAL ROUTING TESTS")
    print("="*50)
    
    for i, test_input in enumerate(additional_tests, 1):
        print(f"\nAdditional Test {i}: {test_input}")
        state = router_workflow.invoke({"input": test_input})
        print(f"Decision: {state.get('decision', 'No decision')}")
        print(f"Output: {state.get('output', 'No output')[:200]}...")  # Truncate long outputs