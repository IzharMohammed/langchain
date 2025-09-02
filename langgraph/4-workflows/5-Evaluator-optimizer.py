"""
Evaluator-Optimizer Pattern in LangGraph 
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Annotated, List
import operator
from typing_extensions import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
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
    model="qwen/qwen3-32b", 
    temperature=0.1,
    timeout=10,  # Add timeout
    max_retries=2  # Add retry logic
)
# llm = ChatOpenAI(model="gpt-4o")  # Uncomment if using OpenAI

# Test the LLM connection
result = llm.invoke("Hello")
print("LLM connection test:", result.content)

# Graph state for evaluator-optimizer workflow
class State(TypedDict):
    """State definition for evaluator-optimizer workflow"""
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    """Schema for joke evaluation feedback"""
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )

# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback)

# Node functions for evaluator-optimizer workflow

def llm_call_generator(state: State):
    """LLM generates a joke based on topic and feedback"""
    
    # Generate joke with optional feedback incorporation
    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    """LLM evaluates the joke and provides structured feedback"""
    
    # Evaluate the joke using structured output
    grade = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

# Conditional edge function to route back to joke generator or end based upon feedback
def route_joke(state: State):
    """Route back to joke generator or end based upon evaluation feedback"""
    
    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"

# Build the evaluator-optimizer workflow
optimizer_builder = StateGraph(State)

# Add nodes to the graph
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {  # Name returned by route_joke : Name of next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# Compile the workflow
optimizer_workflow = optimizer_builder.compile()

# Visualize the graph (for Jupyter notebook)
# Note: This requires IPython and may not work in all Python environments
try:
    graph_image = optimizer_workflow.get_graph().draw_mermaid_png()
    display(Image(graph_image))
except Exception as e:
    print(f"Graph visualization not available: {e}")
    print("Graph structure:")
    print("START → llm_call_generator (joke generation)")
    print("llm_call_generator → llm_call_evaluator (joke evaluation)")
    print("llm_call_evaluator → END (if accepted) or llm_call_generator (if rejected with feedback)")

# Test the evaluator-optimizer workflow
test_topics = [
    "Agentic AI system",
    "programmers and coffee",
    "artificial intelligence"
]

for i, topic in enumerate(test_topics, 1):
    print(f"\nTest {i}: {topic}")
    state = optimizer_workflow.invoke({"topic": topic})
    
    print(f"Final joke: {state.get('joke', 'No joke generated')}")
    print(f"Evaluation: {state.get('funny_or_not', 'No evaluation')}")
    if state.get("feedback"):
        print(f"Feedback: {state.get('feedback')}")
    print(f"Iterations: {len(state.get('feedback', '').split('Iteration')) if state.get('feedback') else 1}")

# Key Concepts Section
print("\n" + "="*50)
print("EVALUATOR-OPTIMIZER PATTERN IN LANGGRAPH")
print("="*50)

print("""
- Iterative Refinement: One LLM generates content while another provides evaluation and feedback
- Feedback Loop: Content is repeatedly refined based on evaluation criteria
- Quality Control: Ensures output meets specific quality standards
- Adaptive Improvement: Content evolves through multiple iterations
""")

# When to Use Evaluator-Optimizer Pattern
print("\n" + "="*50)
print("WHEN TO USE EVALUATOR-OPTIMIZER PATTERN")
print("="*50)

print("""
- Clear evaluation criteria exist for the content
- Iterative refinement provides measurable value improvement
- Human feedback can demonstrably improve LLM responses
- The LLM can provide meaningful feedback on its own outputs
- Analogous to human iterative writing/editing processes
""")

# Benefits of Evaluator-Optimizer Pattern
print("\n" + "="*50)
print("BENEFITS OF EVALUATOR-OPTIMIZER PATTERN")
print("="*50)

print("""
- Quality Assurance: Systematic evaluation ensures high-quality output
- Continuous Improvement: Multiple iterations refine content progressively
- Objective Standards: Consistent evaluation against predefined criteria
- Automated Refinement: Reduces need for human intervention in editing
- Adaptive Learning: System can learn from feedback over time
""")

# Advanced Evaluator-Optimizer Techniques
def demonstrate_advanced_techniques():
    """Show advanced evaluator-optimizer patterns"""
    
    print("\n" + "="*50)
    print("ADVANCED EVALUATOR-OPTIMIZER TECHNIQUES")
    print("="*50)
    
    # Multi-criteria evaluation
    print("\n1. Multi-criteria Evaluation:")
    print("   - Evaluate against multiple quality dimensions")
    print("   - Weighted scoring for different criteria")
    
    # Progressive refinement
    print("\n2. Progressive Refinement:")
    print("   - Different evaluation criteria at each iteration")
    print("   - Focus on different aspects in sequence")
    
    # Confidence scoring
    print("\n3. Confidence Scoring:")
    print("   - LLM provides confidence scores for evaluations")
    print("   - Automatic termination when confidence threshold met")
    
    # Ensemble evaluation
    print("\n4. Ensemble Evaluation:")
    print("   - Multiple evaluators with different perspectives")
    print("   - Consensus-based decision making")

# Real-world use cases
def evaluator_optimizer_use_cases():
    """Show practical applications of evaluator-optimizer pattern"""
    
    print("\n" + "="*50)
    print("PRACTICAL EVALUATOR-OPTIMIZER USE CASES")
    print("="*50)
    
    # Content creation
    print("\n1. Content Creation:")
    print("   - Joke generation with humor evaluation")
    print("   - Story writing with coherence and engagement checks")
    
    # Code generation
    print("\n2. Code Development:")
    print("   - Code generation with quality and efficiency evaluation")
    print("   - Bug detection and correction feedback loops")
    
    # Creative writing
    print("\n3. Creative Writing:")
    print("   - Poetry generation with rhyme and meter evaluation")
    print("   - Marketing copy with tone and effectiveness assessment")
    
    # Educational content
    print("\n4. Educational Content:")
    print("   - Quiz question generation with difficulty evaluation")
    print("   - Explanation generation with clarity assessment")

if __name__ == "__main__":
    print("Evaluator-Optimizer Pattern in LangGraph")
    print("Script converted from Jupyter notebook")
    
    # Run demonstrations
    demonstrate_advanced_techniques()
    evaluator_optimizer_use_cases()
    
    # Additional test cases
    additional_tests = [
        "quantum computing",
        "machine learning algorithms",
        "data scientists"
    ]
    
    print("\n" + "="*50)
    print("ADDITIONAL EVALUATOR-OPTIMIZER TESTS")
    print("="*50)
    
    for i, topic in enumerate(additional_tests, 1):
        print(f"\nAdditional Test {i}: {topic}")
        state = optimizer_workflow.invoke({"topic": topic})
        
        print(f"Final joke: {state.get('joke', 'No joke generated')}")
        print(f"Evaluation: {state.get('funny_or_not', 'No evaluation')}")
        if state.get("feedback"):
            print(f"Feedback: {state.get('feedback')}")
        print(f"Iterations: {len(state.get('feedback', '').split('Iteration')) if state.get('feedback') else 1}")