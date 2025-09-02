"""
Orchestrator-Worker Pattern in LangGraph 
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
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display, Markdown

# Load environment variables
load_dotenv()

# Set up API keys
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Uncomment if using OpenAI
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the language model
llm = ChatGroq(model="qwen-qwq-32b")
# llm = ChatOpenAI(model="gpt-4o")  # Uncomment if using OpenAI

# Test the LLM connection
result = llm.invoke("Hello")
print("LLM connection test:", result.content)

# Schema for structured output to use in planning
class Section(BaseModel):
    """Schema for report sections"""
    name: str = Field(description="Name for this section of the report")
    description: str = Field(description="Brief Overview of the main topics and concepts of the section")

class Sections(BaseModel):
    """Schema for multiple sections"""
    sections: List[Section] = Field(
        description="Sections of the report"
    )

# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Sections)

# Graph state for orchestrator
class State(TypedDict):
    """State definition for orchestrator workflow"""
    topic: str  # Report topic
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_report: str  # Final report

# Worker state
class WorkerState(TypedDict):
    """State definition for worker nodes"""
    section: Section
    completed_sections: Annotated[list, operator.add]

# Node functions for orchestrator-worker workflow

def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""
    
    # Generate report sections using structured output
    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report."),
            HumanMessage(content=f"Here is the report topic: {state['topic']}"),
        ]
    )

    print("Report Sections:", report_sections)

    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    """Worker writes a section of the report"""
    
    # Generate section content
    section = llm.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
            ),
            HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
            ),
        ]
    )

    # Write the updated section to completed sections
    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    """Synthesize full report from sections"""
    
    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed sections to string with separators
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}

# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan using Send API"""
    
    # Kick off section writing in parallel via Send() API
    # This dynamically creates worker nodes for each section
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

# Build the orchestrator-worker workflow
orchestrator_worker_builder = StateGraph(State)

# Add nodes to the graph
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# Compile the workflow
orchestrator_worker = orchestrator_worker_builder.compile()

# Visualize the graph (for Jupyter notebook)
# Note: This requires IPython and may not work in all Python environments
try:
    graph_image = orchestrator_worker.get_graph().draw_mermaid_png()
    display(Image(graph_image))
except Exception as e:
    print(f"Graph visualization not available: {e}")
    print("Graph structure:")
    print("START → orchestrator (plan generation)")
    print("orchestrator → llm_call (parallel worker creation via Send API)")
    print("llm_call → synthesizer (result aggregation)")
    print("synthesizer → END")

# Test the orchestrator-worker workflow
test_topics = [
    "Create a report on Agentic AI RAGs",
    "Write about the future of renewable energy",
    "Analyze the impact of blockchain technology"
]

for i, topic in enumerate(test_topics, 1):
    print(f"\nTest {i}: {topic}")
    state = orchestrator_worker.invoke({"topic": topic})
    
    print(f"Generated {len(state.get('sections', []))} sections")
    print(f"Final report length: {len(state.get('final_report', ''))} characters")
    
    # Display the final report for the first test
    if i == 1:
        display(Markdown(state["final_report"]))

# Key Concepts Section
print("\n" + "="*50)
print("ORCHESTRATOR-WORKER PATTERN IN LANGGRAPH")
print("="*50)

print("""
- Dynamic Task Breakdown: A central LLM (orchestrator) dynamically breaks down complex tasks
- Parallel Execution: Delegates subtasks to worker LLMs that execute in parallel
- Result Synthesis: Combines worker outputs into a final coherent result
- Flexibility: Subtasks aren't pre-defined but determined based on specific input
""")

# When to Use Orchestrator-Worker Pattern
print("\n" + "="*50)
print("WHEN TO USE ORCHESTRATOR-WORKER PATTERN")
print("="*50)

print("""
- Complex tasks where subtasks can't be predicted in advance
- Tasks requiring dynamic planning and execution
- Content generation with multiple interdependent sections
- Code generation where file changes depend on the specific task
- Any workflow where the number and nature of subtasks vary by input
""")

# Benefits of Orchestrator-Worker Pattern
print("\n" + "="*50)
print("BENEFITS OF ORCHESTRATOR-WORKER PATTERN")
print("="*50)

print("""
- Adaptability: Handles varying task complexity dynamically
- Parallelism: Workers execute simultaneously for efficiency
- Specialization: Each worker can be optimized for specific subtasks
- Scalability: Easy to add more workers for complex tasks
- Maintainability: Clear separation between planning and execution
""")

# Advanced Orchestrator-Worker Techniques
def demonstrate_advanced_techniques():
    """Show advanced orchestrator-worker patterns"""
    
    print("\n" + "="*50)
    print("ADVANCED ORCHESTRATOR-WORKER TECHNIQUES")
    print("="*50)
    
    # Hierarchical orchestration
    print("\n1. Hierarchical Orchestration:")
    print("   - Multiple levels of orchestration for complex workflows")
    print("   - Workers can themselves become orchestrators")
    
    # Dynamic worker configuration
    print("\n2. Dynamic Worker Configuration:")
    print("   - Workers with different capabilities and specializations")
    print("   - Orchestrator selects appropriate workers for each task")
    
    # Result validation and iteration
    print("\n3. Result Validation and Iteration:")
    print("   - Orchestrator validates worker outputs")
    print("   - Failed tasks can be reassigned or reworked")
    
    # Resource-aware orchestration
    print("\n4. Resource-Aware Orchestration:")
    print("   - Orchestrator considers computational resources")
    print("   - Dynamic load balancing between workers")

# Real-world use cases
def orchestrator_worker_use_cases():
    """Show practical applications of orchestrator-worker pattern"""
    
    print("\n" + "="*50)
    print("PRACTICAL ORCHESTRATOR-WORKER USE CASES")
    print("="*50)
    
    # Content generation
    print("\n1. Content Generation:")
    print("   - Research paper writing with multiple sections")
    print("   - Blog post creation with different content types")
    
    # Code generation and refactoring
    print("\n2. Code Development:")
    print("   - Multi-file code generation projects")
    print("   - Code refactoring across multiple modules")
    
    # Data analysis pipelines
    print("\n3. Data Analysis:")
    print("   - Complex data processing workflows")
    print("   - Multi-step analytical reporting")
    
    # Customer service automation
    print("\n4. Customer Service:")
    print("   - Multi-step customer inquiry resolution")
    print("   - Dynamic response generation based on context")

if __name__ == "__main__":
    print("Orchestrator-Worker Pattern in LangGraph")
    print("Script converted from Jupyter notebook")
    
    # Run demonstrations
    demonstrate_advanced_techniques()
    orchestrator_worker_use_cases()
    
    # Additional test cases
    additional_tests = [
        "Comprehensive analysis of quantum computing applications",
        "Detailed guide on implementing machine learning pipelines",
        "Market research report on electric vehicle adoption trends"
    ]
    
    print("\n" + "="*50)
    print("ADDITIONAL ORCHESTRATOR-WORKER TESTS")
    print("="*50)
    
    for i, topic in enumerate(additional_tests, 1):
        print(f"\nAdditional Test {i}: {topic}")
        state = orchestrator_worker.invoke({"topic": topic})
        
        print(f"Generated {len(state.get('sections', []))} sections")
        print(f"Final report length: {len(state.get('final_report', ''))} characters")
        
        # Display a preview of the first section
        if state.get("completed_sections"):
            first_section_preview = state["completed_sections"][0][:200] + "..." if len(state["completed_sections"][0]) > 200 else state["completed_sections"][0]
            print(f"First section preview: {first_section_preview}")