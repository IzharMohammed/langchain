"""
=============================================================================
COMPREHENSIVE LANGGRAPH MULTI-TOOL CHATBOT
=============================================================================

This module demonstrates a complete implementation of a LangGraph-based chatbot
that integrates multiple tools and demonstrates key concepts:

CORE CONCEPTS COVERED:
1. State Management with Message Reducers
2. Tool Integration (Mathematical, Research, Web Search)
3. Conditional Routing and Graph Flow Control
4. LLM Integration with Tool Binding
5. Graph Compilation and Execution

ARCHITECTURE OVERVIEW:
- State: Manages conversation history using message reducers
- Nodes: Process state (LLM reasoning, tool execution)
- Edges: Define flow between nodes (conditional and direct)
- Tools: External capabilities (math, research, web search)

WORKFLOW:
User Input → LLM Processing → Tool Decision → Tool Execution → Response
"""

# =============================================================================
# ENVIRONMENT SETUP AND IMPORTS
# =============================================================================

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
# Set up API keys from environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Core LangChain imports for message handling and chat models
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_groq import ChatGroq

# Type system imports for state definition
from typing_extensions import TypedDict
from typing import Annotated

# LangGraph core components
from langgraph.graph.message import add_messages  # Message reducer function
from langgraph.graph import StateGraph, END, START  # Graph building components
from langgraph.prebuilt import ToolNode, tools_condition  # Pre-built utilities

# Tool imports for external capabilities
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_tavily import TavilySearch


# =============================================================================
# TOOL DEFINITIONS AND CONFIGURATION
# =============================================================================

def add(a: int, b: int) -> int:
    """
    Mathematical addition tool for basic arithmetic operations.
    
    This demonstrates how to create custom tools that the LLM can call
    when it needs to perform specific operations.
    
    Args:
        a: First integer to add
        b: Second integer to add
    """
    return a + b


# Configure ArXiv tool for academic paper searches
# ArXiv is a repository of academic papers in physics, mathematics, computer science, etc.
api_wrapper_arxiv = ArxivAPIWrapper(
    top_k_results=2,  # Return top 2 most relevant papers
    doc_content_chars_max=500  # Limit content to 500 characters for efficiency
)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# Configure Wikipedia tool for general knowledge queries
# Wikipedia provides broad encyclopedic information
api_wrapper_wiki = WikipediaAPIWrapper(
    top_k_results=1,  # Return top 1 most relevant article
    doc_content_chars_max=500  # Limit content to 500 characters
)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# Configure Tavily for real-time web search
# Tavily is optimized for LLM integration and provides current information
tavily = TavilySearch(
    max_results=5,  # Return up to 5 search results
    topic="general"  # General topic search (not specialized)
    # Additional configuration options available:
    # include_answer=True,          # Include direct answers when available
    # include_raw_content=True,     # Include full page content
    # include_images=True,          # Include image results
    # search_depth="advanced",      # Depth of search (basic/advanced)
    # time_range="week",           # Time constraint (day/week/month/year)
)

# Combine all tools into a single list for easy management
tools = [add, arxiv, wiki, tavily]


# =============================================================================
# LLM CONFIGURATION AND TOOL BINDING
# =============================================================================

# Initialize the Large Language Model
# Using Groq's hosted Qwen model for fast inference
llm = ChatGroq(model="qwen/qwen3-32b")

# Bind tools to the LLM
# This allows the LLM to understand what tools are available and when to use them
# The LLM will automatically generate tool calls when appropriate
llm_with_tools = llm.bind_tools(tools)

"""
TOOL BINDING EXPLANATION:
When tools are bound to an LLM:
1. The LLM receives function schemas describing each tool
2. Based on user input, the LLM decides which tools (if any) to call
3. The LLM generates structured tool calls with appropriate parameters
4. These tool calls are executed by the ToolNode
5. Results are fed back to the LLM for final response generation
"""


# =============================================================================
# STATE SCHEMA DEFINITION
# =============================================================================

class State(TypedDict):
    """
    Defines the graph state schema using TypedDict.
    
    STATE MANAGEMENT CONCEPTS:
    - State persists throughout the entire conversation flow
    - Messages are accumulated using the add_messages reducer
    - Each node can read from and write to the state
    - State updates are merged, not replaced (thanks to reducers)
    
    The add_messages reducer:
    - Automatically appends new messages to the existing list
    - Handles message deduplication and ordering
    - Maintains conversation context across multiple turns
    """
    messages: Annotated[list[AnyMessage], add_messages]


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def tool_calling_llm(state: State):
    """
    Primary LLM node that processes user input and decides on tool usage.
    
    NODE FUNCTION CONCEPTS:
    - Nodes are the processing units of the graph
    - They receive the current state as input
    - They return state updates (partial state objects)
    - State updates are automatically merged with existing state
    
    TOOL CALLING PROCESS:
    1. LLM analyzes the conversation history
    2. Determines if tools are needed to answer the query
    3. If tools needed: generates tool calls with parameters
    4. If no tools needed: generates direct response
    
    Args:
        state: Current conversation state containing message history
        
    Returns:
        dict: State update with LLM's response (may include tool calls)
    """
    # Invoke the LLM with the current message history
    # The LLM will analyze the messages and decide whether to use tools
    response = llm_with_tools.invoke(state["messages"])
    
    # Return state update - this will be merged with existing state
    return {"messages": [response]}


# =============================================================================
# GRAPH CONSTRUCTION AND COMPILATION
# =============================================================================

def build_graph():
    """
    Constructs and compiles the LangGraph workflow.
    
    GRAPH CONCEPTS:
    - Graphs define the flow of execution between nodes
    - Nodes perform processing (LLM calls, tool execution)
    - Edges define transitions between nodes
    - Conditional edges allow dynamic routing based on node outputs
    
    FLOW EXPLANATION:
    START → tool_calling_llm → [conditional routing] → tools (if needed) → END
                            → END (if no tools needed)
    
    Returns:
        Compiled and ready-to-execute graph
    """
    # Initialize the graph builder with our state schema
    builder = StateGraph(State)
    
    # Add the LLM node for processing user queries
    builder.add_node("tool_calling_llm", tool_calling_llm)
    
    # Add the tool execution node
    # ToolNode automatically handles execution of any tool calls
    # generated by the LLM node
    builder.add_node("tools", ToolNode(tools))
    
    # Define the graph flow:
    
    # 1. Entry point: Always start with the LLM node
    builder.add_edge(START, "tool_calling_llm")
    
    # 2. Conditional routing from LLM node
    # tools_condition is a pre-built function that:
    # - Checks if the LLM generated any tool calls
    # - Routes to "tools" node if tool calls exist
    # - Routes to END if no tool calls (direct response)
    builder.add_conditional_edges(
        "tool_calling_llm",  # Source node
        tools_condition      # Condition function for routing
    )
    
    # 3. After tool execution, always go to END
    # Tools provide their results back to the conversation
    builder.add_edge("tools", END)
    
    # Compile the graph for execution
    # Compilation optimizes the graph and prepares it for invocation
    return builder.compile()


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

def demonstrate_capabilities():
    """
    Demonstrates various capabilities of the multi-tool chatbot.
    
    TEST SCENARIOS:
    1. Mathematical computation (custom tool)
    2. General knowledge query (no tools needed)
    3. Academic research (ArXiv tool)
    4. Real-time information (Tavily web search)
    """
    
    print("Building and compiling the graph...")
    graph = build_graph()
    
    print("Graph successfully compiled!")
    print("=" * 80)
    
    # Test Case 1: Mathematical computation
    print("TEST 1: Mathematical Computation")
    print("Query: 'What is 2 plus 2?'")
    print("-" * 40)
    
    messages = graph.invoke({"messages": HumanMessage(content="What is 2 plus 2?")})
    for message in messages["messages"]:
        message.pretty_print()
    
    print("=" * 80)
    
    # Test Case 2: General knowledge (no tools needed)
    print("TEST 2: General Knowledge Query")
    print("Query: 'What is machine learning?'")
    print("-" * 40)
    
    messages = graph.invoke({"messages": HumanMessage(content="What is machine learning?")})
    for message in messages["messages"]:
        message.pretty_print()
    
    print("=" * 80)
    
    # Test Case 3: Academic research
    print("TEST 3: Academic Research")
    print("Query: 'Find a summary of the Attention Is All You Need paper and then add 100 and 50'")
    print("-" * 40)
    
    messages = graph.invoke({"messages": HumanMessage(
        content="Find a summary of the 'Attention Is All You Need' paper and then add 100 and 50"
    )})
    for message in messages["messages"]:
        message.pretty_print()
    
    print("=" * 80)
    
    # Test Case 4: Real-time web search
    print("TEST 4: Real-time Information")
    print("Query: 'What is the latest news about One Piece?'")
    print("-" * 40)
    
    messages = graph.invoke({"messages": HumanMessage(
        content="What is the latest news about One Piece?"
    )})
    for message in messages["messages"]:
        message.pretty_print()
    
    print("=" * 80)


# =============================================================================
# ADVANCED CONCEPTS EXPLANATION
# =============================================================================

"""
KEY LANGGRAPH CONCEPTS DEMONSTRATED:

1. STATE MANAGEMENT:
   - TypedDict defines the structure of data flowing through the graph
   - Annotated types with reducers handle state updates intelligently
   - add_messages reducer automatically manages conversation history

2. TOOL INTEGRATION:
   - Tools extend LLM capabilities beyond text generation
   - bind_tools() makes tools available to the LLM
   - ToolNode handles actual tool execution
   - Multiple tool types: custom functions, API wrappers, web search

3. CONDITIONAL ROUTING:
   - tools_condition automatically routes based on LLM output
   - If LLM generates tool calls → route to tools node
   - If LLM generates direct response → route to END
   - This enables dynamic, context-aware conversation flow

4. GRAPH COMPILATION:
   - Converts the graph definition into an executable workflow
   - Optimizes execution paths and validates graph structure
   - Results in a callable object that processes inputs through the defined flow

5. MESSAGE FLOW:
   - HumanMessage: User input
   - AIMessage: LLM responses (may contain tool calls)
   - ToolMessage: Results from tool execution
   - All messages accumulate in state for context

BENEFITS OF THIS ARCHITECTURE:
- Modular: Easy to add/remove tools
- Scalable: Can handle complex multi-step reasoning
- Flexible: Conditional routing adapts to different query types
- Maintainable: Clear separation of concerns
- Extensible: New nodes and edges can be added easily

WHEN TO USE DIFFERENT TOOLS:
- add(): Simple mathematical operations
- arxiv: Academic research, scientific papers
- wiki: General knowledge, encyclopedic information
- tavily: Current events, real-time information, web search

ERROR HANDLING:
- Tool errors are automatically handled by ToolNode
- Invalid tool calls result in error messages back to LLM
- LLM can retry or provide alternative responses
"""


# =============================================================================
# EXECUTION POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main execution block - runs when script is executed directly.
    
    This demonstrates the complete workflow and shows how different
    types of queries are handled by the system.
    """
    
    # print(__doc__)  # Print the module documentation
    
    try:
        demonstrate_capabilities()
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Please ensure all API keys are properly configured in your .env file:")
        print("- GROQ_API_KEY")
        print("- TAVILY_API_KEY")


# =============================================================================
# ADDITIONAL USAGE EXAMPLES AND EXTENSIONS
# =============================================================================

"""
EXTENDING THE CHATBOT:

1. Adding More Tools:
   def weather_tool(location: str) -> str:
       # Weather API integration
       pass
   
   tools.append(weather_tool)

2. Adding Custom Nodes:
   def preprocessing_node(state: State):
       # Custom preprocessing logic
       return {"messages": [...]}
   
   builder.add_node("preprocess", preprocessing_node)

3. Adding Memory/Persistence:
   class StateWithMemory(TypedDict):
       messages: Annotated[list[AnyMessage], add_messages]
       user_preferences: dict
       conversation_summary: str

4. Error Handling Node:
   def error_handler(state: State):
       # Handle errors gracefully
       return {"messages": [AIMessage(content="I encountered an error...")]}

5. Multi-turn Conversations:
   # The current implementation already supports multi-turn conversations
   # through the add_messages reducer and state persistence

PERFORMANCE CONSIDERATIONS:
- Tool selection affects response time
- ArXiv/Wikipedia: Medium latency (API calls)
- Tavily: Variable latency (depends on web search complexity)
- Custom tools: Low latency (local execution)

DEBUGGING TIPS:
- Use message.pretty_print() to inspect conversation flow
- Check state["messages"] to see full conversation history
- Monitor tool calls in LLM responses
- Verify API keys are correctly configured
"""