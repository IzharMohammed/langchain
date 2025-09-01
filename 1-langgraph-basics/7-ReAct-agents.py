"""
=============================================================================
REACT AGENT ARCHITECTURE IMPLEMENTATION
=============================================================================

This module demonstrates a complete ReAct (Reasoning and Acting) agent using LangGraph.
ReAct is a paradigm that combines reasoning traces and task-specific actions in large
language models to solve complex problems through iterative cycles.

REACT PARADIGM OVERVIEW:
ReAct agents follow a structured pattern of:
1. REASONING: The agent thinks about the current situation and plans next steps
2. ACTION: The agent takes a specific action (uses a tool or provides answer)
3. OBSERVATION: The agent processes the results from the action
4. REPEAT: Continue the cycle until the task is complete

KEY DIFFERENCES FROM TRADITIONAL AGENTS:
- Traditional: Input → LLM → Tools → Output (linear)
- ReAct: Input → LLM → Tools → LLM → Tools → ... → Output (iterative)

CORE REACT CHARACTERISTICS:
1. ITERATIVE REASONING: Agent can reason multiple times during task execution
2. TOOL INTEGRATION: Seamlessly incorporates external tool results into reasoning
3. SELF-CORRECTION: Agent can adjust approach based on intermediate results
4. TRANSPARENT THINKING: Reasoning process is visible and traceable
5. ADAPTIVE WORKFLOW: Flow adapts based on task complexity and tool outcomes

ARCHITECTURE BENEFITS:
- Handles complex multi-step problems
- Self-correcting when tools provide unexpected results
- Transparent decision-making process
- Robust error handling and recovery
- Natural conversation flow with tool integration
"""

# =============================================================================
# ENVIRONMENT SETUP AND IMPORTS
# =============================================================================

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
# Set up API keys from environment variables for comprehensive tool access
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable LangSmith tracing for ReAct debugging
os.environ["LANGCHAIN_PROJECT"] = "ReAct-agent"  # Project name for organized tracing

# Core LangChain imports for ReAct message handling and chat models
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_groq import ChatGroq

# Type system imports for ReAct state definition
from typing_extensions import TypedDict
from typing import Annotated

# LangGraph core components for ReAct graph construction
from langgraph.graph.message import add_messages  # Message reducer for conversation continuity
from langgraph.graph import StateGraph, END, START  # Graph building components
from langgraph.prebuilt import ToolNode, tools_condition  # Pre-built utilities for tool routing

# External tool imports for ReAct agent capabilities
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_tavily import TavilySearch

# Memory saver
from langgraph.checkpoint.memory import MemorySaver

# =============================================================================
# REACT AGENT TOOL DEFINITIONS
# =============================================================================

def add(a: int, b: int) -> int:
    """
    Mathematical addition tool for ReAct agent arithmetic operations.
    
    REACT TOOL INTEGRATION:
    This tool demonstrates how simple computational tasks are integrated into
    the ReAct reasoning cycle. The agent can:
    1. Reason about needing mathematical calculation
    2. Act by calling this tool with appropriate parameters
    3. Observe the result and incorporate it into further reasoning
    4. Continue reasoning or provide final answer
    
    Args:
        a: First integer to add
        b: Second integer to add
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """
    Mathematical multiplication tool extending ReAct agent capabilities.
    
    REACT MULTI-TOOL REASONING:
    This tool works in combination with other mathematical tools, allowing
    the ReAct agent to perform complex calculations through sequential
    reasoning and action cycles:
    
    Example ReAct Flow:
    1. REASON: "I need to multiply 5 by 3, then add 10"
    2. ACT: Call multiply(5, 3)
    3. OBSERVE: Result is 15
    4. REASON: "Now I need to add 10 to 15"
    5. ACT: Call add(15, 10)
    6. OBSERVE: Final result is 25
    
    Args:
        a: First integer to multiply
        b: Second integer to multiply
        
    """
    return a * b


# Configure ArXiv tool for ReAct academic research capabilities
# REACT RESEARCH INTEGRATION:
# ArXiv tool enables the ReAct agent to perform iterative research where
# it can search, analyze results, and perform follow-up searches based on findings
api_wrapper_arxiv = ArxivAPIWrapper(
    top_k_results=2,  # Return top 2 papers for focused ReAct reasoning
    doc_content_chars_max=500  # Limit content for efficient processing in ReAct cycles
)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# Configure Wikipedia tool for ReAct general knowledge integration
# REACT KNOWLEDGE SYNTHESIS:
# Wikipedia tool allows the agent to gather background information and
# synthesize it with other tool results in subsequent reasoning cycles
api_wrapper_wiki = WikipediaAPIWrapper(
    top_k_results=1,  # Single focused result for clear ReAct reasoning
    doc_content_chars_max=500  # Balanced content length for ReAct processing
)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# Configure Tavily for ReAct real-time information gathering
# REACT REAL-TIME REASONING:
# Tavily enables the agent to get current information and reason about
# it in context with other tools and previous conversation history
tavily = TavilySearch(
    max_results=5,  # Multiple results for comprehensive ReAct analysis
    topic="general"  # General search suitable for diverse ReAct scenarios
    # ReAct-optimized configuration:
    # - Balanced result count for thorough but efficient reasoning
    # - General topic allows flexibility in agent decision-making
    # - Results feed directly into subsequent reasoning cycles
)

# REACT TOOL ECOSYSTEM:
# Combine all tools into a unified ecosystem for the ReAct agent
# This allows the agent to reason about which tools to use and when,
# creating sophisticated multi-tool workflows
tools = [add, multiply, arxiv, wiki, tavily]


# =============================================================================
# REACT LLM CONFIGURATION AND TOOL BINDING
# =============================================================================

# Initialize the Large Language Model for ReAct reasoning
# REACT LLM SELECTION:
# Using Groq's Qwen model for fast inference, crucial for ReAct agents
# that may require multiple LLM calls in a single conversation
llm = ChatGroq(model="qwen/qwen3-32b")

# REACT TOOL BINDING PROCESS:
# Bind tools to the LLM to enable ReAct functionality
# This creates the foundation for the Reasoning-Acting cycle
llm_with_tools = llm.bind_tools(tools)

"""
REACT TOOL BINDING EXPLANATION:
In a ReAct architecture, tool binding is more sophisticated than traditional agents:

1. SCHEMA UNDERSTANDING: The LLM receives detailed schemas for each tool,
   enabling it to reason about which tool to use in different scenarios

2. CONTEXTUAL DECISION MAKING: The LLM can analyze conversation history
   and current context to determine the most appropriate tool

3. PARAMETER REASONING: The agent reasons about what parameters to pass
   to tools based on the current conversation state

4. SEQUENTIAL TOOL USAGE: The agent can plan multi-step tool usage and
   execute them in logical sequences

5. RESULT INTEGRATION: Tool results are fed back into the reasoning process,
   allowing the agent to adapt its approach based on outcomes

REACT CYCLE WITH TOOLS:
Input → Reason about tools needed → Act (use tool) → Observe results → 
Reason about next action → Act again (if needed) → Final response
"""


# =============================================================================
# REACT STATE SCHEMA DEFINITION
# =============================================================================

class State(TypedDict):
    """
    ReAct agent state schema using TypedDict for structured conversation management.
    
    REACT STATE MANAGEMENT PRINCIPLES:
    
    1. CONVERSATION CONTINUITY: State preserves the entire reasoning and acting
       history, allowing the agent to reference previous thoughts and actions
    
    2. MESSAGE ACCUMULATION: Each ReAct cycle adds messages to the state:
       - Human messages (input/questions)
       - AI messages (reasoning and responses)
       - Tool messages (action results)
    
    3. CONTEXT PRESERVATION: The add_messages reducer ensures that each
       ReAct iteration has access to the complete conversation context
    
    4. REASONING TRACE: The accumulated messages create a complete trace
       of the agent's reasoning process, making it auditable and debuggable
    
    REACT STATE FLOW:
    Initial State → Reasoning Message → Tool Action → Tool Result Message → 
    Further Reasoning → More Actions (if needed) → Final Response
    
    The add_messages reducer in ReAct context:
    - Maintains chronological order of reasoning and actions
    - Prevents message duplication across ReAct cycles
    - Enables the agent to learn from its own previous actions
    - Supports complex multi-turn conversations with tool usage
    """
    messages: Annotated[list[AnyMessage], add_messages]


# =============================================================================
# REACT NODE FUNCTIONS
# =============================================================================

def tool_calling_llm(state: State):
    """
    Core ReAct reasoning and action node - the heart of the ReAct agent.
    
    REACT NODE FUNCTIONALITY:
    This node embodies the "Reasoning" part of ReAct (Reasoning and Acting).
    Unlike traditional agents that make single decisions, this node can be
    called multiple times in a conversation, enabling iterative reasoning.
    
    REACT REASONING PROCESS:
    1. CONTEXT ANALYSIS: Analyzes complete conversation history including
       previous reasoning steps and tool results
    
    2. SITUATION ASSESSMENT: Evaluates current state and determines if more
       information or actions are needed to complete the task
    
    3. TOOL DECISION MAKING: Reasons about which tools (if any) would be
       most helpful for the current step
    
    4. ACTION PLANNING: If tools are needed, formulates specific tool calls
       with appropriate parameters based on reasoning
    
    5. RESPONSE GENERATION: If sufficient information is available, generates
       a comprehensive final response
    
    REACT ITERATIVE CAPABILITY:
    This node can process:
    - Initial user queries (start reasoning)
    - Tool results (continue reasoning with new information)
    - Follow-up questions (extend reasoning chain)
    - Clarification requests (refine reasoning approach)
    
    NODE EXECUTION IN REACT CYCLE:
    Input State → Reasoning Analysis → Decision (Tool Call OR Final Answer) → 
    Output State Update
    
    Args:
        state: Current ReAct conversation state with complete message history
        
    Returns:
        dict: State update containing either:
              - Tool calls (to continue ReAct cycle)
              - Final response (to end ReAct cycle)
    """
    # REACT REASONING INVOCATION:
    # Pass complete conversation history to LLM for comprehensive reasoning
    # This includes previous reasoning steps, tool results, and user interactions
    response = llm_with_tools.invoke(state["messages"])
    
    # REACT STATE UPDATE:
    # Add the LLM's reasoning/action decision to the conversation state
    # This response may contain:
    # 1. Tool calls (triggering the "Acting" part of ReAct)
    # 2. Final answer (completing the ReAct cycle)
    # 3. Intermediate reasoning (continuing the thought process)
    return {"messages": [response]}


# =============================================================================
# REACT GRAPH CONSTRUCTION AND COMPILATION
# =============================================================================

def build_graph():
    """
    Constructs and compiles the ReAct agent graph with iterative capabilities.
    
    REACT GRAPH ARCHITECTURE:
    This graph implements the core ReAct pattern through strategic edge design:
    
    TRADITIONAL AGENT FLOW:
    START → LLM → Tools → END (linear, single-pass)
    
    REACT AGENT FLOW:
    START → LLM → Tools → LLM → Tools → ... → LLM → END (iterative, multi-pass)
    
    GRAPH COMPONENTS EXPLANATION:
    
    1. REASONING NODE (tool_calling_llm):
       - Handles all reasoning and decision-making
       - Can be invoked multiple times per conversation
       - Decides whether to use tools or provide final answer
    
    2. ACTION NODE (tools):
       - Executes tool calls generated by reasoning node
       - Feeds results back into the reasoning cycle
       - Handles multiple tool types seamlessly
    
    3. CONDITIONAL ROUTING (tools_condition):
       - Determines if LLM wants to use tools (continue ReAct cycle)
       - Or if LLM is ready to provide final answer (end ReAct cycle)
    
    REACT CYCLE IMPLEMENTATION:
    The key to ReAct is the edge from tools back to the reasoning node:
    - After tools execute, control returns to reasoning node
    - Agent can then reason about tool results and decide next steps
    - This creates the iterative Reasoning-Acting pattern
    
    EDGE FLOW DETAILS:
    1. START → tool_calling_llm: Begin reasoning process
    2. tool_calling_llm → [conditional]:
       - If tool calls detected → route to tools node (Acting phase)
       - If no tool calls → route to END (Final answer ready)
    3. tools → tool_calling_llm: Return to reasoning with tool results
    
    This creates the ReAct loop: Reason → Act → Observe → Reason → ...
    
    Returns:
        Compiled ReAct graph ready for iterative conversation execution
    """
    # Initialize ReAct graph builder with conversation state schema
    builder = StateGraph(State)
    
    # REACT CORE NODES:
    
    # Add reasoning node - handles all LLM-based thinking and decision making
    # This node can be called multiple times per conversation in ReAct pattern
    builder.add_node("tool_calling_llm", tool_calling_llm)
    
    # Add action node - executes tools and feeds results back to reasoning
    # ToolNode automatically handles all tool execution from LLM decisions
    builder.add_node("tools", ToolNode(tools))
    
    # REACT EDGE CONFIGURATION:
    
    # 1. CONVERSATION ENTRY POINT:
    # Always begin with reasoning node to analyze user input
    builder.add_edge(START, "tool_calling_llm")
    
    # 2. REASONING TO ACTION ROUTING:
    # Conditional edge implements ReAct decision-making:
    # - tools_condition examines LLM output for tool calls
    # - Routes to "tools" if agent wants to take action
    # - Routes to END if agent has sufficient information for final answer
    builder.add_conditional_edges(
        "tool_calling_llm",  # Source: reasoning node
        tools_condition      # Condition: check for tool usage intent
    )
    
    # 3. ACTION TO REASONING LOOP (KEY REACT FEATURE):
    # After tools execute, return to reasoning node with results
    # This creates the iterative cycle that defines ReAct agents
    # The agent can now reason about tool results and decide next steps
    builder.add_edge("tools", "tool_calling_llm")
    
    # Adding memory
    memory = MemorySaver()
    # REACT GRAPH COMPILATION:
    # Compile graph into optimized executable form
    # Validates ReAct cycle integrity and optimizes execution paths
    return builder.compile(checkpointer=memory)


# =============================================================================
# REACT AGENT TESTING AND DEMONSTRATION
# =============================================================================

def demonstrate_capabilities():
    """
    Demonstrates ReAct agent capabilities across various scenarios.
    
    REACT TESTING SCENARIOS:
    These tests showcase different aspects of ReAct reasoning:
    
    1. SIMPLE TOOL USAGE: Basic Reasoning → Acting → Response pattern
    2. NO-TOOL REASONING: Pure reasoning without external actions
    3. COMPLEX MULTI-TOOL: Multiple Reasoning-Acting cycles
    4. REAL-TIME INTEGRATION: Current information gathering and reasoning
    
    Each test demonstrates how the ReAct agent:
    - Reasons about the problem
    - Decides on appropriate actions
    - Processes action results
    - Continues reasoning or provides final answer
    """
    
    print("Building and compiling the ReAct agent graph...")
    graph = build_graph()
    
    print("ReAct agent successfully compiled!")
    print("=" * 80)
    
    # REACT TEST CASE 1: Simple Mathematical Reasoning and Acting
    print("REACT TEST 1: Simple Mathematical Reasoning")
    print("Query: 'What is 2 plus 2?'")
    print("Expected ReAct Flow: Reason about math → Act (use add tool) → Final answer")
    print("-" * 40)
    
    messages = graph.invoke({"messages": HumanMessage(content="What is 2 plus 2?")})
    for message in messages["messages"]:
        message.pretty_print()
    
    print("=" * 80)
    
    # REACT TEST CASE 2: Pure Reasoning (No Actions Needed)
    print("REACT TEST 2: Pure Reasoning Query")
    print("Query: 'What is machine learning?'")
    print("Expected ReAct Flow: Reason about query → Determine no tools needed → Direct answer")
    print("-" * 40)
    
    messages = graph.invoke({"messages": HumanMessage(content="What is machine learning?")})
    for message in messages["messages"]:
        message.pretty_print()
    
    print("=" * 80)
    
    # REACT TEST CASE 3: Complex Multi-Tool Reasoning and Acting
    print("REACT TEST 3: Complex Multi-Tool ReAct Cycle")
    print("Query: 'Find a summary of the Attention Is All You Need paper and then add 100 and 50'")
    print("Expected ReAct Flow:")
    print("  1. Reason about research need → Act (search ArXiv)")
    print("  2. Reason about paper results → Reason about math need")
    print("  3. Act (use add tool) → Reason about final response → Answer")
    print("-" * 40)
    
    messages = graph.invoke({"messages": HumanMessage(
        content="Find a summary of the 'Attention Is All You Need' paper and then add 100 and 50"
    )})
    for message in messages["messages"]:
        message.pretty_print()
    
    print("=" * 80)
    
    # REACT TEST CASE 4: Real-Time Information ReAct Cycle
    print("REACT TEST 4: Real-Time Information Gathering")
    print("Query: 'What is the latest news about One Piece?'")
    print("Expected ReAct Flow: Reason about current info need → Act (web search) → ")
    print("                     Reason about results → Final comprehensive answer")
    print("-" * 40)
    
    messages = graph.invoke({"messages": HumanMessage(
        content="What is the latest news about One Piece?"
    )})
    for message in messages["messages"]:
        message.pretty_print()
    
    print("=" * 80)


def demonstrate_capabilities_with_memory():
    config={"configurable":{"thread_id":"1"}}
    graph_memory = build_graph()

    messages=[HumanMessage(content="Add 12 and 13")]

    messages=graph_memory.invoke({"messages":messages},config=config)
    for m in messages["messages"]:
        m.pretty_print()

    messages=[HumanMessage(content="multiply that number to 100")]
    
    messages=graph_memory.invoke({"messages":messages},config=config)
    print(messages)
    for m in messages["messages"]:
        m.pretty_print()

# =============================================================================
# REACT ARCHITECTURE CONCEPTS AND EXPLANATIONS
# =============================================================================

"""
=============================================================================
COMPREHENSIVE REACT ARCHITECTURE ANALYSIS
=============================================================================

REACT PARADIGM DEEP DIVE:

1. REASONING AND ACTING SYNERGY:
   ReAct agents excel because they interleave reasoning and acting:
   - REASONING: "I need current information about X"
   - ACTING: Search for information about X
   - OBSERVATION: Process search results
   - REASONING: "Based on these results, I should also check Y"
   - ACTING: Search for information about Y
   - OBSERVATION: Process additional results
   - REASONING: "Now I have enough information to provide a comprehensive answer"

2. ITERATIVE PROBLEM SOLVING:
   Unlike traditional agents that make single passes, ReAct agents can:
   - Adjust their approach based on intermediate results
   - Perform multi-step problem decomposition
   - Self-correct when initial approaches don't yield sufficient information
   - Build comprehensive understanding through multiple information gathering cycles

3. TOOL INTEGRATION IN REACT CONTEXT:
   Tools in ReAct agents serve as extensions of reasoning:
   - Mathematical tools: Enable quantitative reasoning steps
   - Research tools: Provide factual grounding for reasoning
   - Web search tools: Incorporate current information into reasoning
   - Custom tools: Extend reasoning capabilities to domain-specific tasks

4. CONVERSATION CONTINUITY:
   ReAct maintains conversation context across multiple reasoning cycles:
   - Previous reasoning steps inform current decisions
   - Tool results become part of the reasoning foundation
   - Multi-turn conversations build on accumulated knowledge
   - Complex tasks are broken down into manageable reasoning steps

REACT ADVANTAGES OVER TRADITIONAL APPROACHES:

1. TRANSPARENCY: 
   - Reasoning process is visible and auditable
   - Decision-making steps can be traced and understood
   - Tool usage rationale is clear from conversation flow

2. ADAPTABILITY:
   - Agent can change approach based on intermediate results
   - Failed tool calls lead to alternative reasoning paths
   - Complex problems are decomposed dynamically

3. ROBUSTNESS:
   - Self-correcting when tools provide unexpected results
   - Can handle partial information and reason about gaps
   - Graceful degradation when tools are unavailable

4. SCALABILITY:
   - Easy to add new tools without changing core reasoning logic
   - Complex workflows emerge from simple reasoning patterns
   - Can handle increasingly sophisticated task requirements

REACT IMPLEMENTATION PATTERNS:

1. SINGLE-CYCLE REACT:
   Query → Reason → Act → Respond
   (Simple tasks requiring one tool usage)

2. MULTI-CYCLE REACT:
   Query → Reason → Act → Reason → Act → ... → Respond
   (Complex tasks requiring multiple information gathering steps)

3. BRANCHED REACT:
   Query → Reason → Multiple parallel actions → Synthesize → Respond
   (Tasks requiring information from multiple sources)

4. RECURSIVE REACT:
   Query → Reason → Act → New sub-query → Nested ReAct cycle → Respond
   (Tasks that spawn related sub-tasks)

WHEN TO USE REACT AGENTS:

IDEAL SCENARIOS:
- Multi-step problem solving requiring external information
- Tasks where reasoning about tool results is important
- Situations requiring transparent decision-making processes
- Complex workflows that benefit from adaptive approaches

LESS SUITABLE SCENARIOS:
- Simple, single-step tasks (adds unnecessary complexity)
- Highly time-sensitive applications (multiple LLM calls add latency)
- Tasks with well-defined, fixed workflows (traditional agents more efficient)

REACT DEBUGGING AND MONITORING:

The ReAct pattern provides natural debugging capabilities:
- Each message in the conversation shows a step in the reasoning process
- Tool calls and results are explicitly visible
- Failed reasoning paths can be identified and corrected
- Performance bottlenecks can be traced to specific reasoning or acting steps

OPTIMIZATION STRATEGIES FOR REACT AGENTS:

1. TOOL SELECTION OPTIMIZATION:
   - Provide clear, focused tool descriptions
   - Optimize tool response formats for reasoning consumption
   - Implement tool result caching for repeated queries

2. REASONING EFFICIENCY:
   - Use faster LLM models for intermediate reasoning steps
   - Implement reasoning shortcuts for common patterns
   - Cache reasoning patterns for similar task types

3. CONVERSATION MANAGEMENT:
   - Implement conversation summarization for very long interactions
   - Prune irrelevant historical context when appropriate
   - Balance context preservation with processing efficiency

REACT AGENT EXTENSIONS:

The current implementation can be extended with:
- Memory systems for long-term reasoning patterns
- Learning mechanisms to improve tool selection over time
- Parallel tool execution for independent reasoning branches
- Custom reasoning strategies for domain-specific tasks
- Integration with external knowledge bases for enhanced reasoning
"""


# =============================================================================
# REACT EXECUTION AND ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main execution block for ReAct agent demonstration.
    
    REACT EXECUTION PHILOSOPHY:
    This execution demonstrates the ReAct agent's capability to handle
    diverse scenarios through adaptive reasoning and acting cycles.
    
    The demonstrations show:
    1. How ReAct handles different complexity levels
    2. The iterative nature of reasoning and acting
    3. The transparency of the decision-making process
    4. The integration of multiple tools in coherent workflows
    
    REACT DEBUGGING FEATURES:
    - LangSmith tracing enabled for detailed cycle analysis
    - Message pretty-printing shows complete reasoning trace
    - Error handling provides insight into reasoning failures
    """
    
    try:
        # Execute comprehensive ReAct agent demonstrations
        # demonstrate_capabilities()
        demonstrate_capabilities_with_memory() 

        print("\n" + "=" * 80)
        print("REACT AGENT DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("""
Key ReAct Features Demonstrated:
✅ Iterative Reasoning-Acting cycles
✅ Multi-tool integration and coordination  
✅ Adaptive problem-solving approaches
✅ Transparent decision-making processes
✅ Self-correcting behavior with tool results
✅ Complex task decomposition and execution

ReAct Agent Benefits Observed:
- Handles both simple and complex queries effectively
- Provides traceable reasoning for all decisions
- Adapts approach based on intermediate results
- Maintains conversation context across multiple cycles
- Integrates multiple information sources seamlessly
        """)
        
    except Exception as e:
        print(f"ReAct Agent Execution Error: {e}")
        print("\nREACT TROUBLESHOOTING CHECKLIST:")
        print("- Verify all API keys in .env file:")
        print("  • GROQ_API_KEY (for LLM reasoning)")
        print("  • TAVILY_API_KEY (for web search actions)")
        print("  • LANGCHAIN_API_KEY (for tracing, optional)")
        print("- Check network connectivity for tool actions")
        print("- Ensure sufficient API quotas for iterative calls")
        print("- Verify LangGraph and LangChain versions are compatible")


# =============================================================================
# REACT AGENT EXTENSION EXAMPLES
# =============================================================================

"""
EXTENDING THE REACT AGENT:

1. ADDING DOMAIN-SPECIFIC TOOLS:
   def code_analyzer(code: str) -> str:
       '''Analyze code quality and suggest improvements'''
       # Tool implementation
       pass
   
   # Add to tools list for software development ReAct agent
   tools.append(code_analyzer)

2. IMPLEMENTING CUSTOM REASONING STRATEGIES:
   def strategic_reasoning_node(state: State):
       '''Custom reasoning node with domain-specific logic'''
       # Implement specialized reasoning patterns
       return {"messages": [...]}

3. ADDING MEMORY AND LEARNING:
   class ReActStateWithMemory(TypedDict):
       messages: Annotated[list[AnyMessage], add_messages]
       reasoning_patterns: dict  # Store successful reasoning patterns
       tool_effectiveness: dict  # Track tool success rates
       user_preferences: dict    # Learn user-specific preferences

4. IMPLEMENTING PARALLEL REASONING:
   # Multiple reasoning nodes for different aspects of complex problems
   builder.add_node("research_reasoning", research_focused_node)
   builder.add_node("calculation_reasoning", math_focused_node)

5. ADDING CONVERSATION SUMMARIZATION:
   def summarize_conversation(state: State):
       '''Compress long conversations while preserving key reasoning'''
       # Implement conversation summarization logic
       return {"messages": [summarized_message]}

REACT PERFORMANCE MONITORING:

Track key ReAct metrics:
- Average reasoning cycles per query
- Tool usage patterns and effectiveness
- Reasoning-to-action ratios
- User satisfaction with reasoning transparency
- Task completion rates across different complexity levels

REACT AGENT DEPLOYMENT CONSIDERATIONS:

1. LATENCY MANAGEMENT: Multiple LLM calls increase response time
2. COST OPTIMIZATION: More LLM calls increase API costs
3. RATE LIMITING: Handle API rate limits across reasoning cycles
4. ERROR RECOVERY: Robust handling of tool failures in reasoning cycles
5. CONVERSATION LIMITS: Manage very long reasoning chains
6. CACHING STRATEGIES: Cache reasoning patterns and tool results

The ReAct architecture provides a powerful foundation for building
sophisticated AI agents that can reason transparently and act effectively
across a wide range of complex tasks.
"""