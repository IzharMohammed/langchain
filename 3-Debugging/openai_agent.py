# from typing import Annotated
# from typing_extensions import TypedDict
# # from langchain_openai import ChatOpenAI
# from langgraph.graph import END, START
# from langgraph.graph.state import StateGraph
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode
# from langchain_core.tools import tool
# from langchain_core.messages import BaseMessage
# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


# class State(TypedDict):
#     messages:Annotated[list[BaseMessage],add_messages]

# # model=ChatOpenAI(temperature=0)
# model=ChatGroq(model="Llama3-8b-8192")

# def make_default_graph():
#     graph_workflow=StateGraph(State)

#     def call_model(state):
#         return {"messages":[model.invoke(state['messages'])]}
    
#     graph_workflow.add_node("agent", call_model)
#     graph_workflow.add_edge("agent", END)
#     graph_workflow.add_edge(START, "agent")

#     agent=graph_workflow.compile()
#     return agent

# def make_alternative_graph():
#     """Make a tool-calling agent"""

#     @tool
#     def add(a: float, b: float):
#         """Adds two numbers."""
#         return a + b

#     tool_node = ToolNode([add])
#     model_with_tools = model.bind_tools([add])
#     def call_model(state):
#         return {"messages": [model_with_tools.invoke(state["messages"])]}

#     def should_continue(state: State):
#         if state["messages"][-1].tool_calls:
#             return "tools"
#         else:
#             return END

#     graph_workflow = StateGraph(State)

#     graph_workflow.add_node("agent", call_model)
#     graph_workflow.add_node("tools", tool_node)
#     graph_workflow.add_edge("tools", "agent")
#     graph_workflow.add_edge(START, "agent")
#     graph_workflow.add_conditional_edges("agent", should_continue)

#     agent = graph_workflow.compile()
#     return agent

# agent=make_alternative_graph()

# Fixed openai_agent.py - Using Groq instead of OpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# State definition
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize model
model = ChatGroq(model="Llama3-8b-8192", temperature=0)

def make_default_graph():
    """Create a simple chat agent without tools"""
    graph_workflow = StateGraph(State)
    
    def call_model(state):
        return {"messages": [model.invoke(state['messages'])]}
    
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")
    
    return graph_workflow.compile()

def make_tool_agent():
    """Create an agent with tool-calling capabilities"""
    
    # Define tools
    @tool
    def add(a: float, b: float) -> float:
        """Adds two numbers together."""
        return a + b
    
    @tool
    def multiply(a: float, b: float) -> float:
        """Multiplies two numbers together."""
        return a * b
    
    @tool
    def get_weather(city: str) -> str:
        """Get weather information for a city (mock function)."""
        return f"The weather in {city} is sunny and 25°C"
    
    # Create tool node and bind tools to model
    tools = [add, multiply, get_weather]
    tool_node = ToolNode(tools)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state):
        """Call the model with tool capabilities"""
        return {"messages": [model_with_tools.invoke(state["messages"])]}
    
    def should_continue(state: State):
        """Decide whether to continue to tools or end"""
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        else:
            return END
    
    # Build the graph
    graph_workflow = StateGraph(State)
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    
    # Add edges
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)
    
    return graph_workflow.compile()

def make_multi_agent_system():
    """Create a multi-agent system with different roles"""
    
    # Tools for the system
    @tool
    def calculate(expression: str) -> str:
        """Safely evaluate mathematical expressions."""
        try:
            # Simple calculator - only allow basic operations
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return str(result)
            else:
                return "Invalid expression"
        except:
            return "Error in calculation"
    
    @tool
    def search_knowledge(query: str) -> str:
        """Search for information (mock function)."""
        return f"Here's what I found about '{query}': This is mock search result data."
    
    tools = [calculate, search_knowledge]
    tool_node = ToolNode(tools)
    
    # Different models for different agents
    analyst_model = model.bind_tools(tools)
    coordinator_model = model
    
    def analyst_agent(state):
        """Analyst agent with tool access"""
        system_prompt = "You are an analyst. Use tools to help answer questions accurately."
        messages = state["messages"]
        if not any("analyst" in str(msg) for msg in messages):
            messages = [{"role": "system", "content": system_prompt}] + messages
        return {"messages": [analyst_model.invoke(messages)]}
    
    def coordinator_agent(state):
        """Coordinator agent that manages the conversation"""
        system_prompt = "You are a coordinator. Summarize and provide final responses."
        messages = state["messages"]
        return {"messages": [coordinator_model.invoke(messages)]}
    
    def route_decision(state: State):
        """Decide which agent should handle the request"""
        last_message = state["messages"][-1]
        
        # Check if we need tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Route based on message content
        content = str(last_message.content).lower()
        if any(keyword in content for keyword in ['calculate', 'math', 'number', 'compute']):
            return "analyst"
        else:
            return "coordinator"
    
    # Build multi-agent graph
    graph_workflow = StateGraph(State)
    graph_workflow.add_node("analyst", analyst_agent)
    graph_workflow.add_node("coordinator", coordinator_agent)
    graph_workflow.add_node("tools", tool_node)
    
    # Add edges
    graph_workflow.add_edge(START, "coordinator")
    graph_workflow.add_edge("tools", "analyst")
    graph_workflow.add_edge("analyst", END)
    graph_workflow.add_edge("coordinator", END)
    graph_workflow.add_conditional_edges("coordinator", route_decision)
    
    return graph_workflow.compile()

# Default export for langgraph dev
agent = make_tool_agent()

# Alternative agents (you can switch by changing the export)
# agent = make_default_graph()
# agent = make_multi_agent_system()

if __name__ == "__main__":
    # Test the agent locally
    print("Testing LangGraph Agent...")
    
    # Test simple conversation
    result = agent.invoke({"messages": "Hello! What can you help me with?"})
    print("Simple chat:", result["messages"][-1].content)
    
    # Test tool usage
    result = agent.invoke({"messages": "What's 25 + 37?"})
    print("Tool usage:", result["messages"][-1].content)
    
    # Test weather tool
    result = agent.invoke({"messages": "What's the weather in New York?"})
    print("Weather query:", result["messages"][-1].content)

# Start development server
# langgraph dev

# Start on different port
# langgraph dev --port 3000

# Enable debug mode
# langgraph dev --debug

# Test locally
# python openai_agent.py