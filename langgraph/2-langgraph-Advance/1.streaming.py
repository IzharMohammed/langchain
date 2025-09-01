from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import os
import asyncio
import traceback
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
print(os.getenv("GROQ_API_KEY"))
# Reducers
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

# FIXED: Use a supported Groq model
# Available models: llama3-8b-8192, llama3-70b-4096, mixtral-8x7b-32768, gemma-7b-it
llm_groq = ChatGroq(
    model="qwen/qwen3-32b",  # Changed from qwen/qwen3-32b
    temperature=0.1,
    timeout=30,  # Add timeout
    max_retries=2  # Add retry logic
)

print("Model initialized:", llm_groq)

# Test basic invocation with error handling
try:
    rs = llm_groq.invoke("Hey I am Izhar, I like to solve real world problems using programming")
    print("Basic invocation successful:")
    print("Response:", rs.content)
    print("Type:", type(rs))
except Exception as e:
    print(f"Error in basic invocation: {e}")
    print("Traceback:", traceback.format_exc())

# Creating nodes with error handling
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

def superbot(state: State):
    """Enhanced superbot with error handling and logging"""
    try:
        print(f"Superbot called with state: {state}")
        
        # Get the messages from state
        messages = state["messages"]
        print(f"Processing {len(messages)} messages")
        
        # Call the LLM
        response = llm_groq.invoke(messages)
        print(f"LLM response type: {type(response)}")
        print(f"LLM response content preview: {str(response.content)[:100]}...")
        
        return {"messages": [response]}
        
    except Exception as e:
        print(f"Error in superbot: {e}")
        print("Traceback:", traceback.format_exc())
        # Return an error message instead of crashing
        from langchain_core.messages import AIMessage
        error_msg = AIMessage(content=f"Sorry, I encountered an error: {str(e)}")
        return {"messages": [error_msg]}

# Build graph
graph = StateGraph(State)
graph.add_node("superBot", superbot)
graph.add_edge(START, "superBot")
graph.add_edge("superBot", END)

# Compile with error handling
try:
    graph_builder = graph.compile(checkpointer=memory)
    print("Graph compiled successfully")
except Exception as e:
    print(f"Error compiling graph: {e}")
    exit(1)

print("\n" + "="*60)
print("TESTING DIFFERENT STREAMING METHODS")
print("="*60)

# Test 1: Basic invocation
print("\n1. BASIC INVOCATION TEST")
print("-" * 30)
try:
    config = {"configurable": {"thread_id": "test_basic"}}
    result = graph_builder.invoke(
        {"messages": "Hey, I am Izhar and I like to solve real world problems"}, 
        config
    )
    print("Basic invocation result:")
    print(f"Messages count: {len(result['messages'])}")
    print(f"Last message: {result['messages'][-1].content[:200]}...")
except Exception as e:
    print(f"Error in basic invocation: {e}")
    print("Traceback:", traceback.format_exc())

# Test 2: Streaming with 'updates' mode
print("\n2. STREAMING WITH 'updates' MODE")
print("-" * 30)
try:
    config = {"configurable": {"thread_id": "test_updates"}}
    chunk_count = 0
    for chunk in graph_builder.stream(
        {"messages": "What are some interesting programming projects I can work on?"}, 
        config, 
        stream_mode="updates"
    ):
        chunk_count += 1
        print(f"Update chunk {chunk_count}: {chunk}")
        
        # Safety break to prevent infinite loops
        if chunk_count > 10:
            print("Breaking due to too many chunks")
            break
            
    print(f"Total update chunks received: {chunk_count}")
    
except Exception as e:
    print(f"Error in updates streaming: {e}")
    print("Traceback:", traceback.format_exc())

# Test 3: Streaming with 'values' mode
print("\n3. STREAMING WITH 'values' MODE")
print("-" * 30)
try:
    config = {"configurable": {"thread_id": "test_values"}}
    chunk_count = 0
    for chunk in graph_builder.stream(
        {"messages": "Tell me about machine learning applications"}, 
        config, 
        stream_mode="values"
    ):
        chunk_count += 1
        print(f"Values chunk {chunk_count}:")
        print(f"  Keys: {list(chunk.keys()) if isinstance(chunk, dict) else 'Not a dict'}")
        if isinstance(chunk, dict) and 'superBot' in chunk:
            messages = chunk['superBot'].get('messages', [])
            if messages:
                print(f"  Last message content: {messages[-1].content[:100]}...")
        
        # Safety break
        if chunk_count > 10:
            print("Breaking due to too many chunks")
            break
            
    print(f"Total values chunks received: {chunk_count}")
    
except Exception as e:
    print(f"Error in values streaming: {e}")
    print("Traceback:", traceback.format_exc())

# Test 4: Debug streaming mode
print("\n4. STREAMING WITH 'debug' MODE")
print("-" * 30)
try:
    config = {"configurable": {"thread_id": "test_debug"}}
    chunk_count = 0
    for chunk in graph_builder.stream(
        {"messages": "Explain async programming briefly"}, 
        config, 
        stream_mode="debug"
    ):
        chunk_count += 1
        print(f"Debug chunk {chunk_count}: {chunk}")
        
        # Safety break
        if chunk_count > 10:
            print("Breaking due to too many chunks")
            break
            
    print(f"Total debug chunks received: {chunk_count}")
    
except Exception as e:
    print(f"Error in debug streaming: {e}")
    print("Traceback:", traceback.format_exc())

# Test 5: Async streaming
print("\n5. ASYNC STREAMING TEST")
print("-" * 30)

async def test_async_streaming():
    try:
        config = {"configurable": {"thread_id": "test_async"}}
        chunk_count = 0
        
        async for chunk in graph_builder.astream(
            {"messages": "What's the future of AI programming?"}, 
            config,
            stream_mode="updates"
        ):
            chunk_count += 1
            print(f"Async chunk {chunk_count}: {chunk}")
            
            # Safety break
            if chunk_count > 10:
                print("Breaking due to too many chunks")
                break
                
        print(f"Total async chunks received: {chunk_count}")
        
    except Exception as e:
        print(f"Error in async streaming: {e}")
        print("Traceback:", traceback.format_exc())

# Test 6: Async events streaming
print("\n6. ASYNC EVENTS STREAMING TEST")
print("-" * 30)

async def test_async_events():
    try:
        config = {"configurable": {"thread_id": "test_async_events"}}
        event_count = 0
        
        async for event in graph_builder.astream_events(
            {"messages": "Explain LangGraph in simple terms"}, 
            config, 
            version="v2"
        ):
            event_count += 1
            print(f"Event {event_count}: {event.get('event', 'unknown')} - {event.get('name', 'no_name')}")
            
            # Safety break
            if event_count > 20:
                print("Breaking due to too many events")
                break
                
        print(f"Total events received: {event_count}")
        
    except Exception as e:
        print(f"Error in async events streaming: {e}")
        print("Traceback:", traceback.format_exc())

# Run async tests
print("\n" + "="*60)
print("RUNNING ASYNC TESTS")
print("="*60)

async def run_all_async_tests():
    await test_async_streaming()
    await test_async_events()

try:
    asyncio.run(run_all_async_tests())
except Exception as e:
    print(f"Error running async tests: {e}")
    print("Traceback:", traceback.format_exc())

# Test 7: Conversation continuity
print("\n7. CONVERSATION CONTINUITY TEST")
print("-" * 30)
try:
    config = {"configurable": {"thread_id": "conversation_test"}}
    
    # First message
    print("First message:")
    result1 = graph_builder.invoke(
        {"messages": "Hello, I'm Izhar. I'm a programmer."}, 
        config
    )
    print(f"Response 1: {result1['messages'][-1].content[:100]}...")
    
    # Second message (should remember context)
    print("\nSecond message (continuing conversation):")
    result2 = graph_builder.invoke(
        {"messages": "What programming languages would you recommend for me?"}, 
        config
    )
    print(f"Response 2: {result2['messages'][-1].content[:100]}...")
    print(f"Total messages in thread: {len(result2['messages'])}")
    
except Exception as e:
    print(f"Error in conversation continuity test: {e}")
    print("Traceback:", traceback.format_exc())

# async for event in graph_builder.astream_events({"messages":"Hyy, i am izhar and i like to solve real world problems"},config,version="v2"):
    # print(event)



# STREAMING MODES IN LANGGRAPH:

# 1. 'updates' mode:
#    - Returns only the updates/changes made at each step
#    - Shows what each node produced
#    - Best for: Monitoring node outputs, debugging individual steps

# 2. 'values' mode:
#    - Returns the complete state after each step
#    - Shows the accumulated state
#    - Best for: Seeing the full conversation history, final results

# 3. 'debug' mode:
#    - Returns detailed debugging information
#    - Shows internal graph execution details
#    - Best for: Debugging graph structure, understanding execution flow

# 4. Multiple modes:
#    - Can combine multiple modes like ['updates', 'values']
#    - Returns data for all specified modes
#    - Best for: Comprehensive monitoring

# ASYNC VS SYNC STREAMING:

# Synchronous streaming:
# - Uses: graph_builder.stream()
# - Blocks execution until each chunk is ready
# - Good for: Simple applications, sequential processing

# Asynchronous streaming:
# - Uses: graph_builder.astream() or graph_builder.astream_events()
# - Non-blocking, allows other operations to continue
# - Good for: Web applications, concurrent processing, better performance

# STREAM EVENTS (astream_events):
# - Provides more granular event information
# - Shows start/end of nodes, intermediate steps
# - Version "v2" provides enhanced event data
# - Best for: Fine-grained monitoring, performance analysis

# MEMORY AND THREADING:
# - Each thread_id maintains separate conversation state
# - Memory persists across multiple invocations
# - Different threads are isolated from each other
# - Use same thread_id to continue conversations

# ERROR HANDLING:
# - Wrap streaming in try-catch blocks
# - Handle network interruptions gracefully
# - Consider timeout mechanisms for long-running streams

# PERFORMANCE TIPS:
# - Use async streaming for better concurrency
# - Choose appropriate stream_mode for your needs
# - Consider chunk processing time vs. user experience
# - Monitor memory usage with long conversations
#     """)