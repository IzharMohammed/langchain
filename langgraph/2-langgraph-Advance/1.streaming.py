from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## Reducers
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:Annotated[list,add_messages]


llm_groq=ChatGroq(model="Llama3-8b-8192")
print(llm_groq)
rs = llm_groq.invoke("Hyy i am izhar, i like to solve real world problems using programming")
print(rs)

## creating nodes
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

def superbot(state:State):
    return {"messages":[llm_groq.invoke(state["messages"])]}

graph=StateGraph(State)

## node
graph.add_node("superBot",superbot)

## Edges
graph.add_edge(START,"superBot")
graph.add_edge("superBot",END) 

graph_builder = graph.compile(checkpointer=memory)

## Invocation

config={"configurable":{"thread_id":"1"}}

graph_builder.invoke({"messages":"Hyy, i am izhar and i like to solve real world problems"},config)

## Streaming the responses with stream method
# Create a thread
config={"configurable":{"thread_id":"2"}}

for chunk in graph_builder.stream({"messages":"Hyy, i am izhar and i like to solve real world problems"},config,stream_mode="updates"):
    print(chunk)

for chunk in graph_builder.stream({"messages":"Hyy, i am izhar and i like to solve real world problems"},config,stream_mode="values"):
    print(chunk)

## streaming with astream method
config={"configurable":{"thread_id":"3"}}

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