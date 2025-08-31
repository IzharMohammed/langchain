from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_tavily import TavilySearch
from langchain_community.tools.tavily_search import TavilySearchResults

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
# print(arxiv.name)
# print(arxiv.invoke("Data structures and algorithms"))

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
# print(wiki.name)
# print(wiki.invoke("What is one piece"))

from dotenv import load_dotenv
load_dotenv()

import os

os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


tavily = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

# tavily=TavilySearchResults()
# tavily.invoke("")

# print(tavily.invoke({"query":"When will bleach thousand year boold war part-4 will be releasing"}))

def add(a: int, b: int) -> int:
    """
    Add two integers.
    
    Args:
        a (int): First integer
        b (int): Second integer
    """
    return a + b

## combine all tools in the list
tools=[add,arxiv,wiki,tavily]

## Initialize my LLM model
from langchain_groq import ChatGroq
llm=ChatGroq(model="qwen/qwen3-32b")

llm_with_tools=llm.bind_tools(tools)


from langchain_core.messages import AIMessage, HumanMessage
# llm_with_tools.invoke([HumanMessage(content=f"When will one piece ends")])

## state schema
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

## Entire chatbot with langGraph
from langgraph.graph import StateGraph,END,START
from langgraph.prebuilt import ToolNode,tools_condition

## Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

builder.add_edge(START,"tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm",tools_condition)
builder.add_edge("tools",END)

graph=builder.compile()

messages = graph.invoke({"messages":HumanMessage(content="what is 2 plus 2")})
for m in messages["messages"]:
    m.pretty_print()

print("-------------------------------------------------------------------------")

messages = graph.invoke({"messages": HumanMessage(content="What is machine learning?")})
for m in messages["messages"]:
    m.pretty_print()

print("-------------------------------------------------------------------------")

messages = graph.invoke({"messages": HumanMessage(content="Find a summary of the 'Attention Is All You Need' paper and then add 100 and 50")})
for m in messages["messages"]:
    m.pretty_print()

print("-------------------------------------------------------------------------")

messages = graph.invoke({"messages": HumanMessage(content="What is the latest news about one piece ")})
for m in messages["messages"]:
    m.pretty_print()