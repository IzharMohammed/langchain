# ===================================================================
# LANGCHAIN CHATBOT TUTORIAL: FROM BASIC TO ADVANCED
# This tutorial covers: Basic LLM usage, Message History, Prompt Templates,
# and Conversation Management with detailed explanations
# ===================================================================

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This allows us to store sensitive API keys outside our code
load_dotenv()

# Retrieve Groq API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
print(f"API Key loaded: {'✓' if groq_api_key else '✗'}")

# ===================================================================
# SECTION 1: BASIC LLM INITIALIZATION
# ===================================================================
# Initialize the Groq LLM model
# Groq provides fast inference for open-source models like Gemma2
from langchain_groq import ChatGroq

model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
print("Model initialized:", model)

# ===================================================================
# SECTION 2: BASIC MESSAGE INTERACTION (STATELESS)
# ===================================================================
# Import message types - these represent different roles in conversation
from langchain_core.messages import HumanMessage, AIMessage

# Single message interaction - model has NO memory of previous conversations
print("\n=== BASIC SINGLE MESSAGE ===")
response1 = model.invoke([HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer")])
print("Response 1:", response1.content)

# Multi-message interaction - we manually provide conversation context
print("\n=== MANUAL CONVERSATION CONTEXT ===")
response2 = model.invoke([
    HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer"),
    AIMessage(content="Hello Krish! It's nice to meet you. \n\nAs a Chief AI Engineer, what kind of projects are you working on these days? \n\nI'm always eager to learn more about the exciting work being done in the field of AI.\n"),
    HumanMessage(content="Hey What's my name and what do I do?")
])
print("Response 2:", response2.content)

# ===================================================================
# SECTION 3: MESSAGE HISTORY (STATEFUL CONVERSATIONS)
# ===================================================================
"""
KEY CONCEPT: MESSAGE HISTORY
- Without message history, each interaction is independent
- With message history, the model remembers previous conversations
- Each session has a unique ID to separate different conversations
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Store for maintaining multiple conversation sessions
# Each session_id will have its own conversation history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves or creates a chat history for a given session.
    This allows multiple users or conversation threads to coexist.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap our model with message history capability
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# Configuration specifies which conversation session to use
config = {"configurable": {"session_id": "chat1"}}

print("\n=== STATEFUL CONVERSATION - SESSION 1 ===")
# First message in chat1 session
response3 = with_message_history.invoke(
    [HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer")],
    config=config
)
print("First message response:", response3.content)

# Second message - model should remember the name "Krish"
response4 = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
print("Name recall response:", response4.content)

# ===================================================================
# SECTION 4: MULTIPLE SESSIONS (SESSION ISOLATION)
# ===================================================================
"""
KEY CONCEPT: SESSION ISOLATION
- Different session IDs maintain separate conversation histories
- This allows multiple users or topics to be handled independently
"""

print("\n=== SESSION ISOLATION DEMO ===")
# Create a new session (chat2) - should NOT know about Krish
config1 = {"configurable": {"session_id": "chat2"}}

response5 = with_message_history.invoke(
    [HumanMessage(content="Whats my name")],
    config=config1
)
print("New session - unknown name:", response5.content)

# Introduce a different name in chat2 session
response6 = with_message_history.invoke(
    [HumanMessage(content="Hey My name is John")],
    config=config1
)
print("Introduced John:", response6.content)

# Verify chat2 remembers John, not Krish
response7 = with_message_history.invoke(
    [HumanMessage(content="Whats my name")],
    config=config1
)
print("Chat2 name recall:", response7.content)

# ===================================================================
# SECTION 5: PROMPT TEMPLATES
# ===================================================================
"""
KEY CONCEPT: PROMPT TEMPLATES
- Structure how we format inputs to the LLM
- Allow for consistent system instructions
- Enable dynamic content insertion with variables
- MessagesPlaceholder allows inserting conversation history
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Basic prompt template with system message
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all the question to the best of your ability"),
    MessagesPlaceholder(variable_name="messages")  # Placeholder for conversation messages
])

# Create a chain: prompt template → model
chain = prompt | model  # LCEL syntax

print("\n=== PROMPT TEMPLATE BASIC ===")
response8 = chain.invoke({"messages": [HumanMessage(content="Hi My name is Krish")]})
print("Templated response:", response8.content)

# Wrap the chain with message history
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "chat3"}}
response9 = with_message_history.invoke(
    [HumanMessage(content="Hi My name is Krish")],
    config=config
)
print("Templated + History:", response9.content)

# Test memory in templated chain
response10 = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
print("Templated memory recall:", response10.content)

# ===================================================================
# SECTION 6: ADVANCED PROMPT TEMPLATES WITH VARIABLES
# ===================================================================
"""
KEY CONCEPT: DYNAMIC PROMPT VARIABLES
- Templates can include variables beyond just messages
- Enables customization like language, tone, or domain-specific instructions
- Variables are passed as a dictionary to the chain
"""

print("\n=== ADVANCED PROMPT WITH LANGUAGE VARIABLE ===")
# Enhanced prompt template with language variable
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model

# Test with Hindi language
response11 = chain.invoke({
    "messages": [HumanMessage(content="Hi My name is Krish")], 
    "language": "Hindi"
})
print("Hindi response:", response11.content)

# ===================================================================
# SECTION 7: MESSAGE HISTORY WITH MULTIPLE INPUT KEYS
# ===================================================================
"""
KEY CONCEPT: COMPLEX INPUT HANDLING
- When chain has multiple inputs (messages + other variables)
- Must specify which key contains the conversation messages
- Uses input_messages_key parameter
"""

print("\n=== COMPLEX INPUT WITH MESSAGE HISTORY ===")
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"  # Specify which key contains the messages
)

config = {"configurable": {"session_id": "chat4"}}
response12 = with_message_history.invoke(
    {'messages': [HumanMessage(content="Hi,I am Krish")], "language": "Hindi"},
    config=config
)
print("Complex input - Hindi intro:", response12.content)

response13 = with_message_history.invoke(
    {"messages": [HumanMessage(content="whats my name?")], "language": "Hindi"},
    config=config,
)
print("Complex input - Hindi name recall:", response13.content)

# ===================================================================
# SECTION 8: CONVERSATION HISTORY MANAGEMENT
# ===================================================================
"""
KEY CONCEPT: MESSAGE TRIMMING
- LLMs have token limits (context windows)
- Long conversations can exceed these limits
- trim_messages helps manage conversation length
- Strategies: keep recent messages, preserve system messages, etc.
"""

from langchain_core.messages import SystemMessage, trim_messages

print("\n=== MESSAGE TRIMMING DEMO ===")
# Configure trimmer to keep only recent messages within token limit
trimmer = trim_messages(
    max_tokens=45,           # Maximum tokens to keep
    strategy="last",         # Keep the most recent messages
    token_counter=model,     # Use model's tokenizer for counting
    include_system=True,     # Always keep system message
    allow_partial=False,     # Don't cut messages in half
    start_on="human"        # Ensure conversation starts with human message
)

# Sample long conversation to demonstrate trimming
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# See which messages survive trimming
trimmed = trimmer.invoke(messages)
print(f"Original messages: {len(messages)}, After trimming: {len(trimmed)}")
for msg in trimmed:
    print(f"  {msg.__class__.__name__}: {msg.content}")

# ===================================================================
# SECTION 9: ADVANCED CHAIN WITH TRIMMING
# ===================================================================
"""
KEY CONCEPT: RUNNABLE COMPOSITION
- RunnablePassthrough.assign allows modifying input data mid-chain
- itemgetter extracts specific keys from input dictionary
- Creates complex processing pipelines with LCEL
"""

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

print("\n=== ADVANCED CHAIN WITH TRIMMING ===")
# Complex chain: input → trim messages → apply prompt → model
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)  # Trim messages
    | prompt    # Apply prompt template
    | model     # Send to LLM
)

# Test trimming effect on memory
response14 = chain.invoke({
    "messages": messages + [HumanMessage(content="What ice cream do i like")],
    "language": "English"
})
print("Ice cream memory (should remember):", response14.content)

response15 = chain.invoke({
    "messages": messages + [HumanMessage(content="what math problem did i ask")],
    "language": "English",
})
print("Math memory (might be trimmed):", response15.content)

# ===================================================================
# SECTION 10: COMPLETE SYSTEM WITH TRIMMING AND HISTORY
# ===================================================================
"""
KEY CONCEPT: PRODUCTION-READY CHATBOT
- Combines all features: templates, history, trimming
- Manages token limits automatically
- Maintains conversation state across interactions
- Suitable for real-world applications
"""

print("\n=== COMPLETE CHATBOT SYSTEM ===")
# Final chatbot with all features
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "chat5"}}

# Test with pre-loaded conversation + new question
response16 = with_message_history.invoke({
    "messages": messages + [HumanMessage(content="whats my name?")],
    "language": "English",
}, config=config)
print("Complete system - name recall:", response16.content)

# Follow-up question (should use trimmed history)
response17 = with_message_history.invoke({
    "messages": [HumanMessage(content="what math problem did i ask?")],
    "language": "English",
}, config=config)
print("Complete system - math recall:", response17.content)

# ===================================================================
# SUMMARY OF KEY CONCEPTS
# ===================================================================
"""
🎯 KEY CONCEPTS COVERED:

1. **Basic LLM Usage**: Direct model invocation with messages
2. **Message Types**: HumanMessage, AIMessage, SystemMessage
3. **Stateless vs Stateful**: Memory-less vs conversation history
4. **Session Management**: Multiple isolated conversations
5. **Prompt Templates**: Structured input formatting
6. **Variable Injection**: Dynamic prompt customization
7. **LCEL Chains**: Composing components with pipe operator
8. **Message History**: Persistent conversation memory
9. **Token Management**: Trimming for context window limits
10. **Production Patterns**: Combining all features effectively

🔄 TYPICAL FLOW:
User Input → Session Lookup → Message Trimming → Prompt Template → 
LLM Processing → Response → Save to History → Return to User

🏗️ ARCHITECTURE BENEFITS:
- Scalable: Handle multiple users/sessions
- Memory Efficient: Automatic message trimming
- Flexible: Customizable prompts and behavior
- Maintainable: Modular component design
- Production Ready: Error handling and state management
"""

print("\n" + "="*50)
print("TUTORIAL COMPLETE! 🎉")
print("You now understand LangChain chatbot fundamentals!")
print("="*50)