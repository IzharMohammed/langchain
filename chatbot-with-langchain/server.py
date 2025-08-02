import os
from dotenv import load_dotenv
load_dotenv()  # loading all the environment variable

groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key

from langchain_groq import ChatGroq
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
model

from langchain_core.messages import HumanMessage
model.invoke([HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer")])

from langchain_core.messages import AIMessage
model.invoke(
    [
        HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer"),
        AIMessage(content="Hello Krish! It's nice to meet you. \n\nAs a Chief AI Engineer, what kind of projects are you working on these days? \n\nI'm always eager to learn more about the exciting work being done in the field of AI.\n"),
        HumanMessage(content="Hey What's my name and what do I do?")
    ]
)

### Message History
# We can use a Message History class to wrap our model and make it stateful. 
# This will keep track of inputs and outputs of the model, and store them in some datastore. 
# Future interactions will then load those messages and pass them into the chain as part of the input.

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable": {"session_id": "chat1"}}

response = with_message_history.invoke(
    [HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer")],
    config=config
)

response.content

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

## change the config-->session id
config1 = {"configurable": {"session_id": "chat2"}}
response = with_message_history.invoke(
    [HumanMessage(content="Whats my name")],
    config=config1
)
response.content

response = with_message_history.invoke(
    [HumanMessage(content="Hey My name is John")],
    config=config1
)
response.content

response = with_message_history.invoke(
    [HumanMessage(content="Whats my name")],
    config=config1
)
response.content

### Prompt templates
# Prompt Templates help to turn raw user information into a format that the LLM can work with. 
# In this case, the raw user input is just a message, which we are passing to the LLM. 
# Let's now make that a bit more complicated. First, let's add in a system message with some 
# custom instructions (but still taking messages as input). Next, we'll add in more input besides just the messages.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant.Amnswer all the question to the nest of your ability"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | model

chain.invoke({"messages": [HumanMessage(content="Hi My name is Krish")]})

with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "chat3"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi My name is Krish")],
    config=config
)

response

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

response.content

## Add more complexity
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

response = chain.invoke({"messages": [HumanMessage(content="Hi My name is Krish")], "language": "Hindi"})
response.content

# Let's now wrap this more complicated chain in a Message History class. 
# This time, because there are multiple keys in the input, we need to specify the correct key 
# to use to save the chat history.

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

config = {"configurable": {"session_id": "chat4"}}
repsonse = with_message_history.invoke(
    {'messages': [HumanMessage(content="Hi,I am Krish")], "language": "Hindi"},
    config=config
)
repsonse.content

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="whats my name?")], "language": "Hindi"},
    config=config,
)

response.content

### Managing the Conversation History
# One important concept to understand when building chatbots is how to manage conversation history. 
# If left unmanaged, the list of messages will grow unbounded and potentially overflow the context window of the LLM. 
# Therefore, it is important to add a step that limits the size of the messages you are passing in.
# 'trim_messages' helper to reduce how many messages we're sending to the model. 
# The trimmer allows us to specify how many tokens we want to keep, along with other parameters 
# like if we want to always keep the system message and whether to allow partial messages

from langchain_core.messages import SystemMessage, trim_messages
trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)
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
trimmer.invoke(messages)

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

response = chain.invoke(
    {
    "messages": messages + [HumanMessage(content="What ice cream do i like")],
    "language": "English"
    }
)
response.content

response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what math problem did i ask")],
        "language": "English",
    }
)
response.content

## Lets wrap this in the MEssage History
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)
config = {"configurable": {"session_id": "chat5"}}

response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)

response.content

response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="what math problem did i ask?")],
        "language": "English",
    },
    config=config,
)

response.content