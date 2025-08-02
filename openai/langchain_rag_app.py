#!/usr/bin/env python
# coding: utf-8

"""
Simple Gen AI Application Using LangChain

This script demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline:
1. Loads documents from a web source
2. Processes and chunks the documents
3. Creates vector embeddings and stores them
4. Sets up a retrieval chain for question answering
5. Provides a simple interface to query the documents

Key Components:
- Document Loading: WebBaseLoader
- Text Splitting: RecursiveCharacterTextSplitter 
- Embeddings: OpenAIEmbeddings
- Vector Store: FAISS
- LLM: OpenAI's GPT-4
- Retrieval Chain: Combines retrieval and generation
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure API keys and LangSmith tracking
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable LangSmith tracing
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")  # Project name in LangSmith

# ----------------------
# DATA INGESTION PIPELINE
# ----------------------

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def setup_data_pipeline():
    """
    Sets up the document processing pipeline:
    1. Loads documents from a URL
    2. Splits them into chunks
    3. Creates vector embeddings
    4. Stores them in a vector database
    
    Returns:
        FAISS: Vector store with document embeddings
    """
    # Load documents from web URL
    print("Loading documents from web...")
    loader = WebBaseLoader("https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")
    docs = loader.load()
    
    # Split documents into chunks wit overlap
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Size of each chunk in characters
        chunk_overlap=200,    # Overlap between chunks for context preservation
        separators=["\n\n", "\n", " ", ""]  # Text splitting boundaries
    )
    documents = text_splitter.split_documents(docs)
    
    # Create embeddings and vector store
    print("Creating vector embeddings...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

# ----------------------
# RETRIEVAL & GENERATION
# ----------------------

from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

def setup_retrieval_chain(vectorstore):
    """
    Creates a retrieval-augmented generation chain:
    1. Sets up LLM (GPT-4)
    2. Creates prompt template
    3. Combines document chain with retriever
    
    Args:
        vectorstore: FAISS vector store with document embeddings
    
    Returns:
        Runnable: Configured retrieval chain
    """
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o")
    
    # Create prompt template for contextual question answering
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        
        Question: {input}"""
    )
    
    # Chain that processes documents with the LLM
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retriever from vector store
    retriever = vectorstore.as_retriever()
    
    # Combine retriever and document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# ----------------------
# MAIN APPLICATION FLOW
# ----------------------

def main():
    """
    Main execution flow:
    1. Sets up data pipeline
    2. Configures retrieval chain
    3. Processes sample queries
    """
    # Set up data processing pipeline
    print("\nSetting up data pipeline...")
    vectorstore = setup_data_pipeline()
    
    # Configure retrieval chain
    print("Configuring retrieval chain...")
    retrieval_chain = setup_retrieval_chain(vectorstore)
    
    # Sample queries
    queries = [
        "What are the two types of usage limits in LangSmith?",
        "How can I set usage limits in LangSmith?",
        "What optimization techniques does LangSmith offer?"
    ]
    
    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        response = retrieval_chain.invoke({"input": query})
        print(f"Answer: {response['answer']}")
        print("\nRelevant Context:")
        for i, doc in enumerate(response['context'], 1):
            print(f"\nDocument {i} (Source: {doc.metadata['source']}):")
            print(doc.page_content[:300] + "...")  # Print first 300 chars of each doc

if __name__ == "__main__":
    main()



## Complete Flow:-
'''
1. Prompt Template Creation

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    
    Question: {input}"""
)
What This Does:

Creates a structured template for how the LLM should process the question and context

The template has two variables:

{context}: Will be filled with retrieved documents

{input}: Will contain the user's question

Why It's Important:

Forces the LLM to only use the provided context (prevents hallucination)

Clearly separates context from question for better model understanding

The XML-like <context> tags help the model identify document boundaries

Example Resulting Prompt:

Answer the following question based only on the provided context:
<context>
[Document 1 content about LangSmith limits...]
[Document 2 content about usage graphs...]
</context>

Question: What are LangSmith's usage limits?


2. Document Chain Creation

document_chain = create_stuff_documents_chain(llm, prompt)
What This Does:

Creates a chain that:

Takes a list of documents

"Stuffs" them into the prompt's {context} variable

Passes the formatted prompt to the LLM

Returns the LLM's response

Key Characteristics:

"Stuffing" means concatenating all relevant documents together

Handles document formatting automatically

Manages the token limit constraints

Visual Flow:

Documents → [Format into context] → [Combine with question] → LLM → Answer

3. Retriever Setup

retriever = vectorstore.as_retriever()

What This Does:

Creates a search interface for the vector store

Can be configured with:

search_type ("similarity", "mmr", etc.)

search_kwargs (like k=4 for number of docs to retrieve)

Default Behavior:

Uses similarity search

Returns top 4 most relevant documents by default

Maintains document metadata

4. Retrieval Chain Assembly

retrieval_chain = create_retrieval_chain(retriever, document_chain)
What This Creates:

User Question → [Retriever] → Relevant Docs → [Document Chain] → LLM Answer
Detailed Execution Flow:

User submits a question ("What are LangSmith's usage limits?")

Retriever:

Converts question to embedding vector

Finds most similar document chunks in FAISS index

Returns top k relevant document snippets

Document Chain:

Inserts these documents into the prompt template

Formats the complete prompt with question

Sends to GPT-4 for processing

LLM:

Analyzes question in context of provided docs

Generates answer strictly from the context

Returns formatted response
'''