"""
RAG (Retrieval-Augmented Generation) Application with LangChain

This script demonstrates:
1. Document ingestion and vector storage using ChromaDB
2. Semantic search capabilities
3. Question answering using Groq's Llama3 model
4. Both synchronous and asynchronous operations

Flow of the Application:
1. Document Ingestion -> 2. Vector Embedding -> 3. Storage -> 4. Retrieval -> 5. Generation
"""

from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# ========== DOCUMENT SETUP ==========
# Create sample documents about pets
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc", "animal": "dog"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc", "animal": "cat"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc", "animal": "goldfish"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "birds-pets-doc", "animal": "parrot"},
    )
]

print("\n=== Sample Documents ===")
for doc in documents:
    print(f"\nContent: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")

# ========== VECTOR STORE SETUP ==========
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

"""
What's happening here:
- Convert text to numerical vectors (embeddings)
- Store in ChromaDB for fast similarity search
- This becomes our "searchable knowledge base"
"""

# Initialize embeddings (using a small efficient model)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create Chroma vector store in memory
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="pets-collection"
)

print("\n=== Vector Store Created ===")
print(f"Number of documents stored: {vector_store._collection.count()}")

# ========== LLM SETUP ==========
from langchain_groq import ChatGroq

# Initialize Groq's Llama3 model
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="Llama3-8b-8192",
    temperature=0.5
)

print("\n=== LLM Initialized ===")
print(f"Model: {llm.model_name}")

# ========== RAG CHAIN SETUP ==========
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

"""
RAG Chain Explained:
1. Retrieval: Finds relevant documents (vector_store.as_retriever())
2. Augmentation: Combines question + context in prompt
3. Generation: LLM produces final answer

The '|' symbols chain these steps together into a pipeline
"""
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

# Build the pipeline:
rag_chain = (
    # Step 1: Retrieve relevant docs and pass through question
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()} 
    | ChatPromptTemplate.from_template(template)  # Step 2: Format prompt
    | llm  # Step 3: Generate answer
    | StrOutputParser()  # Clean output
)

# ========== QUESTION ANSWERING ==========
def answer_question(question: str):
    """Helper function to answer questions using RAG"""
    print(f"\nQuestion: {question}")
    
    # Get relevant documents
    relevant_docs = vector_store.similarity_search(question, k=1)
    print("\n=== Relevant Documents ===")
    for doc in relevant_docs:
        print(f"\nContent: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
    
    # Generate answer
    answer = rag_chain.invoke(question)
    print(f"\nAnswer: {answer}")
    return answer

# Example questions
questions = [
    "What are the characteristics of dogs?",
    "Which pets are good for beginners?",
    "What can parrots do?",
    "Tell me about cats"
]

# Answer questions
print("\n=== RAG DEMONSTRATION ===")
for question in questions:
    answer_question(question)

# ========== ASYNC OPERATIONS ==========
import asyncio

async def async_answer_question(question: str):
    """Async version of question answering"""
    print(f"\n[ASYNC] Question: {question}")
    
    # Async document retrieval
    relevant_docs = await vector_store.asimilarity_search(question, k=1)
    print("\n=== Relevant Documents (Async) ===")
    for doc in relevant_docs:
        print(f"\nContent: {doc.page_content}")
    
    # Async answer generation
    answer = await rag_chain.ainvoke(question)
    print(f"\nAnswer: {answer}")
    return answer

# Run async questions
async def run_async_demo():
    print("\n=== ASYNC RAG DEMONSTRATION ===")
    await async_answer_question("What makes cats different from dogs?")

asyncio.run(run_async_demo())

# ========== RETRIEVAL WITH SCORES ==========
print("\n=== SIMILARITY SEARCH WITH SCORES ===")
results_with_scores = vector_store.similarity_search_with_score("bird pets", k=2)
for doc, score in results_with_scores:
    print(f"\nScore: {score:.3f}")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")

"""
Expected Output Flow:
1. Shows the user's question
2. Displays which document chunks were retrieved
3. Presents the LLM's generated answer
4. All answers are grounded in the provided documents
"""

# ========== KEY RAG CONCEPTS ==========
"""
Why RAG?
- Combines the strengths of:
  * Retrieval: Precise, up-to-date information lookup
  * Generation: Natural language understanding

When to use RAG?
- When you need answers based on specific documents
- When your knowledge base changes frequently
- When you need traceability (can see source docs)
"""