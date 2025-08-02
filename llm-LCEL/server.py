# LangChain Translation Server with LCEL (LangChain Expression Language)
# This application creates a FastAPI server that provides translation services using Groq's LLM

# Import necessary components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Groq API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq LLM model
# Gemma2-9b-It is Google's open-source language model hosted on Groq's infrastructure
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# === STEP 1: CREATE PROMPT TEMPLATE ===
# This template defines how we structure our translation requests
system_template = "Translate the following into {language}:"

# ChatPromptTemplate creates a structured prompt with system and user messages
# System message: Sets the context/instruction for the AI
# User message: Contains the actual text to be translated
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),  # System instruction with language parameter
    ('user', '{text}')           # User input with text parameter
])

# === STEP 2: CREATE OUTPUT PARSER ===
# StrOutputParser converts the model's response into a simple string
# This removes any metadata and returns just the translated text
parser = StrOutputParser()

# === STEP 3: CREATE LCEL CHAIN ===
# LCEL (LangChain Expression Language) allows chaining components using the | operator
# This creates a pipeline: prompt → model → parser
chain = prompt_template | model | parser

# === STEP 4: CREATE FASTAPI APPLICATION ===
# Define the FastAPI application with metadata
app = FastAPI(
    title="Langchain Server",
    version="1.0", 
    description="A simple API server using Langchain runnable interfaces"
)

# === STEP 5: ADD CHAIN ROUTES ===
# add_routes automatically creates REST API endpoints for the chain
# This creates endpoints like POST /chain/invoke, POST /chain/stream, etc.
add_routes(
    app,
    chain,
    path="/chain"
)

# === STEP 6: RUN THE SERVER ===
# Start the server when script is run directly
if __name__ == "__main__":
    import uvicorn
    # Run FastAPI server on localhost:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)

"""
=== WHAT IS LCEL (LangChain Expression Language)? ===

LCEL is a declarative way to compose LangChain components into chains using the pipe operator (|).
It provides several key benefits:

1. **Streaming Support**: Automatically supports streaming responses
2. **Async Support**: Built-in async/await support for better performance
3. **Parallelization**: Can run components in parallel when possible
4. **Fallbacks**: Easy error handling and fallback mechanisms
5. **Tracing**: Built-in observability and debugging capabilities

=== APPLICATION FLOW EXPLANATION ===

1. **Initialization Phase**:
   - Load environment variables (API keys)
   - Initialize Groq LLM model
   - Create prompt template with placeholders
   - Set up output parser
   - Chain components using LCEL

2. **Request Processing Flow**:
   When a user makes a POST request to /chain/invoke with JSON like:
   {"input": {"language": "Spanish", "text": "Hello world"}}

   The flow is:
   Step 1: prompt_template receives {"language": "Spanish", "text": "Hello world"}
   Step 2: Template formats: "Translate the following into Spanish: Hello world"
   Step 3: Formatted prompt sent to Groq model
   Step 4: Model generates translation
   Step 5: StrOutputParser extracts just the text response
   Step 6: Final result returned to user

3. **Available Endpoints** (created automatically by add_routes):
   - POST /chain/invoke - Single translation request
   - POST /chain/batch - Multiple translations at once
   - POST /chain/stream - Streaming translation response
   - GET /chain/config - Chain configuration info

=== EXAMPLE USAGE ===

curl -X POST "http://127.0.0.1:8000/chain/invoke" \
     -H "Content-Type: application/json" \
     -d '{"input": {"language": "French", "text": "Good morning"}}'

Response: "Bonjour"

=== ADVANTAGES OF THIS ARCHITECTURE ===

1. **Modular**: Each component (prompt, model, parser) is separate and reusable
2. **Scalable**: FastAPI provides high-performance async handling
3. **Observable**: LCEL provides built-in tracing and monitoring
4. **Flexible**: Easy to modify any component without affecting others
5. **Production-Ready**: Includes proper error handling and API documentation
"""