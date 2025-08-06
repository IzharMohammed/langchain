from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness",
        metadata={"source":"mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space",
        metadata={"source":"mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are r popular pets for beginners, requiring relatively simple care",
        metadata={"source":"fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech",
        metadata={"source":"birds-pets-doc"},
    )
]

# print(documents)

## Vector stores
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
groq_api_key=os.getenv("GROQ_API_KEY")

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

llm=ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192")
print(f"llm:- {llm}")

vectorStore = Chroma.from_documents(documents,embedding=embeddings)
print(f"vector store:- {vectorStore}")

similaritySearch = vectorStore.similarity_search("cat")
print(f"similarity search:- {similaritySearch}")

## Async query
similaritySearch = vectorStore.asimilarity_search("cat")
print(f"similarity search async:- {similaritySearch}")

similaritySearch = vectorStore.similarity_search_with_score("cat")
print(f"similarity search with score:- {similaritySearch}")

## Retrievers