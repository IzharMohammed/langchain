import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text="I am izhar studying btech 4h year, computer science from kurnool,Andhra Pradesh"
query_result=embeddings.embed_query(text)

print(query_result)

print("length",len(query_result))

print("**************************************")

doc_result=embeddings.embed_documents([text,"This is not a test document"])

print(doc_result[0])