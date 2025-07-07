import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

loader=TextLoader("anime.txt")
data=loader.load()
print(data)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
splits=text_splitter.split_documents(data)

print(json.dumps([{
    "content": split.page_content,
    "metadata": split.metadata,
    "length": len(split.page_content)
} for split in splits], indent=2))

embedding=OllamaEmbeddings(model="gemma:2b")
vectordb=Chroma.from_documents(documents=splits,embedding=embedding)
print("vectordb",vectordb)