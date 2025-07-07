from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader=TextLoader("anime.txt")
documents=loader.load()
text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30)
docs=text_splitter.split_documents(documents)

print(docs)