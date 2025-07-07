from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader=TextLoader("anime.txt")
documents=loader.load()
text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30)
docs=text_splitter.split_documents(documents)

# print(docs)

embeddings=OllamaEmbeddings(model="gemma:2b")
db=FAISS.from_documents(docs,embeddings)
# print(db)

## querying
query = "Tell me about the main story and theme of Bleach anime."
docs=db.similarity_search(query)
print(docs[0].page_content)

'''
As a Retriever

we can also convert the vectorstore into a Retriever class. This allows us to easily use it
in other Langchain methods, which largely work with retrievers
'''
retriever=db.as_retriever()
docs = retriever.invoke(query)
print(docs[0].page_content)

## Similarity search with score
docs_and_score=db.similarity_search_with_score(query)
print(docs_and_score)

## Similarity search with vector 
print("************************************")
embeddings_vector=embeddings.embed_query(query)
print(embeddings_vector)

docs_score=db.similarity_search_by_vector(embeddings_vector)
print(docs_score)

print("***********************************")
## Saving and Loading
db.save_local("faiss_index")

new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
docs=new_db.similarity_search(query)
print("here",docs)