from langchain_ollama import OllamaEmbeddings


embeddings=(
    OllamaEmbeddings(model="gemma:2b")
)

r1=embeddings.embed_documents([
    "i am izhar",
    "A final year student from a tier 3 college from Andhra Pradesh, kurnool"
])

print(r1[0])
print("Length of embeddings",len(r1[0]))

print(embeddings.embed_query("In which year is izhar ??"))