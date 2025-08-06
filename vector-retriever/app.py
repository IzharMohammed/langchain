from langchain_core.documents import Document

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

print(documents)
## Vector stores
from langchain_chroma import Chroma
