from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

data = pd.read_csv("MovieDescriptions.csv")
embeddings = OllamaEmbeddings(model = "mxbai-embed-large")
data_location = "chroma_langchain_db"
add_docs = not os.path.exists(data_location)

if add_docs:
    documents = []
    ids = []

    limit = 100
    counter = 0

    for i, row in data.iterrows():
        counter += 1
        if counter > limit:
            break
        document = Document(
            page_content = row["title"] + " " + row["desc"],
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name = "movie_descriptions",
    persist_directory = data_location,
    embedding_function = embeddings
)

if add_docs:
    vector_store.add_documents(documents = documents, ids = ids)

retriever = vector_store.as_retriever(search_kwargs = {"k":3})