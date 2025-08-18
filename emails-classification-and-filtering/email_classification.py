from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import os, json, sys


def create_documents():
    """
    Read all json files and create a list of `Documents`
    each containing a json file
    """
    documents: list[Document] = []

    if os.path.exists("./emails") is True:
        email_json_list = os.listdir("./emails")
        for i, file_name in enumerate(email_json_list, 1):
            with open(f"./emails/{file_name}", "r") as f:
                json_file = json.load(f)
                documents.append(
                    Document(
                        page_content=json_file["body"],
                        metadata={
                            "source": file_name,
                            "subject": json_file["subject"],
                            "sender": json_file["sender"],
                            "date": json_file["date"],
                        },
                    )
                )
            if i % 50 == 0:
                print(f"Created document for {i}'th file")
        return documents
    else:
        print("./emails directory not found")
        sys.exit(1)


def create_vector_store(documents: list[Document] | None = None):
    """
    Create vector store and store embedded documents
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma(
        collection_name="emails_collection",
        embedding_function=embeddings,
        persist_directory="./emails.db",
    )
    if documents is not None:
        vector_store.add_documents(documents=documents)
    return vector_store


def search_vector_store(store: Chroma, query: str, num_results=100):
    """
    Search Vector store using a query
    """
    return store.similarity_search(query=query, k=num_results)


if __name__ == "__main__":
    store: Chroma = None
    if os.path.exists("./emails.db") is True:
        store = create_vector_store()
    else:
        documents = create_documents()
        store = create_vector_store(documents)

    result = search_vector_store(store, "Job Application", 200)
    result_json = []
    for result_doc in result:
        result_json.append({
            "page_content": result_doc.page_content,
            "metadata": result_doc.metadata
        })

    with open("results.json", "w") as result_file:
        json.dump(result_json, result_file)
