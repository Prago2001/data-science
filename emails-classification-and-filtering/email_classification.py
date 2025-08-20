from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.output_parsers.json import SimpleJsonOutputParser
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


def create_context(result: list[Document], start: int, end: int):
    """
    Create a context for the GPT model
    """
    print(f"Creating context for emails from {start} to {end}")
    context = ""
    for i, result_doc in enumerate(result[start:end], 1):
        context += f"Email {i}\n\n"
        context += f"Sender: {result_doc.metadata['sender']}\n"
        context += f"Date: {result_doc.metadata['date']}\n"
        context += f"Subject: {result_doc.metadata['subject']}\n"
        context += f"{result_doc.page_content}\n\n"
    return context


if __name__ == "__main__":
    store: Chroma = None
    if os.path.exists("./emails.db") is True:
        store = create_vector_store()
    else:
        documents = create_documents()
        store = create_vector_store(documents)

    result = search_vector_store(
        store, "Text saying applied to a job opening successfully", 200
    )

    # Initialize a GPT model
    model = OllamaLLM(model="llama3.1:8b", verbose=True)

    template = """
    The below context is a text containing email messages. Give a response in JSON format only. \n\n.{context}
    If you don't know the answer, just say that you don't know, don't try to make up an answer. \n\nQuestion: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    json_parser = SimpleJsonOutputParser()

    chain = prompt | model | json_parser
    question = """
    Strictly return a JSON object containing the list of all companies which Pratik has applied to. An object in the JSON object should contain the following keys - Sender, Date, Subject, Company Name.
    If company name can't be determined then just return an empty string as it's value.
    """

    result_answers = []
    for split_context_start in range(0, 191, 10):
        try:
            context = create_context(
                result, split_context_start, split_context_start + 10
            )
            answer = chain.invoke({"question": question, "context": context})

            result_answers.extend(answer)
        except Exception as e:
            print(
                f"Error while creating JSON object for the context {split_context_start} to {split_context_start + 10}"
            )
            print(str(e))

    with open("results.json", "w") as result_file:
        json.dump(result_answers, result_file)
