import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document


def get_embeddings_model():
    """Initialize and return OpenAI embeddings."""
    return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

def get_llm():
    """Initialize and return the OpenAI language model."""
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text):
    # Split by paragraphs (double newlines)
    return [chunk.strip() for chunk in text.split('\n') if chunk.strip()]


def get_vectorstore():
    return Chroma(
        persist_directory="./chroma_db",  # Use consistent, lowercase directory name
        embedding_function=get_embeddings_model()  # Correct argument name
    )

def add_to_vectorstore(chunks, ids):
    from langchain_core.documents import Document
    vectordb = get_vectorstore()
    docs = [Document(page_content=chunk, metadata={"id": id_}) for chunk, id_ in zip(chunks, ids)]
    vectordb.add_documents(docs)
    vectordb.persist()


def get_retriever():
    vectordb = get_vectorstore()
    return vectordb.as_retriever()
