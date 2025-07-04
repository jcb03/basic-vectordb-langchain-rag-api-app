import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from lanchain.chains import RetrievalQA

def get_embeddings_model():
    """Initialize and return OpenAI embeddings."""
    return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

def get_llm():
    """Initialize and return the OpenAI language model."""
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text, chunk_size=300):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return splitter.split_text(text)

def get_vectorstore():
    return Chroma(
        persist_directory="./chroma_DB",
        embedding_funciton=get_embeddings_model()
    )

def add_to_vectorstore(chunks, ids):
    vectordb = get_vectorstore()
    docs = [{"page_content": chunk, "metadata": {"id": id_}} for chunk, id_ in zip(chunks, ids)]
    vectordb.add_documents(docs)
    vectordb.persist()

def get_retriever():
    vectordb = get_vectorstore()
    return vectordb.as_retriever()