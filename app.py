import streamlit as st
import os
from dotenv import load_dotenv
from db import init_db, insert_chunk, get_chunks_by_ids
from vector_utils import chunk_text, add_to_vectorstore, get_retriever, get_llm
from langchain.chains import RetrievalQA

load_dotenv()
conn = init_db()

st.title("🔎 Minimal RAG Q&A App (LangChain + Chroma + SQLite)")

# 1. Upload and Index Documents
st.header("1. Upload Document")
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
if uploaded_file:
    text = uploaded_file.read().decode()
    chunks = chunk_text(text)
    title = uploaded_file.name
    ids = []
    for chunk in chunks:
        id_ = str(insert_chunk(conn, title, chunk))
        ids.append(id_)
    add_to_vectorstore(chunks, ids)
    st.success(f"Indexed {len(chunks)} chunks from {title}")

# 2. Ask a Question
st.header("2. Ask a Question")
question = st.text_input("Your question:")
if st.button("Get Answer") and question:
    retriever = get_retriever()
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain(question)
    st.write("**Answer:**", result["result"])
    st.write("---")
    st.write("**Context Chunks:**")
    for doc in result["source_documents"]:
        st.write(f"- {doc.page_content}")

# 3. View Indexed Chunks
st.header("3. View Indexed Chunks")
if st.button("Show all chunks"):
    c = conn.cursor()
    c.execute("SELECT id, title, chunk FROM docs")
    rows = c.fetchall()
    for row in rows:
        st.write(f"**{row[1]}** (ID: {row[0]}): {row[2]}")
