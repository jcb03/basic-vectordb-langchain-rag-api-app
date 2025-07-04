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


footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: #f0f2f6;
    color: #262730;
    text-align: center;
    padding: 10px 0 5px 0;
    font-size: 1.1rem;
    z-index: 100;
}
.footer a {
    margin: 0 10px;
    text-decoration: none;
    color: inherit;
}
.footer img {
    vertical-align: middle;
    height: 28px;
    width: 28px;
    margin-bottom: 2px;
}
</style>
<div class="footer">
    <a href="https://www.linkedin.com/in/jai-chaudhary-54bb86221/" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" />
    </a>
    <a href="https://github.com/jcb03?tab=repositories" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" />
    </a>
    <a href="https://learn.microsoft.com/en-us/users/jaichaudhary-6371/" target="_blank">
        <img src="https://learn.microsoft.com/favicon.ico" alt="Microsoft Learn" />
    </a>
    <span style="margin-left: 20px; font-weight: bold;">Jai Chaudhary</span>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
