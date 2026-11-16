import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="PDF Chat AI", layout="centered")
st.title("📄 PDF AI Chatbot")
st.write("Upload your PDF and ask questions!")

@st.cache_resource
def get_embeddings():
    """Create and cache the HuggingFace embeddings model.

    Caching avoids re‑loading the model on every interaction, which speeds up
    subsequent PDF uploads.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def process_pdf(uploaded_file):
    """Load a PDF, split it into chunks, embed the chunks and return a retriever.

    The uploaded file is written to a temporary location on disk because
    ``PyPDFLoader`` expects a file path. After the vector store is built, the
    temporary file is removed to keep the container clean.
    """
    temp_path = "temp_uploaded.pdf"
    try:
        # Write the uploaded bytes to a temporary file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load, split and embed the document
        docs = PyPDFLoader(temp_path).load()
        splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=get_embeddings())
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    finally:
        # Ensure the temporary file is removed even if an error occurs
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                # Silently ignore cleanup errors; they are non‑critical
                pass

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file:
    if "retriever" not in st.session_state:
        with st.spinner("Processing PDF..."):
            st.session_state.retriever = process_pdf(uploaded_file)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant.
Answer ONLY using the provided context.
If the answer is not present in the context, reply:
'I couldn't find the answer in the uploaded PDF.'

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    q = st.chat_input("Ask a question...")
    if q:
        with st.chat_message("user"):
            st.write(q)
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            try:
                ans = rag_chain.invoke(q)
                st.write(ans)
            except Exception as e:
                st.exception(e)
                ans = f"Error: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
else:
    st.info("👈 Upload a PDF from the sidebar.")
