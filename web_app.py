
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env")
    st.stop()

st.set_page_config(page_title="PDF Chat AI", layout="centered")
st.title("📄 PDF AI Chatbot")
st.write("Upload your PDF and ask questions!")

def process_pdf(uploaded_file):
    temp_pdf = "temp_uploaded.pdf"
    with open(temp_pdf, "wb") as f:
        f.write(uploaded_file.getbuffer())

    docs = PyPDFLoader(temp_pdf).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    ).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file:

    st.sidebar.success("✅ PDF Uploaded")

    if "retriever" not in st.session_state:
        with st.spinner("Processing PDF..."):
            st.session_state.retriever = process_pdf(uploaded_file)
        st.sidebar.success("🧠 PDF Indexed Successfully")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply:
"I couldn't find the answer in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
""")
    
    

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": st.session_state.retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    st.divider()
    st.subheader("💬 Chat")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question...")

    if question:

        with st.chat_message("user"):
            st.write(question)

        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.invoke(question)
                    st.write(response)
                except Exception as e:
                    st.exception(e)
                    response = f"Error: {e}"

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )

else:
    st.info("👈 Upload a PDF from the sidebar.")

    st.session_state.pop("retriever", None)
    st.session_state.pop("chat_history", None)
