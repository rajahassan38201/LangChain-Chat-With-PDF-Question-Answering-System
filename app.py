import os
import tempfile
import shutil
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Patch numpy for ChromaDB compatibility with Python 3.13+
if not hasattr(np, "uint"):
    np.uint = np.uint32

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate API Key
if not GOOGLE_API_KEY:
    st.error("❌ Google API Key is missing. Please check your environment variables.")
    st.stop()

# Initialize LLM and Embedding Models
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY
)
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Configure Streamlit UI
st.set_page_config(page_title="RAG System - PDF Q&A", layout="centered")
st.title("📄 PDF Question Answering with LangChain & ChromaDB")

# Initialize Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.session_state.previous_file = None
    st.session_state.answer = None

# Upload and Process PDF
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file:
    if st.session_state.previous_file != uploaded_file.name:
        st.session_state.previous_file = uploaded_file.name
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
        st.session_state.answer = None

    if st.session_state.vectorstore is None:
        st.success("✅ PDF uploaded successfully. Processing now...")

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load and Split Text
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # Create Chroma Vector Store
        chroma_dir = tempfile.mkdtemp()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=chroma_dir
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Build Retrieval QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        # Save in session
        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = qa_chain

# Question Answering UI
if st.session_state.qa_chain:
    st.subheader("🔍 Ask a question about the PDF")
    user_question = st.text_input("Enter your question:")

    if user_question:
        with st.spinner("💬 Generating answer..."):
            response = st.session_state.qa_chain({"query": user_question})
            answer = response["result"].strip()
            st.session_state.answer = answer

    if st.session_state.answer:
        st.markdown("### ✅ Answer:")
        concise = ". ".join(st.session_state.answer.split(". ")[:5])
        st.write(concise + ("." if not concise.endswith(".") else ""))
else:
    st.info("📎 Please upload a PDF file to get started.")
