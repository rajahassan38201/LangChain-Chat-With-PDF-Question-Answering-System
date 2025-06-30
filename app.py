import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import tempfile
import shutil

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini Model and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Streamlit UI Configuration
st.set_page_config(page_title="RAG System - PDF Q&A", layout="centered")
st.title("📄 PDF Question Answering System Using LangChain + Chroma DB")

# Initialize session state for vector store and QA chain
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.session_state.previous_file = None

# Upload and Process PDF File
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    if st.session_state.previous_file != uploaded_file.name:
        st.session_state.previous_file = uploaded_file.name
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
        st.session_state.answer = None

    if st.session_state.vectorstore is None:
        st.success("PDF uploaded successfully and processing started!")

        # Save PDF to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load and split PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)

        # Use Chroma as vector store
        chroma_dir = tempfile.mkdtemp()
        vectorstore = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=chroma_dir)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Save to session state
        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = qa_chain

# Display QA Interface
if st.session_state.qa_chain:
    st.subheader("Ask a question from the PDF")
    user_question = st.text_input("Enter your question")

    if user_question:
        with st.spinner("Generating concise answer..."):
            result = st.session_state.qa_chain({"query": user_question})
            answer = result["result"].strip()
            st.session_state.answer = answer

    if st.session_state.answer:
        st.markdown("### Answer:")
        concise = ". ".join(st.session_state.answer.split(". ")[:5])
        st.write(concise + ("." if not concise.endswith(".") else ""))
else:
    st.info("Please upload a PDF file to get started.")
