# Langchain Chat With PDF

Certainly! Below is a **professional README file** for your project **Chat With PDF LangChain Application (RAG)** written in clear and easy English:

---

# 📄 Chat With PDF - LangChain RAG Application

## 🧠 Overview

**Chat With PDF** is an interactive question-answering application that allows users to upload PDF files and ask questions related to the content. The system uses **LangChain** and **Retrieval-Augmented Generation (RAG)** techniques powered by Large Language Models (LLMs) to provide accurate and relevant answers based on the uploaded documents.

---

## 🚀 Features

* Upload any PDF document and chat with its contents
* Ask questions in natural language
* Retrieves context from PDF using advanced embeddings
* Provides accurate answers with Retrieval-Augmented Generation (RAG)
* Easy-to-use Streamlit interface

---

## 🛠️ Technologies Used

* **Python 3.10+**
* **LangChain**
* **Google Generative AI**
* **Streamlit**
* **PyMuPDF (fitz)**
* **FAISS / Chroma for Vector Storage**

---

## 📁 Project Structure

```
chat-with-pdf-langchain/
│
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API Keys)
└── README.md                # Project documentation
```

---

## ⚙️ Installation

1. **Clone the Repository**

```bash
git clone https://github.com/rajahassan38201/LangChain-Chat-With-PDF-Question-Answering-System.git
cd chat-with-pdf-langchain
```

2. **Create Virtual Environment (Optional but Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Set Environment Variables**
   Create a `.env` file and add your API key:

```
GOOGLE_API_KEY=your_google_genai_key_here
```

---

## ▶️ Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Upload a PDF document

3. Ask any question related to the content of the PDF

---

## 🧠 How It Works

* The uploaded PDF is split into text chunks
* Each chunk is converted into a vector using embeddings
* These vectors are stored in a vector database FAISS
* When a user asks a question, the most relevant chunks are retrieved
* The LLM answers the question using both the question and retrieved context

---

## 📌 Use Cases

* Research document summarization
* Legal or policy document exploration
* Technical manual support
* Education and learning tools

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

---


