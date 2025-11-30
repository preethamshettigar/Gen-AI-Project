# LangChain RAG Chatbot 🤖

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, and **Groq**. This application allows users to chat interactively with a specific PDF document ("Overview of LLM.pdf"), leveraging the power of the **Llama 3** model to answer questions based on the document's content.

## 🚀 Features

* **Interactive Chat Interface**: Built using Streamlit for a clean, chat-like experience.
* **RAG Architecture**: Retrieves relevant context from the PDF document to ground the LLM's responses.
* **High-Performance LLM**: Uses Groq's API to access the `llama3-8b-8192` model for fast and accurate inference.
* **Vector Embeddings**: Utilizes `HuggingFaceEmbeddings` (`all-MiniLM-L12-v2`) for efficient semantic search.
* **Session Memory**: Maintains chat history within the active session.

## 🛠️ Tech Stack

* **Python 3.8+**
* **Streamlit**: Frontend UI.
* **LangChain**: Framework for orchestration and RAG logic.
* **Groq**: LLM inference provider.
* **ChromaDB**: (Implicitly used by `VectorstoreIndexCreator`) for vector storage.
* **HuggingFace**: For embedding models.

## 📋 Prerequisites

Before running the application, ensure you have the following:

1. **Groq API Key**: Sign up at [Groq Console](https://console.groq.com/) to get your free API key.
2. **PDF Document**: Ensure you have a file named `Overview of LLM.pdf` or upload any PDF in the root directory.
