import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

st.title("RAG Chatbot!")

#Setup a session state variable to hold all the history messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_name = "./Overview of LLM.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    #create Chunks of PDF using vector DB - chromadb
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name = 'all-MiniLM-L12-v2'), 
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)).from_loaders(loaders)
    return index.vectorstore

prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message('user').markdown(prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role': 'user','content': prompt })
    
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, 
                                            the most accurate and most precise answers. Answer the following Question: {user_prompt}.
                                            Start the answer directly. No small talk please""")
    
    # Use Groq as LLM provider
    llm = ChatGroq(
        groq_api_key = os.getenv("GROQ_API_KEY"),
        model_name = "llama3-8b-8192" ,
        temperature=0.5
    )
    
    try:
        vectorstore =get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load document")

        chain = RetrievalQA.from_chain_type(
            llm, 
            chain_type='stuff', 
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True)
        
        result = chain({"query": prompt})
        response = result["result"] #Extract the answer
        
    except Exception as e: 
        st.error(f"Error: {str(e)}")

    #chain = groq_sys_prompt | llm | StrOutputParser()
    #response = chain.invoke({"user_prompt": prompt})
    #response = "I am your assistant "
    st.chat_message('assistant').markdown(response)
    # Store the user prompt in state
    st.session_state.messages.append(
        {'role': 'assistant','content': response })
    
    
    