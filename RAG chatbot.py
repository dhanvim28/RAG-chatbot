import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.web import SimpleWebPageReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import MockLLM  # <--- FIX: Import the free placeholder LLM

# 1. SETUP FREE LOCAL EMBEDDING MODEL (Runs on the server for free)
@st.cache_resource
def load_free_embed_model():
    # Downloads a tiny, powerful open-source mathematical mapping model (~100MB)
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Assign the free models globally so LlamaIndex never triggers OpenAI
Settings.embed_model = load_free_embed_model()
Settings.llm = MockLLM()  # <--- FIX: Forces LlamaIndex to skip checking for OpenAI keys!

# Setup a temporary directory to store uploaded files
TEMP_DIR = "./uploaded_files"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

st.set_page_config(page_title="Free Universal RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Free Universal RAG Chatbot")
st.subheader("No API Keys needed! Feed me websites or data files, then ask questions.")

# Initialize session state for chat history and engine
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

# --- SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("📁 Source Material")
    
    # 1. File Upload Section
    uploaded_files = st.file_uploader(
        "Upload data files (PDF, TXT, CSV, DOCX, etc.)", 
        accept_multiple_files=True
    )
    
    # 2. Website URL Section
    urls_input = st.text_area(
        "Paste Website URLs (one per line)", 
        placeholder="https://example.com\nhttps://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    )
    
    # Process Button
    if st.button("Build Brain 🧠", use_container_width=True):
        documents = []
        
        # Clear out old files
        if os.path.exists(TEMP_DIR):
            for filename in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.error(f"Error cleaning temp directory: {e}")

        # Save files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(TEMP_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.spinner("Parsing uploaded files locally..."):
                file_docs = SimpleDirectoryReader(TEMP_DIR).load_data()
                documents.extend(file_docs)
        
        # Parse URLs
        if urls_input.strip():
            urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
            with st.spinner("Scraping websites..."):
                try:
                    web_docs = SimpleWebPageReader(html_to_text=True).load_data(urls)
                    documents.extend(web_docs)
                except Exception as e:
                    st.error(f"Error scraping websites: {e}")

        # Build Index using the free embed model
        if documents:
            with st.spinner("Building Knowledge Base Index..."):
                index = VectorStoreIndex.from_documents(documents)
                
                # Convert into a context retriever (gathers the top 3 closest chunks)
                st.session_state.chat_engine = index.as_query_engine(similarity_top_k=3)
            st.success("Brain successfully built! Ask your questions on the right. 👉")
        else:
            st.warning("Please provide at least one file or website URL.")

# --- MAIN SCREEN: CHAT INTERFACE ---

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user interaction
if user_query := st.chat_input("Ask something about your data..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        if st.session_state.chat_engine is None:
            response_text = "⚠️ Please feed me data using the sidebar and click 'Build Brain' before asking questions."
            st.markdown(response_text)
        else:
            with st.spinner("Searching matching document chunks..."):
                # Pulls relevant passages right out of your PDFs/websites instantly
                response = st.session_state.chat_engine.query(user_query)
                
                # Format a friendly layout displaying the exact source text found
                response_text = f"### Found Information:\n"
                
                # Display individual context nodes extracted from documents
                if hasattr(response, "source_nodes") and response.source_nodes:
                    for i, node in enumerate(response.source_nodes):
                        file_info = node.node.metadata.get('file_name', 'Web/Unknown Source')
                        response_text += f"**Source {i+1} (From: {file_info}):**\n>{node.node.get_content()}\n\n"
                else:
                    response_text += "No matching information chunks found in your documents."
                    
                st.markdown(response_text)
                
    st.session_state.messages.append({"role": "assistant", "content": response_text})
