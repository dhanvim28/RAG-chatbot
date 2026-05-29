import streamlit as st
import os
import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.openai import OpenAI

# Setup a temporary directory to store uploaded files
TEMP_DIR = "./uploaded_files"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

st.set_page_config(page_title="Universal RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Universal RAG Chatbot")
st.subheader("Feed me websites or data files, then ask questions!")

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
        
        # Clear out any old files from previous runs
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                st.error(f"Error cleaning temp directory: {e}")

        # Save freshly uploaded files to the temp directory
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(TEMP_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.spinner("Parsing uploaded files..."):
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

        # Build Index if documents exist
        if documents:
            with st.spinner("Building Knowledge Base Index..."):
                index = VectorStoreIndex.from_documents(documents)
                
                # Configure the persistent chat engine
                st.session_state.chat_engine = index.as_chat_engine(
                    chat_mode="condense_plus_context",
                    llm=OpenAI(model="gpt-4o", temperature=0.2),
                    context_prompt=(
                        "You are a helpful assistant. You have access to custom data files and websites provided by the user.\n"
                        "Always answer queries using the provided context. If the answer cannot be found in the context, "
                        "say 'I cannot find that in the provided data.' Do not make things up.\n"
                        "Context:\n{context_str}"
                    )
                )
            st.success("Brain successfully built! Start chatting on the right. 👉")
        else:
            st.warning("Please provide at least one file or website URL.")

# --- MAIN SCREEN: CHAT INTERFACE ---

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user interaction
if user_query := st.chat_input("Ask something about your data..."):
    # Render user message instantly
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Generate bot response using the RAG engine
    with st.chat_message("assistant"):
        if st.session_state.chat_engine is None:
            response_text = "⚠️ Please feed me data using the sidebar and click 'Build Brain' before asking questions."
            st.markdown(response_text)
        else:
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(user_query)
                response_text = response.response
                st.markdown(response_text)
                
    st.session_state.messages.append({"role": "assistant", "content": response_text})