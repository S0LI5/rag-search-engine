import streamlit as st
import os
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
import pypdf
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from chromadb.config import Settings

# Configuration
CHROMA_PATH = "chroma"
DATA_PATH = "data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Page config
st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def query_rag(query_text: str, top_k: int = 5):
    """Query the RAG system"""
    try:
        # Prepare the DB
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Search the DB
        results = db.similarity_search_with_score(query_text, k=top_k)
        
        if not results:
            return "No relevant information found in the knowledge base.", []
        
        # Build context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Create prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Get LLM response
        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)
        
        # Get sources
        sources = []
        for doc, score in results:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "score": f"{score:.4f}"
            })
        
        return response_text, sources
        
    except Exception as e:
        return f"Error: {str(e)}", []

def load_and_process_pdf(uploaded_file):
    """Load and process uploaded PDF"""
    try:
        # Save uploaded file temporarily
        temp_path = Path(DATA_PATH) / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load PDF
        loader = PyPDFLoader(str(temp_path))
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add to ChromaDB
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        
        # Calculate chunk IDs
        chunks_with_ids = calculate_chunk_ids(chunks)
        
        # Add to database
        new_chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
        db.add_documents(chunks_with_ids, ids=new_chunk_ids)
        db.persist()
        
        return True, len(chunks)
        
    except Exception as e:
        return False, str(e)

def calculate_chunk_ids(chunks):
    """Calculate unique IDs for chunks"""
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    
    return chunks

def get_database_stats():
    """Get database statistics"""
    if not os.path.exists(CHROMA_PATH):
        return None
    
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        results = db.get()
        
        # Count documents
        sources = set()
        for metadata in results["metadatas"]:
            sources.add(metadata.get("source", "Unknown"))
        
        return {
            "total_chunks": len(results["ids"]),
            "total_documents": len(sources),
            "documents": list(sources)
        }
    except:
        return None

# Main UI
st.title("🤖 RAG Knowledge Base Search")
st.markdown("Ask questions about your documents using AI-powered search")

# Sidebar
with st.sidebar:
    st.header("📚 Document Management")
    
    # Database stats
    stats = get_database_stats()
    if stats:
        st.metric("Total Documents", stats["total_documents"])
        st.metric("Total Chunks", stats["total_chunks"])
        
        with st.expander("📄 View Documents"):
            for doc in stats["documents"]:
                st.text(f"• {os.path.basename(doc)}")
    else:
        st.info("No documents in database yet")
    
    st.divider()
    
    # Upload section
    st.subheader("📤 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=['pdf'],
        help="Upload a PDF document to add to the knowledge base"
    )
    
    if uploaded_file:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                success, result = load_and_process_pdf(uploaded_file)
                
                if success:
                    st.success(f"✅ Added {result} chunks to database!")
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result}")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    top_k = st.slider("Number of sources", 1, 10, 5)
    
    # Clear database
    if st.button("🗑️ Clear Database", type="secondary"):
        if os.path.exists(CHROMA_PATH):
            try:
                # Initialize the client with settings that allow reset
                client = chromadb.PersistentClient(
                    path=CHROMA_PATH,
                    settings=Settings(allow_reset=True) # <--- ADD THIS LINE
                )
                client.reset()
                st.success("Database cleared successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing database: {e}")
        else:
            st.info("Database does not exist.")
            
# Main content area
tab1, tab2 = st.tabs(["💬 Chat", "📊 Chat History"])

with tab1:
    # Query input
    query = st.text_input(
        "Ask a question:",
        placeholder="What are the main technical requirements?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear Chat")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    # Process query
    if search_button and query:
        if not os.path.exists(CHROMA_PATH):
            st.error("⚠️ No documents in database. Please upload documents first!")
        else:
            with st.spinner("Searching knowledge base..."):
                response, sources = query_rag(query, top_k=top_k)
                
                # Add to history
                st.session_state.chat_history.append({
                    "query": query,
                    "response": response,
                    "sources": sources
                })
    
    # Display latest response
    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]
        
        st.markdown("### 💡 Answer")
        st.write(latest["response"])
        
        if latest["sources"]:
            st.markdown("### 📚 Sources")
            for i, source in enumerate(latest["sources"], 1):
                with st.expander(f"Source {i} - {os.path.basename(source['source'])} (Page {source['page']}) - Score: {source['score']}"):
                    st.text(source["content"])

with tab2:
    # Display chat history
    if not st.session_state.chat_history:
        st.info("No chat history yet. Ask a question to get started!")
    else:
        for i, item in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {item['query'][:50]}..."):
                st.markdown("**Question:**")
                st.write(item['query'])
                st.markdown("**Answer:**")
                st.write(item['response'])
                if item['sources']:
                    st.markdown("**Sources:**")
                    for source in item['sources']:
                        st.caption(f"• {os.path.basename(source['source'])} (Page {source['page']})")

# Footer
st.divider()
st.caption("Powered by Ollama, ChromaDB, and LangChain")
