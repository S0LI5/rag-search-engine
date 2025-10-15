import gradio as gr
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
import tempfile
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str, top_k: int = 5):
    """Query the RAG system"""
    try:
        if not os.path.exists(CHROMA_PATH):
            return "⚠️ No documents in database. Please upload documents first!", ""
        
        # Prepare the DB
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Search the DB
        results = db.similarity_search_with_score(query_text, k=top_k)
        
        if not results:
            return "No relevant information found.", ""
        
        # Build context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Create prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Get LLM response
        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)
        
        # Format sources
        sources_text = "**Sources:**\n\n"
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            sources_text += f"{i}. **{os.path.basename(source)}** (Page {page}) - Score: {score:.4f}\n"
            sources_text += f"   Preview: {doc.page_content[:150]}...\n\n"
        
        return response_text, sources_text
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

def upload_pdf(file):
    """Upload and process PDF"""
    try:
        if file is None:
            return "⚠️ Please select a file"
        
        # Create data directory
        Path(DATA_PATH).mkdir(exist_ok=True)
        
        # Get file path
        file_path = Path(file.name)
        
        # Use pypdf directly
        import pypdf
        
        # Read PDF
        documents = []
        with open(file_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                documents.append({
                    'page_content': text,
                    'metadata': {
                        'source': str(file_path),
                        'page': page_num
                    }
                })
        
        # Convert to Document objects
        from langchain.schema import Document
        doc_objects = [Document(page_content=doc['page_content'], 
                               metadata=doc['metadata']) 
                      for doc in documents]
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
        )
        chunks = text_splitter.split_documents(doc_objects)
        
        # Calculate chunk IDs
        chunks_with_ids = calculate_chunk_ids(chunks)
        
        # Add to ChromaDB
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        
        new_chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
        db.add_documents(chunks_with_ids, ids=new_chunk_ids)
        
        return f"✅ Successfully added {len(chunks)} chunks from {file_path.name}"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

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

def get_stats():
    """Get database statistics"""
    if not os.path.exists(CHROMA_PATH):
        return "📊 No database found"
    
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        results = db.get()
        
        sources = set()
        for metadata in results["metadatas"]:
            sources.add(metadata.get("source", "Unknown"))
        
        stats = f"📊 **Database Statistics:**\n\n"
        stats += f"- Total Chunks: {len(results['ids'])}\n"
        stats += f"- Total Documents: {len(sources)}\n\n"
        stats += "**Documents:**\n"
        for doc in sources:
            stats += f"  • {os.path.basename(doc)}\n"
        
        return stats
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="RAG Knowledge Base", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🤖 RAG Knowledge Base Search")
    gr.Markdown("Ask questions about your documents using AI-powered search")
    
    with gr.Tab("💬 Query"):
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="What are the main technical requirements?",
                    lines=2
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of sources to retrieve"
                )
                query_button = gr.Button("🔍 Search", variant="primary")
            
            with gr.Column(scale=1):
                stats_output = gr.Markdown(value=get_stats())
                refresh_button = gr.Button("🔄 Refresh Stats")
        
        answer_output = gr.Textbox(label="Answer", lines=5)
        sources_output = gr.Markdown(label="Sources")
        
        query_button.click(
            fn=query_rag,
            inputs=[query_input, top_k_slider],
            outputs=[answer_output, sources_output]
        )
        
        refresh_button.click(
            fn=get_stats,
            outputs=stats_output
        )
    
    with gr.Tab("📤 Upload Documents"):
        file_input = gr.File(
            label="Upload PDF Document",
            file_types=[".pdf"]
        )
        upload_button = gr.Button("📁 Process Document", variant="primary")
        upload_output = gr.Textbox(label="Status", lines=2)
        
        upload_button.click(
            fn=upload_pdf,
            inputs=file_input,
            outputs=upload_output
        )
    
    with gr.Tab("ℹ️ Info"):
        gr.Markdown("""
        ## How to Use
        
        1. **Upload Documents**: Go to the "Upload Documents" tab and upload PDF files
        2. **Ask Questions**: Go to the "Query" tab and ask questions about your documents
        3. **View Sources**: See which documents and pages were used to generate the answer
        
        ## Features
        
        - ✅ Upload multiple PDF documents
        - ✅ Semantic search using embeddings
        - ✅ AI-powered answer generation
        - ✅ Source attribution with page numbers
        - ✅ Adjustable number of sources
        
        ## Technology Stack
        
        - **Embeddings**: Ollama (nomic-embed-text)
        - **LLM**: Ollama (mistral)
        - **Vector DB**: ChromaDB
        - **Framework**: LangChain
        """)

# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
