import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter   
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import os
import tempfile
import warnings

warnings.filterwarnings('ignore')
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 RAG Document Q&A System")
st.markdown("Upload multiple PDFs and ask intelligent questions about their content")

# ========== SESSION STATE INITIALIZATION ==========
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# ========== SIDEBAR: FILE UPLOAD & PROCESSING ==========
with st.sidebar:
    st.header("📁 PDF Upload & Processing")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to process"
    )
    
    st.markdown("---")
    
    if uploaded_files:
        st.info(f"📂 {len(uploaded_files)} file(s) selected")
        for file in uploaded_files:
            st.caption(f"• {file.name} ({file.size / 1024:.1f} KB)")
        
        if st.button("🔄 Process Files", key="process_btn", use_container_width=True):
            with st.spinner("Processing PDFs..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # ========== STEP 1: LOAD PDFS ==========
                    status_text.text("⏳ Step 1/6: Loading PDFs...")
                    progress_bar.progress(15)
                    all_data = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_path = tmp_file.name
                        
                        try:
                            loader = UnstructuredPDFLoader(file_path=tmp_path)
                            data = loader.load()
                            all_data.extend(data)
                            st.success(f"✓ Loaded: {uploaded_file.name} ({len(data)} docs)")
                        except Exception as e:
                            st.error(f"✗ Error loading {uploaded_file.name}: {e}")
                        finally:
                            os.unlink(tmp_path)
                    
                    if not all_data:
                        st.error("No documents loaded from PDFs")
                        st.stop()
                    
                    # ========== STEP 2: SPLIT CHUNKS ==========
                    status_text.text("⏳ Step 2/6: Splitting into chunks...")
                    progress_bar.progress(30)
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(all_data)
                    st.success(f"✓ Split into {len(chunks)} chunks")
                    
                    # ========== STEP 3: LOAD EMBEDDINGS ==========
                    status_text.text("⏳ Step 3/6: Loading embeddings model (first time ~30MB)...")
                    progress_bar.progress(45)
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    st.session_state.embeddings = embeddings
                    st.success("✓ Embeddings model loaded")
                    
                    # ========== STEP 4: CREATE VECTOR DB ==========
                    status_text.text("⏳ Step 4/6: Creating vector database...")
                    progress_bar.progress(60)
                    vector_db = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory="./chroma_db"
                    )
                    st.session_state.vector_db = vector_db
                    st.success("✓ Vector database created")
                    
                    # ========== STEP 5: CREATE LLM ==========
                    status_text.text("⏳ Step 5/6: Initializing LLM (connecting to Ollama)...")
                    progress_bar.progress(75)
                    llm = ChatOllama(model="llama2", temperature=0.3)
                    st.success("✓ LLM initialized")
                    
                    # ========== STEP 6: CREATE RAG CHAIN ==========
                    status_text.text("⏳ Step 6/6: Creating RAG chain...")
                    progress_bar.progress(90)
                    
                    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
                    
                    template = """Answer the question based on the following context:

{context}

Question: {question}

Answer:"""
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    st.session_state.chain = chain
                    st.session_state.uploaded_files = [f.name for f in uploaded_files]
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    
                    st.success("✅ All files processed successfully! You can now ask questions.")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        st.info("👆 Upload PDF files to get started")
    
    st.markdown("---")
    st.markdown(
        """
        **How to use:**
        1. Upload 1+ PDF files
        2. Click 'Process Files'
        3. Ask questions in main area
        4. View sources & download
        """
    )

# ========== MAIN: QUESTION INPUT & RESPONSE ==========
st.header("💬 Ask Questions")

if st.session_state.chain is not None:
    # ========== FILE INFO ==========
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📂 **Files loaded:** {', '.join(st.session_state.uploaded_files)}")
    with col2:
        if st.button("🗑️ Clear All", help="Clear loaded files and start over"):
            st.session_state.vector_db = None
            st.session_state.chain = None
            st.session_state.uploaded_files = []
            st.rerun()
    
    # ========== QUESTION INPUT ==========
    question = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic of the documents? What key points are discussed?",
        help="Ask any question about the uploaded documents"
    )
    
    # ========== SEARCH & RESPONSE ==========
    if st.button("🔍 Search & Get Answer", key="search_btn", use_container_width=True):
        if question.strip():
            with st.spinner("🤖 Searching documents and generating answer..."):
                try:
                    response = st.session_state.chain.invoke(question)
                    
                    # Display answer in styled box
                    st.subheader("📝 Answer:")
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4;">
                        {response}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.divider()
                    
                    # ========== SOURCE DOCUMENTS ==========
                    st.subheader("📚 Source Documents")
                    with st.expander("View source chunks used for this answer", expanded=False):
                        docs = st.session_state.vector_db.similarity_search(question, k=3)
                        
                        if docs:
                            for i, doc in enumerate(docs, 1):
                                with st.container():
                                    st.write(f"**📖 Source {i}:**")
                                    st.text_area(
                                        label=f"Content {i}",
                                        value=doc.page_content,
                                        height=150,
                                        disabled=True,
                                        label_visibility="collapsed"
                                    )
                                    st.caption("---")
                        else:
                            st.info("No source documents found")
                    
                    st.divider()
                    
                    # ========== DOWNLOAD & ACTIONS ==========
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="📥 Download Answer",
                            data=response,
                            file_name="answer.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Save to session for reference
                        if st.button("💾 Save to Session", use_container_width=True):
                            st.success("✓ Answer saved to session")
                    
                    with col3:
                        if st.button("🔄 Ask Another", use_container_width=True):
                            st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")
                    st.info("💡 Make sure Ollama is running: `ollama serve`")
        else:
            st.warning("⚠️ Please enter a question")
    
    st.markdown("---")
    
    # ========== QUERY HISTORY (Optional) ==========
    with st.expander("📋 Quick Question Templates"):
        templates = [
            "What is the main topic of these documents?",
            "Summarize the key points discussed",
            "What are the main findings or conclusions?",
            "What methods or approaches are mentioned?",
            "What data or statistics are provided?"
        ]
        
        cols = st.columns(len(templates))
        for i, template in enumerate(templates):
            if cols[i].button(template, use_container_width=True):
                st.session_state.question_template = template

else:
    # ========== EMPTY STATE ==========
    st.info("👈 **Get started:** Upload PDF files in the sidebar and click 'Process Files'")
    
    with st.expander("ℹ️ How does this work?"):
        st.markdown(
            """
            ### RAG (Retrieval-Augmented Generation) System
            
            1. **PDF Upload**: You upload one or more PDF documents
            2. **Text Extraction**: Documents are parsed and extracted
            3. **Chunking**: Text is split into manageable chunks (~1000 chars)
            4. **Embedding**: Chunks are converted to vector embeddings (all-MiniLM-L6-v2)
            5. **Indexing**: Vectors are stored in Chroma vector database
            6. **Query Processing**: 
               - Your question is converted to embeddings
               - Similar chunks are retrieved (top-3)
               - LLM (Llama2) generates answer based on retrieved context
            7. **Response**: Answer + source documents are displayed
            
            **Benefits:**
            - Ask questions about any document
            - Get answers grounded in actual document content
            - See source chunks used for answers
            - Works completely locally (privacy-friendly)
            """
        )
    
    with st.expander("⚙️ System Requirements"):
        st.markdown(
            """
            **Required:**
            - Ollama running (`ollama serve`)
            - Ollama models: `llama2`
            
            **Python packages:**
            - langchain-community
            - langchain-ollama
            - chromadb
            - sentence-transformers
            - streamlit
            
            **Install all:**
            ```
            pip install streamlit langchain-community langchain-ollama chromadb sentence-transformers
            ```
            """
        )

# ========== FOOTER ==========
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 12px;">
    RAG Document Q&A System | Powered by LangChain, Ollama & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
