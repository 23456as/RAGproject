import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
import os
import tempfile
from PyPDF2 import PdfReader


st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chain" not in st.session_state:
    st.session_state.chain = None


with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get free: https://console.groq.com/keys")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("## 📁 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
    
    # Check files before processing
    if uploaded_files:
        for file in uploaded_files:
            st.caption(f"📄 {file.name} ({file.size / 1024:.1f} KB)")
    
    if uploaded_files and groq_api_key and st.button("📤 Process Files"):
        with st.spinner("Processing..."):
            try:
                all_data = []
                valid_files = 0
                
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getbuffer())
                        tmp_path = tmp.name
                    
                    try:
                        # Check if PDF is encrypted
                        with open(tmp_path, "rb") as f:
                            reader = PdfReader(f)
                            if reader.is_encrypted:
                                st.sidebar.warning(f"⚠️ {file.name} is encrypted, skipping")
                                os.unlink(tmp_path)
                                continue
                            
                            # Check if PDF has pages
                            if len(reader.pages) == 0:
                                st.sidebar.warning(f"⚠️ {file.name} has no pages, skipping")
                                os.unlink(tmp_path)
                                continue
                        
                        # Load with PDFPlumberLoader
                        loader = PDFPlumberLoader(file_path=tmp_path)
                        docs = loader.load()
                        
                        if docs:
                            all_data.extend(docs)
                            valid_files += 1
                            st.sidebar.success(f"✓ {file.name} ({len(docs)} pages)")
                        else:
                            st.sidebar.warning(f"⚠️ {file.name} loaded but no text extracted")
                        
                    except Exception as e:
                        st.sidebar.error(f"✗ {file.name}: {str(e)[:50]}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                
                if not all_data:
                    st.sidebar.error("No valid documents loaded!")
                    st.stop()
                
                st.sidebar.info(f"Processing {valid_files} file(s)...")
                
                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(all_data)
                st.sidebar.success(f"✓ Created {len(chunks)} chunks")
                
                # Create embeddings
                st.sidebar.info("Loading embeddings...")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.sidebar.success("✓ Embeddings ready")
                
                # Create vector database
                st.sidebar.info("Creating vector database...")
                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                st.sidebar.success("✓ Vector DB created")
                
                # Create retriever and LLM
                retriever = vector_db.as_retriever(search_kwargs={"k": 3})
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    groq_api_key=groq_api_key
                )
                
                # Create chain
                template = """Answer based on the provided context. Be specific and concise.

Context:
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
                st.session_state.vector_db = vector_db
                st.sidebar.success("✅ System Ready!")
                st.session_state.messages = []
                
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")


st.title("💬 RAG Document Chat")


if st.session_state.chain:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about your documents...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain.invoke(user_input)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

else:
    st.info("👈 Upload PDFs in the sidebar to get started")
    st.markdown("""
    ## 🚀 Getting Started
    
    1. **Get API Key** → [https://console.groq.com/keys](https://console.groq.com/keys)
    2. **Paste Key** → In sidebar (password field)
    3. **Upload PDFs** → Click "Choose PDFs"
    4. **Process** → Click "📤 Process Files"
    5. **Chat** → Ask your questions!
    
    ### Supported PDF Types
    - ✅ Text-based PDFs
    - ✅ Scanned PDFs (image-based)
    - ✅ Multiple PDFs at once
    - ❌ Encrypted PDFs (will be skipped)
    """)
